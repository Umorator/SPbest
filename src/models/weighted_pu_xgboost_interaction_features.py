"""
Weighted PU Learning with XGBoost - Core Module
Place this in: src/weighted_pu_xgboost.py
"""

import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, precision_recall_curve, auc, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Import get_optimals from the same src directory
from src.get_optimals import get_optimals

"""
Weighted PU Learning with XGBoost - Core Module with SMOTE
Place this in: src/weighted_pu_xgboost.py
"""

import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, precision_recall_curve, auc, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Import get_optimals from the same src directory
from src.get_optimals import get_optimals


class WeightedPUClassifier:
    """
    Weighted Positive-Unlabeled Learning with XGBoost
    Now with SMOTE for handling class imbalance
    """
    
    def __init__(self, xgb_params=None, random_state=42, use_smote=True, smote_k_neighbors=3):
        self.xgb_params = xgb_params or {}
        self.random_state = random_state
        self.model = None
        self.alpha = None
        self.raw_alpha = None  # Store raw alpha before clipping
        self.feature_names = None
        self.use_smote = use_smote
        self.smote_k_neighbors = smote_k_neighbors
        self.smote = SMOTE(random_state=random_state, k_neighbors=smote_k_neighbors)
        
    def estimate_prior(self, X_pos, X_neg, X_unlabeled, df_train=None):
        """
        Step 1: Estimate the proportion of positives in unlabeled data
        using Elkan-Noto method with protein group size weighting
        
        The Elkan-Noto estimator: alpha = mean(p_unlabeled) / mean(p_positives)
        where p_positives are predictions on known positive examples
        """
        print("  Estimating prior (alpha) using Elkan-Noto method...")
        
        # Combine known data
        X_known = np.vstack([X_pos, X_neg])
        y_known = np.array([1]*len(X_pos) + [0]*len(X_neg))
        
        # Create protein weights for known data if df_train provided
        sample_weights = None
        if df_train is not None and 'Author-Protein' in df_train.columns:
            print("  Using protein group weights for prior estimation...")
            
            # Get indices for known data (positives + negatives)
            pos_indices = df_train[df_train['label_PNU'] == 1].index
            neg_indices = df_train[df_train['label_PNU'] == 0].index
            known_indices = list(pos_indices) + list(neg_indices)
            
            # Calculate protein group sizes
            protein_sizes = df_train.groupby('Author-Protein').size()
            
            # Create protein weights (1/size) for known samples
            protein_weights = []
            for idx in known_indices:
                protein = df_train.loc[idx, 'Author-Protein']
                size = protein_sizes[protein]
                protein_weights.append(1.0 / size)
            
            sample_weights = np.array(protein_weights)
            # Normalize
            sample_weights = sample_weights * (len(sample_weights) / sample_weights.sum())
            
            print(f"    Protein weights - min: {sample_weights.min():.4f}, "
                f"max: {sample_weights.max():.4f}, mean: {sample_weights.mean():.4f}")
        
        # Use simpler model to avoid overfitting
        temp_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            random_state=self.random_state,
            verbosity=0,
            use_label_encoder=False,
            n_jobs=-1
        )
        
        try:
            # Train with weights if provided
            if sample_weights is not None:
                temp_model.fit(X_known, y_known, sample_weight=sample_weights)
            else:
                temp_model.fit(X_known, y_known)
            
            # Get predictions on known positives (for calibration)
            pos_probs = temp_model.predict_proba(X_pos)[:, 1]
            mean_pos_prob = np.mean(pos_probs)
            
            # Get predictions on unlabeled data
            unlabeled_probs = temp_model.predict_proba(X_unlabeled)[:, 1]
            mean_unlabeled_prob = np.mean(unlabeled_probs)
            
            # Elkan-Noto estimator: alpha = mean(unlabeled) / mean(positives)
            # This corrects for classifier bias
            if mean_pos_prob > 0:
                raw_alpha = mean_unlabeled_prob / mean_pos_prob
            else:
                print("    Warning: Mean positive probability is zero, using fallback")
                raw_alpha = mean_unlabeled_prob
            
            # Store raw alpha before clipping
            self.raw_alpha = raw_alpha
            
            # Apply reasonable bounds
            alpha = np.clip(raw_alpha, 0.01, 0.99)
            
            print(f"  Mean prob on positives: {mean_pos_prob:.3f}")
            print(f"  Mean prob on unlabeled: {mean_unlabeled_prob:.3f}")
            print(f"  Raw alpha (unlabeled/positives): {raw_alpha:.3f}")
            print(f"  Clipped alpha: {alpha:.3f}")
            
        except Exception as e:
            print(f"  Prior estimation failed: {e}, using default 0.3")
            alpha = 0.3
            self.raw_alpha = alpha
        
        return alpha
    
    def prepare_weighted_data(self, X_pos, X_neg, X_unlabeled, alpha):
        """
        Step 2: Combine all data with appropriate labels and weights
        """
        # Combine features
        X_all = np.vstack([X_pos, X_neg, X_unlabeled])
        
        # Create labels (unlabeled treated as negative)
        y_all = np.array(
            [1] * len(X_pos) + 
            [0] * len(X_neg) + 
            [0] * len(X_unlabeled)
        )
        
        # Calculate weights: unlabeled get (1-alpha), known get 1
        unlabeled_weight = 1 - alpha
        
        sample_weights = np.array(
            [1.0] * len(X_pos) + 
            [1.0] * len(X_neg) + 
            [unlabeled_weight] * len(X_unlabeled)
        )
        
        print(f"  Unlabeled weight: {unlabeled_weight:.3f}")
        print(f"  Total training samples: {len(X_all)}")
        print(f"  Positive samples (original): {len(X_pos)}")
        print(f"  Negative samples (original): {len(X_neg) + len(X_unlabeled)}")
        
        return X_all, y_all, sample_weights
    
    def _apply_smote(self, X_pos, X_neg, X_unlabeled, pos_weight=1.0):
        """
        Apply SMOTE to balance the positive class
        
        Returns augmented data with labels and weights
        """
        # Combine known negatives and unlabeled as the negative class
        X_negative = np.vstack([X_neg, X_unlabeled]) if len(X_neg) > 0 else X_unlabeled
        
        # Prepare data for SMOTE
        X_combined = np.vstack([X_pos, X_negative])
        y_combined = np.array([1] * len(X_pos) + [0] * len(X_negative))
        
        # Calculate target number of positives (aim for 70% of negative count)
        target_pos_count = int(len(X_negative) * 0.7)
        
        if len(X_pos) < target_pos_count and len(X_pos) >= self.smote_k_neighbors:
            print(f"  Applying SMOTE: {len(X_pos)} → {target_pos_count} positives")
            
            # Set sampling strategy
            self.smote.set_params(sampling_strategy={1: target_pos_count})
            
            try:
                # Apply SMOTE
                X_resampled, y_resampled = self.smote.fit_resample(X_combined, y_combined)
                
                # Separate back into positives and negatives
                pos_mask_resampled = y_resampled == 1
                X_pos_augmented = X_resampled[pos_mask_resampled]
                
                # The negatives remain unchanged (we don't want to oversample negatives)
                X_negative_augmented = X_negative
                
                print(f"    SMOTE successful. New positive count: {len(X_pos_augmented)}")
                
                # Create weights (original positives get weight 1, synthetic get lower weight)
                pos_weights = np.array([pos_weight] * len(X_pos) + 
                                      [0.5] * (len(X_pos_augmented) - len(X_pos)))
                
                return X_pos_augmented, X_negative_augmented, pos_weights
                
            except Exception as e:
                print(f"    SMOTE failed: {e}, using original data")
                return X_pos, X_negative, np.array([pos_weight] * len(X_pos))
        else:
            print(f"  SMOTE not applied (positives: {len(X_pos)}, needed: {target_pos_count})")
            return X_pos, X_negative, np.array([pos_weight] * len(X_pos))
    
    def fit(self, X_pos, X_neg, X_unlabeled, X_val=None, y_val=None, 
            protein_groups=None, df_train=None):
        """
        Fit the Weighted PU model with protein group size weighting and SMOTE
        
        Parameters:
        -----------
        X_pos : array-like
            Positive examples
        X_neg : array-like
            Known negative examples (can be empty)
        X_unlabeled : array-like
            Unlabeled examples
        X_val, y_val : array-like, optional
            Validation set for early stopping
        protein_groups : array-like, optional
            Protein group labels for each sample (in same order as combined data)
        df_train : DataFrame, optional
            Original training dataframe with 'Author-Protein' column
        """
        # Handle case with no known negatives
        if len(X_neg) == 0:
            print("\nNo known negatives found. Creating pseudo-negatives...")
            # More sophisticated pseudo-negative selection
            if len(X_unlabeled) > len(X_pos) * 3:
                n_pseudo_neg = len(X_pos) * 2
            else:
                n_pseudo_neg = len(X_unlabeled) // 2
            
            np.random.seed(self.random_state)
            neg_indices = np.random.choice(len(X_unlabeled), n_pseudo_neg, replace=False)
            X_neg = X_unlabeled[neg_indices]
            # Remove these from unlabeled
            unlabeled_mask = np.ones(len(X_unlabeled), dtype=bool)
            unlabeled_mask[neg_indices] = False
            X_unlabeled = X_unlabeled[unlabeled_mask]
            
            print(f"  Using {len(X_neg)} samples as known negatives")
            print(f"  Remaining unlabeled: {len(X_unlabeled)}")
        
        # Step 1: Estimate prior
        self.alpha = self.estimate_prior(
            X_pos, X_neg, X_unlabeled, 
            df_train=df_train
        )
        
        # Step 2: Apply SMOTE if enabled
        if self.use_smote:
            print("\n  Applying SMOTE to balance positive class...")
            pos_weight = 1.0  # Base weight for positives
            X_pos_aug, X_negative, pos_weights = self._apply_smote(
                X_pos, X_neg, X_unlabeled, pos_weight
            )
            
            # Separate negatives back into known and unlabeled if needed for weights
            if len(X_neg) > 0:
                X_neg_aug = X_negative[:len(X_neg)]
                X_unlabeled_aug = X_negative[len(X_neg):]
            else:
                X_neg_aug = np.array([]).reshape(0, X_pos.shape[1])
                X_unlabeled_aug = X_negative
        else:
            print("\n  SMOTE disabled, using original data...")
            X_pos_aug = X_pos
            X_neg_aug = X_neg
            X_unlabeled_aug = X_unlabeled
            pos_weights = np.array([1.0] * len(X_pos))
        
        # Step 3: Prepare weighted data with augmented positives
        print("\n  Preparing weighted data with SMOTE-augmented positives...")
        
        # Combine all data
        X_all = np.vstack([X_pos_aug, X_neg_aug, X_unlabeled_aug])
        
        # Create labels
        y_all = np.array(
            [1] * len(X_pos_aug) + 
            [0] * len(X_neg_aug) + 
            [0] * len(X_unlabeled_aug)
        )
        
        # Calculate PU weights (unlabeled get lower weight)
        unlabeled_weight = 1 - self.alpha
        pu_weights = np.array(
            list(pos_weights) +  # Use SMOTE weights for positives
            [1.0] * len(X_neg_aug) + 
            [unlabeled_weight] * len(X_unlabeled_aug)
        )
        
        print(f"  Positive samples after SMOTE: {len(X_pos_aug)} (original: {len(X_pos)})")
        print(f"  Known negatives: {len(X_neg_aug)}")
        print(f"  Unlabeled samples: {len(X_unlabeled_aug)}")
        print(f"  Unlabeled weight: {unlabeled_weight:.3f}")
        
        # Step 4: Add protein group size weighting if df_train is provided
        if df_train is not None and 'Author-Protein' in df_train.columns:
            print("\n  Adding protein group size weights...")
            
            # For SMOTE-augmented data, we need to handle protein weights carefully
            # Original positives get their protein weights, synthetic ones get average weight
            all_indices = []
            
            # Get indices for original positives
            pos_indices = df_train[df_train['label_PNU'] == 1].index
            all_indices.extend(pos_indices)
            
            # Get indices for negatives
            neg_indices = df_train[df_train['label_PNU'] == 0].index
            all_indices.extend(neg_indices)
            
            # Get indices for unlabeled
            unlabeled_indices = df_train[df_train['label_PNU'] == -1].index
            all_indices.extend(unlabeled_indices)
            
            # Calculate protein group sizes
            protein_sizes = df_train.groupby('Author-Protein').size()
            
            # Create protein weights for original samples
            protein_weights_original = []
            for idx in all_indices:
                protein = df_train.loc[idx, 'Author-Protein']
                size = protein_sizes[protein]
                protein_weights_original.append(1.0 / size)
            
            protein_weights_original = np.array(protein_weights_original)
            
            # For SMOTE-augmented positives, assign average protein weight
            if self.use_smote and len(X_pos_aug) > len(X_pos):
                avg_protein_weight = np.mean(protein_weights_original[:len(X_pos)])
                n_synthetic = len(X_pos_aug) - len(X_pos)
                
                # Combine protein weights: original + synthetic + negatives + unlabeled
                protein_weights = np.concatenate([
                    protein_weights_original[:len(X_pos)],  # Original positives
                    [avg_protein_weight] * n_synthetic,     # Synthetic positives
                    protein_weights_original[len(X_pos):len(X_pos)+len(X_neg)],  # Original negatives
                    protein_weights_original[len(X_pos)+len(X_neg):]  # Original unlabeled
                ])
            else:
                protein_weights = protein_weights_original
            
            # Combine with PU weights (multiply)
            final_weights = pu_weights * protein_weights
            
            # Normalize to keep scale reasonable
            final_weights = final_weights * (len(final_weights) / final_weights.sum())
            
            print(f"  Protein weights - min: {protein_weights.min():.4f}, "
                f"max: {protein_weights.max():.4f}, "
                f"mean: {protein_weights.mean():.4f}")
            print(f"  Final weights - min: {final_weights.min():.4f}, "
                f"max: {final_weights.max():.4f}, "
                f"mean: {final_weights.mean():.4f}")
        else:
            final_weights = pu_weights
            print("\n  No protein group weighting applied")
        
        # Step 5: Train model
        print("\n  Training XGBoost with weights...")
        
        # Prepare default params if none provided
        params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'gamma': 0.2,
            'reg_alpha': 0.5,
            'reg_lambda': 2,
            'random_state': self.random_state,
            'verbosity': 0,
            'eval_metric': 'aucpr',
            'early_stopping_rounds': 50,
            'use_label_encoder': False
        }
        # Update with user params
        params.update(self.xgb_params)
        
        # Create DMatrix with combined weights
        dtrain = xgb.DMatrix(X_all, label=y_all, weight=final_weights)
        
        # Setup validation if provided
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))
        
        # Train model
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=evals,
            verbose_eval=False,
            early_stopping_rounds=params['early_stopping_rounds'] if len(evals) > 1 else None
        )
        
        # Store feature names if provided
        if hasattr(X_pos, 'columns'):
            self.feature_names = X_pos.columns.tolist()
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance scores"""
        if feature_names is None:
            feature_names = self.feature_names
        
        importance_scores = self.model.get_score(importance_type='gain')
        
        if feature_names:
            importance_df = pd.DataFrame([
                {'feature': feature_names[int(f[1:])], 'importance': score}
                for f, score in importance_scores.items()
            ]).sort_values('importance', ascending=False)
        else:
            importance_df = pd.DataFrame([
                {'feature': f, 'importance': score}
                for f, score in importance_scores.items()
            ]).sort_values('importance', ascending=False)
        
        return importance_df


def load_and_prepare_data(config_file, 
                         use_pnu=True, 
                         create_interactions=True,
                         max_features=500):
    """
    Load data from config file with enhanced feature engineering
    
    Parameters:
    -----------
    config_file : str
        Path to configuration JSON file
    use_pnu : bool
        Whether to use PNU mode for labeling
    create_interactions : bool
        Whether to create interaction features between protein and SP features
    max_features : int
        Maximum number of features to keep after selection
    
    Returns:
    --------
    df : DataFrame
        Complete dataset with all features
    feature_cols : list
        List of selected feature column names
    cfg : dict
        Configuration dictionary
    """
    np.random.seed(42)
    
    with open(config_file) as f:
        cfg = json.load(f)
    
    # ===== LABELS =====
    print("\n🔬 Generating labels using get_optimals...")
    df_labels = get_optimals(
        config_file,
        plot=cfg.get("plot", False),
        pnu_mode=use_pnu
    )
    
    # ===== LOAD FEATURES =====
    print(f"\n📂 Loading features...")
    df_prot_features = pd.read_csv(cfg["protein_features_file"])
    df_sp_features = pd.read_csv(cfg["sp_features_file"])
    df_clusters = pd.read_csv(cfg["cluster_file"])
    
    # ===== MERGE =====
    print("\n🔗 Merging data...")
    df = df_labels.merge(df_sp_features, on='id', how='left')
    df = df.merge(df_prot_features, on='id', how='left', suffixes=('_sp', '_prot'))
    
    # ===== FEATURE GROUPS =====
    sp_feature_cols = [c for c in df_sp_features.columns if c != 'id' and c in df.columns]
    prot_feature_cols = [c for c in df_prot_features.columns if c != 'id' and c in df.columns]
    
    proba_cols = [c for c in sp_feature_cols if '_proba' in c or c.endswith('_proba')]
    non_proba_sp_cols = [c for c in sp_feature_cols if c not in proba_cols]
    
    # ===== INTERACTIONS =====
    interaction_features = []
    
    if create_interactions and prot_feature_cols and non_proba_sp_cols:
        print("\n🔄 Creating interaction features...")
        
        # Extract base feature names (remove prefixes)
        prot_base = [c.replace('protein_', '') for c in prot_feature_cols]
        sp_base = [c.replace('sp_', '') for c in non_proba_sp_cols]
        common = set(prot_base).intersection(sp_base)
        
        # Create pairwise interactions for common features
        for name in common:
            p_col = f"protein_{name}"
            s_col = f"sp_{name}"
            
            if p_col in df.columns and s_col in df.columns:
                df[f'interact_{name}_product'] = df[p_col] * df[s_col]
                df[f'interact_{name}_diff'] = np.abs(df[p_col] - df[s_col])
                df[f'interact_{name}_sum'] = df[p_col] + df[s_col]
                df[f'interact_{name}_ratio'] = df[p_col] / (df[s_col] + 1e-8)
                
                interaction_features += [
                    f'interact_{name}_product',
                    f'interact_{name}_diff',
                    f'interact_{name}_sum',
                    f'interact_{name}_ratio'
                ]
        
        # Create global similarity metrics
        if common:
            prot_mat = np.column_stack([df[f"protein_{n}"] for n in common])
            sp_mat = np.column_stack([df[f"sp_{n}"] for n in common])
            
            # Cosine similarity
            prot_norm = prot_mat / (np.linalg.norm(prot_mat, axis=1, keepdims=True) + 1e-8)
            sp_norm = sp_mat / (np.linalg.norm(sp_mat, axis=1, keepdims=True) + 1e-8)
            df['global_cosine_similarity'] = np.sum(prot_norm * sp_norm, axis=1)
            
            # Euclidean distance (negative for consistency - larger is better)
            df['global_euclidean_distance'] = -np.linalg.norm(prot_mat - sp_mat, axis=1)
            
            interaction_features += ['global_cosine_similarity', 'global_euclidean_distance']
        
        print(f"  Created {len(interaction_features)} interaction features")
    
    # ===== FEATURE LIST =====
    all_features = proba_cols + non_proba_sp_cols + prot_feature_cols + interaction_features
    
    print(f"\n📊 Initial features: {len(all_features)}")
    
    # Remove constant features
    valid_feats = [c for c in all_features if c in df.columns and df[c].nunique() > 1]
    print(f"🧹 Non-constant features: {len(valid_feats)}")
    
    # ===== MERGE CLUSTERS + CREATE FOLDS =====
    df = df.merge(df_clusters, on='Author-Protein', how='left')
    df = create_cluster_folds(df, n_folds=5)
    
    # ===== CLUSTER-AWARE FEATURE SELECTION =====
    if len(valid_feats) > max_features:
        print(f"\n🔬 Cluster-aware feature selection ({len(valid_feats)} → {max_features})")
        
        X = df[valid_feats].values
        
        # Use label for feature selection (binary classification)
        if use_pnu and 'label_PNU' in df.columns:
            # For PNU mode, use only labeled data for feature selection
            labeled_mask = df['label_PNU'] != -1
            X_labeled = X[labeled_mask]
            y_labeled = df.loc[labeled_mask, 'label'].values
            fold_labels = df.loc[labeled_mask, 'fold'].values
        else:
            X_labeled = X
            y_labeled = df['label'].values
            fold_labels = df['fold'].values
        
        importance_acc = np.zeros(len(valid_feats))
        
        # Use the existing cluster-aware folds for feature selection
        for fold in range(5):  # Using the same 5 folds from create_cluster_folds
            print(f"  Selection fold {fold+1}/5")
            
            # Train on all folds except current one
            train_mask = fold_labels != fold
            X_train_fold = X_labeled[train_mask]
            y_train_fold = y_labeled[train_mask]
            
            if len(X_train_fold) > 0 and len(np.unique(y_train_fold)) > 1:
                model = xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                model.fit(X_train_fold, y_train_fold)
                importance_acc += model.feature_importances_
            else:
                print(f"    Warning: Fold {fold} has insufficient training data")
        
        # Average importance across folds
        importance_avg = importance_acc / 5
        top_idx = np.argsort(importance_avg)[-max_features:]
        selected_features = [valid_feats[i] for i in top_idx]
        
        print(f"📊 Importance range: [{importance_avg.min():.4f}, {importance_avg.max():.4f}]")
    else:
        selected_features = valid_feats
        print(f"\n📊 Keeping all {len(selected_features)} features")
    
    # ===== FINAL CLEANUP =====
    final_features = [c for c in selected_features if df[c].nunique() > 1]
    
    print(f"\n✅ FINAL FEATURE COUNT: {len(final_features)}")
    if len(final_features) <= 20:
        print(f"   Features: {', '.join(final_features[:10])}...")
    
    return df, final_features, cfg


def create_cluster_folds(df, n_folds=5):
    """
    Create cluster-aware cross-validation folds
    Ensures entire clusters stay together in either train or test
    """
    if "fold" in df.columns:
        df = df.drop("fold", axis=1)
    
    df["fold"] = -1
    
    # Get unique clusters
    clusters = df["cluster"].unique()
    
    # Shuffle clusters randomly
    np.random.seed(42)
    np.random.shuffle(clusters)
    
    # Assign each entire cluster to a fold
    for i, cluster in enumerate(clusters):
        fold_num = i % n_folds
        cluster_mask = df["cluster"] == cluster
        df.loc[cluster_mask, "fold"] = fold_num
    
    # Print fold statistics
    print("\nFold distribution (TEST sets):")
    for fold_i in range(n_folds):
        fold_data = df[df["fold"] == fold_i]
        if 'label_PNU' in fold_data.columns:
            true_data = fold_data[fold_data['label_PNU'] != -1]
            pos_count = (true_data['label'] == 1).sum() if 'label' in true_data.columns else 0
            neg_count = (true_data['label'] == 0).sum() if 'label' in true_data.columns else 0
            unlabeled_count = (fold_data['label_PNU'] == -1).sum()
            print(f"  Fold {fold_i}: {len(fold_data)} total ({unlabeled_count} unlabeled, {pos_count} pos, {neg_count} neg)")
        else:
            pos_count = (fold_data['label'] == 1).sum() if 'label' in fold_data.columns else 0
            print(f"  Fold {fold_i}: {len(fold_data)} samples, {pos_count} positives ({pos_count/len(fold_data)*100:.1f}%)")
    
    return df


def evaluate_predictions(y_true, y_pred_proba, threshold=0.5):
    """
    Evaluate model predictions on TRUE labels only at a fixed threshold
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities
    threshold : float, default=0.5
        Fixed threshold for binary classification
    
    Returns comprehensive metrics including:
    - PR-AUC
    - F1 score (at fixed threshold)
    - Precision, Recall
    - Specificity, Accuracy, Balanced Accuracy
    - Confusion Matrix
    """
    # Calculate precision-recall curve (still useful for PR-AUC)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Calculate predictions at fixed threshold
    y_pred_bin = (y_pred_proba > threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    
    # Calculate all metrics
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_accuracy = (recall_val + specificity) / 2
    
    # Calculate F1 at fixed threshold
    if precision_val + recall_val > 0:
        f1_score_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)
    else:
        f1_score_val = 0.0
    
    return {
        'pr_auc': pr_auc,
        'f1_score': f1_score_val,  # Now at fixed threshold
        'threshold_used': threshold,  # Store the threshold used
        'precision': precision_val,
        'recall': recall_val,
        'specificity': specificity,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'y_pred_proba': y_pred_proba,
        'y_pred_bin': y_pred_bin
    }


def split_pu_data(df, label_col='label_PNU'):
    """
    Split data into positives, known negatives, and unlabeled based on PNU labels
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with PNU labels (1=positive, 0=negative, -1=unlabeled)
    label_col : str
        Name of the column containing PNU labels
    
    Returns:
    --------
    pos_mask, neg_mask, unlabeled_mask
    """
    # Create mask for each category
    pos_mask = df[label_col] == 1
    neg_mask = df[label_col] == 0
    unlabeled_mask = df[label_col] == -1
    
    print(f"\nPU Data Split:")
    print(f"  Positives (label=1): {pos_mask.sum()}")
    print(f"  Known Negatives (label=0): {neg_mask.sum()}")
    print(f"  Unlabeled (label=-1): {unlabeled_mask.sum()}")
    
    return pos_mask, neg_mask, unlabeled_mask