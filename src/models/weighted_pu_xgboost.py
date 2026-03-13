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

class WeightedPUClassifier:
    """
    Weighted Positive-Unlabeled Learning with XGBoost
    """
    
    def __init__(self, xgb_params=None, random_state=42):
        self.xgb_params = xgb_params or {}
        self.random_state = random_state
        self.model = None
        self.alpha = None
        self.raw_alpha = None  # Store raw alpha before clipping
        self.feature_names = None
        
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
        
        # Optional: Adjust weights based on class imbalance
        pos_ratio = len(X_pos) / (len(X_pos) + len(X_neg) + len(X_unlabeled))
        if pos_ratio < 0.1:  # If very imbalanced
            # Give positives slightly higher weight
            pos_weight = 1.0
            neg_weight = 1.0
        else:
            pos_weight = 1.0
            neg_weight = 1.0
        
        sample_weights = np.array(
            [pos_weight] * len(X_pos) + 
            [neg_weight] * len(X_neg) + 
            [unlabeled_weight] * len(X_unlabeled)
        )
        
        print(f"  Unlabeled weight: {unlabeled_weight:.3f}")
        print(f"  Positive weight: {pos_weight:.3f}")
        print(f"  Negative weight: {neg_weight:.3f}")
        print(f"  Total training samples: {len(X_all)}")
        
        return X_all, y_all, sample_weights
    
    def fit(self, X_pos, X_neg, X_unlabeled, X_val=None, y_val=None, 
            protein_groups=None, df_train=None):
        """
        Fit the Weighted PU model with protein group size weighting
        
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
        
        # Step 1: Estimate prior (now with protein weights)
        self.alpha = self.estimate_prior(
            X_pos, X_neg, X_unlabeled, 
            df_train=df_train  # Pass the training dataframe
        )
        
        # Step 2: Prepare weighted data
        X_all, y_all, pu_weights = self.prepare_weighted_data(
            X_pos, X_neg, X_unlabeled, self.alpha
        )
        
        # Step 3: Add protein group size weighting if df_train is provided
        if df_train is not None and 'Author-Protein' in df_train.columns:
            print("\n  Adding protein group size weights...")
            
            # Get protein groups for all training samples
            # We need to reconstruct the order: positives + negatives + unlabeled
            all_indices = []
            
            # Get indices for positives
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
            
            # Create protein weights (1/size) for each sample
            protein_weights = []
            for idx in all_indices:
                protein = df_train.loc[idx, 'Author-Protein']
                size = protein_sizes[protein]
                protein_weights.append(1.0 / size)
            
            protein_weights = np.array(protein_weights)
            
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
        
        # Step 4: Train model
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


def load_and_prepare_data(config_file, use_pnu=True):
    """
    Load data from config file, optionally using get_optimals for labeling
    
    Config parameters that affect this function:
    - random_undersampling: bool (default: False) - Whether to apply random undersampling
    - target_ratio: float (default: 3.0) - Target ratio of unlabeled:positive samples
    - random_seed: int (default: 42) - Random seed for reproducibility
    - include_sp_seq: bool (default: False) - Whether to include SP sequence features
    - sp_seq_file: str - Path to SP sequence features file (required if include_sp_seq=True)
    """
    with open(config_file) as f:
        cfg = json.load(f)
    
    if use_pnu:
        print("\nGenerating labels using get_optimals with PNU mode...")
        # Use get_optimals to generate labels
        df_labels = get_optimals(
            config_file, 
            plot=cfg.get("plot", False), 
            pnu_mode=True
        )
        # Save the labeled data with a default name
        labeled_path = "outputs/labeled.csv"
        df_labels.to_csv(labeled_path, index=False)
        print(f"Labels saved to: {labeled_path}")
    else:
        # Load pre-labeled data (if you ever need it)
        labeled_path = cfg.get("labeled_data", "outputs/labeled.csv")
        print(f"\nLoading pre-labeled data from: {labeled_path}")
        df_labels = pd.read_csv(labeled_path)
    
    # Load features and clusters
    print(f"Loading features from: {cfg['features_file']}")
    df_features = pd.read_csv(cfg["features_file"])
    
    print(f"Loading clusters from: {cfg['cluster_file']}")
    df_clusters = pd.read_csv(cfg["cluster_file"])
    
    # Merge labels with features and clusters
    df = df_labels.merge(df_features, on="id").merge(df_clusters, on="Author-Protein")
    
    # Track feature columns
    feature_cols = [c for c in df_features.columns if c != "id"]
    
    # Check if we should include SP sequence features
    include_sp_seq = cfg.get("include_sp_seq", False)
    
    if include_sp_seq:
        sp_seq_file = cfg.get("sp_seq_file")
        if sp_seq_file is None:
            raise ValueError("include_sp_seq=True but no sp_seq_file provided in config")
        
        print(f"\nLoading SP sequence features from: {sp_seq_file}")
        df_sp_seq = pd.read_csv(sp_seq_file)
        
        # Verify the SP sequence file has an 'id' column for merging
        if 'id' not in df_sp_seq.columns:
            raise ValueError(f"SP sequence file {sp_seq_file} must contain an 'id' column")
        
        # Get SP feature columns (excluding 'id')
        sp_feature_cols = [c for c in df_sp_seq.columns if c != "id"]
        
        print(f"  Found {len(sp_feature_cols)} SP sequence features")
        print(f"  SP features: {sp_feature_cols[:5]}{'...' if len(sp_feature_cols) > 5 else ''}")
        
        # Merge SP sequence features
        original_len = len(df)
        df = df.merge(df_sp_seq, on="id", how="left")
        
        # Check if any IDs didn't have SP features
        merged_len = len(df)
        if merged_len != original_len:
            print(f"  Warning: Merge changed dataframe size from {original_len} to {merged_len}")
        
        # Check for missing SP features
        missing_sp = df[sp_feature_cols].isnull().any(axis=1).sum()
        if missing_sp > 0:
            print(f"  Warning: {missing_sp} rows ({missing_sp/len(df)*100:.1f}%) have missing SP sequence features")
        
        # Add SP features to feature_cols
        feature_cols.extend(sp_feature_cols)
        print(f"  Added {len(sp_feature_cols)} SP sequence features to feature set")
    else:
        print("\nSkipping SP sequence features (include_sp_seq=False)")
    
    # Check if random undersampling is enabled in config
    random_undersampling = cfg.get("random_undersampling", False)
    
    if random_undersampling:
        # Get parameters from config with defaults
        target_ratio = cfg.get("target_ratio", 3.0)  # Default 1:3 positive:unlabeled
        random_seed = cfg.get("random_seed", 42)
        
        print("\n" + "="*60)
        print(f"APPLYING RANDOM UNDERSAMPLING FOR UNLABELED DATA")
        print(f"Target ratio: 1:{target_ratio} (positive:unlabeled)")
        print(f"Random seed: {random_seed}")
        print("="*60)
        
        # Make sure we're working with PNU labels
        if 'label_PNU' not in df.columns:
            print("Warning: label_PNU column not found. Using 'label' column instead.")
            df['label_PNU'] = df['label']  # Fallback
        
        # Store original counts
        original_total = len(df)
        original_pos = (df['label_PNU'] == 1).sum()
        original_neg = (df['label_PNU'] == 0).sum()
        original_unlabeled = (df['label_PNU'] == -1).sum()
        
        print(f"\nOriginal data distribution:")
        print(f"  Positives (label_PNU=1): {original_pos}")
        print(f"  Negatives (label_PNU=0): {original_neg}")
        print(f"  Unlabeled (label_PNU=-1): {original_unlabeled}")
        print(f"  Total: {original_total}")
        
        # Initialize list to store sampled dataframes per group
        sampled_dfs = []
        
        # Group by 'Author-Protein'
        for group_name, group_df in df.groupby('Author-Protein'):
            # Split group into positive, negative, and unlabeled
            positives = group_df[group_df['label_PNU'] == 1]
            negatives = group_df[group_df['label_PNU'] == 0]
            unlabeled = group_df[group_df['label_PNU'] == -1]
            
            # Calculate target unlabeled count (target_ratio × positives)
            target_unlabeled = int(len(positives) * target_ratio)
            
            # If we have more unlabeled than target, undersample
            if len(unlabeled) > target_unlabeled:
                unlabeled_sampled = unlabeled.sample(n=target_unlabeled, random_state=random_seed)
                print(f"\nGroup '{group_name}':")
                print(f"  Positives: {len(positives)}")
                print(f"  Unlabeled original: {len(unlabeled)} -> sampled: {target_unlabeled} (ratio 1:{target_unlabeled/len(positives) if len(positives) > 0 else 0:.1f})")
            else:
                # Keep all unlabeled if we don't have enough
                unlabeled_sampled = unlabeled
                if len(positives) > 0:
                    actual_ratio = len(unlabeled) / len(positives) if len(positives) > 0 else 0
                    print(f"\nGroup '{group_name}':")
                    print(f"  Positives: {len(positives)}")
                    print(f"  Unlabeled: {len(unlabeled)} (keeping all - actual ratio 1:{actual_ratio:.1f})")
                else:
                    print(f"\nGroup '{group_name}': No positives, keeping all {len(unlabeled)} unlabeled samples")
            
            # Combine positives, negatives (keep all), and sampled unlabeled
            sampled_group = pd.concat([positives, negatives, unlabeled_sampled])
            sampled_dfs.append(sampled_group)
        
        # Combine all sampled groups
        df_sampled = pd.concat(sampled_dfs, ignore_index=True)
        
        # Print final statistics
        final_pos = (df_sampled['label_PNU'] == 1).sum()
        final_neg = (df_sampled['label_PNU'] == 0).sum()
        final_unlabeled = (df_sampled['label_PNU'] == -1).sum()
        final_total = len(df_sampled)
        
        print("\n" + "-"*40)
        print("FINAL DATA DISTRIBUTION AFTER UNDERSAMPLING:")
        print("-"*40)
        print(f"  Positives (label_PNU=1): {final_pos}")
        print(f"  Negatives (label_PNU=0): {final_neg}")
        print(f"  Unlabeled (label_PNU=-1): {final_unlabeled}")
        print(f"  Total: {final_total}")
        
        if final_pos > 0:
            print(f"  Positive:Unlabeled ratio: 1:{final_unlabeled/final_pos:.2f} (target: 1:{target_ratio})")
            print(f"  Reduction in unlabeled samples: {original_unlabeled - final_unlabeled} ({((original_unlabeled - final_unlabeled)/original_unlabeled*100):.1f}% reduction)")
        
        if final_neg > 0 and final_pos > 0:
            print(f"  Positive:Negative ratio: 1:{final_neg/final_pos:.2f}")
        
        # Replace original df with sampled version
        df = df_sampled
    else:
        print("\n" + "="*60)
        print("SKIPPING RANDOM UNDERSAMPLING (disabled in config)")
        print("="*60)
        print(f"Using all data: {len(df)} total samples")
    
    print(f"\n📊 Final dataset: {len(df)} samples with {len(feature_cols)} features")
    
    return df, feature_cols, cfg


def create_cluster_folds(df, n_folds=5):
    """
    Create cluster-aware cross-validation folds
    Ensures entire clusters stay together in either train or test
    """
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
        # Only count TRUE positives and negatives for statistics
        if 'label_PNU' in fold_data.columns:
            true_data = fold_data[fold_data['label_PNU'] != -1]
            pos_count = (true_data['label'] == 1).sum()
            neg_count = (true_data['label'] == 0).sum()
            unlabeled_count = (fold_data['label_PNU'] == -1).sum()
            print(f"  Fold {fold_i}: {len(fold_data)} total ({unlabeled_count} unlabeled, {pos_count} pos, {neg_count} neg)")
        else:
            pos_count = (fold_data['label'] == 1).sum()
            print(f"  Fold {fold_i}: {len(fold_data)} samples, {pos_count} positives ({pos_count/len(fold_data)*100:.1f}%)")
    
    return df


def evaluate_predictions(y_true, y_pred_proba):
    """
    Evaluate model predictions on TRUE labels only
    
    Returns comprehensive metrics including:
    - PR-AUC
    - F1 score (best threshold)
    - Precision, Recall
    - Specificity, Accuracy, Balanced Accuracy
    - Confusion Matrix
    """
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Find best F1 score
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = thresholds[best_idx]
    
    # Calculate predictions at best threshold
    y_pred_bin = (y_pred_proba > best_thresh).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    
    # Calculate all metrics
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_accuracy = (recall_val + specificity) / 2
    
    return {
        'pr_auc': pr_auc,
        'best_f1': best_f1,
        'best_thresh': best_thresh,
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