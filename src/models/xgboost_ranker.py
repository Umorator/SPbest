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

# Import get_optimals_continuous from the same src directory
from src.get_optimals_continuous import get_optimals as get_optimals_continuous

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import ndcg_score
# ==================== NEW XGBRanker CLASS ====================
class XGBRanker:
    """
    XGBoost Ranker for SP ranking per protein.
    """
    def __init__(self, xgb_params=None, random_state=42):
        self.xgb_params = xgb_params or {}
        self.random_state = random_state
        self.model = None
        self.feature_names = None

    def fit(self, X, y, groups, group_weight=None,
        X_val=None, y_val=None, groups_val=None):
        """
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,) – relevance scores
        groups : array-like, shape (n_samples,) – group (protein) identifier
        group_weight : array-like, shape (n_groups,) optional – one weight per group
        X_val, y_val, groups_val : optional validation data (no weights needed)
        """
        # Sort data by group
        df = pd.DataFrame({'group': groups, 'y': y})
        df['idx'] = np.arange(len(X))
        df_sorted = df.sort_values('group')
        sorted_idx = df_sorted['idx'].values
        X_sorted = X[sorted_idx]
        y_sorted = y[sorted_idx]
        
        # Get unique groups and their sizes in sorted order
        unique_groups, group_sizes = np.unique(df_sorted['group'].values, return_counts=True)
        
        dtrain = xgb.DMatrix(X_sorted, label=y_sorted)
        dtrain.set_group(group_sizes)
        
        # Apply group weights if provided - FIXED: set at group level, not instance level
        if group_weight is not None:
            print(f"  Using group weights: {len(group_weight)} groups")
            print(f"    Group weight range: [{group_weight.min():.4f}, {group_weight.max():.4f}]")
            
            # For ranking objectives, set weights at the group level
            # The weights should be aligned with the group structure
            dtrain.set_weight(group_weight)
            
            print(f"    Group weights shape: {group_weight.shape}")
            print(f"    Number of groups: {len(group_sizes)}")

        evals = [(dtrain, 'train')]
        if X_val is not None:
            # Sort validation data
            df_val = pd.DataFrame({'group': groups_val, 'y': y_val})
            df_val['idx'] = np.arange(len(X_val))
            df_val_sorted = df_val.sort_values('group')
            val_sorted_idx = df_val_sorted['idx'].values
            X_val_sorted = X_val[val_sorted_idx]
            y_val_sorted = y_val[val_sorted_idx]
            
            group_sizes_val = df_val_sorted.groupby('group').size().values
            
            dval = xgb.DMatrix(X_val_sorted, label=y_val_sorted)
            dval.set_group(group_sizes_val)
            # No weights for validation set - just for monitoring
            
            evals.append((dval, 'val'))

        params = self.xgb_params.copy()
        params['objective'] = 'rank:pairwise'
        params.setdefault('eval_metric', 'ndcg')
        params.pop('use_label_encoder', None)

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 500),
            evals=evals,
            early_stopping_rounds=params.get('early_stopping_rounds', 50),
            verbose_eval=False
        )
        return self
    
    def predict(self, X):
        """Return ranking scores."""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def get_feature_importance(self, feature_names=None):
        """Return feature importance (gain)."""
        imp = self.model.get_score(importance_type='gain')
        if feature_names is None:
            feature_names = self.feature_names
        if feature_names:
            df = pd.DataFrame([
                {'feature': feature_names[int(f[1:])], 'importance': v}
                for f, v in imp.items()
            ]).sort_values('importance', ascending=False)
        else:
            df = pd.DataFrame([
                {'feature': f, 'importance': v}
                for f, v in imp.items()
            ]).sort_values('importance', ascending=False)
        return df


def load_and_prepare_data(config_file,
                         use_pnu=True,
                         create_interactions=True,
                         use_continuous_relevance=True,
                         max_features=1000):

    np.random.seed(42)

    with open(config_file) as f:
        cfg = json.load(f)

    # ===== LABELS =====
    print("\n🔬 Generating labels...")
    df_labels = get_optimals_continuous(
        config_file,
        plot=cfg.get("plot", False),
        pnu_mode=use_pnu,
        use_continuous_relevance=use_continuous_relevance
    )

    # ===== LOAD FEATURES =====
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

        prot_base = [c.replace('protein_', '') for c in prot_feature_cols]
        sp_base = [c.replace('sp_', '') for c in non_proba_sp_cols]
        common = set(prot_base).intersection(sp_base)

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

        if common:
            prot_mat = np.column_stack([df[f"protein_{n}"] for n in common])
            sp_mat = np.column_stack([df[f"sp_{n}"] for n in common])

            prot_norm = prot_mat / (np.linalg.norm(prot_mat, axis=1, keepdims=True) + 1e-8)
            sp_norm = sp_mat / (np.linalg.norm(sp_mat, axis=1, keepdims=True) + 1e-8)

            df['global_cosine_similarity'] = np.sum(prot_norm * sp_norm, axis=1)
            df['global_euclidean_distance'] = -np.linalg.norm(prot_mat - sp_mat, axis=1)

            interaction_features += ['global_cosine_similarity', 'global_euclidean_distance']

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
        y = df['relevance'].values

        importance_acc = np.zeros(len(valid_feats))

        for fold in range(5):
            print(f"  Fold {fold+1}/5")

            train_mask = df["fold"] != fold

            X_train = X[train_mask]
            y_train = y[train_mask]

            model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            importance_acc += model.feature_importances_

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

    return df, final_features, cfg

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

def cross_validated_feature_selection_cluster(df, feature_names,
                                              target_col='relevance',
                                              n_folds=5,
                                              max_features=1000,
                                              random_state=42):
    """
    Cluster-aware cross-validated feature selection (NO leakage)
    Uses precomputed 'fold' column
    """
    import numpy as np
    import xgboost as xgb
    
    X_all = df[feature_names].values
    y_all = df[target_col].values
    
    feature_importance_accumulator = np.zeros(len(feature_names))
    
    print(f"\n🔁 Running {n_folds}-fold CLUSTER-AWARE CV for feature selection...")

    for fold in range(n_folds):
        print(f"  Fold {fold+1}/{n_folds}")
        
        train_mask = df["fold"] != fold
        
        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        feature_importance_accumulator += model.feature_importances_

    # Average importance
    avg_importance = feature_importance_accumulator / n_folds
    
    # Select top features
    top_indices = np.argsort(avg_importance)[-max_features:]
    selected_features = [feature_names[i] for i in top_indices]
    
    print(f"\n📊 Importance range: [{avg_importance.min():.4f}, {avg_importance.max():.4f}]")
    
    return selected_features, avg_importance

def evaluate_ranking(y_true_grouped, y_score_grouped, k=None):
    ndcg_scores = []
    for y_true, y_score in zip(y_true_grouped, y_score_grouped):
        if np.sum(y_true) == 0:
            # No relevant items → NDCG is 0 by definition
            ndcg_scores.append(0.0)
        else:
            ndcg = ndcg_score([y_true], [y_score], k=k)
            ndcg_scores.append(ndcg)
    return np.mean(ndcg_scores)


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

def analyze_per_protein_performance(df_fold, y_true, y_pred_scores, protein_col='Author-Protein', fold_i=None):
    """
    Analyze ranking performance for each protein individually
    
    Parameters:
    -----------
    df_fold : DataFrame with protein information
    y_true : array of true relevance values
    y_pred_scores : array of predicted scores
    protein_col : column name for protein identifiers
    fold_i : fold number for context
    
    Returns:
    --------
    DataFrame with per-protein metrics
    """
    from scipy.stats import spearmanr
    
    # Create temporary dataframe with predictions
    temp_df = df_fold.copy()
    temp_df['true_relevance'] = y_true
    temp_df['pred_score'] = y_pred_scores
    
    results = []
    
    for protein in temp_df[protein_col].unique():
        protein_data = temp_df[temp_df[protein_col] == protein]
        
        prot_true = protein_data['true_relevance'].values
        prot_pred = protein_data['pred_score'].values
        n_samples = len(prot_true)
        
        # For continuous relevance, we don't need to check n_positives > 0
        # We can evaluate ranking even if all values are low
        
        # Calculate NDCG at various k
        ndcg_1 = ndcg_score([prot_true], [prot_pred], k=1) if n_samples >= 1 else 0
        ndcg_3 = ndcg_score([prot_true], [prot_pred], k=3) if n_samples >= 3 else ndcg_1
        ndcg_5 = ndcg_score([prot_true], [prot_pred], k=5) if n_samples >= 5 else ndcg_3
        ndcg_10 = ndcg_score([prot_true], [prot_pred], k=10) if n_samples >= 10 else ndcg_5
        
        # Calculate Spearman correlation (for all SPs)
        try:
            spearman_corr, spearman_p = spearmanr(prot_true, prot_pred)
        except:
            spearman_corr, spearman_p = 0, 1.0
        
        # For continuous relevance, we can also calculate:
        # - Pearson correlation (linear relationship)
        # - MSE/MAE between normalized scores
        try:
            from scipy.stats import pearsonr
            pearson_corr, pearson_p = pearsonr(prot_true, prot_pred)
        except:
            pearson_corr, pearson_p = 0, 1.0
        
        # Calculate if top prediction has highest true relevance
        top_idx = np.argmax(prot_pred)
        top_true_value = prot_true[top_idx]
        max_true_value = prot_true.max()
        top_is_best = abs(top_true_value - max_true_value) < 1e-6
        
        # Calculate reciprocal rank based on threshold
        # For continuous, we can define "relevant" as top 25% or above median
        threshold = np.percentile(prot_true, 75) if len(prot_true) > 3 else prot_true.max() * 0.8
        sorted_indices = np.argsort(prot_pred)[::-1]
        reciprocal_rank = 0
        for rank, idx in enumerate(sorted_indices, 1):
            if prot_true[idx] >= threshold:
                reciprocal_rank = 1.0 / rank
                break
        
        results.append({
            'fold': fold_i,
            'protein': protein,
            'n_samples': n_samples,
            'true_mean': prot_true.mean(),
            'true_std': prot_true.std(),
            'ndcg_1': ndcg_1,
            'ndcg_3': ndcg_3,
            'ndcg_5': ndcg_5,
            'ndcg_10': ndcg_10,
            'spearman': spearman_corr,
            'spearman_p': spearman_p,
            'pearson': pearson_corr,
            'pearson_p': pearson_p,
            'top_is_best': top_is_best,
            'reciprocal_rank': reciprocal_rank
        })
    
    return pd.DataFrame(results)


def print_protein_summary(protein_df, top_n=10):
    """
    Print a summary of per-protein performance
    """
    print(f"\n{'='*70}")
    print(f"📊 PER-PROTEIN PERFORMANCE ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n✅ Best performing proteins (NDCG@5):")
    best = protein_df.nlargest(min(top_n, len(protein_df)), 'ndcg_5')
    for _, row in best.iterrows():
        print(f"  {row['protein']}: NDCG@5={row['ndcg_5']:.3f}, "
              f"n={int(row['n_samples'])}, "
              f"Spearman={row['spearman']:.3f}")
    
    print(f"\n❌ Worst performing proteins (NDCG@5):")
    worst = protein_df.nsmallest(min(top_n, len(protein_df)), 'ndcg_5')
    for _, row in worst.iterrows():
        print(f"  {row['protein']}: NDCG@5={row['ndcg_5']:.3f}, "
              f"n={int(row['n_samples'])}, "
              f"Spearman={row['spearman']:.3f}")
    
    # Correlation with sample size
    if 'n_samples' in protein_df.columns and 'ndcg_5' in protein_df.columns:
        print(f"\n📈 Correlation between sample size and NDCG@5: "
              f"{protein_df['n_samples'].corr(protein_df['ndcg_5']):.3f}")