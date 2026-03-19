#!/usr/bin/env python3
"""
XGBoost Ranking Model for SP Selection - Training Script
Place this in: scripts/run_ranking.py
Run with: python scripts/run_ranking.py --config config.json
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import from the core module
from src.models.weighted_pu_xgboost import (
    XGBRanker,
    load_and_prepare_data, 
    create_cluster_folds,
    evaluate_ranking,
    analyze_per_protein_performance,
    print_protein_summary
)


def create_output_dir(base_dir="outputs"):
    """Create a timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Results will be saved to: {output_dir}")
    return output_dir


def print_fold_distribution(df, n_folds):
    """Print detailed fold distribution with ranking perspective"""
    print("\n" + "-"*70)
    print("📊 FOLD DISTRIBUTION (TEST SETS) - RANKING VIEW")
    print("-"*70)
    
    for fold_i in range(n_folds):
        fold_data = df[df["fold"] == fold_i]
        
        print(f"\nFold {fold_i}:")
        print(f"  📦 Total: {len(fold_data)} samples")
        print(f"  🔢 Proteins: {fold_data['Author-Protein'].nunique()}")
        
        # Count positives per protein
        pos_per_protein = fold_data[fold_data['relevance'] == 1].groupby('Author-Protein').size()
        print(f"  ✅ Proteins with positives: {len(pos_per_protein)}")
        if len(pos_per_protein) > 0:
            print(f"  📈 Avg positives per protein: {pos_per_protein.mean():.2f}")
        print(f"  📊 Total positives: {fold_data['relevance'].sum()}")


def print_fold_ranking_summary(fold_i, metrics):
    """Print ranking metrics for a fold"""
    print(f"\n📊 Fold {fold_i} Ranking Results:")
    print(f"  🎯 NDCG@1: {metrics['ndcg_1']:.4f}")
    print(f"  🎯 NDCG@3: {metrics['ndcg_3']:.4f}")
    print(f"  🎯 NDCG@5: {metrics['ndcg_5']:.4f}")
    print(f"  🎯 NDCG@10: {metrics['ndcg_10']:.4f}")
    if 'mrr' in metrics:
        print(f"  🎯 MRR: {metrics['mrr']:.4f}")


def calculate_mrr(y_true_grouped, y_score_grouped):
    """Calculate Mean Reciprocal Rank"""
    rr_scores = []
    for y_true, y_score in zip(y_true_grouped, y_score_grouped):
        if np.sum(y_true) == 0:
            rr_scores.append(0.0)
        else:
            # Get rank of first positive
            sorted_indices = np.argsort(y_score)[::-1]
            for rank, idx in enumerate(sorted_indices, 1):
                if y_true[idx] == 1:
                    rr_scores.append(1.0 / rank)
                    break
    return np.mean(rr_scores)


def print_final_ranking_summary(fold_results, output_dir):
    """Print final ranking summary across all folds"""
    print("\n" + "="*70)
    print("🎯 FINAL RANKING RESULTS ACROSS ALL FOLDS")
    print("="*70)
    
    metrics = ['ndcg_1', 'ndcg_3', 'ndcg_5', 'ndcg_10', 'mrr']
    summary = {}
    
    for metric in metrics:
        values = [r[metric] for r in fold_results]
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    # Print summary
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print(f"  Mean: {summary[metric]['mean']:.4f} (±{summary[metric]['std']:.4f})")
        print(f"  Range: [{summary[metric]['min']:.4f} - {summary[metric]['max']:.4f}]")
        print(f"  Per fold: {[f'{v:.4f}' for v in summary[metric]['values']]}")
    
    # Save summary to file
    summary_df = pd.DataFrame({
        'fold': range(len(fold_results)),
        **{metric: [r[metric] for r in fold_results] for metric in metrics}
    })
    
    summary_path = os.path.join(output_dir, "ranking_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📁 Detailed results saved to: {summary_path}")
    
    return summary


def save_model(model, fold_i, models_dir):
    """Save trained model for later use"""
    model_path = os.path.join(models_dir, f"ranker_fold_{fold_i}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model_path


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost Ranker for SP selection')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Base directory to save outputs')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')
    parser.add_argument('--save_models', action='store_true', default=True,
                       help='Save trained models for later use')
    parser.add_argument('--use_pnu', action='store_true', default=True,
                       help='Use get_optimals with PNU mode for labeling')
    parser.add_argument('--balance_proteins', action='store_true', default=True,
                       help='Apply protein balancing weights (1/group_size)')
    parser.add_argument('--create_interactions', action='store_true', default=True,
                       help='Create protein-SP interaction features')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 XGBOOST RANKING FOR SIGNAL PEPTIDE SELECTION")
    print("="*70)
    
    # Create timestamped output directory
    output_dir = create_output_dir(args.output_dir)
    
    # Create subdirectories
    fold_splits_dir = os.path.join(output_dir, "fold_splits")
    models_dir = os.path.join(output_dir, "models")
    protein_analysis_dir = os.path.join(output_dir, "protein_analysis")
    os.makedirs(fold_splits_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(protein_analysis_dir, exist_ok=True)
    
    # Load data with interactions
    print("\n📂 Loading and preparing data...")
    df, feature_cols, cfg = load_and_prepare_data(
        args.config, 
        use_pnu=args.use_pnu,
        create_interactions=args.create_interactions
    )
    n_folds = cfg.get("n_folds", 5)
    xgb_params = cfg.get("xgboost_params", {})
    
    print(f"\n📊 Dataset shape: {df.shape}")
    print(f"🔢 Features: {len(feature_cols)}")
    
    # Create relevance column for ranking (1 for positives, 0 for everything else)
    if 'label_PNU' in df.columns:
        df['relevance'] = (df['label_PNU'] == 1).astype(int)
        print(f"\n📋 Ranking Relevance Distribution:")
        print(f"  ✅ Relevant (positives): {df['relevance'].sum()}")
        print(f"  ❌ Non-relevant: {(df['relevance'] == 0).sum()}")
        print(f"  🔢 Unique proteins: {df['Author-Protein'].nunique()}")
        print(f"  📈 Avg relevant per protein: {df.groupby('Author-Protein')['relevance'].sum().mean():.2f}")
    else:
        df['relevance'] = df['label']
        print(f"\n📋 Using 'label' as relevance")
    
    # Create folds
    print("\n🔀 Creating cluster-based folds...")
    df = create_cluster_folds(df, n_folds)
    print_fold_distribution(df, n_folds)
    
    # Save fold assignment
    fold_assignment_path = os.path.join(output_dir, "fold_assignment.csv")
    columns_to_save = ['id', 'Author-Protein', 'cluster', 'label', 'relevance', 'fold']
    if 'label_PNU' in df.columns:
        columns_to_save.append('label_PNU')
    df[columns_to_save].to_csv(fold_assignment_path, index=False)
    print(f"\n📁 Fold assignment saved to: {fold_assignment_path}")
    
    # Save feature names
    feature_names_path = os.path.join(output_dir, "feature_names.csv")
    pd.DataFrame({'feature': feature_cols}).to_csv(feature_names_path, index=False)
    
    # Train per fold
    fold_results = []
    all_predictions = []
    all_models = []
    all_protein_performances = []
    
    for fold_i in range(n_folds):
        print(f"\n{'='*70}")
        print(f"🔷 FOLD {fold_i + 1}/{n_folds}")
        print(f"{'='*70}")
        
        # Split data
        train_idx = df[df["fold"] != fold_i].index
        test_idx = df[df["fold"] == fold_i].index
        
        X_train = df.loc[train_idx, feature_cols].values
        y_train = df.loc[train_idx, 'relevance'].values
        groups_train = df.loc[train_idx, 'Author-Protein'].values
        
        X_test = df.loc[test_idx, feature_cols].values
        y_test = df.loc[test_idx, 'relevance'].values
        groups_test = df.loc[test_idx, 'Author-Protein'].values
        test_ids = df.loc[test_idx, 'id'].values if 'id' in df.columns else None
        
        print(f"\n🎯 Training set: {len(X_train)} samples, {df.loc[train_idx, 'Author-Protein'].nunique()} proteins")
        print(f"🎯 Test set: {len(X_test)} samples, {df.loc[test_idx, 'Author-Protein'].nunique()} proteins")
        
        # In run_ranking.py, when calculating protein balancing weights:
        # Calculate protein balancing weights
        if args.balance_proteins:
            print("\n⚖️ Calculating protein balancing weights...")
            
            # Get unique proteins in training set in the order they'll appear
            protein_order = df.loc[train_idx].groupby('Author-Protein').size().index.tolist()
            
            # Calculate weights (inverse of size)
            protein_sizes = df.loc[train_idx].groupby('Author-Protein').size()
            protein_weights = 1.0 / protein_sizes
            
            # Normalize
            protein_weights = protein_weights * (len(protein_weights) / protein_weights.sum())
            
            # Create group_weight array in the correct order
            group_weight = np.array([protein_weights[prot] for prot in protein_order])
            
            print(f"    Group weights - min: {group_weight.min():.4f}, "
                f"max: {group_weight.max():.4f}, mean: {group_weight.mean():.4f}")
        else:
            group_weight = None

        # Train ranker
        ranker = XGBRanker(xgb_params=xgb_params)
        ranker.fit(
            X=X_train,
            y=y_train,
            groups=groups_train,
            group_weight=group_weight,  # Pass group weights
            X_val=X_test,
            y_val=y_test,
            groups_val=groups_test  # No weights for validation
        )
        all_models.append(ranker)
        
        # Save model if requested
        if args.save_models:
            model_path = save_model(ranker, fold_i, models_dir)
            print(f"  💾 Model saved to: {model_path}")
        
        # Predict scores
        y_pred_scores = ranker.predict(X_test)
        
        # Group predictions by protein for evaluation
        test_df = df.loc[test_idx].copy()
        test_df['pred_score'] = y_pred_scores
        grouped = test_df.groupby('Author-Protein')
        
        y_true_groups = [group['relevance'].values for _, group in grouped]
        y_score_groups = [group['pred_score'].values for _, group in grouped]
        
        # Calculate ranking metrics
        metrics = {
            'ndcg_1': evaluate_ranking(y_true_groups, y_score_groups, k=1),
            'ndcg_3': evaluate_ranking(y_true_groups, y_score_groups, k=3),
            'ndcg_5': evaluate_ranking(y_true_groups, y_score_groups, k=5),
            'ndcg_10': evaluate_ranking(y_true_groups, y_score_groups, k=10),
            'mrr': calculate_mrr(y_true_groups, y_score_groups)
        }
        
        print_fold_ranking_summary(fold_i, metrics)
        fold_results.append(metrics)
        
        # Analyze per-protein performance
        protein_perf = analyze_per_protein_performance(
            test_df, y_test, y_pred_scores, 
            protein_col='Author-Protein', fold_i=fold_i
        )
        all_protein_performances.append(protein_perf)
        
        # Print per-protein summary for this fold
        print_protein_summary(protein_perf, top_n=5)
        
        # Save per-protein results for this fold
        protein_perf.to_csv(
            os.path.join(protein_analysis_dir, f"fold_{fold_i}_protein_performance.csv"), 
            index=False
        )
        
        # Save predictions if requested
        if args.save_predictions and test_ids is not None:
            fold_predictions = pd.DataFrame({
                'id': test_ids,
                'protein': test_df['Author-Protein'].values,
                'true_relevance': y_test,
                'pred_score': y_pred_scores,
                'fold': fold_i
            })
            all_predictions.append(fold_predictions)
        
        # Feature importance for first fold
        if fold_i == 0:
            print("\n🔥 Top 20 most important features:")
            importance_df = ranker.get_feature_importance(feature_cols)
            print(importance_df.head(20))
            importance_df.to_csv(
                os.path.join(output_dir, "feature_importance.csv"), 
                index=False
            )
    
    # Final summary
    summary = print_final_ranking_summary(fold_results, output_dir)
    
    # Combine and analyze all per-protein performances
    if all_protein_performances:
        all_protein_df = pd.concat(all_protein_performances, ignore_index=True)
        all_protein_df.to_csv(os.path.join(output_dir, "all_protein_performance.csv"), index=False)
        
        print("\n" + "="*70)
        print("📊 OVERALL PER-PROTEIN PERFORMANCE SUMMARY")
        print("="*70)
        
        # Summary statistics
        print(f"\nAverage across all protein-fold combinations ({len(all_protein_df)} entries):")
        for metric in ['ndcg_1', 'ndcg_3', 'ndcg_5', 'ndcg_10', 'spearman', 'reciprocal_rank']:
            if metric in all_protein_df.columns:
                mean_val = all_protein_df[metric].mean()
                std_val = all_protein_df[metric].std()
                print(f"  {metric}: {mean_val:.4f} (±{std_val:.4f})")
        
        # Per-protein consistency (if a protein appears in multiple folds)
        if 'protein' in all_protein_df.columns:
            protein_consistency = all_protein_df.groupby('protein').agg({
                'ndcg_5': ['mean', 'std'],
                'n_positives': 'first',
                'fold': 'count'
            }).round(4)
            protein_consistency.columns = ['ndcg5_mean', 'ndcg5_std', 'n_positives', 'n_folds']
            protein_consistency = protein_consistency.sort_values('ndcg5_mean', ascending=False)
            protein_consistency.to_csv(os.path.join(output_dir, "protein_consistency.csv"))
            
            print("\n📈 Best performing proteins (avg NDCG@5):")
            for prot, row in protein_consistency.head(10).iterrows():
                print(f"  {prot}: {row['ndcg5_mean']:.3f} (±{row['ndcg5_std']:.3f}) "
                      f"with {int(row['n_positives'])} positives")
    
    # Create fold split files
    print("\n" + "="*70)
    print("📁 CREATING FOLD SPLIT FILES")
    print("="*70)
    
    for fold_i in range(n_folds):
        train_idx = df[df["fold"] != fold_i].index
        test_idx = df[df["fold"] == fold_i].index
        
        fold_split_df = pd.DataFrame({
            'id': df.loc[train_idx, 'id'].values.tolist() + df.loc[test_idx, 'id'].values.tolist(),
            'relevance': df.loc[train_idx, 'relevance'].values.tolist() + df.loc[test_idx, 'relevance'].values.tolist(),
            'protein': df.loc[train_idx, 'Author-Protein'].values.tolist() + df.loc[test_idx, 'Author-Protein'].values.tolist(),
            'cluster': df.loc[train_idx, 'cluster'].values.tolist() + df.loc[test_idx, 'cluster'].values.tolist(),
            'train': [1] * len(train_idx) + [0] * len(test_idx)
        })
        
        fold_split_path = os.path.join(fold_splits_dir, f"fold_{fold_i}_split.csv")
        fold_split_df.to_csv(fold_split_path, index=False)
        
        print(f"  ✅ Fold {fold_i}: {len(train_idx)} train, {len(test_idx)} test samples")
    
    # Save all predictions
    if args.save_predictions and all_predictions:
        all_pred_df = pd.concat(all_predictions, ignore_index=True)
        all_pred_df.to_csv(os.path.join(output_dir, "all_predictions.csv"), index=False)
        print(f"\n📁 Predictions saved to: {os.path.join(output_dir, 'all_predictions.csv')}")
    
    # Save configuration
    config_copy_path = os.path.join(output_dir, "config_used.json")
    with open(config_copy_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    
    # Save model information
    model_info = {
        'n_folds': n_folds,
        'features_used': len(feature_cols),
        'feature_names': feature_cols,
        'ranking_metrics': {k: {'mean': v['mean'], 'std': v['std']} 
                           for k, v in summary.items()},
        'saved_model_paths': [f"models/ranker_fold_{i}.pkl" for i in range(n_folds)] if args.save_models else None
    }
    model_info_path = os.path.join(output_dir, "model_info.json")
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n📁 Fold split files saved in: {fold_splits_dir}")
    print(f"📁 Models saved in: {models_dir}")
    print(f"📁 Protein analysis saved in: {protein_analysis_dir}")
    
    print(f"\n{'='*70}")
    print("✅ RANKING TRAINING COMPLETE!")
    print(f"📁 All results saved to: {output_dir}")
    print('='*70)
    
    return summary


def load_saved_model(model_path):
    """Helper function to load a saved model for later analysis"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_for_new_protein(model, protein_features, all_sp_features, sp_ids=None):
    """
    Predict SP rankings for a new protein
    
    Parameters:
    -----------
    model : trained XGBRanker
    protein_features : array-like, shape (n_features,)
        Feature vector for the new protein
    all_sp_features : array-like, shape (n_sp, n_features)
        Feature vectors for all candidate SPs
    sp_ids : list, optional
        IDs for the SPs
    
    Returns:
    --------
    DataFrame with SPs ranked by predicted suitability
    """
    # Combine protein features with each SP (assuming concatenated features)
    n_sp = len(all_sp_features)
    protein_features_tiled = np.tile(protein_features, (n_sp, 1))
    
    # If features are concatenated protein+SP, you might need to combine them
    # This is a placeholder - adjust to your actual feature structure
    combined_features = np.hstack([protein_features_tiled, all_sp_features])
    
    # Predict scores
    scores = model.predict(combined_features)
    
    # Create ranked DataFrame
    results = pd.DataFrame({
        'sp_id': sp_ids if sp_ids is not None else range(n_sp),
        'score': scores
    }).sort_values('score', ascending=False).reset_index(drop=True)
    
    results['rank'] = range(1, len(results) + 1)
    
    return results


if __name__ == "__main__":
    main()