#!/usr/bin/env python3
"""
Weighted PU Learning with XGBoost - Training Script
Place this in: scripts/run_weighted_pu.py
Run with: python scripts/run_weighted_pu.py --config config.json
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.weighted_pu_xgboost_interaction_features import (
    WeightedPUClassifier, 
    load_and_prepare_data, 
    create_cluster_folds,
    evaluate_predictions,
    split_pu_data
)


def create_output_dir(base_dir="outputs"):
    """Create a timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Results will be saved to: {output_dir}")
    return output_dir


def print_fold_distribution(df, n_folds):
    """Print detailed fold distribution"""
    print("\n" + "-"*70)
    print("📊 FOLD DISTRIBUTION (TEST SETS)")
    print("-"*70)
    
    for fold_i in range(n_folds):
        fold_data = df[df["fold"] == fold_i]
        
        if 'label_PNU' in fold_data.columns:
            # Get true labels (excluding unlabeled)
            true_data = fold_data[fold_data['label_PNU'] != -1]
            pos_count = (true_data['label'] == 1).sum() if 'label' in true_data.columns else 0
            neg_count = (true_data['label'] == 0).sum() if 'label' in true_data.columns else 0
            unlabeled_count = (fold_data['label_PNU'] == -1).sum()
            
            print(f"\nFold {fold_i}:")
            print(f"  📦 Total: {len(fold_data)} samples")
            print(f"  ✅ True Positives: {pos_count}")
            print(f"  ❌ True Negatives: {neg_count}")
            print(f"  ❓ Unlabeled: {unlabeled_count}")
            if pos_count + neg_count > 0:
                pos_ratio = pos_count/(pos_count+neg_count)*100
                print(f"  📈 Positive Ratio (in true data): {pos_ratio:.1f}%")
        else:
            pos_count = (fold_data['label'] == 1).sum() if 'label' in fold_data.columns else 0
            print(f"Fold {fold_i}: {len(fold_data)} samples, {pos_count} positives ({pos_count/len(fold_data)*100:.1f}%)")


def print_fold_summary(fold_i, results):
    """Print comprehensive fold results"""
    print(f"\n📊 Fold {fold_i} Results:")
    print(f"  🎯 PR-AUC: {results['pr_auc']:.4f}")
    print(f"  📈 F1 Score: {results['f1_score']:.4f} (threshold={results['threshold_used']:.2f})")
    print(f"  ✅ Precision: {results['precision']:.4f}")
    print(f"  🔍 Recall: {results['recall']:.4f}")
    print(f"  📐 Accuracy: {results['accuracy']:.4f}")
    print(f"  🛡️  Specificity: {results['specificity']:.4f}")
    print(f"  ⚖️  Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    
    # Print confusion matrix
    cm = results['confusion_matrix']
    print(f"  📊 Confusion Matrix:")
    print(f"     TN: {cm['true_negatives']:4d}  FP: {cm['false_positives']:4d}")
    print(f"     FN: {cm['false_negatives']:4d}  TP: {cm['true_positives']:4d}")


def print_final_summary(fold_results, output_dir):
    """Print final summary across all folds and save to file"""
    print("\n" + "="*70)
    print("🎯 FINAL RESULTS ACROSS ALL FOLDS")
    print("="*70)
    
    # Collect all metrics
    metrics = ['pr_auc', 'f1_score', 'precision', 'recall', 'accuracy', 'specificity', 'balanced_accuracy']
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
        'pr_auc': [r['pr_auc'] for r in fold_results],
        'f1': [r['f1_score'] for r in fold_results],
        'precision': [r['precision'] for r in fold_results],
        'recall': [r['recall'] for r in fold_results],
        'accuracy': [r['accuracy'] for r in fold_results],
        'specificity': [r['specificity'] for r in fold_results],
        'balanced_accuracy': [r['balanced_accuracy'] for r in fold_results],
        'threshold_used': [r['threshold_used'] for r in fold_results]
    })
    
    summary_path = os.path.join(output_dir, "fold_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📁 Detailed results saved to: {summary_path}")
    
    # Save mean results as text
    mean_results_path = os.path.join(output_dir, "mean_results.txt")
    with open(mean_results_path, 'w', encoding='utf-8') as f:
        f.write("FINAL RESULTS - WEIGHTED PU LEARNING\n")
        f.write("="*40 + "\n\n")
        for metric in metrics:
            f.write(f"{metric.upper()}: {summary[metric]['mean']:.4f} (±{summary[metric]['std']:.4f})\n")
            f.write(f"  Range: [{summary[metric]['min']:.4f} - {summary[metric]['max']:.4f}]\n\n")
    
    return summary


def save_model(model, fold_i, models_dir):
    """Save trained model for later use"""
    model_path = os.path.join(models_dir, f"model_fold_{fold_i}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model_path


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost with Weighted PU learning')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Base directory to save outputs (subfolder with date will be created)')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')
    parser.add_argument('--save_models', action='store_true', default=True,
                       help='Save trained models for later use')
    parser.add_argument('--use_pnu', action='store_true', default=True,
                       help='Use get_optimals with PNU mode for labeling')
    parser.add_argument('--create_interactions', action='store_true', default=True,
                       help='Create interaction features between protein and SP features')
    parser.add_argument('--max_features', type=int, default=1000,
                       help='Maximum number of features to keep after selection')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 WEIGHTED PU LEARNING WITH XGBOOST")
    print("="*70)
    
    # Create timestamped output directory
    output_dir = create_output_dir(args.output_dir)
    
    # Create subdirectories
    fold_splits_dir = os.path.join(output_dir, "fold_splits")
    models_dir = os.path.join(output_dir, "models")
    features_dir = os.path.join(output_dir, "features")
    os.makedirs(fold_splits_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    # Load config to check parameters
    with open(args.config) as f:
        cfg = json.load(f)
    
    # Use config values if available, otherwise use command line args
    create_interactions = cfg.get("create_interactions", args.create_interactions)
    max_features = cfg.get("max_features", args.max_features)
    
    # Load data with enhanced feature engineering
    print("\n📂 Loading and preparing data with feature engineering...")
    df, feature_cols, cfg = load_and_prepare_data(
        args.config, 
        use_pnu=args.use_pnu,
        create_interactions=create_interactions,
        max_features=max_features
    )
    
    n_folds = cfg.get("n_folds", 5)
    xgb_params = cfg.get("xgboost_params", {})
    
    print(f"\n📊 Dataset shape: {df.shape}")
    print(f"🔢 Features: {len(feature_cols)}")
    
    # Save feature information
    feature_info = {
        'total_features': len(feature_cols),
        'feature_names': feature_cols,
        'interactions_created': create_interactions
    }
    
    # Categorize features by type
    proba_features = [f for f in feature_cols if '_proba' in f]
    sp_features = [f for f in feature_cols if f.startswith('sp_') and '_proba' not in f]
    protein_features = [f for f in feature_cols if f.startswith('protein_')]
    interaction_features = [f for f in feature_cols if f.startswith('interact_') or 
                           f in ['global_cosine_similarity', 'global_euclidean_distance']]
    
    feature_categories = {
        'probability_features': len(proba_features),
        'sp_non_proba_features': len(sp_features),
        'protein_features': len(protein_features),
        'interaction_features': len(interaction_features)
    }
    
    print("\n📊 Feature Categories:")
    for cat, count in feature_categories.items():
        print(f"  {cat}: {count}")
    
    # Save feature categorization
    with open(os.path.join(features_dir, "feature_categories.json"), 'w') as f:
        json.dump(feature_categories, f, indent=2)
    
    # Save all feature names by category
    pd.DataFrame({'feature': proba_features, 'type': 'probability'}).to_csv(
        os.path.join(features_dir, "probability_features.csv"), index=False)
    pd.DataFrame({'feature': sp_features, 'type': 'sp_non_proba'}).to_csv(
        os.path.join(features_dir, "sp_features.csv"), index=False)
    pd.DataFrame({'feature': protein_features, 'type': 'protein'}).to_csv(
        os.path.join(features_dir, "protein_features.csv"), index=False)
    pd.DataFrame({'feature': interaction_features, 'type': 'interaction'}).to_csv(
        os.path.join(features_dir, "interaction_features.csv"), index=False)
    
    # Check if we have PNU labels
    if 'label_PNU' in df.columns:
        print(f"\n📋 PNU Label Distribution:")
        print(f"  ✅ Positives (1): {(df['label_PNU'] == 1).sum()}")
        print(f"  ❌ Negatives (0): {(df['label_PNU'] == 0).sum()}")
        print(f"  ❓ Unlabeled (-1): {(df['label_PNU'] == -1).sum()}")
        label_col = 'label_PNU'
    else:
        print(f"\n📋 Standard Label Distribution:")
        print(f"  ✅ Positives (1): {(df['label'] == 1).sum()}")
        print(f"  ❓ Negatives/Unlabeled (0): {(df['label'] == 0).sum()}")
        label_col = 'label'
    
    # Create folds
    print("\n🔀 Creating cluster-based folds...")
    df = create_cluster_folds(df, n_folds)
    print_fold_distribution(df, n_folds)
    
    # Save fold assignment
    fold_assignment_path = os.path.join(output_dir, "fold_assignment.csv")
    columns_to_save = ['id', 'Author-Protein', 'cluster', 'label', 'fold']
    if 'label_PNU' in df.columns:
        columns_to_save.append('label_PNU')
    df[columns_to_save].to_csv(fold_assignment_path, index=False)
    print(f"\n📁 Fold assignment saved to: {fold_assignment_path}")
    
    # Save feature names for later use
    feature_names_path = os.path.join(output_dir, "feature_names.csv")
    pd.DataFrame({'feature': feature_cols}).to_csv(feature_names_path, index=False)
    
    # Train per fold
    fold_results = []
    all_predictions = []
    all_models = []
    alphas = []
    raw_alphas = []
    
    for fold_i in range(n_folds):
        print(f"\n{'='*70}")
        print(f"🔷 FOLD {fold_i + 1}/{n_folds}")
        print(f"{'='*70}")
        
        # Split data
        train_idx = df[df["fold"] != fold_i].index
        test_idx = df[df["fold"] == fold_i].index
        
        X_train = df.loc[train_idx, feature_cols].values
        X_test = df.loc[test_idx, feature_cols].values
        
        # For evaluation, we ONLY use true labels (ignore unlabeled in test set)
        if 'label_PNU' in df.columns:
            # Get true labels for test set (excluding unlabeled)
            test_true_mask = df.loc[test_idx, 'label_PNU'] != -1
            X_test_true = X_test[test_true_mask]
            y_test_true = df.loc[test_idx, 'label'].values[test_true_mask]
            test_ids_true = df.loc[test_idx, 'id'].values[test_true_mask] if 'id' in df.columns else None
            test_true_indices = test_idx[test_true_mask]
            
            print(f"\n🎯 Test set (true data only): {len(X_test_true)} samples")
            print(f"   ✅ Positives: {(y_test_true == 1).sum()}")
            print(f"   ❌ Negatives: {(y_test_true == 0).sum()}")
        else:
            X_test_true = X_test
            y_test_true = df.loc[test_idx, 'label'].values
            test_ids_true = df.loc[test_idx, 'id'].values if 'id' in df.columns else None
            test_true_indices = test_idx
            test_true_mask = np.ones(len(X_test), dtype=bool)
        
        # Separate training data for PU learning
        if label_col == 'label_PNU':
            pos_mask, neg_mask, unlabeled_mask = split_pu_data(
                df.loc[train_idx], label_col=label_col
            )
            
            X_pos = X_train[pos_mask]
            X_neg = X_train[neg_mask] if neg_mask.any() else np.array([]).reshape(0, X_train.shape[1])
            X_unlabeled = X_train[unlabeled_mask] if unlabeled_mask.any() else np.array([]).reshape(0, X_train.shape[1])
            
            pos_indices = train_idx[pos_mask]
            neg_indices = train_idx[neg_mask] if neg_mask.any() else np.array([])
            unlabeled_indices = train_idx[unlabeled_mask] if unlabeled_mask.any() else np.array([])
            
        else:
            # Standard approach: positives are label=1, everything else is unlabeled
            pos_mask = df.loc[train_idx, 'label'] == 1
            X_pos = X_train[pos_mask]
            X_neg = np.array([]).reshape(0, X_train.shape[1])
            X_unlabeled = X_train[~pos_mask]
            
            pos_indices = train_idx[pos_mask]
            neg_indices = np.array([])
            unlabeled_indices = train_idx[~pos_mask]
        
        # Train Weighted PU model with protein group weighting
        pu_model = WeightedPUClassifier(xgb_params=xgb_params,use_smote=True,smote_k_neighbors=5)

        pu_model.fit(
            X_pos, X_neg, X_unlabeled, 
            X_val=X_test_true, y_val=y_test_true,
            df_train=df.loc[train_idx]  # Pass the training dataframe with Author-Protein
        )
        
        # Store model and alpha
        all_models.append(pu_model)
        alphas.append(pu_model.alpha)
        if hasattr(pu_model, 'raw_alpha'):
            raw_alphas.append(pu_model.raw_alpha)
        
        # Save model if requested
        if args.save_models:
            model_path = save_model(pu_model, fold_i, models_dir)
            print(f"  💾 Model saved to: {model_path}")
        
        # Predict only on true test data
        y_pred_proba = pu_model.predict_proba(X_test_true)
        
        # Evaluate only on true labels
        results = evaluate_predictions(y_test_true, y_pred_proba, threshold=0.5)
        print_fold_summary(fold_i, results)
        
        fold_results.append(results)
        
        # Save fold predictions if requested
        if args.save_predictions and test_ids_true is not None:
            fold_predictions = pd.DataFrame({
                'id': test_ids_true,
                'index': test_true_indices,
                'true_label': y_test_true,
                'pred_proba': y_pred_proba,
                'pred_label': results['y_pred_bin'],
                'fold': fold_i
            })
            all_predictions.append(fold_predictions)
        
        # Feature importance for first fold
        if fold_i == 0:
            print("\n🔥 Top 20 most important features:")
            importance_df = pu_model.get_feature_importance(feature_cols)
            print(importance_df.head(20))
            
            # Save feature importance
            importance_df.to_csv(
                os.path.join(output_dir, "feature_importance.csv"), 
                index=False
            )
            
            # Also save categorized importance
            importance_df['feature_type'] = importance_df['feature'].apply(
                lambda x: 'interaction' if x.startswith('interact_') or x in ['global_cosine_similarity', 'global_euclidean_distance']
                else 'probability' if '_proba' in x
                else 'sp' if x.startswith('sp_')
                else 'protein' if x.startswith('protein_')
                else 'other'
            )
            
            # Save importance by category
            for ftype in importance_df['feature_type'].unique():
                type_importance = importance_df[importance_df['feature_type'] == ftype]
                type_importance.to_csv(
                    os.path.join(features_dir, f"importance_{ftype}.csv"),
                    index=False
                )
    
    # Final summary
    summary = print_final_summary(fold_results, output_dir)
    
    # Print alpha information
    print("\n" + "="*70)
    print("📊 PRIOR (ALPHA) INFORMATION")
    print("="*70)
    print(f"Alphas per fold: {[f'{a:.3f}' for a in alphas]}")
    print(f"Mean alpha: {np.mean(alphas):.3f} (±{np.std(alphas):.3f})")
    if raw_alphas:
        print(f"Raw alphas per fold: {[f'{a:.3f}' for a in raw_alphas]}")
        print(f"Mean raw alpha: {np.mean(raw_alphas):.3f} (±{np.std(raw_alphas):.3f})")
    
    # Create fold split files with real IDs
    print("\n" + "="*70)
    print("📁 CREATING FOLD SPLIT FILES WITH REAL IDS")
    print("="*70)
    
    for fold_i in range(n_folds):
        # Get indices for this fold
        train_idx = df[df["fold"] != fold_i].index
        test_idx = df[df["fold"] == fold_i].index
        
        # Create dataframe with all information
        fold_split_df = pd.DataFrame({
            'id': df.loc[train_idx, 'id'].values.tolist() + df.loc[test_idx, 'id'].values.tolist(),
            'label': df.loc[train_idx, 'label'].values.tolist() + df.loc[test_idx, 'label'].values.tolist(),
            'cluster': df.loc[train_idx, 'cluster'].values.tolist() + df.loc[test_idx, 'cluster'].values.tolist(),
            'train': [1] * len(train_idx) + [0] * len(test_idx)
        })
        
        # Add PNU column if it exists
        if 'label_PNU' in df.columns:
            fold_split_df['PNU'] = df.loc[train_idx, 'label_PNU'].values.tolist() + df.loc[test_idx, 'label_PNU'].values.tolist()
        
        # Sort by train (1 first) then by id for consistency
        fold_split_df = fold_split_df.sort_values(['train', 'id'], ascending=[False, True])
        
        # Save to fold_splits directory
        fold_split_path = os.path.join(fold_splits_dir, f"fold_{fold_i}_split.csv")
        fold_split_df.to_csv(fold_split_path, index=False)
        
        # Also save a version with only test IDs for easy reference
        test_ids_path = os.path.join(fold_splits_dir, f"fold_{fold_i}_test_ids.csv")
        test_df = fold_split_df[fold_split_df['train'] == 0]
        if 'PNU' in test_df.columns:
            test_df[['id', 'PNU', 'label', 'cluster']].to_csv(test_ids_path, index=False)
        else:
            test_df[['id', 'label', 'cluster']].to_csv(test_ids_path, index=False)
        
        print(f"  ✅ Fold {fold_i}: {len(train_idx)} train, {len(test_idx)} test samples")
        if len(train_idx) > 0 and len(test_idx) > 0:
            print(f"     ├─ Train IDs: {fold_split_df[fold_split_df['train']==1]['id'].iloc[0]} ... {fold_split_df[fold_split_df['train']==1]['id'].iloc[-1]}")
            print(f"     └─ Test IDs:  {fold_split_df[fold_split_df['train']==0]['id'].iloc[0]} ... {fold_split_df[fold_split_df['train']==0]['id'].iloc[-1]}")
    
    # Save all predictions
    if args.save_predictions and all_predictions:
        all_pred_df = pd.concat(all_predictions, ignore_index=True)
        all_pred_df.to_csv(os.path.join(output_dir, "all_predictions.csv"), index=False)
        print(f"\n📁 Predictions saved to: {os.path.join(output_dir, 'all_predictions.csv')}")
    
    # Save configuration used
    config_copy_path = os.path.join(output_dir, "config_used.json")
    with open(config_copy_path, 'w', encoding='utf-8') as f:
        # Add feature info to config
        cfg['features_used'] = {
            'total': len(feature_cols),
            'by_category': feature_categories
        }
        json.dump(cfg, f, indent=2)
    print(f"\n📁 Configuration saved to: {config_copy_path}")
    
    # Save model information
    model_info = {
        'n_folds': n_folds,
        'features_used': len(feature_cols),
        'feature_names': feature_cols,
        'feature_categories': feature_categories,
        'alphas': [float(a) for a in alphas],
        'raw_alphas': [float(a) for a in raw_alphas] if raw_alphas else None,
        'mean_alpha': float(np.mean(alphas)),
        'std_alpha': float(np.std(alphas)),
        'saved_model_paths': [f"models/model_fold_{i}.pkl" for i in range(n_folds)] if args.save_models else None,
        'interactions_created': create_interactions
    }
    model_info_path = os.path.join(output_dir, "model_info.json")
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2)
    
    # Create a README file with experiment details
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("WEIGHTED PU LEARNING EXPERIMENT\n")
        f.write("="*40 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config file: {args.config}\n\n")
        f.write("FEATURE ENGINEERING:\n")
        f.write(f"- Interactions created: {create_interactions}\n")
        f.write(f"- Max features: {max_features}\n")
        f.write(f"- Final features: {len(feature_cols)}\n\n")
        f.write("FEATURE CATEGORIES:\n")
        for cat, count in feature_categories.items():
            f.write(f"  {cat}: {count}\n")
        f.write("\nRESULTS:\n")
        for metric in ['pr_auc','f1_score', 'precision', 'recall']:
            f.write(f"  {metric}: {summary[metric]['mean']:.4f} (±{summary[metric]['std']:.4f})\n")
    
    print(f"\n📁 Fold split files saved in: {fold_splits_dir}")
    print(f"📁 Models saved in: {models_dir}")
    print(f"📁 Feature info saved in: {features_dir}")
    print(f"📄 README.txt created")
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print(f"📁 All results saved to: {output_dir}")
    print('='*70)
    
    return summary


def load_saved_model(model_path):
    """Helper function to load a saved model for later analysis"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == "__main__":
    main()