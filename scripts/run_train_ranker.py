#!/usr/bin/env python3
"""
Run 5-fold cross-validation for SP ranking model
Usage: 
    python scripts/run_train_ranker.py [config_file] [--pairs PAIRS_PATH] [--no-plot]
"""

import sys
import os
import argparse
import json
from datetime import datetime
import joblib  # for saving the scaler

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_ranker import train_ranking_model_cv

def create_run_folder(base_dir='outputs'):
    """Create a timestamped folder for this run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    
    # Create subfolders
    os.makedirs(os.path.join(run_folder, 'models'), exist_ok=True)
    os.makedirs(os.path.join(run_folder, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(run_folder, 'results'), exist_ok=True)
    
    return run_folder, timestamp

def save_config_copy(config_file, run_folder):
    """Save a copy of the config file in the run folder"""
    import shutil
    dest = os.path.join(run_folder, 'config_used.json')
    shutil.copy2(config_file, dest)
    print(f"📄 Config saved to: {dest}")

def main():
    parser = argparse.ArgumentParser(description='Run 5-fold cross-validation for SP ranking model')
    
    # Config file options
    parser.add_argument('config', nargs='?', default='config.json',
                       help='Path to config file (default: config.json)')
    parser.add_argument('--config', dest='config_alt', 
                       help='Alternative way to specify config file')
    
    # Data options
    parser.add_argument('--pairs', default=None,
                       help='Path to pairs CSV (overrides config setting)')
    
    # Training options
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Disable feature normalization')
    
    args = parser.parse_args()
    
    # Determine which config argument to use
    config_file = args.config_alt if args.config_alt else args.config
    
    # Check if config exists
    if not os.path.exists(config_file):
        print(f"❌ Error: Config file '{config_file}' not found!")
        print(f"   Please provide a valid config file path.")
        sys.exit(1)
    
    # Load config to get paths
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Determine pairs path (command line overrides config)
    if args.pairs:
        pairs_path = args.pairs
    else:
        pairs_path = config.get('pairs_path', 'outputs/ranking_pairs.csv')
    
    if not os.path.exists(pairs_path):
        print(f"❌ Pairs file not found: {pairs_path}")
        sys.exit(1)
    
    # Create timestamped run folder
    run_folder, timestamp = create_run_folder()
    
    # Save config copy
    save_config_copy(config_file, run_folder)
    
    # Get other config settings
    add_sp = config.get('add_sp', True)
    
    print("="*70)
    print("🔬 SP RANKING MODEL - 5-FOLD CROSS-VALIDATION")
    print("="*70)
    print(f"📁 Config file: {config_file}")
    print(f"📊 Pairs file: {pairs_path}")
    print(f"🧬 SP Features: {'Enabled' if add_sp else 'Disabled'}")
    print(f"📈 Plotting: {'Disabled' if args.no_plot else 'Enabled'}")
    print(f"⚙️  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"⚙️  Learning rate: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"⚙️  Normalization: {'Disabled' if args.no_normalize else 'Enabled'}")
    print(f"📁 Run folder: {run_folder}")
    print("="*70)
    
    # Override output paths in config for this run
    config['run_folder'] = run_folder
    config['models_dir'] = os.path.join(run_folder, 'models')
    config['plots_dir'] = os.path.join(run_folder, 'plots')
    config['results_dir'] = os.path.join(run_folder, 'results')
    
    # Save updated config
    with open(os.path.join(run_folder, 'config_run.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run 5-fold cross-validation
    print("\n📊 Running 5-fold cross-validation by protein clusters...")
    print("="*70)
    
    try:
        result = train_ranking_model_cv(
            config_path=config_file,
            pairs_path=pairs_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            normalize=not args.no_normalize,
            run_folder=run_folder
        )
        
        # Handle tuple return (fold_results, scaler) or single return
        if isinstance(result, tuple):
            fold_results, scaler = result
            # Save scaler for future inference
            joblib.dump(scaler, os.path.join(run_folder, "feature_scaler.joblib"))
            print(f"💾 Feature scaler saved to: {os.path.join(run_folder, 'feature_scaler.joblib')}")
        else:
            fold_results = result
        
        # Print final summary per fold including new metrics
        print("\n" + "="*70)
        print("✅ CROSS-VALIDATION COMPLETE - FOLD SUMMARY")
        print("="*70)
        for fold, res in fold_results.items():
            spearman = res.get('mean_spearman', float('nan'))
            top1 = res.get('top1_accuracy', float('nan'))
            print(f"Fold {fold}: "
                  f"Val Acc={res['val_accuracy']:.4f}, "
                  f"Weighted Acc={res['val_weighted_accuracy']:.4f}, "
                  f"High Conf Acc={res['val_high_conf_accuracy']:.4f}, "
                  f"Spearman={spearman:.4f}, "
                  f"Top1={top1:.4f}")
        
        print("\n📁 All outputs saved to: {run_folder}")
        print(f"   📊 Results CSV: {os.path.join(run_folder, 'results', 'cv_results.csv')}")
        print(f"   📈 Plots: {os.path.join(run_folder, 'plots')}")
        print(f"   💾 Models: {os.path.join(run_folder, 'models')}")
        print(f"   📄 Config: {os.path.join(run_folder, 'config_used.json')}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
