#!/usr/bin/env python3
"""
Run script for creating ranking pairs from SP data.
Usage: python run_create_pairs.py [config_file]
"""

import sys
import os
import argparse
import sys
import json

# Add repo root to module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.create_rank_pairs import create_rank_pairs


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create ranking pairs from SP libraries')
    
    # Accept config both as positional and named argument
    parser.add_argument('config', nargs='?', default='config.json', 
                       help='Path to config file (default: config.json)')
    parser.add_argument('--config', dest='config_alt', 
                       help='Alternative way to specify config file')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--output', '-o', default='outputs/ranking_pairs.csv',
                       help='Output path for pairs CSV (default: outputs/ranking_pairs.csv)')
    
    args = parser.parse_args()
    
    # Determine which config argument to use
    config_file = args.config_alt if args.config_alt else args.config
    
    # Check if config exists
    if not os.path.exists(config_file):
        print(f"❌ Error: Config file '{config_file}' not found!")
        print(f"   Please provide a valid config file path.")
        print(f"\nUsage examples:")
        print(f"   python run_create_pairs.py configs/get_optimals.json")
        print(f"   python run_create_pairs.py --config configs/get_optimals.json")
        sys.exit(1)
    
    # Load config to display settings
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    add_sp = config.get('add_sp', True)
    
    print("="*60)
    print("🔬 SP RANKING PAIRS GENERATOR")
    print("="*60)
    print(f"📁 Config file: {config_file}")
    print(f"📊 Plotting: {'Disabled' if args.no_plot else 'Enabled'}")
    print(f"🧬 SP Features (from config): {'Enabled' if add_sp else 'Disabled'}")
    print(f"💾 Output: {args.output}")
    print("="*60)
    
    try:
        # Run the pair creation
        pairs_df = create_rank_pairs(
            config_path=config_file, 
            plot=not args.no_plot
            # add_sp now comes from config file!
        )
        
        # Save to specified output
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        pairs_df.to_csv(args.output, index=False)
        
        print("\n" + "="*60)
        print("✅ SUCCESS!")
        print("="*60)
        print(f"📊 Total pairs created: {len(pairs_df):,}")
        print(f"📁 Pairs saved to: {args.output}")
        print(f"📈 Pair type breakdown:")
        for pair_type, count in pairs_df['pair_type'].value_counts().items():
            print(f"   - {pair_type}: {count:,} ({count/len(pairs_df)*100:.1f}%)")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()