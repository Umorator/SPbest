import sys
import os
import json
import pandas as pd

# Add repo root to module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.get_optimals import get_optimals

config_path = "configs/get_optimals.json"

# Load config to get parameters
print(f"Loading config from: {config_path}")
with open(config_path, 'r') as f:
    config = json.load(f)

# Get parameters from config (with defaults if not present)
plot = config.get("plot", True)  # Default to True if not specified
pnu_mode = config.get("pnu_mode", False)  # Default to False if not specified

print(f"Running get_optimals with: plot={plot}, pnu_mode={pnu_mode}")
df_labeled = get_optimals(config_path, plot=plot, pnu_mode=pnu_mode)

# Get output path from config or use default
output_path = config.get("labeled_data", "outputs/labeled.csv")
df_labeled.to_csv(output_path, index=False)

# Quick verification
print(f"\nFile saved to: {output_path}")
print(f"\nDataset shape: {df_labeled.shape}")
print(f"Columns: {df_labeled.columns.tolist()}")

print(f"\nLabel distribution (original binary):")
print(df_labeled['label'].value_counts().sort_index())

if pnu_mode and 'label_PNU' in df_labeled.columns:
    print(f"\nPNU label distribution (1=positive, 0=negative, -1=unlabeled):")
    print(df_labeled['label_PNU'].value_counts().sort_index())
    
    # Show first few rows with PNU info
    print(f"\nFirst 5 rows (with PNU):")
    print(df_labeled[['Author-Protein', 'SP name', 'enzyme_activity', 'label', 'label_PNU']].head())
else:
    # Show first few rows without PNU
    print(f"\nFirst 5 rows:")
    print(df_labeled[['Author-Protein', 'SP name', 'enzyme_activity', 'label']].head())

print("\nDone!")