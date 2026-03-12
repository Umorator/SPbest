import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np

def get_optimals(config_path: str, plot: bool = True, pnu_mode: bool = False):
    """
    Load CSV from JSON config and label enzyme_activity.
    
    Two modes:
    1. Standard mode (pnu_mode=False): 
       - label: 1 for optimal, 0 for non-optimal
    
    2. PNU mode (pnu_mode=True):
       - label: Original binary classification (1=optimal, 0=non-optimal)
       - label_PNU: 1=positive (optimal), 0=negative (non-optimal with numeric value), 
                   -1=unlabeled (no enzyme activity data)
    
    Rules:
    - Group by 'Author-Protein'.
    - If all enzyme_activity numeric: label 1 if > group mean, else 0.
    - If non-numeric present: numerics=1, non-numerics=0.
    - Special cases:
        * 'Khadye - E7FHY4': specific SPs labeled as 1 (even if non-numeric)
        * 'Ying - EF634454.1': all entries labeled 1
    - In PNU mode:
        * If enzyme_activity is numeric -> label_PNU = label (0 or 1)
        * If enzyme_activity is non-numeric (missing/string) -> label_PNU = -1 (unlabeled)
        * EXCEPT for Khadye special SPs: these are ALWAYS positive (label_PNU=1) even if non-numeric
    """

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    csv_path = config.get('csv_path')
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV path not found in config: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Required columns
    required_cols = ['Author-Protein', 'enzyme_activity', 'SP name']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Special SPs for Khadye
    khadye_sps = ['cith','lytb','ywsb','ybdg','phob','abna','ykoj','ykwd','yobv','ybbe','ywad','apre']

    def label_group(group_df):
        """Apply labeling rules per Author-Protein group."""
        name = group_df.name
        if name == 'Khadye - E7FHY4':
            return group_df['SP name'].apply(lambda x: 1 if x in khadye_sps else 0)
        elif name == 'Ying - EF634454.1':
            return pd.Series(1, index=group_df.index)
        else:
            numeric_activity = pd.to_numeric(group_df['enzyme_activity'], errors='coerce')
            if numeric_activity.notna().all():
                mean_val = numeric_activity.mean()
                return (numeric_activity > mean_val).astype(int)
            else:
                return numeric_activity.notna().astype(int)

    # Apply labeling per group
    df['label'] = df.groupby('Author-Protein', group_keys=False).apply(label_group)
    
    # Add PNU column if requested
    if pnu_mode:
        # Create label_PNU column
        # First, check if enzyme_activity is numeric
        df['is_numeric'] = pd.to_numeric(df['enzyme_activity'], errors='coerce').notna()
        
        # Default rule: if numeric -> use label, if non-numeric -> -1
        df['label_PNU'] = np.where(
            df['is_numeric'],
            df['label'],
            -1
        )
        
        # SPECIAL CASE: Khadye specific SPs are always positive (1) even if non-numeric
        khadye_mask = (df['Author-Protein'] == 'Khadye - E7FHY4') & (df['SP name'].isin(khadye_sps))
        df.loc[khadye_mask, 'label_PNU'] = 1
        
        # Drop temporary column
        df = df.drop('is_numeric', axis=1)
        
        print("\nPNU Label Distribution:")
        print(f"  Positives (label_PNU=1): {(df['label_PNU'] == 1).sum()}")
        print(f"  Negatives (label_PNU=0): {(df['label_PNU'] == 0).sum()}")
        print(f"  Unlabeled (label_PNU=-1): {(df['label_PNU'] == -1).sum()}")
        print(f"  Total: {len(df)}")
        
        # Show Khadye special cases
        khadye_pos = df[(df['Author-Protein'] == 'Khadye - E7FHY4') & (df['SP name'].isin(khadye_sps))]
        print(f"\nKhadye special SPs (all set to 1): {len(khadye_pos)}")
        if len(khadye_pos) > 0:
            print(f"  Examples: {khadye_pos['SP name'].tolist()}")

    if plot:
        plot_distribution(df, pnu_mode)

    return df

def plot_distribution(df, pnu_mode=False):
    """
    Plot distribution of labels
    
    For pnu_mode=False: Stacked bar chart of Optimal vs Non-Optimal
    For pnu_mode=True: Stacked bar chart of Positive vs Negative vs Unlabeled
    """
    if pnu_mode:
        # PNU mode: Show distribution of label_PNU
        summary = df.groupby('Author-Protein')['label_PNU'].value_counts().unstack(fill_value=0)
        
        # Ensure all three columns exist
        for col in [1, 0, -1]:
            if col not in summary.columns:
                summary[col] = 0
        
        # Reorder columns: Positive (1), Negative (0), Unlabeled (-1)
        summary = summary[[1, 0, -1]]
        summary['Total'] = summary.sum(axis=1)
        summary = summary.sort_values('Total', ascending=False).drop('Total', axis=1)
        
        # Colors: Blue for positive, gray for negative, light gray for unlabeled
        colors = ['#2c7bb6', '#95a5a6', "#63701a"]
        labels = ['Positive (Optimal)', 'Negative (Non-Optimal)', 'Unlabeled (No Data)']
        title = 'Distribution of Positive, Negative, and Unlabeled SPs by Author-Protein'
        
    else:
        # Standard mode: Show distribution of label
        summary = df.groupby('Author-Protein')['label'].value_counts().unstack(fill_value=0)
        
        # Ensure both columns exist
        for col in [1, 0]:
            if col not in summary.columns:
                summary[col] = 0
        
        summary = summary[[1, 0]]
        summary['Total'] = summary.sum(axis=1)
        summary = summary.sort_values('Total', ascending=False).drop('Total', axis=1)
        
        # Colors: Blue for optimal, gray for non-optimal
        colors = ['#2c7bb6', '#95a5a6']
        labels = ['Optimal SP', 'Non-Optimal SP']
        title = 'Distribution of Optimal vs Non-Optimal SPs by Author-Protein'

    # Create plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(summary)*0.35)))
    
    bars = summary.plot(kind='barh', stacked=True, ax=ax, color=colors, width=0.7,
                        edgecolor='white', linewidth=0.5)

    # Labels and title
    ax.set_xlabel('Number of Entries', fontsize=12, fontweight='normal', 
                  fontfamily='sans-serif', labelpad=10)
    ax.set_ylabel('Author - Protein', fontsize=12, fontweight='normal', 
                  fontfamily='sans-serif', labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', 
                 fontfamily='sans-serif', pad=20, loc='left')

    # Ticks
    ax.tick_params(axis='both', which='major', labelsize=10, length=4, width=0.5)
    ax.tick_params(axis='x', which='minor', length=2, width=0.5)

    # Value labels
    for i, (index, row) in enumerate(summary.iterrows()):
        cumulative = 0
        for j, col in enumerate(summary.columns):
            value = row[col]
            if value > 0:
                x_pos = cumulative + value/2
                # Text color: white on dark colors, black on light colors
                text_color = 'white' if j == 0 else 'black'
                ax.text(x_pos, i, str(int(value)), ha='center', va='center',
                        fontweight='normal', fontsize=9, color=text_color)
                cumulative += value

    # Grid and styling
    ax.xaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Legend
    legend = ax.legend(labels, loc='upper right', frameon=True, 
                       fancybox=False, edgecolor='black', fontsize=11,
                       handlelength=1.0, handletextpad=0.5, borderpad=0.5, 
                       framealpha=0.95)
    for text in legend.get_texts():
        text.set_color('black')
        text.set_fontweight('normal')

    plt.tight_layout()
    
    # Save figure
    mode_str = 'pnu' if pnu_mode else 'standard'
    plt.savefig(f'outputs/figures/enzyme_activity_distribution_{mode_str}.png', 
                dpi=300, bbox_inches='tight', format='png')
    plt.show()
