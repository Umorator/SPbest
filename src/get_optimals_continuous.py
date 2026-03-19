import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np

def get_optimals(config_path: str, plot: bool = True, pnu_mode: bool = False, use_continuous_relevance: bool = True):
    """
    Load CSV from JSON config and label enzyme_activity.
    
    Two modes:
    1. Standard mode (pnu_mode=False): 
       - label: 1 for optimal, 0 for non-optimal
    
    2. PNU mode (pnu_mode=True):
       - label: Original binary classification (1=optimal, 0=non-optimal)
       - label_PNU: 1=positive (optimal), 0=negative (non-optimal with numeric value), 
                   -1=unlabeled (no enzyme activity data)
    
    3. Continuous relevance (use_continuous_relevance=True):
       - Adds 'relevance' column with integer levels (0-4) for ranking
       - For unlabeled SPs, assigns relevance = 0
       - For measured SPs, assigns levels 1-4 based on activity percentiles
    
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
    - In continuous relevance mode:
        * Creates 'relevance' column with integer levels 0-4 for ranking
        * Unlabeled SPs get relevance = 0
        * Measured SPs get levels 1-4 based on activity percentiles within each protein
        * Khadye special SPs get relevance = 3 if they have no numeric value
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
    
    # ===== MODIFIED: Add integer relevance scores for ranking (0-4) =====
    if use_continuous_relevance:
        print("\n📊 Creating integer relevance levels (0-4) for ranking...")
        
        # First, convert enzyme_activity to numeric, coercing errors to NaN
        df['activity_numeric'] = pd.to_numeric(df['enzyme_activity'], errors='coerce')
        
        # Initialize relevance column - default 0 for unlabeled
        df['relevance'] = 0
        
        # Process each protein group
        for protein in df['Author-Protein'].unique():
            protein_mask = df['Author-Protein'] == protein
            protein_data = df[protein_mask]
            
            # Get numeric activities for this protein
            numeric_mask = protein_data['activity_numeric'].notna()
            numeric_activities = protein_data.loc[numeric_mask, 'activity_numeric'].values
            
            if len(numeric_activities) > 0:
                # For proteins with enough measurements, create 4 relevance levels (1-4)
                if len(numeric_activities) >= 4:
                    # Use quartiles to assign levels 1-4
                    q1 = np.percentile(numeric_activities, 25)
                    q2 = np.percentile(numeric_activities, 50)
                    q3 = np.percentile(numeric_activities, 75)
                    
                    # Assign levels based on quartiles
                    for idx in protein_data[numeric_mask].index:
                        activity = df.loc[idx, 'activity_numeric']
                        if activity <= q1:
                            df.loc[idx, 'relevance'] = 1
                        elif activity <= q2:
                            df.loc[idx, 'relevance'] = 2
                        elif activity <= q3:
                            df.loc[idx, 'relevance'] = 3
                        else:
                            df.loc[idx, 'relevance'] = 4
                
                # For proteins with 2-3 measurements, use 2-3 levels
                elif len(numeric_activities) >= 2:
                    # Sort activities
                    sorted_activities = np.sort(numeric_activities)
                    
                    if len(numeric_activities) == 2:
                        # Two levels: low=1, high=2
                        for idx in protein_data[numeric_mask].index:
                            activity = df.loc[idx, 'activity_numeric']
                            if activity == sorted_activities[0]:
                                df.loc[idx, 'relevance'] = 1
                            else:
                                df.loc[idx, 'relevance'] = 2
                    else:  # 3 measurements
                        # Three levels: low=1, middle=2, high=3
                        for idx in protein_data[numeric_mask].index:
                            activity = df.loc[idx, 'activity_numeric']
                            if activity == sorted_activities[0]:
                                df.loc[idx, 'relevance'] = 1
                            elif activity == sorted_activities[1]:
                                df.loc[idx, 'relevance'] = 2
                            else:
                                df.loc[idx, 'relevance'] = 3
                
                # For single measurement, just assign level 2 (middle)
                else:  # len(numeric_activities) == 1
                    df.loc[protein_mask & numeric_mask, 'relevance'] = 2
                
                # Unlabeled SPs already have relevance = 0 (from initialization)
                
            # If protein has no numeric activities, all stay at 0
            # (no action needed)
        
        # Special handling for Khadye special SPs (even if non-numeric, they're positive)
        khadye_mask = (df['Author-Protein'] == 'Khadye - E7FHY4') & (df['SP name'].isin(khadye_sps))
        if khadye_mask.any():
            # For Khadye special SPs with no numeric value, assign level 3
            for idx in df[khadye_mask].index:
                if pd.isna(df.loc[idx, 'activity_numeric']):
                    # These are the special Khadye SPs with no numeric value
                    # Assign them relevance level 3 (above average but not top)
                    df.loc[idx, 'relevance'] = 3
        
        print(f"\n📊 Relevance Level Distribution:")
        print(f"  Level 0 (unlabeled): {(df['relevance'] == 0).sum()}")
        print(f"  Level 1 (lowest): {(df['relevance'] == 1).sum()}")
        print(f"  Level 2: {(df['relevance'] == 2).sum()}")
        print(f"  Level 3: {(df['relevance'] == 3).sum()}")
        print(f"  Level 4 (highest): {(df['relevance'] == 4).sum()}")
        print(f"  Total measured: {(df['relevance'] > 0).sum()}")
        
        # Show per-protein statistics
        protein_stats = df.groupby('Author-Protein').agg({
            'relevance': ['min', 'max', 'mean', 'count'],
            'activity_numeric': lambda x: x.notna().sum()
        }).round(2)
        protein_stats.columns = ['rel_min', 'rel_max', 'rel_mean', 'total_count', 'n_measured']
        
        print("\n📈 Per-protein relevance stats (first 5):")
        print(protein_stats[['n_measured', 'rel_min', 'rel_max', 'rel_mean']].head())
        
        # Drop temporary column
        df = df.drop('activity_numeric', axis=1)

    if plot:
        plot_distribution(df, pnu_mode, use_continuous_relevance)

    return df

def plot_distribution(df, pnu_mode=False, use_continuous_relevance=False):
    """
    Plot distribution of labels
    
    For pnu_mode=False: Stacked bar chart of Optimal vs Non-Optimal
    For pnu_mode=True: Stacked bar chart of Positive vs Negative vs Unlabeled
    If use_continuous_relevance: Also show relevance score distribution
    """
    if use_continuous_relevance:
        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, max(8, len(df['Author-Protein'].unique())*0.35)))
        
        # First subplot: PNU distribution (if available) or binary labels
        ax1 = axes[0]
        if pnu_mode and 'label_PNU' in df.columns:
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
            
            # Colors: Blue for positive, gray for negative, light brown for unlabeled
            colors = ['#2c7bb6', '#95a5a6', "#d3d3d3"]
            labels = ['Positive (Optimal)', 'Negative (Non-Optimal)', 'Unlabeled (No Data)']
            title = 'PNU Distribution by Protein'
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
            title = 'Binary Label Distribution by Protein'
        
        summary.plot(kind='barh', stacked=True, ax=ax1, color=colors, width=0.7,
                    edgecolor='white', linewidth=0.5)
        
        ax1.set_xlabel('Number of Entries', fontsize=12)
        ax1.set_ylabel('Author - Protein', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold', pad=20, loc='left')
        ax1.xaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        ax1.legend(labels, loc='upper right')
        
        # Second subplot: Relevance score distribution
        ax2 = axes[1]
        
        # Prepare data for boxplot
        plot_data = []
        proteins = []
        
        for protein in df['Author-Protein'].unique():
            protein_data = df[df['Author-Protein'] == protein]['relevance'].values
            plot_data.append(protein_data)
            proteins.append(protein[:20] + '...' if len(protein) > 20 else protein)
        
        # Create boxplot
        bp = ax2.boxplot(plot_data, labels=proteins, vert=False, patch_artist=True,
                         showmeans=True, meanline=True)
        
        # Color boxes
        for box in bp['boxes']:
            box.set_facecolor('#2c7bb6')
            box.set_alpha(0.6)
        
        ax2.set_xlabel('Relevance Score', fontsize=12)
        ax2.set_title('Continuous Relevance Scores by Protein', fontsize=14, fontweight='bold', pad=20, loc='left')
        ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        
        plt.tight_layout()
        
    else:
        # Original plotting code (when not using continuous relevance)
        fig, ax = plt.subplots(figsize=(12, max(8, len(df['Author-Protein'].unique())*0.35)))
        
        if pnu_mode and 'label_PNU' in df.columns:
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
            
            # Colors: Blue for positive, gray for negative, light brown for unlabeled
            colors = ['#2c7bb6', '#95a5a6', "#d3d3d3"]
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

        summary.plot(kind='barh', stacked=True, ax=ax, color=colors, width=0.7,
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
    mode_str = 'continuous' if use_continuous_relevance else ('pnu' if pnu_mode else 'standard')
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig(f'outputs/figures/enzyme_activity_distribution_{mode_str}.png', 
                dpi=300, bbox_inches='tight', format='png')
    plt.show()