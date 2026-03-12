import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# --- Load data ---
path_data = r"data\CoSMPAD.csv"
df = pd.read_csv(path_data)
df = df[df['Species'].str.lower().str.startswith('bacillus', na=False)].copy()

# --- Group by protein + reference (same protein can appear in multiple references) ---
dfs = [x for _, x in df.groupby(['UniprotKB/NCBI_POI','Reference']) if len(x) > 1]

# --- Helper: get most common value ---
def most_common(lst):
    return max(set(lst), key=lst.count)

# --- Helper: extract first author from reference ---
def get_first_author(reference):
    """
    Extract first author's last name from reference string
    Handles formats like "Smith,J.", "Smith J.", "Smith et al.", etc.
    """
    if pd.isna(reference):
        return "Unknown"
    
    ref_str = str(reference)
    
    # Common patterns in your data: 'Tsuji', 'Yao', 'Fu,G.', 'Zhang,W.'
    # First, try to split by comma or space
    if ',' in ref_str:
        # Take the part before the first comma (usually the first author's last name)
        first_author = ref_str.split(',')[0].strip()
    elif ' ' in ref_str:
        # Take the part before the first space
        first_author = ref_str.split(' ')[0].strip()
    else:
        # If no separators, use the whole string
        first_author = ref_str
    
    # Remove any trailing punctuation
    first_author = first_author.rstrip('.,;:')
    
    return first_author

# --- Filter each protein/ref group based on dominant conditions ---
df_filtered_list = []
for group in dfs:
    a = group[group.Promoter == most_common(list(group.Promoter.values))]
    a = a[a['time (h)'] == most_common(list(a['time (h)'].values))]
    a = a[a['Host'] == most_common(list(a['Host'].values))]

    if 'Tsuji' in str(group.Reference.unique()):
        a = group[group.Promoter == most_common(list(group.Promoter.values))]
        df_filtered_list.append(a)
    elif 'Yao' in str(group.Reference.unique()):
        a = group[group.Promoter == most_common(list(group.Promoter.values))]
        a = a[a['Host'] == most_common(list(a['Host'].values))]
        df_filtered_list.append(a)
    elif 'Fu,G.' in str(group.Reference.unique()):
        df_filtered_list.append(group)
    elif 'Zhang,W.' in str(group.Reference.unique()):
        b = group[group.Promoter == 'Pglvm']
        df_filtered_list.append(b)
    else:
        a = a[a['cultivation_flask'] == most_common(list(a['cultivation_flask'].values))]
        df_filtered_list.append(a)

# --- Concatenate filtered groups ---
df_all = pd.concat(df_filtered_list, join='outer', axis=0)

# ==========================================================
# --- DATA PROCESSING SECTION (ALL BEFORE PLOT) ---
# ==========================================================
print("\n" + "="*60)
print("DATA PROCESSING")
print("="*60)

# ----------------------------------------------------------
# 1. Create dataset_name column if it doesn't exist
# ----------------------------------------------------------
if 'dataset_name' not in df_all.columns:
    print("Creating 'dataset_name' column...")
    df_all['dataset_name'] = df_all.apply(lambda row: 
        f"{get_first_author(row['Reference'])}_{row['UniprotKB/NCBI_POI']}", axis=1)
    print(f"Created with {df_all['dataset_name'].nunique()} unique values")

# ----------------------------------------------------------
# 2. Create First_Author column
# ----------------------------------------------------------
print("\nCreating First_Author column...")
df_all['First_Author'] = df_all['Reference'].apply(get_first_author)

# ----------------------------------------------------------
# 3. MERGE BROCKMEIER AND CASPERS DATASETS
# ----------------------------------------------------------
print("\n" + "-"*40)
print("MERGING BROCKMEIER AND CASPERS DATASETS")
print("-"*40)

# Check for Brockmeier and Caspers entries with protein C7ZGJ1
brockmeier_pattern = 'Brockmeier'
caspers_pattern = 'Caspers'
c7zgj1_mask = df_all['UniprotKB/NCBI_POI'] == 'C7ZGJ1'

# Find entries based on First_Author and Reference
brockmeier_mask = df_all['First_Author'].str.contains(brockmeier_pattern, case=False, na=False) | \
                  df_all['Reference'].str.contains(brockmeier_pattern, case=False, na=False)
caspers_mask = df_all['First_Author'].str.contains(caspers_pattern, case=False, na=False) | \
               df_all['Reference'].str.contains(caspers_pattern, case=False, na=False)

bc_mask = (brockmeier_mask | caspers_mask) & c7zgj1_mask

print(f"Brockmeier entries found: {len(df_all[brockmeier_mask & c7zgj1_mask])}")
print(f"Caspers entries found: {len(df_all[caspers_mask & c7zgj1_mask])}")

if bc_mask.any():
    # Update First_Author to merged name
    df_all.loc[bc_mask, 'First_Author'] = 'Brockmeier-Caspers'
    
    # Update dataset_name
    df_all.loc[bc_mask, 'dataset_name'] = 'Brockmeier-Caspers_C7ZGJ1'
    
    print(f"Merged {bc_mask.sum()} entries into 'Brockmeier-Caspers'")
else:
    print("No Brockmeier or Caspers entries found with protein C7ZGJ1")

# ----------------------------------------------------------
# 4. Create Author-Protein column (AFTER merging Brockmeier-Caspers)
# ----------------------------------------------------------
print("\nCreating Author-Protein column...")
df_all['Author-Protein'] = df_all['First_Author'] + ' - ' + df_all['UniprotKB/NCBI_POI']

# ----------------------------------------------------------
# 5. AVERAGE GRASSO - P00692 SEQUENCES WITH SAME SP_SEQ
# ----------------------------------------------------------
print("\n" + "-"*40)
print("AVERAGING GRASSO - P00692 SEQUENCES")
print("-"*40)

# Create numeric enzyme activity column
df_all['enzyme_activity_numeric'] = pd.to_numeric(df_all['enzyme_activity'], errors='coerce')

# Identify Grasso - P00692 entries
grasso_mask = (df_all['First_Author'].str.contains('Grasso', case=False, na=False)) & \
              (df_all['UniprotKB/NCBI_POI'] == 'P00692')

if grasso_mask.any():
    # Get the Grasso subset
    grasso_df = df_all[grasso_mask].copy()
    non_grasso_df = df_all[~grasso_mask].copy()
    
    print(f"Grasso entries before averaging: {len(grasso_df)}")
    
    if 'sp_seq' in grasso_df.columns:
        unique_sp_seq = grasso_df['sp_seq'].nunique()
        print(f"Unique sp_seq values: {unique_sp_seq}")
        print(f"Sequences with duplicates: {len(grasso_df) - unique_sp_seq}")
        
        # Create a list to store the averaged rows
        averaged_rows = []
        
        # Group by sp_seq
        for sp_seq, group in grasso_df.groupby('sp_seq'):
            if len(group) == 1:
                # No averaging needed for unique sequences
                averaged_rows.append(group.iloc[0])
            else:
                # Take the first row as template
                first_row = group.iloc[0].copy()
                
                # Calculate mean of enzyme_activity_numeric
                mean_activity = group['enzyme_activity_numeric'].mean()
                
                # Update the values
                first_row['enzyme_activity_numeric'] = mean_activity
                first_row['enzyme_activity'] = mean_activity  # Keep as numeric, not string
                first_row['n_averaged'] = len(group)  # Add count of averaged sequences
                
                averaged_rows.append(first_row)
            
        # Create new dataframe from averaged rows
        grasso_avg = pd.DataFrame(averaged_rows)
        
        print(f"Grasso entries after averaging: {len(grasso_avg)}")
        print(f"Reduced by: {len(grasso_df) - len(grasso_avg)} rows")
        
        # Recombine the datasets
        df_all = pd.concat([non_grasso_df, grasso_avg], ignore_index=True, sort=False)
    else:
        print("Warning: 'sp_seq' column not found")
else:
    print("No Grasso entries found")

# ----------------------------------------------------------
# 6. Ensure numeric column is properly set
# ----------------------------------------------------------
print("\nEnsuring numeric conversion...")
df_all['enzyme_activity_numeric'] = pd.to_numeric(df_all['enzyme_activity_numeric'], errors='coerce')

# ----------------------------------------------------------
# 7. Remove previous Khadye - E7FHY4 entries
# ----------------------------------------------------------
print("\n" + "-"*40)
print("REMOVING OLD KHADYE ENTRIES")
print("-"*40)

old_khadye = df_all[df_all['UniprotKB/NCBI_POI'] == "E7FHY4"]
print(f"Old Khadye entries found: {len(old_khadye)}")

df_all = df_all[df_all['UniprotKB/NCBI_POI'] != "E7FHY4"]
print(f"Rows after removal: {len(df_all)}")

# ----------------------------------------------------------
# 8. Load and merge new Khadye dataset
# ----------------------------------------------------------
print("\n" + "-"*40)
print("LOADING NEW KHADYE DATASET")
print("-"*40)

path_khadye = r"C:\Users\rafae\OneDrive\Documents\PhD_2026\Thesis\Chapter_2_SPbest\SP_best_Repo\data\Takara\Khadye_Bglu_updated.csv"
df_khadye = pd.read_csv(path_khadye)

print(f"New Khadye rows: {len(df_khadye)}")
print(f"New Khadye columns: {len(df_khadye.columns)}")

# Align columns
for col in df_all.columns:
    if col not in df_khadye.columns:
        df_khadye[col] = np.nan

for col in df_khadye.columns:
    if col not in df_all.columns:
        df_all[col] = np.nan

df_khadye = df_khadye[df_all.columns]

# Force correct labeling
df_khadye['First_Author'] = "Khadye"
df_khadye['UniprotKB/NCBI_POI'] = "E7FHY4"
df_khadye['Author-Protein'] = "Khadye - E7FHY4"

if 'enzyme_activity' in df_khadye.columns:
    df_khadye['enzyme_activity_numeric'] = pd.to_numeric(df_khadye['enzyme_activity'], errors='coerce')

# Append
df_all = pd.concat([df_all, df_khadye], ignore_index=True)

print(f"\nTotal rows after Khadye merge: {len(df_all)}")

# ==========================================================
# --- NOW CREATE THE PLOT (ALL PROCESSING DONE) ---
# ==========================================================
# ==========================================================
# --- NOW CREATE THE PLOT (ALL PROCESSING DONE) ---
# ==========================================================
print("\n" + "="*60)
print("CREATING VISUALIZATION")
print("="*60)

# Find enzyme activity column
activity_columns = [col for col in df_all.columns if 'activity' in col.lower() or 
                   'value' in col.lower() or 'enzyme' in col.lower() or 'u/ml' in col.lower()]

if activity_columns:
    activity_col = activity_columns[0]
    print(f"Using column: {activity_col} for enzyme activity")
    
    # Create activity status column (numeric = measured)
    df_all['activity_status'] = pd.to_numeric(df_all[activity_col], errors='coerce').notna()
    
    # Override for Khadye using Class column
    khadye_mask = df_all['Author-Protein'] == "Khadye - E7FHY4"
    if 'Class' in df_all.columns:
        df_all.loc[khadye_mask, 'activity_status'] = (df_all.loc[khadye_mask, 'Class'] == 1)
        print("Khadye activity overridden using Class column")
    
    # Create labels
    df_all['activity_status_label'] = df_all['activity_status'].map({
        True: 'Measured Activity',
        False: 'No Measured Activity'
    })
    
    # Group by Author-Protein
    summary = df_all.groupby('Author-Protein')['activity_status_label'].value_counts().unstack(fill_value=0)
    
    # Ensure both columns exist
    if 'Measured Activity' not in summary.columns:
        summary['Measured Activity'] = 0
    if 'No Measured Activity' not in summary.columns:
        summary['No Measured Activity'] = 0
    
    # Reorder and sort
    summary = summary[['Measured Activity', 'No Measured Activity']]
    summary['Total'] = summary.sum(axis=1)
    summary = summary.sort_values('Total', ascending=False).drop('Total', axis=1)
    
    # Create publication-ready plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(summary) * 0.35)))
    
    # Lighter color palette - orange for measured, gray for not measured
    colors = ['#2c7bb6', '#95a5a6']  # Warm orange for measured, light gray for not measured
    
    # Plot with enhanced styling
    bars = summary.plot(kind='barh', stacked=True, ax=ax, color=colors, width=0.7, 
                       edgecolor='white', linewidth=0.5)
    
    # Axis labels with professional styling
    ax.set_xlabel('Number of Entries', fontsize=12, fontweight='normal', 
                 fontfamily='sans-serif', labelpad=10)
    ax.set_ylabel('Author - Protein', fontsize=12, fontweight='normal', 
                 fontfamily='sans-serif', labelpad=10)
    ax.set_title('Distribution of Enzyme Activity Measurements by Author-Protein', 
                fontsize=14, fontweight='bold', fontfamily='sans-serif', pad=20, loc='left')
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=10, length=4, width=0.5)
    ax.tick_params(axis='x', which='minor', length=2, width=0.5)
    
    # Add value labels with improved visibility
    for i, (index, row) in enumerate(summary.iterrows()):
        cumulative = 0
        for j, col in enumerate(summary.columns):
            value = row[col]
            if value > 0:
                x_pos = cumulative + value/2
                # White text on orange bar, black text on light gray bar
                text_color = 'white' if j == 0 else 'black'
                ax.text(x_pos, i, str(int(value)), 
                       ha='center', va='center', 
                       fontweight='normal', fontsize=9,
                       color=text_color)
                cumulative += value
    
    # Grid styling
    ax.xaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend with custom styling - top right position
    legend = ax.legend(['Activity status: Measured', 'Activity status: No measured'], 
                      loc='upper right', 
                      frameon=True, 
                      fancybox=False, 
                      edgecolor='black', 
                      fontsize=11,
                      handlelength=1.0,
                      handletextpad=0.5,
                      borderpad=0.5,
                      framealpha=0.95)
    
    # Set legend text color
    for text in legend.get_texts():
        text.set_color('black')
        text.set_fontweight('normal')
    
    # Remove top and right spines
    sns.despine(top=True, right=True, left=False, bottom=False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save high-resolution figure
    plt.savefig('enzyme_activity_distribution.png', dpi=300, bbox_inches='tight', format='png', transparent=False)
    print("\nPlots saved as:")
    print("  - enzyme_activity_distribution.pdf (vector format)")
    print("  - enzyme_activity_distribution.png (raster format, 300 DPI)")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total records: {len(df_all):,}")
    print(f"Measured activity: {df_all['activity_status'].sum():,} ({df_all['activity_status'].sum()/len(df_all)*100:.1f}%)")
    print(f"No measured activity: {(~df_all['activity_status']).sum():,} ({(~df_all['activity_status']).sum()/len(df_all)*100:.1f}%)")
    print(f"Unique Author-Protein combinations: {df_all['Author-Protein'].nunique()}")
    
    # Show top combinations
    print("\nTop 5 Author-Protein combinations:")
    top_combinations = summary.copy()
    top_combinations['Total'] = top_combinations.sum(axis=1)
    top_combinations = top_combinations.sort_values('Total', ascending=False).head(5)
    for idx, row in top_combinations.iterrows():
        print(f"  {idx}: {int(row['Total'])} total ({int(row['Measured Activity'])} measured, {int(row['No Measured Activity'])} not measured)")
    
    # Show Brockmeier-Caspers result
    print("\n" + "="*60)
    print("BROCKMEIER-CASPERS MERGE RESULT")
    print("="*60)
    bc_data = df_all[df_all['First_Author'] == 'Brockmeier-Caspers']
    if len(bc_data) > 0:
        print(f"Total Brockmeier-Caspers entries: {len(bc_data)}")
        print(f"Author-Protein value: {bc_data['Author-Protein'].iloc[0]}")
        print(f"Measured activity: {bc_data['activity_status'].sum()}")
        print(f"No measured activity: {(~bc_data['activity_status']).sum()}")
    
    # Show Grasso result
    print("\n" + "="*60)
    print("GRASSO - P00692 AVERAGING RESULT")
    print("="*60)
    grasso_final = df_all[(df_all['First_Author'].str.contains('Grasso', case=False, na=False)) & 
                          (df_all['UniprotKB/NCBI_POI'] == 'P00692')]
    print(f"Total Grasso entries after averaging: {len(grasso_final)}")
    if len(grasso_final) > 0:
        print(f"Unique sp_seq values: {grasso_final['sp_seq'].nunique()}")
        print(f"Measured activity: {grasso_final['activity_status'].sum()} (should be {len(grasso_final)} - all have activity)")
        
        # Check if any have 'n_averaged' column
        if 'n_averaged' in grasso_final.columns:
            avg_counts = grasso_final['n_averaged'].value_counts()
            print(f"\nAveraging summary:")
            for n, count in avg_counts.items():
                if n == 1:
                    print(f"  {count} unique sequences (no averaging needed)")
                else:
                    print(f"  {count} sequences averaged from {n} duplicates each")
        
        print("\nSample of Grasso entries (all should show Measured Activity):")
        for _, row in grasso_final.head(5).iterrows():
            sp_seq_short = row['sp_seq'][:30] + "..." if len(row['sp_seq']) > 30 else row['sp_seq']
            avg_note = f" (avg of {int(row['n_averaged'])} seqs)" if 'n_averaged' in row and row['n_averaged'] > 1 else ""
            print(f"  sp_seq: {sp_seq_short}")
            print(f"    activity_status: {'Measured' if row['activity_status'] else 'NOT MEASURED'} - Value: {row['enzyme_activity_numeric']:.6f}{avg_note}")

print("\n" + "="*60)
print("FINAL DATASET CHECK")
print("="*60)
print(f"Final dataset shape: {df_all.shape}")
print(f"Total rows: {len(df_all):,}")
print(f"Unique Author-Protein: {df_all['Author-Protein'].nunique()}")

# ==========================================================
# --- SAVE FINAL DATASET ---
# ==========================================================
print("\n" + "="*60)
print("SAVING FINAL DATASET")
print("="*60)

# Define output path
output_path = r"outputs\CoSMPAD_autor_ref.csv"  # Adjust path as needed

# Save to CSV
df_all.to_csv(output_path, index=False)
print(f"Dataset saved to: {output_path}")
print(f"Shape: {df_all.shape}")