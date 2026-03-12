import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Load and process the data
path_data = r"data\CoSMPAD.csv"
df = pd.read_csv(path_data)

# Apply filtering code with special handling for specific datasets
dfs = [x for _, x in df.groupby(['UniprotKB/NCBI_POI','Reference']) if len(x) > 1]

def most_common(lst):
    return max(set(lst), key=lst.count)

dfss_filtered = []
special_handling_log = []

for i in dfs:
    ref = i['Reference'].iloc[0]
    protein = i['UniprotKB/NCBI_POI'].iloc[0]
    
    # Apply basic filters (promoter, time, host)
    a = i[i.Promoter == most_common(list(i.Promoter.values))]
    a = a[a['time (h)'] == most_common(list(a['time (h)'].values))]
    a = a[a['Host'] == most_common(list(a['Host'].values))]
    
    # Check for special cases that need different filtering
    is_fu_p06279 = ('Fu' in str(ref) and 'P06279' in str(protein))
    is_zhang_a0a172wbp7 = ('Zhang' in str(ref) and 'A0A172WBP7' in str(protein))
    is_zhang_w_specific = ('Zhang,W.' in str(ref))
    is_yao = ('Yao' in str(ref))
    is_tsuji = ('Tsuji' in str(ref))

    # Track if this dataset gets special filtering treatment
    has_special_treatment = is_fu_p06279 or is_zhang_a0a172wbp7 or is_zhang_w_specific or is_yao or is_tsuji
    
    # Handle different filtering protocols
    if is_tsuji:
        # Tsuji special case
        final = i[i.Promoter == most_common(list(i.Promoter.values))]
        handling = "Tsuji: special (no host/time filters)"
        
    elif is_yao:
        # Yao special case
        a_yao = i[i.Promoter == most_common(list(i.Promoter.values))]
        final = a_yao[a_yao['Host'] == most_common(list(a_yao['Host'].values))]
        handling = "Yao: special (no time filter)"
        
    elif is_zhang_w_specific:
        # Zhang,W. special case
        final = i[i.Promoter == 'Pglvm']
        handling = "Zhang,W.: specific promoter only"
        
    elif is_fu_p06279:
        # Special handling for Fu-P06279: preserve all active SPs
        original_data = i
        
        # Get all SPs that have at least one non-NR activity
        sps_with_activity = []
        if 'enzyme_activity' in original_data.columns:
            for sp in original_data['SP name'].unique():
                sp_data = original_data[original_data['SP name'] == sp]
                if not (sp_data['enzyme_activity'] == "NR").all():
                    sps_with_activity.append(sp)
        
        # Get rows for active SPs (keep ALL conditions)
        if sps_with_activity:
            active_sp_rows = original_data[original_data['SP name'].isin(sps_with_activity)]
        else:
            active_sp_rows = pd.DataFrame()
        
        # For inactive SPs, apply normal filtering
        sps_without_activity = [sp for sp in original_data['SP name'].unique() if sp not in sps_with_activity]
        
        if sps_without_activity:
            inactive_sp_rows = a[a['SP name'].isin(sps_without_activity)]
            if 'cultivation_flask' in inactive_sp_rows.columns:
                inactive_sp_rows = inactive_sp_rows[inactive_sp_rows['cultivation_flask'] == 
                                                    most_common(list(inactive_sp_rows['cultivation_flask'].values))]
        else:
            inactive_sp_rows = pd.DataFrame()
        
        final = pd.concat([active_sp_rows, inactive_sp_rows], ignore_index=True)
        handling = f"Fu-P06279: preserved {len(sps_with_activity)} active SPs"
        
    elif is_zhang_a0a172wbp7:
        # Special handling for Zhang-A0A172WBP7
        original_data = i
        
        # Get all SPs that have at least one non-NR activity
        sps_with_activity = []
        if 'enzyme_activity' in original_data.columns:
            for sp in original_data['SP name'].unique():
                sp_data = original_data[original_data['SP name'] == sp]
                if not (sp_data['enzyme_activity'] == "NR").all():
                    sps_with_activity.append(sp)
        
        # Get rows for active SPs (keep ALL conditions)
        if sps_with_activity:
            active_sp_rows = original_data[original_data['SP name'].isin(sps_with_activity)]
        else:
            active_sp_rows = pd.DataFrame()
        
        # For inactive SPs, apply normal filtering
        sps_without_activity = [sp for sp in original_data['SP name'].unique() if sp not in sps_with_activity]
        
        if sps_without_activity:
            inactive_sp_rows = a[a['SP name'].isin(sps_without_activity)]
            if 'cultivation_flask' in inactive_sp_rows.columns:
                inactive_sp_rows = inactive_sp_rows[inactive_sp_rows['cultivation_flask'] == 
                                                    most_common(list(inactive_sp_rows['cultivation_flask'].values))]
        else:
            inactive_sp_rows = pd.DataFrame()
        
        final = pd.concat([active_sp_rows, inactive_sp_rows], ignore_index=True)
        handling = f"Zhang-A0A172WBP7: preserved {len(sps_with_activity)} active SPs"
        
    else:
        # Standard filtering for all others
        if 'cultivation_flask' in a.columns:
            final = a[a['cultivation_flask'] == most_common(list(a['cultivation_flask'].values))]
        else:
            final = a
        handling = "Standard filtering"
        has_special_treatment = False
    
    dfss_filtered.append(final)
    special_handling_log.append({
        'Protein': protein,
        'Reference': ref[:50] + '...',
        'Handling': handling,
        'Has_Special_Treatment': has_special_treatment,
        'Original_SPs': i['SP name'].nunique(),
        'Final_SPs': final['SP name'].nunique()
    })

df_all = pd.concat(dfss_filtered, join='outer', axis=0)

# Create summary dataframe using SP NAMES
summary_list = []
for df_protein in [x for _, x in df_all.groupby(['UniprotKB/NCBI_POI','Reference'])]:
    protein = df_protein['UniprotKB/NCBI_POI'].iloc[0]
    reference = df_protein['Reference'].iloc[0]
    
    # Find the SP name column
    sp_name_col = None
    possible_names = ['Signal peptide', 'SP_name', 'signal_peptide', 'sp_name', 'Name', 'name']
    for col in possible_names:
        if col in df_protein.columns:
            sp_name_col = col
            break
    
    # If no name column found, fall back to SP name
    if sp_name_col is None:
        sp_name_col = 'SP name'
    
    # Get unique SP names
    unique_sp_names = df_protein[sp_name_col].unique()
    num_sps = len(unique_sp_names)
    
    # Count activities correctly based on SP names
    num_activities = 0
    for sp_name in unique_sp_names:
        sp_data = df_protein[df_protein[sp_name_col] == sp_name]
        
        if 'enzyme_activity' in sp_data.columns:
            if not (sp_data['enzyme_activity'] == "NR").all():
                num_activities += 1
        elif 'activity' in sp_data.columns:
            if sp_data['activity'].notna().any():
                num_activities += 1
    
    # Get first author name from reference
    if ',' in reference:
        first_author = reference.split(',')[0].strip()
        # Clean up common patterns
        first_author = first_author.replace('Fu,G.', 'Fu').replace('Zhang,W.', 'Zhang').replace('Song,Y.', 'Song')
    else:
        first_author = reference[:10]
    
    # Create label: FirstAuthor-ProteinID
    protein_short = protein.split('|')[-1] if '|' in protein else protein
    label = f"{first_author}-{protein_short}"
    
    # Check if this dataset had special filtration
    is_fu_p06279 = ('Fu' in str(reference) and 'P06279' in protein)
    is_zhang_a0a172wbp7 = ('Zhang' in str(reference) and 'A0A172WBP7' in protein)
    is_zhang_w = ('Zhang,W.' in str(reference))
    is_yao = ('Yao' in str(reference))
    is_tsuji = ('Tsuji' in str(reference))
    
    has_special_filtration = is_fu_p06279 or is_zhang_a0a172wbp7 or is_zhang_w or is_yao or is_tsuji
    
    summary_list.append({
        'Protein': protein,
        'Reference': reference,
        'First_Author': first_author,
        'Label': label,
        'Num_SPs': num_sps,
        'Num_Activities': num_activities,
        'Missing_Activities': num_sps - num_activities,
        'Activity_Pct': (num_activities/num_sps*100) if num_sps > 0 else 0,
        'Has_Special_Filtration': has_special_filtration,
        'Filtration_Type': handling if has_special_filtration else 'Standard'
    })

summary_df = pd.DataFrame(summary_list)

# Sort by number of SPs
summary_df = summary_df.sort_values('Num_SPs', ascending=False).reset_index(drop=True)

# Calculate averages
avg_SPs_tested = summary_df['Num_SPs'].mean()
avg_SPs_with_activity = summary_df['Num_Activities'].mean()

print("\n" + "="*80)
print("SPECIAL FILTRATION SUMMARY")
print("="*80)
handling_df = pd.DataFrame(special_handling_log)
print(handling_df[handling_df['Has_Special_Treatment']].to_string())

print("\n" + "="*80)
print(f"Average SPs tested per dataset: {avg_SPs_tested:.1f}")
print(f"Average SPs with activity per dataset: {avg_SPs_with_activity:.1f}")
print("="*80)

# Create the plot
fig, ax = plt.subplots(figsize=(20, 12))

# Create stacked bars
bar_width = 0.8
x_pos = np.arange(len(summary_df))

# Plot activities (measured) in green
bars1 = ax.bar(x_pos, summary_df['Num_Activities'], bar_width, 
                label='SPs with activity data', 
                color='#2ecc71', edgecolor='black', linewidth=0.5)

# Plot missing activities in red with hatch
bars2 = ax.bar(x_pos, summary_df['Missing_Activities'], bar_width, 
                bottom=summary_df['Num_Activities'], 
                label='SPs without activity data', 
                color='#e74c3c', edgecolor='black', linewidth=0.5, 
                hatch='///', alpha=0.7)

# Add total SPs count on top of bars
for i, (idx, row) in enumerate(summary_df.iterrows()):
    total = row['Num_SPs']
    if total > 0:
        ax.text(i, total + 0.5, str(int(total)), 
                ha='center', va='bottom', fontsize=8, fontweight='bold')

# Add activity count
for i, (idx, row) in enumerate(summary_df.iterrows()):
    activities = row['Num_Activities']
    if activities > 0:
        green_height = activities
        total_height = row['Num_SPs']
        
        if green_height > total_height * 0.1:
            ax.text(i, green_height/2, str(int(activities)), 
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        else:
            ax.text(i, green_height + 0.5, str(int(activities)), 
                    ha='center', va='bottom', fontsize=7, color='black', fontweight='bold')

# Customize the plot
ax.set_ylabel('Number of Signal Peptides', fontsize=14, fontweight='bold')
ax.set_xlabel('Author-Protein', fontsize=14, fontweight='bold')
ax.set_title('Signal Peptide Data Completeness: Tested SPs vs. Those with Activity Measurements', 
              fontsize=16, fontweight='bold', pad=20)

# Set x-axis labels
ax.set_xticks(x_pos)
ax.set_xticklabels(summary_df['Label'], rotation=90, fontsize=8)

# Add average lines
ax.axhline(y=avg_SPs_tested, color='#f39c12', linestyle='--', linewidth=2.5, alpha=0.8, 
           label=f'Average SPs tested: {avg_SPs_tested:.1f}')
ax.axhline(y=avg_SPs_with_activity, color='#3498db', linestyle='--', linewidth=2.5, alpha=0.8, 
           label=f'Average SPs with activity: {avg_SPs_with_activity:.1f}')

# Color-code x-axis labels - ONLY for datasets with special filtration
for i, (idx, row) in enumerate(summary_df.iterrows()):
    if row['Has_Special_Filtration']:
        ax.get_xticklabels()[i].set_color('darkred')
        ax.get_xticklabels()[i].set_fontweight('bold')
    else:
        ax.get_xticklabels()[i].set_color('black')

# Legend
legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='black', label='SPs with activity data'),
    Patch(facecolor='#e74c3c', edgecolor='black', hatch='///', alpha=0.7, label='SPs without activity data'),
    Line2D([0], [0], color='#f39c12', linestyle='--', linewidth=2.5, label=f'Avg SPs tested: {avg_SPs_tested:.1f}'),
    Line2D([0], [0], color='#3498db', linestyle='--', linewidth=2.5, label=f'Avg SPs with activity: {avg_SPs_with_activity:.1f}')
]

ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.2, axis='y', linestyle='--')
ax.set_ylim(0, summary_df['Num_SPs'].max() * 1.1)

plt.tight_layout()
plt.savefig('data_sparsity_author_protein.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary
print("\n" + "="*80)
print("DATASETS WITH SPECIAL FILTRATION")
print("="*80)
special_df = summary_df[summary_df['Has_Special_Filtration']]
for _, row in special_df.iterrows():
    print(f"{row['Label']:35} | SPs: {row['Num_SPs']:3} | Activity: {row['Num_Activities']:3} | {row['Activity_Pct']:5.1f}% | {row['Filtration_Type']}")

# Save summary
summary_df.to_csv('data_sparsity_author_protein.csv', index=False)
print(f"\nAnalysis saved to 'data_sparsity_author_protein.csv'")