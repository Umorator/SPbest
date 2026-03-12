import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as pe

# ----------------------------
# Styling
# ----------------------------
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# ----------------------------
# 1. Load data
# ----------------------------
df = pd.read_csv("outputs/CoSMPAD_autor_ref.csv")
df_bacillus = df[df['Species'].str.lower().str.startswith('bacillus', na=False)].copy()
print(f"Total Bacillus rows: {len(df_bacillus):,}")

# ----------------------------
# 2. Separate Khadye
# ----------------------------
khadye_name = 'Khadye - E7FHY4'
khadye_mask = df_bacillus['Author-Protein'].str.strip().str.lower() == khadye_name.lower()
df_khadye = df_bacillus[khadye_mask].copy()
df_rest = df_bacillus[~khadye_mask].copy()
print(f"Rest dataset: {len(df_rest)}, Khadye: {len(df_khadye)}")

# ----------------------------
# 3. Rank rest
# ----------------------------
def calculate_rank_score(group):
    group = group.copy()
    N_total = len(group)
    group['activity_numeric'] = pd.to_numeric(group['enzyme_activity'], errors='coerce')
    nr_mask = group['enzyme_activity'] == "NR"

    if (~nr_mask).sum() > 0:
        numeric_indices = group[~nr_mask].sort_values('activity_numeric', ascending=False).index
        for rank, idx in enumerate(numeric_indices, 1):
            group.loc[idx, 'rank'] = rank
            group.loc[idx, 'rank_score'] = (N_total - rank + 1) / N_total

    group.loc[nr_mask, 'rank'] = np.nan
    group.loc[nr_mask, 'rank_score'] = 0
    group['library_size'] = N_total
    return group

df_rest['group_key'] = df_rest['Author-Protein']
ranked_rest = [calculate_rank_score(g) for _, g in df_rest.groupby('group_key')]
df_rest_ranked = pd.concat(ranked_rest, ignore_index=True)

# ----------------------------
# 4. Rank Khadye (custom order)
# ----------------------------
def calculate_khadye_rank(group):
    group = group.copy()
    N_total = len(group)
    khadye_ranking_order = ['cith','lytb','ywsb','ybdg','phob','Abna','ykoj','ykwd','yobv','ybbe','ywad','apre']

    all_sps = group['SP name'].unique()
    sp_to_rank = {sp: i+1 for i, sp in enumerate(khadye_ranking_order) if sp in all_sps}
    remaining_sps = [sp for sp in all_sps if sp not in sp_to_rank]
    for i, sp in enumerate(remaining_sps, len(sp_to_rank)+1):
        sp_to_rank[sp] = i

    group['rank_key'] = group['SP name'].map(sp_to_rank)
    group['activity_numeric'] = pd.to_numeric(group['enzyme_activity'], errors='coerce').fillna(0)
    numeric_indices = group.sort_values(['rank_key','activity_numeric'], ascending=[True,False]).index

    for rank, idx in enumerate(numeric_indices, 1):
        group.loc[idx, 'rank'] = rank
        group.loc[idx, 'rank_score'] = (N_total - rank + 1) / N_total

    group.loc[group['enzyme_activity']=='NR','rank_score'] = 0
    group['library_size'] = N_total
    group.drop(['rank_key','activity_numeric'], axis=1, inplace=True)
    return group

if len(df_khadye) > 0:
    df_khadye = calculate_khadye_rank(df_khadye)

# ----------------------------
# 5. Merge ranked datasets
# ----------------------------
df_ranked = pd.concat([df_rest_ranked, df_khadye], ignore_index=True)

# ----------------------------
# 6. Calculate SP statistics
# ----------------------------
# ----------------------------
# 6. Calculate SP statistics
# ----------------------------
sp_stats = []
for sp_name, sp_group in df_ranked.groupby('SP name'):
    total_proteins = sp_group['UniprotKB/NCBI_POI'].nunique()
    nr_mask = sp_group['enzyme_activity'] == "NR"
    
    # Get the signal peptide sequence (should be the same for all rows of this SP)
    sp_seq = sp_group['sp_seq'].iloc[0] if len(sp_group) > 0 else ""
    
    proteins_measured = sp_group.loc[~nr_mask, 'UniprotKB/NCBI_POI'].nunique()
    proteins_nr = sp_group.loc[nr_mask, 'UniprotKB/NCBI_POI'].nunique()
    measured_group = sp_group[~nr_mask]
    mean_rank = measured_group['rank_score'].mean() if len(measured_group) > 0 else 0

    sp_stats.append({
        'SP name': sp_name,
        'sp_seq': sp_seq,  # Now correctly gets the sequence for this SP
        'proteins_measured': proteins_measured,
        'proteins_nr': proteins_nr,
        'proteins_total': total_proteins,
        'mean_rank': mean_rank
    })

versatility_df = pd.DataFrame(sp_stats)

# ----------------------------
# 6b. Versatility calculation - Per-protein optimal count
# ----------------------------
# Define optimal performance threshold (top 50% rank = 0.5)
OPTIMAL_THRESHOLD = 0.5
# Define confidence threshold - minimum proteins tested for full confidence
CONFIDENCE_THRESHOLD = 10

# Calculate per-protein optimal performance
optimal_counts = []
for sp_name, sp_group in df_ranked.groupby('SP name'):
    optimal_count = 0
    # Get unique proteins for this SP
    for protein in sp_group['UniprotKB/NCBI_POI'].unique():
        protein_data = sp_group[sp_group['UniprotKB/NCBI_POI'] == protein]
        # Get rank score for this SP in this protein (should be one row)
        rank_score = protein_data['rank_score'].iloc[0]
        if rank_score >= OPTIMAL_THRESHOLD:
            optimal_count += 1
    
    total_proteins = sp_group['UniprotKB/NCBI_POI'].nunique()
    optimal_counts.append({
        'SP name': sp_name,
        'optimal_count': optimal_count,
        'proteins_total_for_optimal': total_proteins
    })

# Merge optimal counts with versatility_df
optimal_df = pd.DataFrame(optimal_counts)
versatility_df = versatility_df.merge(optimal_df[['SP name', 'optimal_count']], on='SP name', how='left')

# Raw versatility = proportion of proteins where SP performed optimally
versatility_df['raw_versatility'] = versatility_df['optimal_count'] / versatility_df['proteins_total']

# Confidence penalty for SPs tested in few proteins
versatility_df['confidence_penalty'] = versatility_df['proteins_total'].apply(
    lambda x: min(1.0, x / CONFIDENCE_THRESHOLD)
)

# Final versatility score
versatility_df['versatility_score'] = versatility_df['raw_versatility'] * versatility_df['confidence_penalty']

# ----------------------------
# 6c. CONTINUOUS UNVERSATILITY CALCULATION
# ----------------------------
# Calculate unversatility as the average "badness" across all tested proteins
# For each protein: 
#   - If NR → contributes 1 (complete failure)
#   - If measured → contributes (1 - rank_score) (how bad the performance was)
# Then average across all proteins tested

unversatility_scores = []
for sp_name, sp_group in df_ranked.groupby('SP name'):
    total_proteins = sp_group['UniprotKB/NCBI_POI'].nunique()
    unversatility_sum = 0
    
    for protein in sp_group['UniprotKB/NCBI_POI'].unique():
        protein_data = sp_group[sp_group['UniprotKB/NCBI_POI'] == protein]
        rank_score = protein_data['rank_score'].iloc[0]
        
        # NR or very poor performance
        if rank_score == 0:  # NR or worst performer
            unversatility_sum += 1
        else:
            unversatility_sum += (1 - rank_score)
    
    unversatility_score = unversatility_sum / total_proteins
    unversatility_scores.append({
        'SP name': sp_name, 
        'unversatility_score': unversatility_score
    })

# Merge continuous unversatility scores
unv_df = pd.DataFrame(unversatility_scores)
versatility_df = versatility_df.merge(unv_df, on='SP name', how='left')

# Sort by versatility for display
versatility_df = versatility_df.sort_values('versatility_score', ascending=False).reset_index(drop=True)

# Display top SPs to verify
print("\nTop 10 Most Versatile SPs:")
print(versatility_df[['SP name', 'proteins_total', 'optimal_count', 'raw_versatility', 
                      'confidence_penalty', 'versatility_score']].head(10).to_string())

print("\nBottom 10 Least Versatile SPs (with continuous unversatility):")
print(versatility_df[['SP name', 'proteins_total', 'mean_rank', 'optimal_count', 
                      'unversatility_score']].tail(10).to_string())
# ----------------------------
# 6d. Collect protein names per SP for Optimal / Measured (not optimal) / NR
# ----------------------------
protein_name_stats = []

for sp_name, sp_group in df_ranked.groupby('SP name'):
    nr_mask = sp_group['enzyme_activity'] == "NR"
    measured_group = sp_group[~nr_mask]

    optimal_mask = measured_group['rank_score'] >= OPTIMAL_THRESHOLD
    measured_not_optimal_mask = measured_group['rank_score'] < OPTIMAL_THRESHOLD

    # Gather protein names (can repeat if needed)
    optimal_proteins = measured_group.loc[optimal_mask, 'Protein name'].tolist()
    measured_not_optimal_proteins = measured_group.loc[measured_not_optimal_mask, 'Protein name'].tolist()
    nr_proteins = sp_group.loc[nr_mask, 'Protein name'].tolist()

    protein_name_stats.append({
        'SP name': sp_name,
        'Optimal Proteins': "; ".join(optimal_proteins),
        'Measured (Not Optimal) Proteins': "; ".join(measured_not_optimal_proteins),
        'NR Proteins': "; ".join(nr_proteins)
    })

protein_names_df = pd.DataFrame(protein_name_stats)

# Merge these columns into your versatility_df
versatility_df = versatility_df.merge(protein_names_df, on='SP name', how='left')

# ----------------------------
# 7. Plots
# ----------------------------
# Create figure with side-by-side histograms - SAME COLOR
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Versatility histogram - steelblue
axes[0].hist(versatility_df['versatility_score'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(versatility_df['versatility_score'].mean(), linestyle='--', color='red', 
                label=f"Mean: {versatility_df['versatility_score'].mean():.3f}")
axes[0].axvline(versatility_df['versatility_score'].median(), linestyle='--', color='green', 
                label=f"Median: {versatility_df['versatility_score'].median():.3f}")
axes[0].set_title("Distribution of Versatility Scores")
axes[0].set_xlabel("Versatility Score")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# Unversatility histogram - ALSO steelblue for consistency
axes[1].hist(versatility_df['unversatility_score'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].axvline(versatility_df['unversatility_score'].mean(), linestyle='--', color='red', 
                label=f"Mean: {versatility_df['unversatility_score'].mean():.3f}")
axes[1].axvline(versatility_df['unversatility_score'].median(), linestyle='--', color='green', 
                label=f"Median: {versatility_df['unversatility_score'].median():.3f}")
axes[1].set_title("Distribution of Unversatility Scores")
axes[1].set_xlabel("Unversatility Score")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig("Distribution_Scores_Comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# Versatility scatter (Top Performers)
fig, ax = plt.subplots(figsize=(12,7))
scatter = ax.scatter(versatility_df['proteins_total'], versatility_df['mean_rank'],
                     c=versatility_df['versatility_score'], cmap="viridis", s=120, alpha=0.8,
                     edgecolors="black", linewidth=0.5)
ax.set_xlabel("Total Proteins Tested")
ax.set_ylabel("Mean Rank Score")
ax.set_title("Signal Peptide Versatility: Top Performers Across Protein Contexts", pad=15)
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
cbar = plt.colorbar(scatter)
cbar.set_label("Versatility Score", fontsize=11)

# Add vertical line for confidence threshold
ax.axvline(x=CONFIDENCE_THRESHOLD, linestyle='--', color='gray', alpha=0.5, 
           label=f'Confidence Threshold (n={CONFIDENCE_THRESHOLD})')

# Label top 15 most versatile SPs
top15 = versatility_df.nlargest(15, 'versatility_score')
texts = []
for _, row in top15.iterrows():
    txt = ax.text(row['proteins_total'], row['mean_rank'], row['SP name'],
                  fontsize=10, weight='bold', color='darkred')
    txt.set_path_effects([pe.Stroke(linewidth=2, foreground='white'), pe.Normal()])
    texts.append(txt)
adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.6), 
            expand_points=(1.3, 1.3), force_text=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("Versatility_Top_Performers.png", dpi=300, bbox_inches="tight")
plt.show()

# Unversatility scatter (Worst Performers - continuous scale)
fig, ax = plt.subplots(figsize=(12,7))

# Filter for SPs tested in at least 5 proteins to avoid single-library wonders
min_proteins_for_consideration = 5
mask = versatility_df['proteins_total'] >= min_proteins_for_consideration

scatter = ax.scatter(versatility_df.loc[mask, 'proteins_total'], 
                     versatility_df.loc[mask, 'mean_rank'],
                     c=versatility_df.loc[mask, 'unversatility_score'], 
                     cmap="Reds", s=150, alpha=0.8,
                     edgecolors="black", linewidth=0.5, vmin=0.5, vmax=1.0)

# Plot the filtered-out points in light gray for context
ax.scatter(versatility_df.loc[~mask, 'proteins_total'], 
           versatility_df.loc[~mask, 'mean_rank'],
           c='lightgray', s=80, alpha=0.4,
           edgecolors='gray', linewidth=0.3, label='Tested in <5 proteins')

ax.set_xlabel("Total Proteins Tested")
ax.set_ylabel("Mean Rank Score")
ax.set_title("Signal Peptide Unversatility: Tested Broadly but Perform Poorly", pad=15)
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
cbar = plt.colorbar(scatter)
cbar.set_label("Unversatility Score", fontsize=11)

# Add vertical line for confidence threshold
ax.axvline(x=CONFIDENCE_THRESHOLD, linestyle='--', color='gray', alpha=0.5, 
           label=f'Confidence Threshold (n={CONFIDENCE_THRESHOLD})')

# Identify truly unversatile SPs: high unversatility score and tested broadly
truly_unversatile = versatility_df[
    (versatility_df['unversatility_score'] > 0.8) & 
    (versatility_df['proteins_total'] >= CONFIDENCE_THRESHOLD)
].sort_values('unversatility_score', ascending=False)

# Label top 10 truly unversatile SPs
texts = []
for _, row in truly_unversatile.head(10).iterrows():
    txt = ax.text(row['proteins_total'], row['mean_rank'], row['SP name'],
                  fontsize=10, weight='bold', color='darkblue')
    txt.set_path_effects([pe.Stroke(linewidth=2, foreground='white'), pe.Normal()])
    texts.append(txt)
adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.6), 
            expand_points=(1.3, 1.3), force_text=0.5)

plt.legend()
plt.tight_layout()
plt.savefig("Unversatility_Worst_Performers.png", dpi=300, bbox_inches="tight")
plt.show()

# ----------------------------
# 8. Output Table (with protein names)
# ----------------------------
display_df = versatility_df.copy()
display_df = versatility_df.copy()
display_df = display_df.rename(columns={
    'SP name':'SP Name',
    'sp_seq':'sp_seq',   # <-- add sequence column
    'proteins_measured':'Measured',
    'proteins_nr':'NR',
    'proteins_total':'Total',
    'mean_rank':'Mean Rank',
    'optimal_count':'Optimal',
    'raw_versatility':'Raw Versatility',
    'confidence_penalty':'Confidence',
    'versatility_score':'Versatility Score',
    'unversatility_score':'Unversatility Score'
})

# Round numeric columns for display
numeric_cols = ['Mean Rank', 'Raw Versatility', 'Confidence', 'Versatility Score', 'Unversatility Score']
display_df[numeric_cols] = display_df[numeric_cols].round(4)

# Save full dataset with protein names
display_df.to_csv("versatility_analysis_complete.csv", index=False)
print("\n" + "="*80)
print("COMPLETE DATASET SAVED TO: versatility_analysis_complete.csv")
print("="*80)

# Top 20 Versatile SPs (include protein names)
print("\n" + "="*80)
print("TOP 20 SIGNAL PEPTIDES BY VERSATILITY SCORE")
print("="*80)
top20_versatile = display_df.sort_values('Versatility Score', ascending=False).head(20)
print(top20_versatile[['SP Name', 'Total', 'Measured', 'NR', 'Optimal', 
                       'Optimal Proteins', 'Measured (Not Optimal) Proteins', 'NR Proteins',
                       'Mean Rank', 'Raw Versatility', 'Confidence', 'Versatility Score']].to_string(index=False))

# Top 20 Unversatile SPs (filtered for those tested in at least 5 proteins, include protein names)
print("\n" + "="*80)
print("TOP 20 TRULY UNVERSATILE SPs (Tested in ≥5 proteins)")
print("="*80)
truly_unv = display_df[display_df['Total'] >= 5].sort_values('Unversatility Score', ascending=False).head(50)
print(truly_unv[['SP Name', 'Total', 'Measured', 'NR', 'Optimal',
                 'Optimal Proteins', 'Measured (Not Optimal) Proteins', 'NR Proteins',
                 'Mean Rank', 'Raw Versatility', 'Confidence', 'Unversatility Score']].to_string(index=False))