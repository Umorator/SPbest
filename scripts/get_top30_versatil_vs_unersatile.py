import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load the versatility analysis file
# ----------------------------
file_path = r"C:\Users\rafae\OneDrive\Documents\PhD_2026\Thesis\Chapter_2_SPbest\SP_best_Repo\versatility_analysis_complete.csv"
df = pd.read_csv(file_path)

print(f"Total SPs in dataset: {len(df)}")
print(f"Columns available: {df.columns.tolist()}")
print("\n" + "="*80)

# ----------------------------
# Filter for minimally tested SPs
# ----------------------------
min_proteins = 3
df_filtered = df[df['Total'] >= min_proteins].copy()

print(f"SPs after filtering (Total >= {min_proteins}): {len(df_filtered)}")

# ----------------------------
# Determine top percentage
# ----------------------------
top_percent = 0.30
top_n = int(len(df_filtered) * top_percent)

print(f"Selecting top {top_percent*100:.0f}% -> {top_n} SPs per class")
print("\n" + "="*80)

# ----------------------------
# Get top versatile SPs
# ----------------------------
top_versatile = df_filtered.nlargest(top_n, 'Versatility Score')[
    ['SP Name', 'sp_seq', 'Measured', 'NR', 'Total',
     'Mean Rank', 'Optimal', 'Versatility Score', 'Unversatility Score']
].copy()

top_versatile['Classification'] = 'Versatile'

print(f"Top {top_n} Versatile SPs:")
print(top_versatile.to_string(index=False))
print("\n" + "="*80)

# ----------------------------
# Get top unversatile SPs
# ----------------------------
top_unversatile = df_filtered.nlargest(top_n, 'Unversatility Score')[
    ['SP Name', 'sp_seq', 'Measured', 'NR', 'Total',
     'Mean Rank', 'Optimal', 'Versatility Score', 'Unversatility Score']
].copy()

top_unversatile['Classification'] = 'Unversatile'

print(f"Top {top_n} Unversatile SPs:")
print(top_unversatile.to_string(index=False))
print("\n" + "="*80)

# ----------------------------
# Combine datasets
# ----------------------------
combined_df = pd.concat([top_versatile, top_unversatile], ignore_index=True)

combined_df['Rank_in_Class'] = combined_df.groupby('Classification').cumcount() + 1

combined_df = combined_df[['Rank_in_Class', 'Classification', 'SP Name', 'sp_seq',
                           'Measured', 'NR', 'Total', 'Mean Rank', 'Optimal',
                           'Versatility Score', 'Unversatility Score']]

numeric_cols = ['Mean Rank', 'Versatility Score', 'Unversatility Score']
combined_df[numeric_cols] = combined_df[numeric_cols].round(4)

print("Combined Dataset:")
print(combined_df.to_string(index=False))

# ----------------------------
# Save CSV
# ----------------------------
percent_label = int(top_percent * 100)

output_path = rf"C:\Users\rafae\OneDrive\Documents\PhD_2026\Thesis\Chapter_2_SPbest\SP_best_Repo\versatile_vs_unversatile_top{percent_label}percent.csv"
combined_df.to_csv(output_path, index=False)

print(f"\n✅ Saved combined dataset to: {output_path}")

# ----------------------------
# Quick statistics
# ----------------------------
print("\n" + "="*80)
print("QUICK STATISTICS:")

print(f"Versatile SPs - Mean versatility: {top_versatile['Versatility Score'].mean():.4f}")
print(f"Versatile SPs - Mean unversatility: {top_versatile['Unversatility Score'].mean():.4f}")
print(f"Versatile SPs - Avg proteins tested: {top_versatile['Total'].mean():.1f}")
print(f"Versatile SPs - Avg optimal count: {top_versatile['Optimal'].mean():.1f}")
print(f"Versatile SPs - Avg Mean Rank: {top_versatile['Mean Rank'].mean():.4f}")

print()

print(f"Unversatile SPs - Mean versatility: {top_unversatile['Versatility Score'].mean():.4f}")
print(f"Unversatile SPs - Mean unversatility: {top_unversatile['Unversatility Score'].mean():.4f}")
print(f"Unversatile SPs - Avg proteins tested: {top_unversatile['Total'].mean():.1f}")
print(f"Unversatile SPs - Avg optimal count: {top_unversatile['Optimal'].mean():.1f}")
print(f"Unversatile SPs - Avg Mean Rank: {top_unversatile['Mean Rank'].mean():.4f}")

# ----------------------------
# Overlap check
# ----------------------------
versatile_names = set(top_versatile['SP Name'])
unversatile_names = set(top_unversatile['SP Name'])

overlap = versatile_names.intersection(unversatile_names)

if overlap:
    print(f"\n⚠️  Warning: {len(overlap)} SPs appear in both lists: {overlap}")
else:
    print("\n✅ No overlap between versatile and unversatile lists - good separation!")

# ----------------------------
# Summary statistics table
# ----------------------------
summary_stats = pd.DataFrame({
    'Metric': ['Mean Versatility', 'Mean Unversatility', 'Avg Proteins Tested',
               'Avg Optimal Count', 'Avg Mean Rank'],
    'Versatile SPs': [
        top_versatile['Versatility Score'].mean(),
        top_versatile['Unversatility Score'].mean(),
        top_versatile['Total'].mean(),
        top_versatile['Optimal'].mean(),
        top_versatile['Mean Rank'].mean()
    ],
    'Unversatile SPs': [
        top_unversatile['Versatility Score'].mean(),
        top_unversatile['Unversatility Score'].mean(),
        top_unversatile['Total'].mean(),
        top_unversatile['Optimal'].mean(),
        top_unversatile['Mean Rank'].mean()
    ]
})

print("\n" + "="*80)
print("SUMMARY STATISTICS COMPARISON:")
print(summary_stats.round(4).to_string(index=False))