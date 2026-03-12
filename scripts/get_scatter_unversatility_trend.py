import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

# Load your data
df = pd.read_csv("versatility_vs_unversatility_features_with_proba.csv")
df.columns

# Rename probability columns for easier handling
proba_cols = [col for col in df.columns if col.startswith('proba_')]
for col in proba_cols:
    df[col.replace('proba_', '')] = df[col]

# Separate classes
versatile = df[df['Classification'] == 'Versatile'].copy()
unversatile = df[df['Classification'] == 'Unversatile'].copy()

print("=== DATA OVERVIEW ===")
print(f"Total samples: {len(df)}")
print(f"Versatile: {len(versatile)} samples")
print(f"Unversatile: {len(unversatile)} samples")


# ============================================
# PLOT 1: Scatterplot with regression lines for both classes (FIXED: both dashed)
# ============================================
plt.figure(figsize=(14, 10))

# Plot versatile (red circles)
plt.scatter(versatile['Sec/SPI'], versatile['TM/Globular'], 
           c='#FF6B6B', alpha=0.7, s=200, label='Versatile', 
           edgecolors='darkred', linewidth=1.5, zorder=2)

# Plot unversatile (blue squares)
plt.scatter(unversatile['Sec/SPI'], unversatile['TM/Globular'], 
           c='#4A90E2', alpha=0.7, s=200, label='Unversatile', 
           edgecolors='darkblue', linewidth=1.5, marker='s', zorder=2)

# Function to add regression line with statistics
def add_regression_line(data, color, label_prefix, line_style='--'):
    x = data['Sec/SPI'].values
    y = data['TM/Globular'].values
    
    # Remove any NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) > 1:
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        
        # Create regression line
        x_line = np.linspace(0, 1, 100)
        y_line = slope * x_line + intercept
        
        # Plot regression line - BOTH NOW USE SAME LINE STYLE (dashed)
        plt.plot(x_line, y_line, color=color, linewidth=3, alpha=0.8, linestyle=line_style,
                label=f'{label_prefix} (R²={r_value**2:.3f}, p={p_value:.2e})')
        
        # Add confidence interval
        from scipy import stats as statsci
        n = len(x_clean)
        t_value = statsci.t.ppf(0.975, n-2)
        y_err = t_value * std_err * np.sqrt(1/n + (x_line - np.mean(x_clean))**2 / np.sum((x_clean - np.mean(x_clean))**2))
        plt.fill_between(x_line, y_line - y_err, y_line + y_err, color=color, alpha=0.1)
        
        return {'slope': slope, 'intercept': intercept, 'r2': r_value**2, 'p_value': p_value}
    return None

# Add regression lines for both classes - BOTH NOW USE DASHED LINE STYLE
unversatile_stats = add_regression_line(unversatile, 'blue', 'Unversatile trend', '--')
versatile_stats = add_regression_line(versatile, 'red', 'Versatile trend', '--')

# Add labels and title
plt.xlabel('Sec/SPI Probability', fontsize=16, fontweight='bold')
plt.ylabel('TM/Globular Probability', fontsize=16, fontweight='bold')
plt.title('Signal Peptide Classification: Contrasting Trends', fontsize=18, fontweight='bold')

# Customize legend
plt.legend(fontsize=12, framealpha=0.9, loc='upper right')

# Set axis limits
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

# Add grid
plt.grid(True, alpha=0.2, linestyle='--')

# Add quadrant lines at 0.5
plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('scatterplot_both_regressions.png', dpi=300, bbox_inches='tight')
plt.show()

# Print regression statistics comparison
print("\n" + "="*60)
print("REGRESSION ANALYSIS COMPARISON")
print("="*60)

print("\n🔵 UNVERSATILE:")
if unversatile_stats:
    print(f"  Slope: {unversatile_stats['slope']:.3f}")
    print(f"  Intercept: {unversatile_stats['intercept']:.3f}")
    print(f"  R²: {unversatile_stats['r2']:.3f}")
    print(f"  P-value: {unversatile_stats['p_value']:.2e}")
    print(f"  Equation: TM/Globular = {unversatile_stats['slope']:.3f} × Sec/SPI + {unversatile_stats['intercept']:.3f}")

print("\n🔴 VERSATILE:")
if versatile_stats:
    print(f"  Slope: {versatile_stats['slope']:.3f}")
    print(f"  Intercept: {versatile_stats['intercept']:.3f}")
    print(f"  R²: {versatile_stats['r2']:.3f}")
    print(f"  P-value: {versatile_stats['p_value']:.2e}")
    print(f"  Equation: TM/Globular = {versatile_stats['slope']:.3f} × Sec/SPI + {versatile_stats['intercept']:.3f}")



# ============================================
# Boxplot versatility
# ============================================
score_column = 'Versatility Score'

plot_df = pd.concat([
    pd.DataFrame({
        'Class': 'Versatile',
        'Score': versatile[score_column]
    }),
    pd.DataFrame({
        'Class': 'Unversatile',
        'Score': unversatile[score_column]
    })
])

# ============================================
# Plot
# ============================================
sns.set(style="whitegrid", context="talk")

plt.figure(figsize=(8,7))

# Boxplot
ax = sns.boxplot(
    data=plot_df,
    x='Class',
    y='Score',
    palette=['#FF6B6B', '#4A90E2'],
    width=0.5,
    fliersize=0
)

# Swarmplot (individual points)
sns.swarmplot(
    data=plot_df,
    x='Class',
    y='Score',
    color='black',
    size=5,
    alpha=0.8
)

# ============================================
# Statistical test
# ============================================
stat, p_val = stats.mannwhitneyu(
    versatile[score_column],
    unversatile[score_column],
    alternative='two-sided'
)

print(f"Mann-Whitney U p-value: {p_val}")

if p_val < 0.001:
    sig = "***"
elif p_val < 0.01:
    sig = "**"
elif p_val < 0.05:
    sig = "*"
else:
    sig = "ns"

# ============================================
# Significance annotation
# ============================================
y_max = plot_df['Score'].max()

plt.plot([0,0,1,1],
         [y_max*1.02, y_max*1.05, y_max*1.05, y_max*1.02],
         lw=1.5,
         color='black')

plt.text(
    0.5,
    y_max*1.06,
    sig,
    ha='center',
    fontsize=18,
    fontweight='bold'
)

plt.ylim(top=y_max*1.1)

# ============================================
# Labels
# ============================================
plt.ylabel("Versatility Score")
plt.xlabel("")
plt.title("Distribution of Versatility Score by Class")

plt.tight_layout()

# Save
plt.savefig(
    "versatility_score_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================
# PLOT 3: Second Choice Analysis with Box Plot for Sec/SPI sequences
# ============================================
# Filter Sec/SPI sequences
# ============================================
versatile_sec = versatile[versatile['Sec/SPI'] > 0.5].copy()
unversatile_sec = unversatile[unversatile['Sec/SPI'] > 0.5].copy()

print("\n" + "="*50)
print("SEC/SPI SEQUENCES ANALYSIS")
print("="*50)

print(f"Versatile Sec/SPI: {len(versatile_sec)} ({len(versatile_sec)/len(versatile)*100:.1f}%)")
print(f"Unversatile Sec/SPI: {len(unversatile_sec)} ({len(unversatile_sec)/len(unversatile)*100:.1f}%)")

# ============================================
# Define probability columns
# ============================================
prob_types = ['Sec/SPI', 'Sec/SPII', 'Sec/SPIII', 'Tat/SPI', 'Tat/SPII', 'TM/Globular']

def get_second_choice(row):
    probs = {pt: row[pt] for pt in prob_types}
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return sorted_probs[1][0]

versatile_sec['second_type'] = versatile_sec.apply(get_second_choice, axis=1)
unversatile_sec['second_type'] = unversatile_sec.apply(get_second_choice, axis=1)

# ============================================
# Combine dataframe
# ============================================
plot_df = pd.concat([
    pd.DataFrame({
        "Class": "Versatile",
        "Second Choice": versatile_sec['second_type']
    }),
    pd.DataFrame({
        "Class": "Unversatile",
        "Second Choice": unversatile_sec['second_type']
    })
])

# Convert counts to percentages
count_df = (
    plot_df
    .groupby(['Class','Second Choice'])
    .size()
    .reset_index(name='count')
)
count_df['percent'] = count_df.groupby('Class')['count'].transform(lambda x: x/x.sum()*100)

# ============================================
# Remove second-choice types that are 0% for both classes
# ============================================
total_percent = count_df.groupby('Second Choice')['percent'].sum().reset_index()
non_zero_types = total_percent[total_percent['percent'] > 0]['Second Choice'].tolist()
count_df = count_df[count_df['Second Choice'].isin(non_zero_types)]

# Ensure the order of bars and hue
second_choice_order = non_zero_types  # only include non-zero types
hue_order = ['Versatile', 'Unversatile']  # Versatile always first

# ============================================
# ============================================
# Plot
# ============================================
sns.set(style="whitegrid", context="talk")

plt.figure(figsize=(9,6))

ax = sns.barplot(
    data=count_df,
    x="Second Choice",
    y="percent",
    hue="Class",
    palette=["#FF6B6B", "#4A90E2"],  # Versatile, Unversatile
    hue_order=["Versatile", "Unversatile"],  # ensure first bar is Versatile
    order=second_choice_order
)

# Labels and title
plt.ylabel("Percentage (%)")
plt.xlabel("Second Most Probable SP Type")
plt.title("Second Most Probable Class for Sec/SPI Sequences")
plt.xticks(rotation=40)

# Add percentage labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f%%", fontsize=10, weight='bold')

# Legend in a box
ax.legend(
    fontsize=10,
    frameon=True,
    facecolor='white',
    edgecolor='black',
    loc='upper right'
)

plt.tight_layout()

# Save figure
plt.savefig(
    "second_choice_class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

import umap
from sklearn.preprocessing import StandardScaler

# ============================================
# DEFINE COLUMNS TO REMOVE
# ============================================

remove_cols = [
    'Rank_in_Class',
    'Classification',
    'SP Name',
    'sp_seq',
    'Measured',
    'NR',
    'Total',
    'Mean Rank',
    'Optimal',
    'Versatility Score',
    'Unversatility Score',
    'sp_seq.1',
    'pred_label_name',
    'ensemble_confidence'
]

# Remove metadata + probability columns
feature_cols = [
    c for c in df.columns
    if c not in remove_cols and not c.startswith("proba_")
]

X = df[feature_cols]

# keep only numeric features
X = X.select_dtypes(include=[np.number])

y = df['Classification']
ids = df['SP Name']

print("Number of features used:", X.shape[1])
print("Features:", list(X.columns))

# ============================================
# STANDARDIZE FEATURES
# ============================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# UMAP PROJECTION
# ============================================

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=5,
    min_dist=0.15,
    metric='euclidean',
    random_state=42
)

X_embedded = umap_model.fit_transform(X_scaled)

plot_df = pd.DataFrame({
    "UMAP1": X_embedded[:,0],
    "UMAP2": X_embedded[:,1],
    "Class": y,
    "SP": ids
})

# ============================================
# PLOT
# ============================================

plt.figure(figsize=(10,8))

sns.scatterplot(
    data=plot_df,
    x="UMAP1",
    y="UMAP2",
    hue="Class",
    palette={
        "Versatile": "#FF6B6B",
        "Unversatile": "#4A90E2"
    },
    s=140,
    edgecolor="black",
    alpha=0.9
)

plt.title("UMAP Projection of Signal Peptide Feature Space", fontsize=16)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("umap_feature_projection.png", dpi=300)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# Focus on unversatile samples with TM/Globular >= 0.5
unversatile_tm = unversatile[unversatile['TM/Globular'] >= 0.5].copy()

# Optional: check how many samples
print(f"Number of unversatile TM/Globular samples: {len(unversatile_tm)}")

plt.figure(figsize=(8,6))

# Scatter
sns.scatterplot(
    data=unversatile_tm,
    x='Sec/SPI',
    y='Mean Rank',
    s=100,
    color='#4A90E2',
    edgecolor='black'
)

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    unversatile_tm['Sec/SPI'], unversatile_tm['Mean Rank']
)

# Regression line
x_vals = np.linspace(0, 1, 100)
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, color='red', linewidth=2, label=f"y={slope:.3f}x+{intercept:.2f}\nR²={r_value**2:.3f}, p={p_value:.2e}")

plt.xlabel("Sec/SPI Probability")
plt.ylabel("Mean Rank")
plt.title("Impact of Sec/SPI Probability on Mean Rank\n(Unversatile TM/Globular ≥ 0.5)")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

# Print regression stats
print(f"Slope: {slope:.3f}")
print(f"Intercept: {intercept:.3f}")
print(f"R²: {r_value**2:.3f}")
print(f"P-value: {p_value:.2e}")