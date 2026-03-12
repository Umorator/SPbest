import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ==========================================================
# Load the undersampled data
# ==========================================================
main_data_dir = Path(r"C:\Users\rafae\OneDrive\Documents\PhD_2026\Thesis\Chapter_2_SPbest\SP_best_Repo\data")
output_csv = main_data_dir / "spbest_undersampled.csv"

# Create output directory for figures
figures_dir = main_data_dir.parent / "outputs" / "undersampled" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

print(f"Figures will be saved to: {figures_dir}")

# Read the CSV file
final_df = pd.read_csv(output_csv)
print(f"Loaded undersampled data with {len(final_df)} rows")

# ==========================================================
# Prepare data for plotting
# ==========================================================
print("\n" + "="*60)
print("PREPARING DATA FOR PLOTTING")
print("="*60)

# Add label string for grouping
final_df['label_str'] = final_df['label'].map({1: 'optimal', 0: 'non-optimal'})

# Get counts per dataset
plot_data = final_df.groupby(['dataset_name', 'label_str']).size().unstack(fill_value=0).reset_index()

# Rename columns
plot_data = plot_data.rename(columns={
    'optimal': 'Num_Optimal',
    'non-optimal': 'Num_NonOptimal'
})

# Ensure both columns exist
for col in ['Num_Optimal', 'Num_NonOptimal']:
    if col not in plot_data.columns:
        plot_data[col] = 0

# Calculate total for sorting
plot_data['Num_SPs'] = plot_data['Num_Optimal'] + plot_data['Num_NonOptimal']

# Create Label column for plotting (Author-Protein format)
plot_data['Label'] = plot_data['dataset_name']

# Sort by total SPs
plot_data = plot_data.sort_values('Num_SPs', ascending=False).reset_index(drop=True)

# Calculate averages
avg_optimal = plot_data['Num_Optimal'].mean()
avg_nonoptimal = plot_data['Num_NonOptimal'].mean()

# ==========================================================
# Create the plot
# ==========================================================
print("\n" + "="*60)
print("CREATING PLOT")
print("="*60)

# Create the plot with adjusted figure size and margins
fig, ax = plt.subplots(figsize=(24, 14))  # Slightly larger figure

# Create stacked bars
bar_width = 0.8
x_pos = np.arange(len(plot_data))

# Plot non-optimal SPs (gray bars at the bottom)
bars1 = ax.bar(x_pos, plot_data['Num_NonOptimal'], bar_width, 
                label='Non-optimal SPs', 
                color='#696969', edgecolor='black', linewidth=0.5)  # Dark gray

# Plot optimal SPs (blue bars on top of non-optimal)
bars2 = ax.bar(x_pos, plot_data['Num_Optimal'], bar_width, 
                bottom=plot_data['Num_NonOptimal'], 
                label='Optimal SPs', 
                color='#1f77b4', edgecolor='black', linewidth=0.5)  # Blue

# Add optimal count inside blue bars or above if too small
for i, (idx, row) in enumerate(plot_data.iterrows()):
    optimal = row['Num_Optimal']
    nonoptimal = row['Num_NonOptimal']
    total = row['Num_SPs']
    
    if optimal > 0:
        # Position for optimal count (middle of blue bar)
        blue_position = nonoptimal + (optimal / 2)
        
        # If blue bar is tall enough, put text inside
        if optimal > total * 0.1:  # If optimal bar is at least 10% of total
            ax.text(i, blue_position, str(int(optimal)), 
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        else:
            # Otherwise put it above the blue bar
            ax.text(i, nonoptimal + optimal + 0.5, str(int(optimal)), 
                    ha='center', va='bottom', fontsize=7, color='#1f77b4', fontweight='bold')

# Add non-optimal count inside gray bars or above if too small
for i, (idx, row) in enumerate(plot_data.iterrows()):
    nonoptimal = row['Num_NonOptimal']
    optimal = row['Num_Optimal']
    total = row['Num_SPs']
    
    if nonoptimal > 0:
        # Position for non-optimal count (middle of gray bar)
        gray_position = nonoptimal / 2
        
        # If gray bar is tall enough, put text inside
        if nonoptimal > total * 0.1:  # If non-optimal bar is at least 10% of total
            ax.text(i, gray_position, str(int(nonoptimal)), 
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
        else:
            # Otherwise put it above the gray bar
            if optimal > 0:
                # If there's a blue bar above, put it just below the blue bar
                ax.text(i, nonoptimal - 0.8 if nonoptimal > 1 else nonoptimal + 0.8, 
                        str(int(nonoptimal)), 
                        ha='center', va='bottom' if nonoptimal > 1 else 'top', 
                        fontsize=7, color='black', fontweight='bold')
            else:
                # If no blue bar, put it above
                ax.text(i, nonoptimal + 0.5, str(int(nonoptimal)), 
                        ha='center', va='bottom', fontsize=7, color='black', fontweight='bold')

# Customize the plot
ax.set_ylabel('Number of Signal Peptides', fontsize=14, fontweight='bold')
ax.set_xlabel('Dataset (Author-Protein)', fontsize=14, fontweight='bold')
ax.set_title('Undersampled Data: Optimal vs Non-Optimal Signal Peptides per Dataset\n(After Cluster-based Undersampling)', 
              fontsize=16, fontweight='bold', pad=20)

# Set x-axis labels
ax.set_xticks(x_pos)
ax.set_xticklabels(plot_data['Label'], rotation=90, fontsize=8, ha='center')

# Add average lines
line1 = ax.axhline(y=avg_nonoptimal, color='#696969', linestyle='--', linewidth=2.5, alpha=0.8,
                   label=f'Average Non-optimal: {avg_nonoptimal:.1f}')
line2 = ax.axhline(y=avg_optimal, color='#1f77b4', linestyle='--', linewidth=2.5, alpha=0.8,
                   label=f'Average Optimal: {avg_optimal:.1f}')

# Legend
legend_elements = [
    Patch(facecolor='#696969', edgecolor='black', label='Non-optimal SPs'),
    Patch(facecolor='#1f77b4', edgecolor='black', label='Optimal SPs'),
    Line2D([0], [0], color='#696969', linestyle='--', linewidth=2.5, label=f'Avg Non-optimal: {avg_nonoptimal:.1f}'),
    Line2D([0], [0], color='#1f77b4', linestyle='--', linewidth=2.5, label=f'Avg Optimal: {avg_optimal:.1f}')
]

ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.2, axis='y', linestyle='--')
ax.set_ylim(0, plot_data['Num_SPs'].max() * 1.15)

# Use subplots_adjust instead of tight_layout for more control
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15)

# Save the plot with bbox_inches='tight' and additional padding
plot_path = figures_dir / "undersampled_optimal_vs_nonoptimal.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.show()
print(f"Plot saved to: {plot_path}")

print("\n" + "="*60)
print("DONE - PLOT CREATED")
print("="*60)