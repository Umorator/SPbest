import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patheffects as pe
from adjustText import adjust_text
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

# ============================================================================
# LOAD DATA
# ============================================================================
file1_path = Path("C:/Users/rafae/OneDrive/Documents/PhD_2026/Thesis/Chapter_2_SPbest/SP_best_Repo/versatility_analysis_complete.csv")
file2_path = Path("C:/Users/rafae/OneDrive/Documents/PhD_2026/Thesis/Chapter_2_SPbest/SP_best_Repo/protein_functional_bins.csv")

# Read the files
df_sp_raw = pd.read_csv(file1_path)
df_bins = pd.read_csv(file2_path)
df_bins
# Clean column names
df_sp_raw.columns = df_sp_raw.columns.str.strip()

# ============================================================================
# FILTER TOP 15 MOST VERSATILE SPs
# ============================================================================
versatility_col = df_sp_raw.columns[9]
df_sp_raw_sorted = df_sp_raw.sort_values(by=versatility_col, ascending=False).head(15)
df_sp_raw_sorted
# ============================================================================
# PROCESS DATA
# ============================================================================
df_sp = pd.DataFrame()
df_sp['SP_Name'] = df_sp_raw_sorted.iloc[:, 0]

def parse_protein_list(protein_string):
    if pd.isna(protein_string) or protein_string == '':
        return []
    protein_string = str(protein_string).replace('"', '').strip()
    if protein_string == '':
        return []
    proteins = [p.strip() for p in protein_string.split(';')]
    return [p for p in proteins if p]

# Parse the three categories
df_sp['Optimal_Proteins'] = df_sp_raw_sorted.iloc[:, 11].apply(parse_protein_list)
df_sp['Measured_Not_Optimal'] = df_sp_raw_sorted.iloc[:, 12].apply(parse_protein_list)
df_sp['NR_Proteins'] = df_sp_raw_sorted.iloc[:, 13].apply(parse_protein_list)

# ============================================================================
# SIMPLIFY FUNCTIONAL BINS
# ============================================================================
bin_mapping = {
    'Nitrogen-metabolism enzyme': 'Others',
    'Non-enzymatic protein': 'Others',
    'Oxidoreductase': 'Oxidoreductases & Transferases',
    'Transferase': 'Oxidoreductases & Transferases',
}

df_bins['Functional_bin'] = df_bins['Functional_bin'].replace(bin_mapping)

# Create protein to bin mapping
protein_to_bin = {}
for _, row in df_bins.iterrows():
    protein_name = row['Protein name'].strip()
    bin_name = row['Functional_bin']
    protein_to_bin[protein_name.lower()] = bin_name
    protein_to_bin[protein_name] = bin_name

# Get all bins
all_bins = sorted(df_bins['Functional_bin'].unique())
total_proteins_per_bin = df_bins['Functional_bin'].value_counts().to_dict()

# ============================================================================
# CREATE STATUS COUNTS
# ============================================================================
def get_status_counts(row):
    status_counts = {bin_name: {'optimal': 0, 'measured': 0, 'nr': 0} for bin_name in all_bins}
    
    def find_bin(protein_name):
        protein_clean = protein_name.strip()
        if protein_clean in protein_to_bin:
            return protein_to_bin[protein_clean]
        if protein_clean.lower() in protein_to_bin:
            return protein_to_bin[protein_clean.lower()]
        for key, bin_name in protein_to_bin.items():
            if isinstance(key, str) and (key.lower() in protein_clean.lower() or protein_clean.lower() in key.lower()):
                return bin_name
        return None
    
    for protein in row['Optimal_Proteins']:
        bin_name = find_bin(protein)
        if bin_name and bin_name in status_counts:
            status_counts[bin_name]['optimal'] += 1
    
    for protein in row['Measured_Not_Optimal']:
        bin_name = find_bin(protein)
        if bin_name and bin_name in status_counts:
            status_counts[bin_name]['measured'] += 1
    
    for protein in row['NR_Proteins']:
        bin_name = find_bin(protein)
        if bin_name and bin_name in status_counts:
            status_counts[bin_name]['nr'] += 1
    
    return status_counts

df_sp['status_counts'] = df_sp.apply(get_status_counts, axis=1)

# ============================================================================
# CALCULATE METRICS FOR BUBBLE PLOT
# ============================================================================
bubble_data = []

for idx, row in df_sp.iterrows():
    sp = row['SP_Name']
    for bin_name in all_bins:
        status = row['status_counts'][bin_name]
        optimal = status['optimal']
        measured = status['measured']
        nr = status['nr']
        total_secreted = optimal + measured + nr
        
        if total_secreted > 0:
            coverage = (total_secreted / total_proteins_per_bin[bin_name]) * 100
            successful = optimal + measured
            quality = (optimal / successful * 100) if successful > 0 else 0
            
            bubble_data.append({
                'Signal_Peptide': sp,
                'Category': bin_name,
                'Coverage': round(coverage, 1),
                'Quality': round(quality, 1),
                'Total_Secreted': total_secreted,
                'Optimal': optimal,
                'Measured': measured,
                'NR': nr
            })

bubble_df = pd.DataFrame(bubble_data)

print(f"\n📊 Total bubbles to label: {len(bubble_df)}")
print("\n📊 Bubbles by category:")
for cat in bubble_df['Category'].unique():
    cat_count = len(bubble_df[bubble_df['Category'] == cat])
    print(f"   • {cat}: {cat_count} bubbles")

# ============================================================================
# DEBUG: Check for Lipid-active enzyme at (50,100)
# ============================================================================
print("\n" + "="*80)
print("DEBUGGING: Looking for Lipid-active enzyme at (50,100)")
print("="*80)

lipid_points = []
for idx, row in bubble_df.iterrows():
    if row['Category'] == 'Lipid-active enzyme' and abs(row['Coverage'] - 50) < 2 and abs(row['Quality'] - 100) < 2:
        lipid_points.append(row)
        print(f"✅ Found Lipid-active enzyme point:")
        print(f"   • SP: {row['Signal_Peptide']}")
        print(f"   • Coordinates: ({row['Coverage']}, {row['Quality']})")
        print(f"   • Total Secreted: {row['Total_Secreted']}")

if not lipid_points:
    print("❌ No Lipid-active enzyme points found at (50,100) in bubble_df!")

# ============================================================================
# AGGLOMERATION FUNCTION - WITH "&" FOR EXACTLY 2 SPS
# ============================================================================
def agglomerate_close_points(df, distance_thresh=8):
    """
    Group points that are very close to each other (same category)
    into a single square label.
    - For 2 SPs: shows "SP1 & SP2"
    - For 3+ SPs: shows "SP1 +N"
    """
    df = df.copy()
    
    # First, identify all unique positions (rounded to 1 decimal to account for floating point)
    df['rounded_x'] = round(df['Coverage'], 1)
    df['rounded_y'] = round(df['Quality'], 1)
    df['pos_key'] = df.apply(lambda row: f"{row['Category']}_{row['rounded_x']}_{row['rounded_y']}", axis=1)
    
    # Group by position to find clusters
    position_groups = df.groupby('pos_key').agg({
        'Signal_Peptide': list,
        'Category': 'first',
        'Coverage': 'first',
        'Quality': 'first',
        'Total_Secreted': 'sum',
        'rounded_x': 'first',
        'rounded_y': 'first'
    }).reset_index()
    
    print(f"\n🔄 Found {len(position_groups)} unique positions")
    
    # DEBUG: Check if Lipid-active enzyme at (50,100) is in position_groups
    print("\n🔍 Checking position_groups for Lipid-active enzyme at (50,100):")
    found_in_groups = False
    for _, row in position_groups.iterrows():
        if (row['Category'] == 'Lipid-active enzyme' and 
            abs(row['rounded_x'] - 50) < 0.2 and 
            abs(row['rounded_y'] - 100) < 0.2):
            found_in_groups = True
            print(f"   ✅ Found in position_groups: {row['Signal_Peptide']} at ({row['rounded_x']}, {row['rounded_y']})")
    
    if not found_in_groups:
        print("   ❌ Lipid-active enzyme NOT found in position_groups!")
    
    agglomerated_data = []
    
    # Process each position group
    for idx, row in position_groups.iterrows():
        sp_list = row['Signal_Peptide']
        cat = row['Category']
        x = row['Coverage']
        y = row['Quality']
        
        if len(sp_list) > 1:
            # Multiple SPs at same/similar position
            # Sort SPs alphabetically for consistency
            sp_list.sort()
            
            # DIFFERENT FORMAT BASED ON NUMBER OF SPS
            if len(sp_list) == 2:
                # Exactly 2 SPs: show "SP1 & SP2"
                display_name = f"{sp_list[0]} & {sp_list[1]}"
                print(f"   → Two SPs at ({x:.1f}, {y:.1f}): {display_name}")
            else:
                # 3 or more SPs: show "SP1 +N"
                display_name = f"{sp_list[0]} +{len(sp_list)-1}"
                print(f"   → Grouped {len(sp_list)} SPs at ({x:.1f}, {y:.1f}): {display_name} ({', '.join(sp_list)})")
            
            # SPECIAL CASE 1: Protease in (42-45,100) - increase distance by 100% (from 7 to 14)
            if cat == 'Protease' and 42 <= x <= 45 and abs(y - 100) < 2:
                label_x = x
                label_y = y + 14  # Double the distance (100% increase)
                direction = 'up'
                print(f"      → SPECIAL: 100% increased distance for Protease")
            
            # SPECIAL CASE 2: Carbohydrate-active enzyme in (55,100) - upper left
            elif cat == 'Carbohydrate-active enzyme' and abs(x - 55) < 2 and abs(y - 100) < 2:
                label_x = x - 12  # Left
                label_y = y + 8   # Up
                direction = 'up-left'
                print(f"      → SPECIAL: Upper left placement for Carbohydrate-active enzyme")
            
            # SPECIAL CASE 3: Lipid-active enzyme in (50,100) - increase distance by 50% (from 7 to 10.5)
            elif cat == 'Lipid-active enzyme' and abs(x - 50) < 2 and abs(y - 100) < 2:
                label_x = x
                label_y = y + 10.5  # 50% increase (7 * 1.5 = 10.5)
                direction = 'up'
                print(f"      → SPECIAL: 50% increased distance for Lipid-active enzyme")
            
            # Default vertical offset based on position index
            elif idx % 2 == 0:
                label_x = x
                label_y = y + 7  # Up
                direction = 'up'
            else:
                label_x = x
                label_y = y - 7  # Down
                direction = 'down'
            
        else:
            # Single SP
            display_name = sp_list[0]
            label_x = x + 1.5
            label_y = y - 1.5
            direction = 'none'
            
            print(f"   → Single SP at ({x:.1f}, {y:.1f}): {display_name}")
        
        agglomerated_data.append({
            'Category': cat,
            'Bubble_X': x,
            'Bubble_Y': y,
            'Label_X': label_x,
            'Label_Y': label_y,
            'Display_Name': display_name,
            'Total_SPs': len(sp_list),
            'SP_Names': sp_list,
            'Total_Secreted': row['Total_Secreted'],
            'Has_Multiple': len(sp_list) > 1,
            'Direction': direction
        })
    
    result_df = pd.DataFrame(agglomerated_data)
    
    # ============================================================================
    # VERIFICATION: Check if every bubble has a label
    # ============================================================================
    print("\n🔍 VERIFYING ALL BUBBLES HAVE LABELS:")
    
    # Create a mapping of original bubbles
    original_bubbles = {}
    for _, row in df.iterrows():
        key = (round(row['Coverage'], 1), round(row['Quality'], 1), row['Signal_Peptide'])
        original_bubbles[key] = row['Category']
    
    # Check each original bubble is represented
    missing = []
    for (x, y, sp), cat in original_bubbles.items():
        found = False
        for _, row in result_df.iterrows():
            if (abs(row['Bubble_X'] - x) < 0.1 and 
                abs(row['Bubble_Y'] - y) < 0.1 and 
                sp in row['SP_Names']):
                found = True
                break
        if not found:
            missing.append((x, y, sp, cat))
    
    if missing:
        print(f"\n⚠️ Found {len(missing)} missing bubbles - FORCE ADDING THEM:")
        for x, y, sp, cat in missing:
            print(f"   • Missing: {sp} at ({x}, {y}) in {cat}")
            # Force add
            new_row = pd.DataFrame([{
                'Category': cat,
                'Bubble_X': x,
                'Bubble_Y': y,
                'Label_X': x + 1.5,
                'Label_Y': y - 1.5,
                'Display_Name': sp,
                'Total_SPs': 1,
                'SP_Names': [sp],
                'Total_Secreted': 1,
                'Has_Multiple': False,
                'Direction': 'none'
            }])
            result_df = pd.concat([result_df, new_row], ignore_index=True)
            print(f"      ✅ Added {sp}")
    else:
        print("   ✅ All bubbles have labels!")
    
    return result_df

# Apply agglomeration
agglomerated_df = agglomerate_close_points(bubble_df, distance_thresh=8)

print(f"\n📊 Final agglomerated labels: {len(agglomerated_df)}")
print(f"📊 Total SPs represented: {sum(agglomerated_df['Total_SPs'])}")

# ============================================================================
# FINAL VERIFICATION - Check Lipid-active enzyme specifically
# ============================================================================
print("\n" + "="*80)
print("FINAL VERIFICATION - Lipid-active enzyme at (50,100)")
print("="*80)

lipid_found = False
for idx, row in agglomerated_df.iterrows():
    if (row['Category'] == 'Lipid-active enzyme' and 
        abs(row['Bubble_X'] - 50) < 2 and 
        abs(row['Bubble_Y'] - 100) < 2):
        lipid_found = True
        distance = abs(row['Label_Y'] - row['Bubble_Y'])
        print(f"✅ Found in final agglomerated_df:")
        print(f"   • Display Name: {row['Display_Name']}")
        print(f"   • Contains SPs: {', '.join(row['SP_Names'])}")
        print(f"   • Label at ({row['Label_X']:.1f}, {row['Label_Y']:.1f})")
        print(f"   • Distance from bubble: {distance:.1f} units (50% increase from 7 to 10.5)")
        print(f"   • Type: {'Grouped' if row['Has_Multiple'] else 'Single'}")

if not lipid_found:
    print("❌ Lipid-active enzyme NOT found in final agglomerated_df!")

# Check for all special cases with their distances
print("\n🔍 Checking all special cases with distances:")
for _, row in agglomerated_df.iterrows():
    # Check Protease at (42-45,100) - 100% increase
    if row['Category'] == 'Protease' and 42 <= row['Bubble_X'] <= 45 and abs(row['Bubble_Y'] - 100) < 2:
        distance = abs(row['Label_Y'] - row['Bubble_Y'])
        print(f"   ✅ Protease special: {row['Display_Name']} at distance {distance:.1f} units (100% increase)")
    
    # Check Carbohydrate-active enzyme at (55,100) - upper left
    if row['Category'] == 'Carbohydrate-active enzyme' and abs(row['Bubble_X'] - 55) < 2 and abs(row['Bubble_Y'] - 100) < 2:
        print(f"   ✅ Carbohydrate-active enzyme special: {row['Display_Name']} at ({row['Label_X']:.1f}, {row['Label_Y']:.1f}) (upper left)")
    
    # Check Lipid-active enzyme at (50,100) - 50% increase
    if row['Category'] == 'Lipid-active enzyme' and abs(row['Bubble_X'] - 50) < 2 and abs(row['Bubble_Y'] - 100) < 2:
        distance = abs(row['Label_Y'] - row['Bubble_Y'])
        print(f"   ✅ Lipid-active enzyme: {row['Display_Name']} at distance {distance:.1f} units (50% increase)")

# Final verification all bubbles have labels
all_good = True
bubble_check = {}
for _, row in bubble_df.iterrows():
    key = (round(row['Coverage'], 1), round(row['Quality'], 1), row['Signal_Peptide'])
    bubble_check[key] = False

for _, row in agglomerated_df.iterrows():
    for sp in row['SP_Names']:
        key = (round(row['Bubble_X'], 1), round(row['Bubble_Y'], 1), sp)
        if key in bubble_check:
            bubble_check[key] = True

missing = [k for k, v in bubble_check.items() if not v]
if missing:
    print(f"\n❌ Still missing {len(missing)} bubbles:")
    for x, y, sp in missing:
        print(f"   • {sp} at ({x}, {y})")
    all_good = False
else:
    print("\n✅ ALL BUBBLES HAVE LABELS!")

# ============================================================================
# CREATE COLOR MAP FOR CATEGORIES (with 'Others' at the end)
# ============================================================================
# Reorder categories to put 'Others' at the end
categories = list(agglomerated_df['Category'].unique())
if 'Others' in categories:
    categories.remove('Others')
    categories.append('Others')

colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
color_map = dict(zip(categories, colors))

# ============================================================================
# CREATE THE BUBBLE PLOT
# ============================================================================
fig, ax = plt.subplots(figsize=(22, 16), constrained_layout=True)
# Plot bubbles at ORIGINAL positions with 10% larger size
bubble_scale_factor = 200 * 1.10

for category in categories:
    cat_bubbles = bubble_df[bubble_df['Category'] == category]
    if len(cat_bubbles) > 0:
        ax.scatter(cat_bubbles['Coverage'], cat_bubbles['Quality'],
                  s=cat_bubbles['Total_Secreted'] * bubble_scale_factor,
                  c=[color_map[category]] * len(cat_bubbles),
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=1.5,
                  label=category,
                  zorder=5)

# Add quadrant lines
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1, zorder=1)
ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=1, zorder=1)

# Draw arrows for groups
for idx, row in agglomerated_df.iterrows():
    if row['Has_Multiple']:
        bubble_color = color_map[row['Category']]
        
        if row['Direction'] == 'up-left':
            # Diagonal arrow for upper-left placement
            ax.annotate('', 
                       xy=(row['Bubble_X'], row['Bubble_Y']),
                       xytext=(row['Label_X'], row['Label_Y']),
                       arrowprops=dict(arrowstyle='->',
                                     color=bubble_color,
                                     lw=2.0,
                                     alpha=0.9,
                                     linestyle='-',
                                     zorder=8))
        else:
            # Strictly vertical arrow for others
            ax.annotate('', 
                       xy=(row['Bubble_X'], row['Bubble_Y']),
                       xytext=(row['Bubble_X'], row['Label_Y']),
                       arrowprops=dict(arrowstyle='->',
                                     color=bubble_color,
                                     lw=2.0,
                                     alpha=0.9,
                                     linestyle='-',
                                     zorder=8))

# Add labels
for idx, row in agglomerated_df.iterrows():
    if row['Has_Multiple']:
        # Square label for grouped SPs
        txt = ax.text(row['Label_X'], row['Label_Y'],
                     row['Display_Name'],
                     fontsize=12, 
                     fontweight='bold',
                     color='black',
                     ha='center', 
                     va='center',
                     bbox=dict(boxstyle='square,pad=0.4', 
                              facecolor='white', 
                              alpha=0.95, 
                              edgecolor=color_map[row['Category']], 
                              linewidth=2.5),
                     zorder=10)
        
        # Direction indicator
        if row['Direction'] == 'up':
            ax.text(row['Label_X'], row['Label_Y'] - 1.5, '↑', 
                   ha='center', va='top', fontsize=10, color=color_map[row['Category']], fontweight='bold')
        elif row['Direction'] == 'down':
            ax.text(row['Label_X'], row['Label_Y'] + 1.5, '↓', 
                   ha='center', va='bottom', fontsize=10, color=color_map[row['Category']], fontweight='bold')
        elif row['Direction'] == 'up-left':
            ax.text(row['Label_X'] + 1.5, row['Label_Y'] - 1.5, '↖', 
                   ha='center', va='center', fontsize=10, color=color_map[row['Category']], fontweight='bold')
    else:
        # Round label for single SPs
        txt = ax.text(row['Label_X'], row['Label_Y'], 
                     row['Display_Name'],
                     fontsize=12, 
                     fontweight='bold',
                     color='black',
                     ha='center', 
                     va='center',
                     bbox=dict(boxstyle='round,pad=0.2', 
                              facecolor='white', 
                              alpha=0.9, 
                              edgecolor=color_map[row['Category']], 
                              linewidth=1.2),
                     zorder=10)

# Customize axes
ax.set_xlabel('Coverage (% of total proteins in category)', fontsize=14, fontweight='bold')
ax.set_ylabel('Quality (% Optimal of successfully secreted)', fontsize=14, fontweight='bold')
ax.set_title('Signal Peptide Performance Map: Coverage vs Quality by Functional Category', 
             fontsize=18, fontweight='bold', pad=40)

ax.set_xlim(-5, 105)
ax.set_ylim(-15, 120)
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, zorder=0)

# Add quadrant explanations
adjust_up = 5

# LOW QUALITY boxes - moved lower to -8
ax.text(25, -8, 'LOW QUALITY\nLOW COVERAGE', 
        ha='center', fontsize=11, fontweight='bold', 
        color='#b71c1c', 
        bbox=dict(boxstyle='round', facecolor='#ffebee', alpha=0.9, 
                 edgecolor='#b71c1c', linewidth=2),
        zorder=20)

ax.text(75, -8, 'LOW QUALITY\nHIGH COVERAGE', 
        ha='center', fontsize=11, fontweight='bold', 
        color='#b71c1c', 
        bbox=dict(boxstyle='round', facecolor='#ffebee', alpha=0.9, 
                 edgecolor='#b71c1c', linewidth=2),
        zorder=20)

# HIGH QUALITY boxes moved UP
ax.text(25, 110 + adjust_up, 'HIGH QUALITY\nLOW COVERAGE', 
        ha='center', fontsize=11, fontweight='bold', 
        color='#2e7d32', 
        bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.9, 
                 edgecolor='#2e7d32', linewidth=2),
        zorder=20)

ax.text(75, 110 + adjust_up, 'HIGH QUALITY\nHIGH COVERAGE', 
        ha='center', fontsize=11, fontweight='bold', 
        color='#1b5e20', 
        bbox=dict(boxstyle='round', facecolor='#c8e6c9', alpha=0.9, 
                 edgecolor='#1b5e20', linewidth=2),
        zorder=20)

# ============================================================================
# LEGEND (ONLY functional categories - NO explanation boxes)
# ============================================================================
handles = []
for category in categories:
    handles.append(plt.Line2D([0], [0], 
                              marker='o', 
                              color='w',
                              markerfacecolor=color_map[category],
                              markersize=12,
                              markeredgecolor='black',
                              markeredgewidth=1,
                              label=category))

legend = ax.legend(handles=handles, 
                    title='Functional Category',
                    bbox_to_anchor=(1.02, 1),
                    loc='upper left',
                    fontsize=11,
                    title_fontsize=12,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    edgecolor='black')

# ========================================================================
# BUBBLE SIZE LEGEND (placed below the functional category legend)
# ========================================================================

size_values = [1, 5, 10]

size_handles = [
    plt.scatter([], [], 
                s=v * bubble_scale_factor,
                facecolors='gray',
                edgecolors='black',
                alpha=0.7)
    for v in size_values
]

size_labels = [f"{v} " for v in size_values]

legend_sizes = ax.legend(size_handles,
                         size_labels,
                         title="Bubble size\n(# secreted proteins)",
                         bbox_to_anchor=(1.035, 0.88),
                         loc='upper left',
                         fontsize=11,
                         title_fontsize=12,
                         frameon=True,
                         fancybox=True,
                         shadow=True,
                         edgecolor='black',
                         labelspacing=4.0,
                         borderpad=2.3,
                         handletextpad=1.8)

legend_sizes._legend_title_box._text.set_ha("center")

ax.add_artist(legend)
plt.savefig(
    "SP_Performance_Map_Final.png",
    dpi=300,
    bbox_inches="tight",
    bbox_extra_artists=(legend, legend_sizes)
)#plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY - ALL POINTS AUTOMATICALLY LABELED")
print("="*80)

print(f"\n📊 Total bubbles: {len(bubble_df)}")
print(f"📊 Total labels: {len(agglomerated_df)}")
print(f"📊 Total SPs represented: {sum(agglomerated_df['Total_SPs'])}")
print(f"\n   • Single SPs (round): {len(agglomerated_df[~agglomerated_df['Has_Multiple']])}")
print(f"   • Grouped SPs (square): {len(agglomerated_df[agglomerated_df['Has_Multiple']])}")

print("\n📊 Special case distances:")
for _, row in agglomerated_df.iterrows():
    if row['Has_Multiple']:
        if row['Category'] == 'Protease' and 42 <= row['Bubble_X'] <= 45 and abs(row['Bubble_Y'] - 100) < 2:
            distance = abs(row['Label_Y'] - row['Bubble_Y'])
            print(f"   • Protease: {distance:.1f} units (100% increase)")
        elif row['Category'] == 'Lipid-active enzyme' and abs(row['Bubble_X'] - 50) < 2 and abs(row['Bubble_Y'] - 100) < 2:
            distance = abs(row['Label_Y'] - row['Bubble_Y'])
            print(f"   • Lipid-active enzyme: {distance:.1f} units (50% increase)")
        elif row['Category'] == 'Carbohydrate-active enzyme' and abs(row['Bubble_X'] - 55) < 2 and abs(row['Bubble_Y'] - 100) < 2:
            print(f"   • Carbohydrate-active enzyme: upper left placement")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("   File saved: 'SP_Performance_Map_Final.png'")
print("="*80)



# ============================================================================
# HEATMAPS: QUALITY AND COVERAGE FOR TOP VERSATILE SPs
# ============================================================================

# After you have df_sp with category_performance from previous steps

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
# ============================================================================
# CALCULATE BIN SIMILARITY MATRIX (Add this before heatmaps)
# ============================================================================
from sklearn.metrics.pairwise import cosine_similarity

# Get all unique bins
all_bins = sorted(df_bins['Functional_bin'].unique())
n_bins = len(all_bins)

# Create a matrix of proteins per bin (one-hot encoding)
# First, get all unique proteins
all_proteins = df_bins['Protein name'].unique()
protein_to_idx = {protein: i for i, protein in enumerate(all_proteins)}

# Initialize bin-protein matrix
bin_protein_matrix = np.zeros((n_bins, len(all_proteins)))

# Fill the matrix
for _, row in df_bins.iterrows():
    bin_name = row['Functional_bin']
    protein = row['Protein name']
    bin_idx = all_bins.index(bin_name)
    protein_idx = protein_to_idx[protein]
    bin_protein_matrix[bin_idx, protein_idx] = 1

# Calculate cosine similarity between bins based on their protein composition
bin_similarity = cosine_similarity(bin_protein_matrix)

# Convert to DataFrame for easy access
bin_similarity_df = pd.DataFrame(
    bin_similarity, 
    index=all_bins, 
    columns=all_bins
)

print("\n=== PROTEIN FAMILY SIMILARITY MATRIX ===")
print(bin_similarity_df.round(3))
# ============================================================================
# STEP 1: Prepare data for heatmaps
# ============================================================================
quality_data = []
coverage_data = []
tested_count_data = []

for _, row in df_sp.iterrows():
    sp = row['SP_Name']
    for bin_name in all_bins:
        perf = row['category_performance'][bin_name]
        if perf is not None and perf['total_tested'] > 0:
            # Quality: % optimal of successful secretions
            quality = perf['optimal_ratio']
            
            # Coverage: % of total proteins in this category that were tested
            total_in_category = total_proteins_per_bin[bin_name]
            coverage = (perf['total_tested'] / total_in_category) * 100
            
            quality_data.append({
                'Signal_Peptide': sp,
                'Category': bin_name,
                'Value': quality,
                'Metric': 'Quality'
            })
            
            coverage_data.append({
                'Signal_Peptide': sp,
                'Category': bin_name,
                'Value': coverage,
                'Metric': 'Coverage'
            })
            
            tested_count_data.append({
                'Signal_Peptide': sp,
                'Category': bin_name,
                'Tested': perf['total_tested'],
                'Total_in_Category': total_in_category
            })
        else:
            # No data for this category
            quality_data.append({
                'Signal_Peptide': sp,
                'Category': bin_name,
                'Value': np.nan,
                'Metric': 'Quality'
            })
            
            coverage_data.append({
                'Signal_Peptide': sp,
                'Category': bin_name,
                'Value': np.nan,
                'Metric': 'Coverage'
            })

# Create DataFrames
quality_df = pd.DataFrame(quality_data)
coverage_df = pd.DataFrame(coverage_data)
tested_df = pd.DataFrame(tested_count_data)

# Pivot for heatmaps
quality_pivot = quality_df.pivot(
    index='Signal_Peptide', 
    columns='Category', 
    values='Value'
)

coverage_pivot = coverage_df.pivot(
    index='Signal_Peptide', 
    columns='Category', 
    values='Value'
)

# Reorder categories if desired (put 'Others' at the end)
if 'Others' in quality_pivot.columns:
    cols = [col for col in quality_pivot.columns if col != 'Others'] + ['Others']
    quality_pivot = quality_pivot[cols]
    coverage_pivot = coverage_pivot[cols]

# ============================================================================
# STEP 2: Create custom colormaps
# ============================================================================
# Quality colormap: Red (bad) to Yellow (medium) to Green (good)
quality_cmap = LinearSegmentedColormap.from_list(
    'quality_cmap', 
    ['#d73027', '#ffffbf', '#1a9850'],  # Red, Yellow, Green
    N=100
)

# Coverage colormap: Light blue to Dark blue (low to high)
coverage_cmap = LinearSegmentedColormap.from_list(
    'coverage_cmap', 
    ['#f7fbff', '#08306b'],  # Very light blue to dark blue
    N=100
)

# ============================================================================
# STEP 3: Create the heatmaps
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

# Heatmap 1: QUALITY (% optimal of successful secretions)
sns.heatmap(
    quality_pivot, 
    annot=True, 
    fmt='.0f', 
    cmap=quality_cmap,
    ax=ax1,
    cbar_kws={'label': 'Quality (% Optimal of Successful)', 'shrink': 0.8},
    linewidths=0.5,
    linecolor='gray',
    vmin=0, 
    vmax=100,
    mask=quality_pivot.isna(),
    annot_kws={'fontsize': 10, 'fontweight': 'bold'}
)

ax1.set_title('QUALITY: % Optimal of Successful Secretions\nby Protein Family', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Protein Functional Category', fontsize=12, fontweight='bold')
ax1.set_ylabel('Signal Peptide', fontsize=12, fontweight='bold')
ax1.tick_params(axis='x', rotation=45, ha='right')

# Heatmap 2: COVERAGE (% of total proteins in category tested)
sns.heatmap(
    coverage_pivot, 
    annot=True, 
    fmt='.0f', 
    cmap=coverage_cmap,
    ax=ax2,
    cbar_kws={'label': 'Coverage (% of Category Proteins Tested)', 'shrink': 0.8},
    linewidths=0.5,
    linecolor='gray',
    vmin=0, 
    vmax=100,
    mask=coverage_pivot.isna(),
    annot_kws={'fontsize': 10, 'fontweight': 'bold'}
)

ax2.set_title('COVERAGE: % of Total Proteins in Category Tested\nby Protein Family', 
              fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Protein Functional Category', fontsize=12, fontweight='bold')
ax2.set_ylabel('Signal Peptide', fontsize=12, fontweight='bold')
ax2.tick_params(axis='x', rotation=45, ha='right')

plt.tight_layout()
plt.savefig('SP_Quality_Coverage_Heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 4: Create a combined "Versatility Score" heatmap
# ============================================================================
# Calculate a combined score: Quality × (Coverage/100) to weight by confidence
combined_pivot = quality_pivot * (coverage_pivot / 100)
combined_pivot = combined_pivot.round(1)

fig, ax = plt.subplots(figsize=(20, 8))

sns.heatmap(
    combined_pivot, 
    annot=True, 
    fmt='.1f', 
    cmap='viridis',
    ax=ax,
    cbar_kws={'label': 'Weighted Score (Quality × Coverage/100)', 'shrink': 0.8},
    linewidths=0.5,
    linecolor='gray',
    vmin=0, 
    vmax=100,
    mask=combined_pivot.isna(),
    annot_kws={'fontsize': 10, 'fontweight': 'bold'}
)

ax.set_title('WEIGHTED VERSATILITY SCORE: Quality × (Coverage/100)\nHigher = Well-tested and High Quality', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Protein Functional Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Signal Peptide', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45, ha='right')

plt.tight_layout()
plt.savefig('SP_Weighted_Versatility_Heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 5: Create summary statistics
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS FOR TOP VERSATILE SPs")
print("="*80)

# Average quality per SP
avg_quality = quality_pivot.mean(axis=1).sort_values(ascending=False)
print("\n📊 AVERAGE QUALITY ACROSS ALL CATEGORIES:")
for sp, val in avg_quality.items():
    print(f"   {sp:<20} {val:.1f}%")

# Average coverage per SP
avg_coverage = coverage_pivot.mean(axis=1).sort_values(ascending=False)
print("\n📊 AVERAGE COVERAGE ACROSS ALL CATEGORIES:")
for sp, val in avg_coverage.items():
    print(f"   {sp:<20} {val:.1f}%")

# Best categories for each SP
print("\n🏆 BEST CATEGORIES FOR EACH SP (Quality > 80%):")
for sp in quality_pivot.index:
    best = []
    for cat in quality_pivot.columns:
        q = quality_pivot.loc[sp, cat]
        c = coverage_pivot.loc[sp, cat]
        if pd.notna(q) and q > 80 and pd.notna(c) and c > 20:  # High quality + decent coverage
            best.append(f"{cat} ({q:.0f}%, {c:.0f}% cov)")
    
    if best:
        print(f"\n{sp}:")
        for b in best[:3]:  # Show top 3
            print(f"   • {b}")

# ============================================================================
# STEP 6: Identify truly versatile SPs (high quality across diverse categories)
# ============================================================================
print("\n" + "="*80)
print("TRULY VERSATILE SPs (High Quality Across Diverse Categories)")
print("="*80)

versatile_sps = []
for sp in quality_pivot.index:
    # Get categories where this SP has quality > 70%
    high_quality_cats = []
    for cat in quality_pivot.columns:
        q = quality_pivot.loc[sp, cat]
        if pd.notna(q) and q > 70:
            high_quality_cats.append(cat)
    
    if len(high_quality_cats) >= 3:
        # Calculate diversity of these categories using bin similarity
        if len(high_quality_cats) >= 2:
            avg_dissimilarity = 0
            pairs = 0
            for i, cat1 in enumerate(high_quality_cats):
                for cat2 in high_quality_cats[i+1:]:
                    sim = bin_similarity_df.loc[cat1, cat2]
                    avg_dissimilarity += (1 - sim)
                    pairs += 1
            
            if pairs > 0:
                diversity = avg_dissimilarity / pairs
                versatile_sps.append({
                    'SP': sp,
                    'categories': len(high_quality_cats),
                    'diversity': diversity,
                    'categories_list': high_quality_cats
                })

versatile_df = pd.DataFrame(versatile_sps).sort_values('diversity', ascending=False)

for _, row in versatile_df.iterrows():
    print(f"\n{row['SP']}:")
    print(f"   • Works well in {row['categories']} categories")
    print(f"   • Category diversity: {row['diversity']:.2f} (higher = more different families)")
    print(f"   • Categories: {', '.join(row['categories_list'])}")