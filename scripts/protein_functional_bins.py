import pandas as pd

# Read the dataset
df = pd.read_csv("outputs/CoSMPAD_autor_ref.csv")
df.columns
# -----------------------------
# Define all possible functional bins
# -----------------------------
functional_bins = [
    "Lipid-active enzyme",
    "Protease",
    "Oxidoreductases & Transferases",  # Combined category
    "Carbohydrate-active enzyme",
    "Others"  # Catch-all for Nitrogen-metabolism enzyme, Non-enzymatic protein, etc.
]

# -----------------------------
# Manual overrides
# -----------------------------
manual_bins = {
    "XynBYG": "Carbohydrate-active enzyme",
    "Siase": "Carbohydrate-active enzyme", 
    "Dac": "Carbohydrate-active enzyme",
    "PG": "Carbohydrate-active enzyme",
    "LipS": "Lipid-active enzyme",
    "BcaPRO": "Protease",
    "MTG": "Oxidoreductases & Transferases",  # Was Transferase
    "MO": "Others",  # Was Non-enzymatic protein
    "MtC1LPMO": "Carbohydrate-active enzyme"
}

# -----------------------------
# Automatic classification
# -----------------------------
def classify_protein(name):
    # Check manual overrides first
    if name in manual_bins:
        return manual_bins[name]

    n = name.lower()

    if any(k in n for k in ["lipase","cutinase","esterase","petase","pld"]):
        return "Lipid-active enzyme"

    if any(k in n for k in [
        "protease","subtilisin","peptidase",
        "aminopeptidase","nattokinase",
        "asparaginase","acylase"
    ]):
        return "Protease"

    if any(k in n for k in ["laccase","aldh","lpmo","cgtase","transglutaminase"]):
        return "Oxidoreductases & Transferases"

    if "nhase" in n:
        return "Others"  # Nitrogen-metabolism enzyme -> Others

    if any(k in n for k in [
        "amylase","amy","xylanase","xyn",
        "glucosidase","mannanase","pullulanase",
        "lyase","pectate","phytase"
    ]):
        return "Carbohydrate-active enzyme"

    # Default fallback
    return "Others"

# -----------------------------
# Apply classification
# -----------------------------
df["functional_bin"] = df["Protein name"].apply(classify_protein)

# -----------------------------
# SIMPLIFY FUNCTIONAL BINS (additional pass to ensure consistency)
# ============================================================================
bin_mapping = {
    'Nitrogen-metabolism enzyme': 'Others',
    'Non-enzymatic protein': 'Others',
    'Oxidoreductase': 'Oxidoreductases & Transferases',
    'Transferase': 'Oxidoreductases & Transferases',
}

df['functional_bin'] = df['functional_bin'].replace(bin_mapping)

# -----------------------------
# Create protein to bin mapping (case-insensitive)
# -----------------------------
protein_to_bin = {}
for _, row in df.iterrows():
    protein_name = row['Protein name'].strip()
    bin_name = row['functional_bin']
    protein_to_bin[protein_name.lower()] = bin_name
    protein_to_bin[protein_name] = bin_name

# -----------------------------
# Display results
# -----------------------------
print("Functional bin distribution:")
print(df["functional_bin"].value_counts())
print("\n" + "="*50)
print("Sample of protein classifications:")
print(df[["Protein name", "functional_bin"]].head(10))

# -----------------------------
# Save back to the same location
# -----------------------------
output_path = "outputs/CoSMPAD_autor_ref.csv"
df.to_csv(output_path, index=False)
print(f"\nFile saved to: {output_path}")

# Optional: Save the protein mapping separately
mapping_df = pd.DataFrame(list(protein_to_bin.items()), columns=['Protein_key', 'Functional_bin'])
mapping_path = "outputs/protein_bin_mapping.csv"
mapping_df.to_csv(mapping_path, index=False)
print(f"Protein mapping saved to: {mapping_path}")