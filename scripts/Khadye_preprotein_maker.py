import pandas as pd

# Path to your file
input_file = r"C:\Users\rafae\OneDrive\Documents\PhD_2026\Thesis\Chapter_2_SPbest\SP_best_Repo\data\Takara\Khadye_Bglu.csv"

# Load CSV
df = pd.read_csv(input_file, sep=";")
# Make sure sequence columns are treated as strings (avoids NaN issues)
sequence_columns = [
    "SP_DNA_seq",
    "DNA_linker",
    "PROTEIN_DNA_SEQ",
    "sp_seq",
    "aa_linker",
    "protein_seq"
]

for col in sequence_columns:
    df[col] = df[col].fillna("").astype(str)

# Create new columns
df["Preprotein_DNA_seq"] = (
    df["SP_DNA_seq"] +
    df["DNA_linker"] +
    df["PROTEIN_DNA_SEQ"]
)

df["Preprotein_seq"] = (
    df["sp_seq"] +
    df["aa_linker"] +
    df["protein_seq"]
)

# Save updated file (you can change the name if you prefer)
output_file = r"C:\Users\rafae\OneDrive\Documents\PhD_2026\Thesis\Chapter_2_SPbest\SP_best_Repo\data\Takara\Khadye_Bglu_updated.csv"
df.to_csv(output_file, index=False)

print("New columns created successfully.")
print("File saved to:", output_file)

