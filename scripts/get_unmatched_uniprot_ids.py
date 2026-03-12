import pandas as pd
import textwrap

# =========================
# FILE PATHS
# =========================

original_ids_path = r"C:\Users\rafae\OneDrive\Documents\PhD_2026\Thesis\Chapter_2_SPbest\SP_best_Repo\bacillus_protein_ids.txt"

uniprot_output_path = r"C:\Users\rafae\OneDrive\Documents\PhD_2026\Thesis\Chapter_2_SPbest\SP_best_Repo\idmapping_accession_A0A0J1IQE9_OR_access_2026_03_04.xlsx"

output_unmatched_path = r"C:\Users\rafae\OneDrive\Documents\PhD_2026\Thesis\Chapter_2_SPbest\SP_best_Repo\unmatched_ids.fasta"

cosmpad_path = r"C:\Users\rafae\OneDrive\Documents\PhD_2026\Thesis\Chapter_2_SPbest\SP_best_Repo\data\CoSMPAD.csv"


# =========================
# STEP 1: READ ORIGINAL IDS
# =========================

with open(original_ids_path, "r") as f:
    content = f.read()

original_ids = set(content.split())
print(f"Total original IDs: {len(original_ids)}")


# =========================
# STEP 2: READ UNIPROT OUTPUT
# =========================

df_uniprot = pd.read_excel(uniprot_output_path)

matched_ids = set(df_uniprot["From"].astype(str).str.strip())
print(f"Matched IDs from UniProt: {len(matched_ids)}")


# =========================
# STEP 3: FIND UNMATCHED IDS
# =========================

unmatched_ids = original_ids - matched_ids
print(f"Unmatched IDs: {len(unmatched_ids)}")


# =========================
# STEP 4: LOAD CoSMPAD DATA
# =========================

df_cosmpad = pd.read_csv(cosmpad_path)

# Clean matching column
df_cosmpad['UniprotKB/NCBI_POI'] = df_cosmpad['UniprotKB/NCBI_POI'].astype(str).str.strip()


# =========================
# STEP 5: WRITE FASTA FILE
# =========================

with open(output_unmatched_path, "w") as f:
    
    for uid in sorted(unmatched_ids):
        
        # Find matching rows
        matches = df_cosmpad[df_cosmpad['UniprotKB/NCBI_POI'] == uid]
        
        if not matches.empty:
            # Select first occurrence
            sequence = matches.iloc[0]['protein_seq']
            
            if pd.notna(sequence):
                sequence = sequence.strip()
                
                # Write FASTA header
                f.write(f">{uid}\n")
                
                # Wrap sequence to 60 characters per line
                wrapped_seq = textwrap.fill(sequence, width=60)
                f.write(wrapped_seq + "\n")
        
        else:
            print(f"WARNING: No sequence found for {uid}")

print("FASTA file saved successfully.")