# src/create_rank_pairs.py
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

def create_rank_pairs(config_path: str, plot: bool = True):
    """
    Create pairwise ranking data from SP libraries.
    
    Parameters:
    - config_path: path to config JSON file
    - plot: whether to generate plots
    
    Config parameters:
    - add_sp: if True, merge SP features; if False, only merge Preprotein features
    """
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Get file paths and parameters from config
    csv_path = config.get('csv_path')
    sp_features_path = config.get('sp_seq')  # SP features
    preprotein_features_path = config.get('Preprotein_seq')  # Preprotein features
    add_sp = config.get('add_sp', True)  # Default to True if not specified
    
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV path not found in config: {csv_path}")
    
    if add_sp and (not sp_features_path or not os.path.exists(sp_features_path)):
        raise FileNotFoundError(f"SP features file not found: {sp_features_path}")
    
    if not preprotein_features_path or not os.path.exists(preprotein_features_path):
        raise FileNotFoundError(f"Preprotein features file not found: {preprotein_features_path}")

    # Load main data
    print(f"Loading main data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check if there's an 'id' column in the main data
    if 'id' not in df.columns:
        print("Warning: No 'id' column found in main data. Creating one...")
        df['id'] = range(len(df))
    
    # Load features (conditionally)
    df_sp_features = None
    if add_sp:
        print(f"Loading SP features from: {sp_features_path}")
        df_sp_features = pd.read_csv(sp_features_path)
        # Ensure SP features have 'id' column
        if 'id' not in df_sp_features.columns:
            # Try to find the ID column
            for col in ['id', 'ID', 'Id', 'sequence_id', 'index', 'Unnamed: 0']:
                if col in df_sp_features.columns:
                    df_sp_features = df_sp_features.rename(columns={col: 'id'})
                    break
            else:
                # Assume first column is ID
                first_col = df_sp_features.columns[0]
                df_sp_features = df_sp_features.rename(columns={first_col: 'id'})
                print(f"    Renamed '{first_col}' to 'id' for SP features")
    
    print(f"Loading Preprotein features from: {preprotein_features_path}")
    df_preprotein_features = pd.read_csv(preprotein_features_path)
    # Ensure Preprotein features have 'id' column
    if 'id' not in df_preprotein_features.columns:
        for col in ['id', 'ID', 'Id', 'sequence_id', 'index', 'Unnamed: 0']:
            if col in df_preprotein_features.columns:
                df_preprotein_features = df_preprotein_features.rename(columns={col: 'id'})
                break
        else:
            first_col = df_preprotein_features.columns[0]
            df_preprotein_features = df_preprotein_features.rename(columns={first_col: 'id'})
            print(f"    Renamed '{first_col}' to 'id' for Preprotein features")

    print(f"\n📋 Configuration:")
    print(f"  - Add SP features: {add_sp}")
    print(f"  - Plot: {plot}")

    # Required columns check
    required_cols = ['Author-Protein', 'enzyme_activity', 'SP name']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Khadye special SPs (in rank order)
    khadye_sps_ranked = ['cith','lytb','ywsb','ybdg','phob','abna','ykoj','ykwd','yobv','ybbe','ywad','apre']
    
    # Initialize list to store all pairs
    all_pairs = []
    
    # Process each library separately - using actual 'id' from the data
    for library_name, group in df.groupby('Author-Protein'):
        print(f"\nProcessing library: {library_name} ({len(group)} SPs)")
        
        # Convert enzyme_activity to numeric, coercing errors to NaN
        group = group.copy()
        group['activity_numeric'] = pd.to_numeric(group['enzyme_activity'], errors='coerce')
        
        # Split into measured (numeric) and unmeasured (non-numeric)
        measured_mask = group['activity_numeric'].notna()
        unmeasured_mask = ~measured_mask
        
        measured_sp = group[measured_mask].sort_values('activity_numeric', ascending=False)
        unmeasured_sp = group[unmeasured_mask]
        
        # SPECIAL CASE: Khadye library - special SPs are treated as measured with known rank order
        if library_name == 'Khadye - E7FHY4':
            print(f"  ⚠️  Khadye library: special SPs have known rank order")
            
            # Mark special SPs and their rank positions
            group['is_special'] = group['SP name'].isin(khadye_sps_ranked)
            group['special_rank'] = group['SP name'].map(
                {name: i for i, name in enumerate(khadye_sps_ranked)}
            )
            
            # Special SPs count
            special_count = group['is_special'].sum()
            nonspecial_count = len(group) - special_count
            print(f"  - Special Khadye SPs (to be treated as measured): {special_count}")
            print(f"  - Non-special Khadye SPs (truly unmeasured): {nonspecial_count}")
            
            # REASSIGN: Special SPs go to measured, non-special go to unmeasured
            special_mask = group['is_special']
            nonspecial_mask = ~special_mask
            
            # Create measured set from special SPs
            measured_sp = group[special_mask].copy()
            # Assign artificial activity values based on rank (higher rank = higher activity)
            max_rank = measured_sp['special_rank'].max()
            measured_sp['activity_numeric'] = max_rank - measured_sp['special_rank'] + 1
            measured_sp = measured_sp.sort_values('activity_numeric', ascending=False)
            
            # Unmeasured set is the non-special SPs
            unmeasured_sp = group[nonspecial_mask]
            
            print(f"  - Now: Measured SPs: {len(measured_sp)} (the 12 special SPs)")
            print(f"  - Now: Unmeasured SPs: {len(unmeasured_sp)} (the other {nonspecial_count} SPs)")
            
            # TYPE 1: Measured-unmeasured pairs (special SPs > non-special SPs) - THESE ARE THE MAIN ONES
            for measured_idx, measured_row in measured_sp.iterrows():
                for unmeasured_idx, unmeasured_row in unmeasured_sp.iterrows():
                    all_pairs.append({
                        'sp_A_id': measured_row['id'],
                        'sp_B_id': unmeasured_row['id'],
                        'target': 1,
                        'pair_type': 'khadye_measured_unmeasured',
                        'confidence': 0.9,  # High confidence that specials are better
                        'library': library_name
                    })
            
            # TYPE 2: Special ordered pairs (using known rank order) - ONLY THESE, not both!
            special_ids = measured_sp.sort_values('special_rank')['id'].tolist()
            for i in range(len(special_ids)):
                for j in range(i+1, len(special_ids)):
                    all_pairs.append({
                        'sp_A_id': special_ids[i],  # higher ranked (better)
                        'sp_B_id': special_ids[j],  # lower ranked
                        'target': 1,
                        'pair_type': 'khadye_special_ordered',
                        'confidence': 1.0,
                        'library': library_name
                    })
            
            print(f"  - Created {len(measured_sp) * len(unmeasured_sp)} measured-unmeasured pairs")
            print(f"  - Created {len(special_ids)*(len(special_ids)-1)//2} special ordered pairs")
            
            # Skip the regular pair generation for this library
            continue
        
        # REGULAR PAIR GENERATION FOR NON-KHADYE LIBRARIES
        print(f"  - Measured SPs: {len(measured_sp)}")
        print(f"  - Unmeasured SPs: {len(unmeasured_sp)}")
        
        # TYPE 1: Measured vs Measured (both have numeric values) - use actual IDs
        measured_ids = measured_sp['id'].tolist()
        for i in range(len(measured_ids)):
            for j in range(i+1, len(measured_ids)):
                id_a = measured_ids[i]
                id_b = measured_ids[j]
                
                # Get activity values to determine order
                act_a = measured_sp[measured_sp['id'] == id_a]['activity_numeric'].iloc[0]
                act_b = measured_sp[measured_sp['id'] == id_b]['activity_numeric'].iloc[0]
                
                if act_a > act_b:
                    sp_a_id, sp_b_id = id_a, id_b
                else:
                    sp_a_id, sp_b_id = id_b, id_a
                
                all_pairs.append({
                    'sp_A_id': sp_a_id,
                    'sp_B_id': sp_b_id,
                    'target': 1,  # A is better than B
                    'pair_type': 'measured_measured',
                    'confidence': 1.0,
                    'library': library_name
                })
        
        # TYPE 2: Measured vs Unmeasured (measured > unmeasured) - use actual IDs
        for measured_id in measured_sp['id']:
            for unmeasured_id in unmeasured_sp['id']:
                all_pairs.append({
                    'sp_A_id': measured_id,
                    'sp_B_id': unmeasured_id,
                    'target': 1,  # measured > unmeasured
                    'pair_type': 'measured_unmeasured',
                    'confidence': 1.0,
                    'library': library_name
                })
    
    # Convert to DataFrame
    pairs_df = pd.DataFrame(all_pairs)
    
    print(f"\n{'='*50}")
    print(f"TOTAL PAIRS CREATED: {len(pairs_df)}")
    print(f"{'='*50}")
    print(f"\nBreakdown by pair type:")
    print(pairs_df['pair_type'].value_counts())
    print(f"\nBreakdown by library:")
    print(pairs_df['library'].value_counts())
    
    # ============= FEATURE MERGING SECTION =============
    print("\nMerging with features...")
    
    # sp_A_id and sp_B_id are already the actual IDs from the data
    # No need to convert to string - they should match the feature files
    
    # Always merge Preprotein features
    print("  - Adding Preprotein features...")
    
    # Rename columns for merge (add prefix)
    df_pre_A = df_preprotein_features.copy()
    df_pre_A.columns = ['sp_A_pre_' + col if col != 'id' else 'sp_A_id' for col in df_pre_A.columns]
    pairs_df = pairs_df.merge(df_pre_A, on='sp_A_id', how='left')
    
    df_pre_B = df_preprotein_features.copy()
    df_pre_B.columns = ['sp_B_pre_' + col if col != 'id' else 'sp_B_id' for col in df_pre_B.columns]
    pairs_df = pairs_df.merge(df_pre_B, on='sp_B_id', how='left')
    
    # Conditionally merge SP features based on config
    if add_sp and df_sp_features is not None:
        print("  - Adding SP features...")
        
        df_sp_A = df_sp_features.copy()
        df_sp_A.columns = ['sp_A_' + col if col != 'id' else 'sp_A_id' for col in df_sp_A.columns]
        pairs_df = pairs_df.merge(df_sp_A, on='sp_A_id', how='left')
        
        df_sp_B = df_sp_features.copy()
        df_sp_B.columns = ['sp_B_' + col if col != 'id' else 'sp_B_id' for col in df_sp_B.columns]
        pairs_df = pairs_df.merge(df_sp_B, on='sp_B_id', how='left')
        
        print(f"  - SP features added")
    else:
        print("  - SP features not added (add_sp=False in config)")
    
    print(f"Final pairs dataset shape: {pairs_df.shape}")
    
    # Count feature columns
    pre_feat_cols = [c for c in pairs_df.columns if 'sp_A_pre_' in c]
    print(f"  - Preprotein feature columns: {len(pre_feat_cols)}")
    
    if add_sp:
        sp_feat_cols = [c for c in pairs_df.columns if 'sp_A_' in c and 'pre' not in c]
        print(f"  - SP feature columns: {len(sp_feat_cols)}")
    
    # Simple plot
    if plot:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        pairs_df['pair_type'].value_counts().plot(kind='bar')
        plt.title('Pairs by Type')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 3, 2)
        pairs_df['library'].value_counts().head(10).plot(kind='bar')
        plt.title('Top 10 Libraries by Pairs')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 3, 3)
        pairs_df['confidence'].hist(bins=20)
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence')
        
        plt.tight_layout()
        
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/pair_check.png')
        plt.show()
    
    return pairs_df