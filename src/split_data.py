import pandas as pd
import json
import os


def split_kfold_clusters(config_path):
    """
    Create 5 folds based on predefined clusters.

    Inputs:
    - labeled dataset (with id, Author-Protein, enzyme_activity, label)
    - cluster file (Author-Protein -> cluster)

    Output:
    - 5 CSV files with the split data
    """

    # -----------------------------
    # Load config
    # -----------------------------
    with open(config_path, "r") as f:
        config = json.load(f)

    data_path = config["labeled_data"]
    cluster_path = config["cluster_file"]
    output_dir = config["fold_output"]

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Load datasets
    # -----------------------------
    df = pd.read_csv(data_path)
    clusters = pd.read_csv(cluster_path)

    # -----------------------------
    # Merge cluster info
    # -----------------------------
    df = df.merge(clusters, on="Author-Protein", how="left")

    if df["cluster"].isna().any():
        missing = df[df["cluster"].isna()]["Author-Protein"].unique()
        print("WARNING: Missing cluster for:", missing)

    # -----------------------------
    # Keep required columns
    # -----------------------------
    df = df[['id', 'Author-Protein', 'enzyme_activity', 'cluster', 'label']]

    # -----------------------------
    # Create folds
    # -----------------------------
    folds = sorted(df["cluster"].unique())

    for fold in folds:

        test = df[df["cluster"] == fold]
        train = df[df["cluster"] != fold]

        train_path = os.path.join(output_dir, f"train_fold_{fold}.csv")
        test_path = os.path.join(output_dir, f"test_fold_{fold}.csv")

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        print(f"Fold {fold}")
        print(f"Train size: {len(train)}")
        print(f"Test size: {len(test)}")
        print("-" * 40)

    print("Finished creating folds.")

    return df