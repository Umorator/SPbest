import pandas as pd


def create_cluster_folds(raw_df, cluster_file, features_df):
    """
    Create cluster folds and attach feature vectors.

    Returns
    -------
    folds : list
        Each fold contains:
        {
          "X_train": features,
          "y_train": labels,
          "X_test": features,
          "y_test": labels
        }
    """

    clusters = pd.read_csv(cluster_file)

    # merge clusters
    df = raw_df.merge(clusters, on="Author-Protein", how="left")

    if df["cluster"].isna().any():
        raise ValueError("Some Author-Protein entries missing cluster")

    folds = []

    for cluster_id in sorted(df["cluster"].unique()):

        train = df[df.cluster != cluster_id]
        test = df[df.cluster == cluster_id]

        # select features by id
        X_train = features_df[features_df["id"].isin(train["id"])]
        X_test = features_df[features_df["id"].isin(test["id"])]

        # align labels
        y_train = train.set_index("id").loc[X_train["id"], "label"]
        y_test = test.set_index("id").loc[X_test["id"], "label"]

        # remove id column from features
        X_train = X_train.drop(columns=["id"])
        X_test = X_test.drop(columns=["id"])

        folds.append({
            "X_train": X_train,
            "y_train": y_train,
            "train_meta": train[["id", "Author-Protein"]].reset_index(drop=True),
            "X_test": X_test,
            "y_test": y_test
        })

    return folds