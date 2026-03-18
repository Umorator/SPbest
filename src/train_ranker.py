# src/train_ranker.py

import os
import json
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from src.ranking_model import SPRankingDataset, SPRankingModel


# ============================================================
# Feature utilities
# ============================================================

def prepare_features(pairs_df, add_sp=True):

    if add_sp:
        feature_cols = [
            c for c in pairs_df.columns
            if (c.startswith("sp_A_") or c.startswith("sp_B_"))
            and not c.endswith("_id")
        ]
    else:
        feature_cols = [
            c for c in pairs_df.columns
            if (c.startswith("sp_A_pre_") or c.startswith("sp_B_pre_"))
        ]

    print(f"Using {len(feature_cols)} feature columns")

    return feature_cols


def normalize_features(train_df, val_df, feature_cols):

    train_df = train_df.copy()
    val_df = val_df.copy()

    a_cols = [c for c in feature_cols if c.startswith("sp_A_")]
    b_cols = [c for c in feature_cols if c.startswith("sp_B_")]

    # ✅ Rename columns to SAME feature space
    a_renamed = [c.replace("sp_A_", "") for c in a_cols]
    b_renamed = [c.replace("sp_B_", "") for c in b_cols]

    train_A = train_df[a_cols].copy()
    train_A.columns = a_renamed

    train_B = train_df[b_cols].copy()
    train_B.columns = b_renamed

    # ✅ Fit on combined SAME-NAME features
    scaler = StandardScaler()
    scaler.fit(pd.concat([train_A, train_B], axis=0))

    # ✅ Transform A
    train_df.loc[:, a_cols] = scaler.transform(train_A)
    val_A = val_df[a_cols].copy()
    val_A.columns = a_renamed
    val_df.loc[:, a_cols] = scaler.transform(val_A)

    # ✅ Transform B
    train_df.loc[:, b_cols] = scaler.transform(train_B)
    val_B = val_df[b_cols].copy()
    val_B.columns = b_renamed
    val_df.loc[:, b_cols] = scaler.transform(val_B)

    return train_df, val_df, scaler


# ============================================================
# Cluster folds
# ============================================================

def split_kfold_clusters(config_path):

    with open(config_path) as f:
        config = json.load(f)

    df = pd.read_csv(config["labeled_data"])
    clusters = pd.read_csv(config["cluster_file"])

    df = df.merge(clusters, on="Author-Protein")

    df = df[['id', 'Author-Protein', 'enzyme_activity', 'cluster', 'label']]

    folds = sorted(df.cluster.unique())

    fold_info = {}

    for fold in folds:

        val_ids = df[df.cluster == fold].id.tolist()
        train_ids = df[df.cluster != fold].id.tolist()

        fold_info[fold] = {
            "train_ids": train_ids,
            "val_ids": val_ids,
            "train_proteins": df[df.cluster != fold]["Author-Protein"].nunique(),
            "val_proteins": df[df.cluster == fold]["Author-Protein"].nunique()
        }

    return df, fold_info


def filter_pairs(pairs_df, ids):

    return pairs_df[
        pairs_df.sp_A_id.isin(ids) &
        pairs_df.sp_B_id.isin(ids)
    ]


# ============================================================
# Validation
# ============================================================

def evaluate_model(model, val_pairs_df, feature_cols, sp_df, device):

    model.eval()

    dataset = SPRankingDataset(val_pairs_df, feature_cols)
    loader = DataLoader(dataset, batch_size=256)

    scores_a = []
    scores_b = []
    targets = []

    with torch.no_grad():

        for batch in loader:

            sp_a = batch["sp_a_features"].to(device)
            sp_b = batch["sp_b_features"].to(device)

            s_a = model(sp_a)
            s_b = model(sp_b)

            scores_a.extend(s_a.cpu().numpy())
            scores_b.extend(s_b.cpu().numpy())
            targets.extend(batch["target"].numpy())

    scores_a = np.array(scores_a).flatten()
    scores_b = np.array(scores_b).flatten()
    targets = np.array(targets).flatten()

    predictions = (scores_a > scores_b).astype(int)

    pair_accuracy = (predictions == targets).mean()

    # ----------------------------
    # protein ranking evaluation
    # ----------------------------

    sp_ids = pd.unique(
        val_pairs_df[["sp_A_id", "sp_B_id"]].values.ravel()
    )

    rows = val_pairs_df.drop_duplicates("sp_A_id")

    feature_matrix = rows[
        [c for c in feature_cols if c.startswith("sp_A_")]
    ].values.astype(np.float32)

    features = torch.tensor(feature_matrix).to(device)

    with torch.no_grad():
        sp_scores = model(features).cpu().numpy().flatten()

    score_map = dict(zip(rows.sp_A_id.values, sp_scores))

    sp_lookup = sp_df[sp_df.id.isin(sp_ids)].copy()

    sp_lookup["score"] = sp_lookup.id.map(score_map)

    protein_spearman = []
    protein_top1 = []

    for protein, group in sp_lookup.groupby("Author-Protein"):

        if len(group) < 3:
            continue

        corr, _ = spearmanr(
            group.enzyme_activity,
            group.score
        )

        if not np.isnan(corr):
            protein_spearman.append(corr)

        true_best = group.loc[group.enzyme_activity.idxmax()].id
        pred_best = group.loc[group.score.idxmax()].id

        protein_top1.append(true_best == pred_best)

    return {
        "accuracy": pair_accuracy,
        "spearman": np.mean(protein_spearman),
        "top1": np.mean(protein_top1)
    }


# ============================================================
# Training
# ============================================================

def train_fold(
        model,
        train_loader,
        val_pairs,
        feature_cols,
        sp_df,
        device,
        epochs=100,
        learning_rate=1e-3,   # ✅ rename
        weight_decay=1e-5     # ✅ add
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    ranking_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction="none")

    best_acc = 0
    patience = 6
    counter = 0

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch in train_loader:

            sp_a = batch["sp_a_features"].to(device)
            sp_b = batch["sp_b_features"].to(device)
            targets = batch["target"].to(device)
            conf = batch["confidence"].to(device)

            s_a = model(sp_a).view(-1)
            s_b = model(sp_b).view(-1)
            ranking_targets = (targets.view(-1) * 2 - 1)

            losses = ranking_loss(s_a, s_b, ranking_targets)

            loss = (losses * conf).mean()

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:

            val = evaluate_model(
                model,
                val_pairs,
                feature_cols,
                sp_df,
                device
            )

            print(
                f"Epoch {epoch+1} | "
                f"Loss {total_loss/len(train_loader):.4f} | "
                f"Acc {val['accuracy']:.3f} | "
                f"Spearman {val['spearman']:.3f} | "
                f"Top1 {val['top1']:.3f}"
            )

            if val["accuracy"] > best_acc:

                best_acc = val["accuracy"]
                counter = 0

            else:

                counter += 1

            if counter >= patience:
                print("Early stopping")
                break


# ============================================================
# Cross validation
# ============================================================

def train_ranking_model_cv(
    config_path,
    pairs_path="outputs/ranking_pairs.csv",
    batch_size=64,
    epochs=100,
    learning_rate=1e-3,          # ✅ NEW
    weight_decay=1e-5,           # ✅ NEW
    hidden_dims=[256,128,64],    # ✅ NEW
    dropout=0.3,                 # ✅ NEW
    normalize=True,              # ✅ NEW
    run_folder=None              # (kept for compatibility)
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs_df = pd.read_csv(pairs_path)

    feature_cols = prepare_features(pairs_df)

    input_dim = len([c for c in feature_cols if c.startswith("sp_A_")])

    cluster_df, fold_info = split_kfold_clusters(config_path)

    sp_df = cluster_df[["id", "Author-Protein", "enzyme_activity"]]

    results = []

    for fold, info in fold_info.items():

        print("\n==============================")
        print(f"FOLD {fold}")
        print("==============================")

        train_pairs = filter_pairs(pairs_df, info["train_ids"])
        val_pairs = filter_pairs(pairs_df, info["val_ids"])

        # ✅ Optional normalization
        if normalize:
            train_pairs, val_pairs, _ = normalize_features(
                train_pairs,
                val_pairs,
                feature_cols
            )

        train_dataset = SPRankingDataset(train_pairs, feature_cols)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # ✅ Model now uses configurable params
        model = SPRankingModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(device)

        # ✅ Pass LR + WD into training
        train_fold(
            model,
            train_loader,
            val_pairs,
            feature_cols,
            sp_df,
            device,
            epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

        val = evaluate_model(
            model,
            val_pairs,
            feature_cols,
            sp_df,
            device
        )

        results.append(val)

    results = pd.DataFrame(results)

    print("\n==============================")
    print("CV RESULTS")
    print("==============================")

    print(results)

    print("\nAverage accuracy:", results.accuracy.mean())
    print("Average spearman:", results.spearman.mean())
    print("Average top1:", results.top1.mean())

    return results

