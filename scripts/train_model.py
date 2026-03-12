import json
import pandas as pd

from src.utils.data_split import create_cluster_folds
from src.models.train_lgbm_pu import train_lgbm_pu


# load config
with open("configs/config_training.json") as f:
    config = json.load(f)

# load data
raw_df = pd.read_csv(config["raw_data"])
features_df = pd.read_csv(config["features_file"])

# create folds
folds = create_cluster_folds(
    raw_df,
    config["cluster_file"],
    features_df
)

print(f"{len(folds)} folds created")

# train model
train_lgbm_pu(
    folds,
    config["lightgbm_params"]
)