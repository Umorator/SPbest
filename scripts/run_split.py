import sys
import os

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.split_data import split_kfold_clusters

split_kfold_clusters("configs/get_optimals.json")