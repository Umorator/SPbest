# src/ranking_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class SPRankingModel(nn.Module):
    """Neural network for ranking SPs"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final layer outputs a single score
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass to get ranking score"""
        return self.network(x).squeeze()
    
    def predict_score(self, features):
        """Predict ranking score for a single SP"""
        self.eval()
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            return self.forward(features).numpy()
        



class SPRankingDataset(Dataset):
    """Dataset for SP ranking pairs"""
    
    def __init__(self, pairs_df, feature_cols, preprotein_only=False, include_library_weights=False):
        """
        Args:
            pairs_df: DataFrame with pairs and features
            feature_cols: list of feature column names to use
            preprotein_only: if True, only use preprotein features (not used currently, kept for compatibility)
            include_library_weights: if True, include library weights in the returned item
        """
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.include_library_weights = include_library_weights
        
        # Separate SP A and SP B features
        self.sp_a_features = []
        self.sp_b_features = []
        self.targets = []
        self.confidences = []
        
        # Optional: store library weights
        if include_library_weights and 'library_weight' in pairs_df.columns:
            self.library_weights = []
        else:
            self.library_weights = None
        
        for idx, row in pairs_df.iterrows():
            # Get features for SP A (all columns starting with sp_A_)
            a_features = row[[c for c in feature_cols if c.startswith('sp_A_')]].values.astype(np.float32)
            
            # Get features for SP B (all columns starting with sp_B_)
            b_features = row[[c for c in feature_cols if c.startswith('sp_B_')]].values.astype(np.float32)
            
            self.sp_a_features.append(a_features)
            self.sp_b_features.append(b_features)
            self.targets.append(row['target'])
            self.confidences.append(row['confidence'])
            
            if include_library_weights and 'library_weight' in pairs_df.columns:
                self.library_weights.append(row['library_weight'])
    
    def __len__(self):
        return len(self.pairs_df)
    
    def __getitem__(self, idx):
        item = {
            'sp_a_features': torch.FloatTensor(self.sp_a_features[idx]),
            'sp_b_features': torch.FloatTensor(self.sp_b_features[idx]),
            'target': torch.FloatTensor([self.targets[idx]]),
            'confidence': torch.FloatTensor([self.confidences[idx]])
        }
        
        # Add library weights if available
        if self.include_library_weights and self.library_weights is not None:
            item['library_weight'] = torch.FloatTensor([self.library_weights[idx]])
        
        return item