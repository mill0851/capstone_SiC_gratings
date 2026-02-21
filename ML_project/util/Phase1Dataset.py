import torch
from torch.utils.data import Dataset
from typing import List
import numpy as np
import pandas as pd


class Phase1Dataset(Dataset):
    """
    Dataset for Phase 1 of LMI Machine Learning Project:
    Geometry ('tlw', 'blw', 'h', 's') --> (lambda_res, Q_max)
    this is done over some specified spectral domain (see phase1_preprocessing)

    Supports:
        > Linear Regression
        > Quadratic Regression
        > MLP
    """

    def __init__(
            self,
            geom_df: pd.DataFrame,
            feat_df: pd.DataFrame,
            normalize_geom: bool = True,
            normalize_feat: bool = True
    ):
        """
        Args:
            geom_df (pd.DataFrame): indexed by run_xxxx, columns: ['tlw', 'blw', 'h', 's']
            feat_df (pd.DataFrame): indexed by run_xxxx, columns: ['lambda', 'Q'], if no
            resonance within the spectral region under study lambda = Q = 0
            normalize_geom (bool, optional): Defaults to True.
            normalize_feat (bool, optional): Defaults to True.
        """

        # Force matching order
        geom_df = geom_df.sort_index()
        feat_df = feat_df.loc[geom_df.index]

        # Convert to numpy
        geom_np = geom_df.values.astype(np.float32)
        feat_np = feat_df.values.astype(np.float32)

        # Resonance mask
        has_res = (feat_np[:,1] > 0).astype(np.float32)

        # Normalization stats
        self.normalize_goem = normalize_geom
        self.normalize_feat = normalize_feat

        # Normalize geometry
        if normalize_geom:
            self.geom_mean = geom_np.mean(axis = 0)
            self.geom_std = geom_np.std(axis = 0) + 1e-8
            geom_np = (geom_np - self.geom_mean) / self.geom_std
        else:
            self.geom_mean = None
            self.geom_std = None

        # Normalize featurs
        if normalize_feat:
            mask = has_res == 1
            self.feat_mean = np.zeros(2, dtype=np.float32)
            self.feat_std = np.ones(2, dtype=np.float32)

            if mask.sum() > 0:
                self.feat_mean = feat_np[mask].mean(axis=0)
                self.feat_std = feat_np[mask].std(axis=0) + 1e-8

                feat_np[mask] = (
                    (feat_np[mask] - self.feat_mean) / self.feat_std
                )
        else:
            self.feat_mean = None
            self.feat_std = None

        # Convert to tensors
        self.geom = torch.tensor(geom_np, dtype=torch.float32)
        self.feat = torch.tensor(feat_np, dtype=torch.float32)
        self.has_resonance = torch.tensor(has_res, dtype=torch.float32)       

    # Inherited methods
    def __len__(self):
        return len(self.geom)

    def __getitem__(self, idx):
        return (
            self.geom[idx],           # geometry (4,)
            self.feat[idx],           # (lambda, Q)
            self.has_resonance[idx]   # 0 or 1
        )

    # Denormalization helpers
    def denormalize_geometry(self, x):
        if not self.normalize_geom:
            return x
        return x * torch.tensor(self.geom_std) + torch.tensor(self.geom_mean)

    def denormalize_feature(self, y):
        if not self.normalize_feat:
            return y
        return y * torch.tensor(self.feat_std) + torch.tensor(self.feat_mean)
