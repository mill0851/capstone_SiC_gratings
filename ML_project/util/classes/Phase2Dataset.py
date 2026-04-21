import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class Phase2Dataset(Dataset):
    """
    Dataset for Phase 2 of LMI Machine Learning Project:
    Geometry ('tlw', 'blw', 'h', 's') --> (lambda_0, Q_0, ..., lambda_{N-1}, Q_{N-1})
    over some specified spectral domain, for N candidate peaks per spectrum
    (see multi_peak_extraction).

    Supports:
        > MLP (multi-peak regression)
    """

    def __init__(
            self,
            geom_df: pd.DataFrame,
            feat_df: pd.DataFrame,
            n_peaks: int,
            normalize_geom: bool = True,
            normalize_feat: bool = True
    ):
        """
        Args:
            geom_df (pd.DataFrame): indexed by run_xxxx, columns: ['tlw', 'blw', 'h', 's']
            feat_df (pd.DataFrame): indexed by run_xxxx, columns from multi_peak_extraction:
                ['lambda_0', ..., 'lambda_{N-1}',
                 'Q_0',      ..., 'Q_{N-1}',
                 'mask_0',   ..., 'mask_{N-1}']
                Slots where no peak was found have lambda = Q = 0 and mask = 0.
            n_peaks (int): number of peaks N (must match columns in feat_df).
            normalize_geom (bool, optional): Defaults to True.
            normalize_feat (bool, optional): Defaults to True.
        """

        # Force matching order
        geom_df = geom_df.sort_index()
        feat_df = feat_df.loc[geom_df.index]

        # Column groups
        lambda_cols = [f'lambda_{i}' for i in range(n_peaks)]
        Q_cols      = [f'Q_{i}'      for i in range(n_peaks)]
        mask_cols   = [f'mask_{i}'   for i in range(n_peaks)]

        # Convert to numpy
        geom_np   = geom_df.values.astype(np.float32)                    # (S, 4)
        lambda_np = feat_df[lambda_cols].values.astype(np.float32)       # (S, N)
        Q_np      = feat_df[Q_cols].values.astype(np.float32)            # (S, N)
        mask_np   = feat_df[mask_cols].values.astype(np.float32)         # (S, N)

        # Config
        self.n_peaks = n_peaks
        self.normalize_geom = normalize_geom
        self.normalize_feat = normalize_feat

        # Normalize geometry
        if normalize_geom:
            self.geom_mean = geom_np.mean(axis=0)
            self.geom_std  = geom_np.std(axis=0) + 1e-8
            geom_np = (geom_np - self.geom_mean) / self.geom_std
        else:
            self.geom_mean = None
            self.geom_std = None

        # Normalize features (only over valid peaks, mask == 1)
        # Single scalar mean/std per channel (lambda, Q), aggregated across all
        # ranks and samples -- mirrors Phase1Dataset's per-channel stats.
        if normalize_feat:
            self.feat_mean = np.zeros(2, dtype=np.float32)  # [lambda_mean, Q_mean]
            self.feat_std  = np.ones(2,  dtype=np.float32)  # [lambda_std,  Q_std]

            valid = mask_np == 1
            if valid.sum() > 0:
                self.feat_mean[0] = lambda_np[valid].mean()
                self.feat_std[0]  = lambda_np[valid].std() + 1e-8
                self.feat_mean[1] = Q_np[valid].mean()
                self.feat_std[1]  = Q_np[valid].std() + 1e-8

                lambda_np = np.where(
                    valid,
                    (lambda_np - self.feat_mean[0]) / self.feat_std[0],
                    lambda_np
                )
                Q_np = np.where(
                    valid,
                    (Q_np - self.feat_mean[1]) / self.feat_std[1],
                    Q_np
                )
        else:
            self.feat_mean = None
            self.feat_std  = None

        # Stack channels -> (S, N, 2), last dim = (lambda, Q)
        feat_np = np.stack([lambda_np, Q_np], axis=-1).astype(np.float32)

        # Convert to tensors
        self.geom = torch.tensor(geom_np, dtype=torch.float32)   # (S, 4)
        self.feat = torch.tensor(feat_np, dtype=torch.float32)   # (S, N, 2)
        self.mask = torch.tensor(mask_np, dtype=torch.float32)   # (S, N)

    def __len__(self):
        return len(self.geom)

    def __getitem__(self, idx):
        return (
            self.geom[idx],   # geometry  (4,)
            self.feat[idx],   # peaks     (N, 2) -> (lambda, Q)
            self.mask[idx]    # per-peak  (N,)   -> 0 or 1
        )

    def denormalize_feature(self, y):
        """
        Args:
            y (torch.Tensor): shape (..., N, 2), last dim is (lambda, Q).
        """
        if not self.normalize_feat:
            return y
        mean = torch.tensor(self.feat_mean, dtype=y.dtype, device=y.device)
        std  = torch.tensor(self.feat_std,  dtype=y.dtype, device=y.device)
        return y * std + mean
