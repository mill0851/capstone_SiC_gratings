import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class PCADataset(Dataset):
    """
    Dataset for PCA-surrogate training:
    Geometry ('tlw', 'blw', 'h', 's') --> (pc_0, pc_1, ..., pc_{K-1})
    where the PC coefficients come from extract_pca(). A held reference to
    the PCA artifacts (mean + components) lets the dataset reconstruct a
    full spectrum from predicted coefficients, so an MLP trained on this
    data can be evaluated in spectrum space without re-fitting PCA.

    Supports:
        > MLP (PCA-coefficient regression, i.e. linear-output spectral surrogate)
    """

    def __init__(
            self,
            geom_df: pd.DataFrame,
            feat_df: pd.DataFrame,
            pca_artifacts: dict,
            normalize_geom: bool = True,
            normalize_feat: bool = True
    ):
        """
        Args:
            geom_df (pd.DataFrame): indexed by run_xxxx, columns ['tlw', 'blw', 'h', 's']
            feat_df (pd.DataFrame): indexed by run_xxxx, columns ['pc_0', ..., 'pc_{K-1}']
                (output of extract_pca)
            pca_artifacts (dict): artifacts dict from extract_pca containing at least
                'mean' (L,), 'components' (K, L), 'wl' (L,). Stored so the dataset
                can reconstruct spectra from predicted coefficients.
            normalize_geom (bool, optional): Defaults to True.
            normalize_feat (bool, optional): Defaults to True. When True, each PC
                channel is standardized to zero-mean unit-variance so the MSE loss
                weights all components equally during training (PC0 otherwise
                dominates because its variance is much larger than later PCs).
        """

        # Force matching order
        geom_df = geom_df.sort_index()
        feat_df = feat_df.loc[geom_df.index]

        # Convert to numpy
        geom_np = geom_df.values.astype(np.float32)     # (S, 4)
        feat_np = feat_df.values.astype(np.float32)     # (S, K)

        # Config
        self.K = feat_np.shape[1]
        self.normalize_geom = normalize_geom
        self.normalize_feat = normalize_feat

        # Keep PCA artifacts for reconstruction (spectrum = mean + coeffs @ components)
        self.pca_mean = pca_artifacts['mean'].astype(np.float32)
        self.pca_components = pca_artifacts['components'][:self.K].astype(np.float32)
        self.wl = pca_artifacts['wl']

        # Normalize geometry
        if normalize_geom:
            self.geom_mean = geom_np.mean(axis=0)
            self.geom_std  = geom_np.std(axis=0) + 1e-8
            geom_np = (geom_np - self.geom_mean) / self.geom_std
        else:
            self.geom_mean = None
            self.geom_std  = None

        # Normalize features (per-PC standardization).
        # PCA coeffs already have ~zero mean by construction (over the fit set),
        # but std varies a lot across components, so standardizing each channel
        # is the meaningful step: MSE then weights every PC equally instead of
        # being dominated by PC0.
        if normalize_feat:
            self.feat_mean = feat_np.mean(axis=0).astype(np.float32)
            self.feat_std  = (feat_np.std(axis=0) + 1e-8).astype(np.float32)
            feat_np = (feat_np - self.feat_mean) / self.feat_std
        else:
            self.feat_mean = None
            self.feat_std  = None

        # Convert to tensors
        self.geom = torch.tensor(geom_np, dtype=torch.float32)   # (S, 4)
        self.feat = torch.tensor(feat_np, dtype=torch.float32)   # (S, K)

    def __len__(self):
        return len(self.geom)

    def __getitem__(self, idx):
        return (
            self.geom[idx],   # geometry (4,)
            self.feat[idx]    # PCA coefficients (K,)
        )

    def denormalize_feature(self, y: torch.Tensor) -> torch.Tensor:
        """
        Map standardized PC coefficients back to raw coefficient space.

        Args:
            y (torch.Tensor): shape (..., K)
        """
        if not self.normalize_feat:
            return y
        mean = torch.tensor(self.feat_mean, dtype=y.dtype, device=y.device)
        std  = torch.tensor(self.feat_std,  dtype=y.dtype, device=y.device)
        return y * std + mean

    def reconstruct_spectrum(self, y: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a full spectrum from (possibly normalized) PC coefficients.
        Inverts feature normalization first, then applies the PCA synthesis
        spectrum = mean + coeffs @ components.

        Args:
            y (torch.Tensor): shape (..., K)
        Returns:
            torch.Tensor: shape (..., L) spectrum on self.wl
        """
        coeffs = self.denormalize_feature(y)
        mean = torch.tensor(self.pca_mean, dtype=y.dtype, device=y.device)
        comps = torch.tensor(self.pca_components, dtype=y.dtype, device=y.device)
        return mean + coeffs @ comps
