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

        # Optional per-sample loss weights. When None, __getitem__ yields
        # (geom, feat); when set, yields (geom, feat, weight) so trainers can
        # apply a weighted MSE without re-plumbing the dataloader pipeline.
        self.sample_weights: torch.Tensor | None = None

        # Cached PCA tensors for reconstruct_spectrum — avoids torch.tensor(np)
        # allocation on every batch (called O(1e8) times across an HPO run).
        self._pca_mean_t  = torch.from_numpy(self.pca_mean)
        self._pca_comps_t = torch.from_numpy(self.pca_components)
        if self.feat_mean is not None:
            self._feat_mean_t = torch.from_numpy(self.feat_mean)
            self._feat_std_t  = torch.from_numpy(self.feat_std)
        else:
            self._feat_mean_t = None
            self._feat_std_t  = None

    def __len__(self):
        return len(self.geom)

    def __getitem__(self, idx):
        if self.sample_weights is None:
            return (
                self.geom[idx],   # geometry (4,)
                self.feat[idx]    # PCA coefficients (K,)
            )
        return (
            self.geom[idx],
            self.feat[idx],
            self.sample_weights[idx],
        )

    def set_sample_weights(self, weights):
        """
        Attach per-sample loss weights (shape (S,)). After this, __getitem__
        yields (geom, feat, weight) and the PcaMLP trainers will pick them up
        when the model has peak_weighted_loss=True.
        """
        if weights is None:
            self.sample_weights = None
            return
        w = torch.as_tensor(weights, dtype=torch.float32)
        if w.shape[0] != self.geom.shape[0]:
            raise ValueError(
                f"weights length {w.shape[0]} != dataset length {self.geom.shape[0]}"
            )
        self.sample_weights = w

    def denormalize_feature(self, y: torch.Tensor) -> torch.Tensor:
        """
        Map standardized PC coefficients back to raw coefficient space.

        Args:
            y (torch.Tensor): shape (..., K)
        """
        if not self.normalize_feat:
            return y
        mean = self._feat_mean_t.to(dtype=y.dtype, device=y.device)
        std  = self._feat_std_t.to(dtype=y.dtype, device=y.device)
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
        mean = self._pca_mean_t.to(dtype=y.dtype, device=y.device)
        comps = self._pca_comps_t.to(dtype=y.dtype, device=y.device)
        return mean + coeffs @ comps
