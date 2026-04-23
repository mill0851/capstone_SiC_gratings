import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class PCAPeakDataset(Dataset):
    """
    Dataset for the multi-task PCA + peak surrogate:
    Geometry ('tlw', 'blw', 'h', 's') --> (pc_0, ..., pc_{K-1}) and
                                          (lambda_res, gamma, A) of the
                                          max-amplitude peak.

    A held reference to the PCA artifacts (mean + components) lets the
    dataset reconstruct a full spectrum from predicted coefficients,
    so the model can be evaluated in spectrum space without re-fitting PCA.

    A per-sample mask is exposed so the peak-head loss can be ignored
    on samples with no resonance in the domain (the (lambda, gamma, A)
    triple is then all zeros).

    Supports:
        > PcaPeakMLP (PCA-coefficient regression + auxiliary peak head)
    """

    def __init__(
            self,
            geom_df: pd.DataFrame,
            pca_df: pd.DataFrame,
            amp_df: pd.DataFrame,
            pca_artifacts: dict,
            normalize_geom: bool = True,
            normalize_pca: bool = False,
            normalize_amp: bool = True
    ):
        """
        Args:
            geom_df (pd.DataFrame): indexed by run_xxxx, columns ['tlw', 'blw', 'h', 's']
            pca_df (pd.DataFrame): indexed by run_xxxx, columns ['pc_0', ..., 'pc_{K-1}']
                (output of extract_pca)
            amp_df (pd.DataFrame): indexed by run_xxxx, columns ['lambda_res', 'gamma', 'A']
                (output of max_A). Rows with no detected peak should be all-zero.
            pca_artifacts (dict): artifacts dict from extract_pca containing at least
                'mean' (L,), 'components' (K, L), 'wl' (L,). Stored so the dataset
                can reconstruct spectra from predicted coefficients.
            normalize_geom (bool, optional): Defaults to True.
            normalize_pca (bool, optional): Defaults to False. When True, each PC
                channel is standardized to zero-mean unit-variance so the MSE loss
                weights all components equally during training (PC0 otherwise
                dominates because its variance is much larger than later PCs).
            normalize_amp (bool, optional): Defaults to True. When True, each
                peak-target channel is standardized using stats computed over the
                masked (peak-present) rows only, so zero-padded rows do not
                bias the mean/std.
        """

        # Force matching order
        geom_df = geom_df.sort_index()
        pca_df = pca_df.loc[geom_df.index]
        amp_df = amp_df.loc[geom_df.index]

        # Convert to numpy
        geom_np = geom_df.values.astype(np.float32)     # (S, 4)
        pca_np  = pca_df.values.astype(np.float32)      # (S, K)
        amp_np  = amp_df.values.astype(np.float32)      # (S, 3)

        # Config
        self.K = pca_np.shape[1]
        self.normalize_geom = normalize_geom
        self.normalize_pca = normalize_pca
        self.normalize_amp = normalize_amp

        # Keep PCA artifacts for spectrum reconstruction
        # (spectrum = pca_spec_mean + coeffs @ pca_components)
        self.pca_spec_mean  = pca_artifacts['mean'].astype(np.float32)
        self.pca_components = pca_artifacts['components'][:self.K].astype(np.float32)
        self.wl = pca_artifacts['wl']

        # Mask must be computed from RAW amp targets, before any normalization
        # shifts zero-padded rows away from zero. (S, 1) so it broadcasts cleanly
        # against per-sample peak losses.
        mask_np = (amp_np.sum(axis=1) != 0).astype(np.float32).reshape(-1, 1)

        # Normalize geometry
        if normalize_geom:
            self.geom_mean = geom_np.mean(axis=0)
            self.geom_std  = geom_np.std(axis=0) + 1e-8
            geom_np = (geom_np - self.geom_mean) / self.geom_std
        else:
            self.geom_mean = None
            self.geom_std  = None

        # Normalize PCA coefficients (per-PC standardization).
        # Stored as pca_norm_* to avoid colliding with pca_spec_mean above.
        if normalize_pca:
            self.pca_norm_mean = pca_np.mean(axis=0).astype(np.float32)
            self.pca_norm_std  = (pca_np.std(axis=0) + 1e-8).astype(np.float32)
            pca_np = (pca_np - self.pca_norm_mean) / self.pca_norm_std
        else:
            self.pca_norm_mean = None
            self.pca_norm_std  = None

        # Normalize peak targets (per-channel standardization).
        # Stats are computed over peak-present rows only so that zero-padded
        # no-peak rows don't bias them. The mask gates the loss so the
        # standardized pad-row values are never seen by the optimizer.
        if normalize_amp:
            valid = mask_np.squeeze(1).astype(bool)
            if valid.any():
                self.amp_mean = amp_np[valid].mean(axis=0).astype(np.float32)
                self.amp_std  = (amp_np[valid].std(axis=0) + 1e-8).astype(np.float32)
            else:
                self.amp_mean = np.zeros(amp_np.shape[1], dtype=np.float32)
                self.amp_std  = np.ones(amp_np.shape[1], dtype=np.float32)
            amp_np = (amp_np - self.amp_mean) / self.amp_std
        else:
            self.amp_mean = None
            self.amp_std  = None

        # Convert to tensors
        self.geom = torch.tensor(geom_np, dtype=torch.float32)   # (S, 4)
        self.pca  = torch.tensor(pca_np,  dtype=torch.float32)   # (S, K)
        self.amp  = torch.tensor(amp_np,  dtype=torch.float32)   # (S, 3)
        self.mask = torch.tensor(mask_np, dtype=torch.float32)   # (S, 1)

    def __len__(self):
        return len(self.geom)

    def __getitem__(self, idx):
        return (
            self.geom[idx],   # geometry (4,)
            self.pca[idx],    # PCA coefficients (K,)
            self.amp[idx],    # max-amp peak target (lambda, gamma, A) (3,)
            self.mask[idx],   # peak-present flag (1,)
        )

    def denormalize_pca(self, y: torch.Tensor) -> torch.Tensor:
        """
        Map standardized PC coefficients back to raw coefficient space.

        Args:
            y (torch.Tensor): shape (..., K)
        """
        if not self.normalize_pca:
            return y
        mean = torch.tensor(self.pca_norm_mean, dtype=y.dtype, device=y.device)
        std  = torch.tensor(self.pca_norm_std,  dtype=y.dtype, device=y.device)
        return y * std + mean

    def denormalize_amp(self, y: torch.Tensor) -> torch.Tensor:
        """
        Map normalized peak targets back to raw (lambda, gamma, A) space.

        Args:
            y (torch.Tensor): shape (..., 3)
        """
        if not self.normalize_amp:
            return y
        mean = torch.tensor(self.amp_mean, dtype=y.dtype, device=y.device)
        std  = torch.tensor(self.amp_std,  dtype=y.dtype, device=y.device)
        return y * std + mean

    def reconstruct_spectrum(self, y: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a full spectrum from (possibly normalized) PC coefficients.
        Inverts coefficient normalization first, then applies the PCA synthesis
        spectrum = pca_spec_mean + coeffs @ pca_components.

        Args:
            y (torch.Tensor): shape (..., K)
        Returns:
            torch.Tensor: shape (..., L) spectrum on self.wl
        """
        coeffs = self.denormalize_pca(y)
        mean = torch.tensor(self.pca_spec_mean, dtype=y.dtype, device=y.device)
        comps = torch.tensor(self.pca_components, dtype=y.dtype, device=y.device)
        return mean + coeffs @ comps
