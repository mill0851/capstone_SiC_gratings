import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class PCAPeakDataset(Dataset):
    """
    Dataset for the multi-task PCA + peak surrogate:
    Geometry ('tlw', 'blw', 'h', 's') --> (pc_0, ..., pc_{K-1}) and
                                          (lambda_k, gamma_k, A_k) for the
                                          top-N peaks by amplitude.

    Peak target layout per sample (length 3N):
        [lambda_0, gamma_0, A_0, lambda_1, gamma_1, A_1, ..., lambda_{N-1}, gamma_{N-1}, A_{N-1}]
    where rank 0 is the highest-amplitude peak (produced by max_N_A). Samples
    with fewer than N peaks are zero-padded on the unused slots.

    A per-peak mask of shape (N,) is exposed so the peak-head loss can be
    applied only where a real peak is present.

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
            geom_df (pd.DataFrame): indexed by run_xxxx, geometry columns.
            pca_df (pd.DataFrame): indexed by run_xxxx, columns ['pc_0', ..., 'pc_{K-1}'].
            amp_df (pd.DataFrame): indexed by run_xxxx, (S, 3N) columns laid out
                [lambda_0, gamma_0, A_0, ..., lambda_{N-1}, gamma_{N-1}, A_{N-1}].
                Unused peak slots are zero-padded.
            pca_artifacts (dict): output of extract_pca containing at least
                'mean' (L,), 'components' (K, L), 'wl' (L,).
            normalize_geom (bool, optional): Defaults to True.
            normalize_pca (bool, optional): Defaults to False. When True, each PC
                channel is standardized to zero-mean unit-variance.
            normalize_amp (bool, optional): Defaults to True. When True, each
                peak's (lambda, gamma, A) triple is standardized using stats
                computed *only* over rows where that specific peak is present,
                so zero-padded slots do not bias the stats.
        """

        # Force matching order
        geom_df = geom_df.sort_index()
        pca_df = pca_df.loc[geom_df.index]
        amp_df = amp_df.loc[geom_df.index]

        # Convert to numpy
        geom_np = geom_df.values.astype(np.float32)     # (S, geom_dim)
        pca_np  = pca_df.values.astype(np.float32)      # (S, K)
        amp_np  = amp_df.values.astype(np.float32)      # (S, 3N)

        assert amp_np.shape[1] % 3 == 0, \
            f"amp_df must have 3N columns (got {amp_np.shape[1]})"

        # Config
        self.K = pca_np.shape[1]
        self.n_peaks = amp_np.shape[1] // 3
        self.normalize_geom = normalize_geom
        self.normalize_pca = normalize_pca
        self.normalize_amp = normalize_amp

        # Keep PCA artifacts for spectrum reconstruction
        # (spectrum = pca_spec_mean + coeffs @ pca_components)
        self.pca_spec_mean  = pca_artifacts['mean'].astype(np.float32)
        self.pca_components = pca_artifacts['components'][:self.K].astype(np.float32)
        self.wl = pca_artifacts['wl']

        # Per-peak mask of shape (S, N): a peak is "present" if any of its three
        # raw values is nonzero. Built from RAW amp targets, before normalization
        # can shift zero-padded rows away from zero.
        amp_reshaped = amp_np.reshape(-1, self.n_peaks, 3)          # (S, N, 3)
        mask_np = (np.abs(amp_reshaped).sum(axis=2) != 0).astype(np.float32)  # (S, N)

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

        # Normalize peak targets (per-peak, per-channel standardization).
        # For each peak k, compute stats using only the rows where peak k is
        # present; apply to columns [3k, 3k+1, 3k+2]. Zero-padded slots will
        # take non-meaningful normalized values but the mask gates the loss,
        # so those never affect training.
        if normalize_amp:
            amp_mean = np.zeros(3 * self.n_peaks, dtype=np.float32)
            amp_std  = np.ones(3 * self.n_peaks, dtype=np.float32)
            for peak_k in range(self.n_peaks):
                valid = mask_np[:, peak_k].astype(bool)
                cols = slice(3 * peak_k, 3 * (peak_k + 1))
                if valid.any():
                    amp_mean[cols] = amp_np[valid, cols].mean(axis=0)
                    amp_std[cols]  = amp_np[valid, cols].std(axis=0) + 1e-8
            self.amp_mean = amp_mean
            self.amp_std  = amp_std
            amp_np = (amp_np - self.amp_mean) / self.amp_std
        else:
            self.amp_mean = None
            self.amp_std  = None

        # Convert to tensors
        self.geom = torch.tensor(geom_np, dtype=torch.float32)   # (S, geom_dim)
        self.pca  = torch.tensor(pca_np,  dtype=torch.float32)   # (S, K)
        self.amp  = torch.tensor(amp_np,  dtype=torch.float32)   # (S, 3N)
        self.mask = torch.tensor(mask_np, dtype=torch.float32)   # (S, N)

    def __len__(self):
        return len(self.geom)

    def __getitem__(self, idx):
        return (
            self.geom[idx],   # geometry (geom_dim,)
            self.pca[idx],    # PCA coefficients (K,)
            self.amp[idx],    # peak targets (3N,)
            self.mask[idx],   # per-peak mask (N,)
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
        Map normalized peak targets back to raw (lambda, gamma, A, ...) space.

        Args:
            y (torch.Tensor): shape (..., 3N)
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
