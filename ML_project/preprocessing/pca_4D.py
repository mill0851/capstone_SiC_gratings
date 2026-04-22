import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from util.data_preprocessing import *
from preprocessing.pipelines import build_pca_pipeline


#### DATA PREPROCESSING PIPELINE - PCA FEATURES ####
# Config
pd.set_option('display.max_columns', 10)
DATA_PATH = './data/batch2'
DOMAIN = (10.25, 11.0)
TEST_IDX = np.arange(0, 1024, 128)
INTERP = 4
K = 10
K_SWEEP = [2, 5, 10, 20, 40, 80]
N_PC_SHOW = 6

# Import Data
geom_labels, geom_table, refl_data, abs_data, refl_bg, abs_bg, wl = import_data(DATA_PATH)
print(f'\ngeometry labels:\n{geom_labels}\n')
print(f'geometry_table:\n{geom_table.head()}\n')
print(f'\ngeometry labels:\n{geom_labels}\n')
print(f'geometry_table:\n{geom_table.head()}\n')
print(f'reflection data:\n{refl_data.head()}\n')
print(f'reflection background:\n{refl_bg.head()}\n')
print(f'absorption data:\n{abs_data.head()}\n')
print(f'absorption background:\n{abs_bg.head()}\n')


# Build and fit preprocessing pipelines
abs_pipe  = build_pca_pipeline(K, DOMAIN, INTERP, background=abs_bg)
refl_pipe = build_pca_pipeline(K, DOMAIN, INTERP, background=refl_bg, flip=True)

abs_ft_pca  = abs_pipe.fit_transform(abs_data)
refl_ft_pca = refl_pipe.fit_transform(refl_data)

abs_pca  = abs_pipe.named_steps['pca'].artifacts_
refl_pca = refl_pipe.named_steps['pca'].artifacts_

# Processed spectra (all steps except PCA) — used by exploration plots below
abs_data  = abs_pipe[:-1].fit_transform(abs_data)
refl_data = refl_pipe[:-1].fit_transform(refl_data)
wl = abs_data.columns.to_numpy(dtype=float)

print(f'Feature table (abs):\n{abs_ft_pca.head()}\n')
print(f'Feature table (refl):\n{refl_ft_pca.head()}\n')


#### EXPLORATION PLOTS ####
# 1) Scree / cumulative explained variance
def plot_scree(artifacts: dict, title: str, k_max: int = 80) -> None:
    evr = artifacts['explained_variance_ratio'][:k_max]
    cum = np.cumsum(evr)
    idx = np.arange(1, len(evr) + 1)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(idx, evr, color='steelblue')
    ax1.set_xlabel('Principal component', fontsize=12)
    ax1.set_ylabel('Explained variance ratio', fontsize=12)
    ax1.set_title(f'{title} - Scree', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    ax2.plot(idx, cum, 'o-', color='darkred')
    for thr in (0.95, 0.99, 0.999):
        ax2.axhline(thr, linestyle='--', alpha=0.5)
        k_hit = int(np.searchsorted(cum, thr) + 1) if (cum >= thr).any() else None
        if k_hit is not None:
            ax2.annotate(f'{thr:g}: K={k_hit}', xy=(k_hit, thr),
                         xytext=(5, -15), textcoords='offset points')
    ax2.set_xlabel('Number of components K', fontsize=12)
    ax2.set_ylabel('Cumulative explained variance', fontsize=12)
    ax2.set_title(f'{title} - Cumulative EV', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_scree(abs_pca, 'Absorption', k_max = 20)


# 2) Shapes of the first N principal components
def plot_pc_shapes(artifacts: dict, title: str, n_show: int = N_PC_SHOW) -> None:
    wl_local = artifacts['wl']
    comps = artifacts['components_full'][:n_show]
    evr = artifacts['explained_variance_ratio'][:n_show]
    plt.figure(figsize=(12, 6))
    for i in range(n_show):
        plt.plot(wl_local, comps[i],
                 label=f'PC{i}  ({evr[i]*100:.1f}% var)')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.xlabel('Wavelength (um)', fontsize=14)
    plt.ylabel('Component weight', fontsize=14)
    plt.title(f'{title} - First {n_show} principal components', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_pc_shapes(abs_pca, 'Absorption')


# 3) Reconstruction fidelity vs K on a single example spectrum
def plot_k_sweep_reconstruction(df: pd.DataFrame, artifacts: dict,
                                sample_idx: int, ks: list[int], title: str) -> None:
    wl_local = artifacts['wl']
    X = df.to_numpy()
    mean = artifacts['mean']
    components_full = artifacts['components_full']
    truth = X[sample_idx]
    centered = truth - mean

    plt.figure(figsize=(12, 6))
    plt.plot(wl_local, truth, 'k-', linewidth=2, label='Truth')
    for K_i in ks:
        comps = components_full[:K_i]
        coeffs = centered @ comps.T
        recon = mean + coeffs @ comps
        plt.plot(wl_local, recon, '--', alpha=0.85, label=f'K={K_i}')
    plt.xlabel('Wavelength (um)', fontsize=14)
    plt.ylabel('Absorption (proxy)', fontsize=14)
    plt.title(f'{title} - Reconstruction vs K  (sample {sample_idx})',
              fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

# Pick a spectrum with a clean peak for visualization
demo_idx = int(TEST_IDX[len(TEST_IDX) // 2])
plot_k_sweep_reconstruction(abs_data, abs_pca, demo_idx, K_SWEEP, 'Absorption')


# 4) Spectral MSE and peak-location MAE vs K (dataset-wide diagnostic)
#    This is the important sanity check: global MSE can look fine while
#    the resonance peak is smeared. Track both.
def peak_wavelengths(data: np.ndarray, wl_local: np.ndarray,
                     height: float = 0.2) -> np.ndarray:
    """Wavelength of the largest-amplitude peak per row, NaN if none found."""
    out = np.full(len(data), np.nan)
    for i, row in enumerate(data):
        pks, props = find_peaks(row, height=height)
        if len(pks) == 0:
            continue
        best = pks[np.argmax(props['peak_heights'])]
        out[i] = wl_local[best]
    return out

def plot_k_sweep_error(df: pd.DataFrame, artifacts: dict,
                       ks: list[int], title: str) -> None:
    wl_local = artifacts['wl']
    X = df.to_numpy()
    mean = artifacts['mean']
    components_full = artifacts['components_full']
    Xc = X - mean

    truth_peaks = peak_wavelengths(X, wl_local)
    truth_valid = ~np.isnan(truth_peaks)

    mse_list = []
    peak_mae_list = []
    for K_i in ks:
        comps = components_full[:K_i]
        recon = mean + (Xc @ comps.T) @ comps
        mse_list.append(np.mean((recon - X) ** 2))

        recon_peaks = peak_wavelengths(recon, wl_local)
        both_valid = truth_valid & ~np.isnan(recon_peaks)
        if both_valid.any():
            peak_mae_list.append(
                np.mean(np.abs(recon_peaks[both_valid] - truth_peaks[both_valid]))
            )
        else:
            peak_mae_list.append(np.nan)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ks, mse_list, 'o-', color='steelblue')
    ax1.set_xlabel('K (components kept)', fontsize=12)
    ax1.set_ylabel('Mean spectral MSE', fontsize=12)
    ax1.set_title(f'{title} - Reconstruction MSE vs K',
                  fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(alpha=0.3)

    ax2.plot(ks, peak_mae_list, 'o-', color='darkred')
    ax2.set_xlabel('K (components kept)', fontsize=12)
    ax2.set_ylabel('Peak-location MAE (um)', fontsize=12)
    ax2.set_title(f'{title} - Peak-location MAE vs K',
                  fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f'{title} K-sweep:')
    for k_i, m, pm in zip(ks, mse_list, peak_mae_list):
        print(f'  K={k_i:3d}  MSE={m:.3e}  peak-MAE={pm:.4f} um')

plot_k_sweep_error(abs_data, abs_pca, K_SWEEP, 'Absorption')


# 5) Latent space scatter: PC1 vs PC2 colored by each geometry parameter.
#    Useful for spotting structure (does the geometry vary smoothly through
#    the latent, or cluster?). If the surrogate learns well, a smooth gradient
#    in these plots is a good sign.
def plot_latent_colored_by_geom(features: pd.DataFrame,
                                geom_table: pd.DataFrame,
                                geom_labels: np.ndarray,
                                title: str) -> None:
    geom_aligned = geom_table.loc[features.index]
    pc0 = features['pc_0'].to_numpy()
    pc1 = features['pc_1'].to_numpy()

    n_g = len(geom_labels)
    cols = 2
    rows = int(np.ceil(n_g / cols))
    _, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axes = np.atleast_2d(axes).ravel()
    for ax, label in zip(axes, geom_labels):
        sc = ax.scatter(pc0, pc1, c=geom_aligned[label].to_numpy(),
                        cmap='viridis', s=10)
        ax.set_xlabel('PC0', fontsize=12)
        ax.set_ylabel('PC1', fontsize=12)
        ax.set_title(f'{title}: colored by {label}', fontsize=13)
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax)
    for ax in axes[n_g:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()

plot_latent_colored_by_geom(abs_ft_pca, geom_table, geom_labels, 'Absorption')


# 6) Reconstruction grid: data (target) vs K-component PCA reconstruction
#    for each sample in TEST_IDX. Direct visual check of PCA fidelity.
def plot_reconstruction_grid(df: pd.DataFrame, artifacts: dict,
                              test_idx: np.ndarray, title: str,
                              recon_modelled: np.ndarray | None = None) -> None:
    wl_local = artifacts['wl']
    X = df.to_numpy()
    mean = artifacts['mean']
    components = artifacts['components']
    K_local = artifacts['K']

    Xc = X - mean
    coeffs = Xc @ components.T
    recons = mean + coeffs @ components

    n_plot = len(test_idx)
    ncols = 4
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()
    for i, idx in enumerate(test_idx):
        ax = axes_flat[i]
        ax.plot(wl_local, X[idx], color='k', label='Data')
        ax.plot(wl_local, recons[idx], color='grey', linestyle='--',
                label=f'True PCA (K={K_local})')
        if recon_modelled is not None:
            ax.plot(wl_local, recon_modelled[i], color='cyan', linestyle='--',
                    label='Modelled')
        ax.set_title(f"idx={int(idx)}", fontsize=9)
        ax.grid(alpha=0.5)
        if i == 0:
            ax.legend(fontsize=9)
    for j in range(n_plot, len(axes_flat)):
        axes_flat[j].axis('off')
    fig.suptitle(f"{title}: Data vs PCA Reconstruction (K={K_local})", fontsize=14)
    fig.tight_layout()
    plt.show()

plot_reconstruction_grid(abs_data, abs_pca, TEST_IDX, 'Absorption')


# Create export dict
pca_ft = {
    "wl": wl,
    "reflection": refl_data,
    "absorption": abs_data,
    "reflection_features": refl_ft_pca,
    "absorption_features": abs_ft_pca,
    "reflection_pca": refl_pca,
    "absorption_pca": abs_pca,
    "geometry_labels": geom_labels,
    "geometry_table": geom_table,
    "K": K,
}

data_config = {
    "data_path": DATA_PATH,
    "domain": DOMAIN,
    "interpolation_density": INTERP,
    "K": K,
}
