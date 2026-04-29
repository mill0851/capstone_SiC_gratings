from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


HERE = Path(__file__).parent

DPT_PATH = HERE / 'NRS06032024_FTIR_dpt_15x_90deg_second_measurements' \
    / 'NRS06032024_TLW-xx_BLW-4_S-4_H-3_dpt' \
    / '2025-06-11_NRS06032024_TLW-5_BLW-4_S-4_H-3.dpt'

MULTI_ANGLE_PATH = HERE / 'multi_angle_scan_T-5_B-4_S-4_H-3.txt'

X_MIN, X_MAX = 10.0, 12.5  # wavelength window (um) for the plot


def load_dpt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """col0 = wavenumber (cm^-1), col1 = intensity. Convert to wavelength (um),
    then normalize y by its max within [X_MIN, X_MAX]."""
    data = np.loadtxt(path)
    wavenumber = data[:, 0]
    y = data[:, 1]
    wavelength_um = 1e4 / wavenumber
    order = np.argsort(wavelength_um)
    wavelength_um, y = wavelength_um[order], y[order]

    win = (wavelength_um >= X_MIN) & (wavelength_um <= X_MAX)
    y = y / y[win].max()
    return wavelength_um, y


def load_multi_angle(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Skip `%`-prefixed header, normalize each of the 6 reflectance columns by its
    in-window max, then average them at each wavelength."""
    data = np.loadtxt(path, comments='%')
    wavelength_um = data[:, 0]
    refl = data[:, 1:7]

    win = (wavelength_um >= X_MIN) & (wavelength_um <= X_MAX)
    refl = refl / refl[win].max(axis=0)
    refl_avg = refl.mean(axis=1)
    return wavelength_um, refl_avg


def load_multi_angle_columns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all 6 reflectance columns and normalize each by its in-window max.
    Returns (wavelength_um, refl) where refl has shape (N, 6)."""
    data = np.loadtxt(path, comments='%')
    wavelength_um = data[:, 0]
    refl = data[:, 1:7]

    win = (wavelength_um >= X_MIN) & (wavelength_um <= X_MAX)
    refl = refl / refl[win].max(axis=0)
    return wavelength_um, refl


def main() -> None:
    wl_dpt, y_dpt = load_dpt(DPT_PATH)
    wl_ma, refl_ma = load_multi_angle(MULTI_ANGLE_PATH)
    wl_ma_all, refl_ma_all = load_multi_angle_columns(MULTI_ANGLE_PATH)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax = axes[0]
    ax.plot(wl_dpt, y_dpt, color='black', lw=1, label='FTIR')
    ax.plot(wl_ma, refl_ma, color='red', lw=1.2, label=f'COMSOL (mean of 6 angles)')
    ax.set_xlim(X_MIN, X_MAX)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14)

    ax = axes[1]
    ax.plot(wl_dpt, y_dpt, color='black', lw=1, label='FTIR')
    angles = ['9.8', '12.56', '15.32', '18.08', '20.84', '23.6']
    for i in range(refl_ma_all.shape[1]):
        ax.plot(wl_ma_all, refl_ma_all[:, i], lw=1.0, label=f'COMSOL ({angles[i]} deg)')
    ax.set_xlim(X_MIN, X_MAX)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14, ncol=2)

    fig.suptitle(f'COMSOL vs FTIR', fontsize=20)
    fig.supxlabel('x (um)', fontsize=16)
    fig.supylabel('y (um)', fontsize=16)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
