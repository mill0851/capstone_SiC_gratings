import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PATH = r"NRS06032024_FTIR_dpt_15x_90deg_second_measurements/NRS06032024_TLW-xx_BLW-4_S-4_H-3_dpt/"

X_MIN, X_MAX = 700.0, 1000.0
PARAM_KEYS = ('TLW', 'BLW', 'S', 'H')


def parse_params(name: str) -> dict[str, str]:
    """Pull TLW/BLW/S/H values out of a filename like '..._TLW-5_BLW-5_S-1_H-3.dpt'."""
    out = {}
    for key in PARAM_KEYS:
        m = re.search(rf'{key}-([A-Za-z0-9]+)', name)
        if m:
            out[key] = m.group(1)
    return out


def plot_dpt_directory(directory: str | Path) -> None:
    directory = Path(directory)
    files = sorted(directory.glob('*.dpt'))
    if not files:
        raise FileNotFoundError(f"No .dpt files in {directory}")

    # Use the directory's params as the canonical set; whichever value differs
    # across files is the "scanned" parameter.
    dir_params = parse_params(directory.name)
    file_params = [parse_params(f.name) for f in files]
    scanned_keys = [
        k for k in PARAM_KEYS
        if len({fp.get(k) for fp in file_params}) > 1
    ]
    if not scanned_keys:
        scanned_keys = [k for k, v in dir_params.items() if v.lower().startswith('x')]
    scan_key = scanned_keys[0] if scanned_keys else 'param'

    title_params = dir_params if dir_params else file_params[0]
    title = ' '.join(f"{k}-{title_params.get(k, '?')}" for k in PARAM_KEYS if k in title_params)

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap('viridis')

    for i, (f, fp) in enumerate(zip(files, file_params)):
        data = np.loadtxt(f)
        x, y = data[:, 0], data[:, 1]
        mask = (x >= X_MIN) & (x <= X_MAX)
        x_win, y_win = x[mask], y[mask]
        if y_win.size == 0:
            continue
        y_norm = y_win / y_win.max()
        label = f"{scan_key}={fp.get(scan_key, '?')}"
        color = cmap(i / max(len(files) - 1, 1))
        ax.plot(x_win, y_norm, color=color, linewidth=1, label=label)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xlabel(r'Wavenumber (cm$^{-1}$)')
    ax.set_ylabel('Normalized intensity (a.u.)')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(title=scan_key)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_dpt_directory(PATH)
