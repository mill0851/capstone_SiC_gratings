import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PATH = r"NRS06032024/square_gratings/"

PARAM_KEYS = ('TLW', 'BLW', 'S', 'H')


def parse_params(name: str) -> dict[str, str]:
    """Pull TLW/BLW/S/H values out of a filename or directory name.
    Handles both `_` and `-` separators between key-value pairs."""
    out = {}
    for key in PARAM_KEYS:
        m = re.search(rf'(?:^|[_-]){key}-([A-Za-z0-9]+)', name)
        if m:
            out[key] = m.group(1)
    return out


def load_afm_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load AFM .txt: skip header lines (x(um), y(nm), blank, X<TAB>Y), parse the rest."""
    data = np.loadtxt(path, skiprows=4)
    return data[:, 0], data[:, 1]


def plot_afm_directory(directory: str | Path) -> None:
    directory = Path(directory)
    files = sorted(directory.glob('*.txt'))
    if not files:
        raise FileNotFoundError(f"No .txt files in {directory}")

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
        x, y = load_afm_txt(f)
        if y.size == 0:
            continue
        label = f"{scan_key}={fp.get(scan_key, '?')}"
        color = cmap(i / max(len(files) - 1, 1))
        ax.plot(x, y, color=color, linewidth=1, label=label)

    ax.set_xlabel(r'x ($\mu$m)')
    ax.set_ylabel('y (nm)')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(title=scan_key)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_afm_directory(PATH)
