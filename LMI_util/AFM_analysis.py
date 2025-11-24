import matplotlib.pyplot as plt
import numpy as np
import json
import re
from pathlib import Path


# Helper Functions for analysis

def load_all_AFM(root_dir, sample_name="Unspecified"):
    afm_data = []
    root_dir = Path(root_dir)

    fname_pattern = re.compile(
        r"TLW-(?P<TLW>[^-]+)-BLW-(?P<BLW>[^-]+)-S-(?P<S>[^-]+)-H-(?P<H>[^-]+)\.txt"
    )

    for path in root_dir.rglob("*.txt"):
        m = fname_pattern.fullmatch(path.name)
        if m is None:
            continue

        TLW = float(m.group("TLW"))
        BLW = float(m.group("BLW"))
        S   = float(m.group("S"))
        H   = float(m.group("H"))

        data = import_data_AFM(path)
        x = data[:, 0].tolist()
        y = data[:, 1].tolist()

        afm_data.append({
        "pTLW": TLW,
        "pBLW": BLW,
        "pS": S,
        "pH": H,
        "AFMx": x,
        "AFMy": y,
        "grating": str(path),
        "sample": sample_name
        })

    return afm_data

def import_data_AFM(path):
    data = np.loadtxt(path, float, skiprows=4)

    # Import lines for units
    with open(path, 'r') as file:
        lines = file.readlines()
        x_units = lines[0]
        y_units = lines[1]

    # Put data in (um)
    if y_units == "y(nm)\n":
        data[:,1] = data[:,1] * 10**-3
    elif x_units == "x(nm)\n":
        data[:,0] = data[:,0] * 10**-3

    # Normalize data
    data[:,1] = data[:,1] - np.min(data[:,1])

    return data

def get_param_pairs_AFM(afm_data, param_x, param_y, plot=False):
    x = [] 
    y = []
    sample_name = None

    for data in afm_data:
        if sample_name is None:
            sample_name = data['sample']

        try:
            x.append(data[param_x])
            y.append(data[param_y])

        except KeyError as e:
            raise KeyError(
                f"Record is missing key {e} for parameters {param_x}, {param_y}. "
                f"Offending record: {data}"
            ) from e

    if plot:
        plt.scatter(x, y)
        plt.xlabel(param_x)
        plt.ylabel(param_y)
        plt.title(f"{param_y} vs {param_x} : {sample_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return np.array(x), np.array(y)

def get_raw_data_AFM(afm_data, TLW, BLW, S, H):
    for data in afm_data:
        if data["pTLW"] == TLW and data["pBLW"] == BLW and data["pS"] == S and data['pH'] == H:
            x = data["AFMx"]
            y = data["AFMy"]
            sample = data["sample"]
        grating = f"TLW-{TLW} BLW-{BLW} S-{S} H-{H}"
    return np.array(x), np.array(y), sample, grating

def moving_average(y, window=5):
    """Simple 1D moving average, edges left as-is if window < 2."""
    if window < 2:
        return y
    # mode='same' keeps same length, convolution with normalized box
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')

def plot_single_AFM(AFMx, AFMy, sample, grating, AFMy_smooth=None, level1=None, level2=None, baseline=None):
    plt.figure(figsize=(15,10))
    plt.plot(AFMx, AFMy, color='r', linewidth=2)
    
    if AFMy_smooth.any():
        plt.plot(AFMx, AFMy_smooth, color='b', linewidth=2)

    if level1 and AFMy_smooth.any():
        y_level = np.empty(len(AFMy))
        y_level.fill(level1 * np.max(AFMy_smooth))
        plt.plot(AFMx, y_level, color='green', linestyle='--', linewidth=2)
    elif level1:
        y_level = np.empty(len(AFMy))
        y_level.fill(level1 * np.max(AFMy))
        plt.plot(AFMx, y_level, color='green', linestyle='--', linewidth=2)

    if level2 and AFMy_smooth.any():
        y_level = np.empty(len(AFMy))
        y_level.fill(level2 * np.max(AFMy_smooth))
        plt.plot(AFMx, y_level, color='green', linestyle='--', linewidth=2)
    elif level2:
        y_level = np.empty(len(AFMy))
        y_level.fill(level2 * np.max(AFMy))
        plt.plot(AFMx, y_level, color='green', linestyle='--', linewidth=2)

    if baseline.any():
        plt.plot(AFMx, baseline, color='purple', linestyle='--', linewidth=2)

    plt.grid(True, alpha=0.5)
    plt.xlabel("Tip Horizontal Position (um)", fontsize=30)
    plt.ylabel("Tip Vertical Position (um)", fontsize=30)
    plt.title(f"AFM: {sample}: {grating}", fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.legend(fontsize=30)
    plt.show()


def estimate_baseline_AFM(AFMx, AFMy_smooth, low_percentile=30, poly_deg=1):
    """
    Estimate a slowly-varying baseline by fitting a polynomial to
    the 'low' parts of the signal (below some percentile).

    Returns:
        baseline : array same shape as y_smooth
    """
    # Pick points that are likely in the troughs
    cutoff = np.percentile(AFMy_smooth, low_percentile)
    mask = AFMy_smooth <= cutoff

    if mask.sum() < poly_deg + 1:
        # not enough points to fit; fall back to simple constant baseline
        baseline_level = AFMy_smooth[mask].mean() if mask.any() else AFMy_smooth.mean()
        return np.full_like(AFMy_smooth, baseline_level)

    # Fit polynomial baseline to low points
    coeffs = np.polyfit(AFMx[mask], AFMy_smooth[mask], deg=poly_deg)
    baseline = np.polyval(coeffs, AFMx)

    return baseline