import matplotlib.pyplot as plt
import numpy as np
import json
import re
from pathlib import Path
import os

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

def plot_single_AFM(AFMx, AFMy, sample, grating, label='no label',
                    AFMy_smooth1=(np.array([]),"no label"),
                    AFMy_smooth2=(np.array([]),'no label'),
                    level1=(0,'no label'),
                    level2=(0,'no label'),
                    baseline=(np.array([]),'no label'),
                    level_curve=0):
    
    plt.figure(figsize=(15,10))
    plt.scatter(AFMx, AFMy, color='r', linewidth=2, label=label)
    
    if AFMy_smooth1[0].any():
        plt.plot(AFMx, AFMy_smooth1[0], color='b', linewidth=2, linestyle='--', label=AFMy_smooth1[1])

    if AFMy_smooth2[0].any():
        plt.plot(AFMx, AFMy_smooth2[0], color='k', linewidth=2, linestyle='--', label=AFMy_smooth2[1])

    if level1[0] and level_curve==1:
        y_level = np.empty(len(AFMy_smooth1[0]))
        y_level.fill(level1[0] * np.max(AFMy_smooth1[0]))
        plt.plot(AFMx, y_level, color='green', linestyle='--', linewidth=2, label=level1[1])
    elif level1[0]:
        y_level = np.empty(len(AFMy))
        y_level.fill(level1[0] * np.max(AFMy))
        plt.plot(AFMx, y_level, color='green', linestyle='--', linewidth=2, label=level1[1])

    if level2[0] and level_curve==2:
        y_level = np.empty(len(AFMy_smooth1[0]))
        y_level.fill(level2[0] * np.max(AFMy_smooth1[0]))
        plt.plot(AFMx, y_level, color='green', linestyle='--', linewidth=2, label=level2[1])
    elif level2[0]:
        y_level = np.empty(len(AFMy))
        y_level.fill(level2[0] * np.max(AFMy))
        plt.plot(AFMx, y_level, color='green', linestyle='--', linewidth=2, label=level2[1])

    if baseline[0].any():
        plt.plot(AFMx, baseline, color='purple', linestyle='--', linewidth=2, label=baseline[1])

    plt.grid(True, alpha=0.5)
    plt.xlabel("Tip Horizontal Position (um)", fontsize=30)
    plt.ylabel("Tip Vertical Position (um)", fontsize=30)
    plt.title(f"AFM: {sample}: {grating}", fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=30)
    # plt.legend(fontsize=30)
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

def segment_structures_with_intersections(
        x, y, baseline,
        eps=0.0,
        min_points=30,
        min_width=1.0):
    """
    Split AFM line scan into complete structures defined as regions
    above a baseline, and include the exact baseline intersection
    points as the first and last point of each segment.

    Parameters
    ----------
    x, y : 1D arrays
        Full line-scan data.
    baseline : float
        Baseline level to segment against.
    eps : float
        Extra offset: treat y > baseline + eps as "above".
        Use small positive eps to ignore tiny crossings.
    min_points : int
        Minimum number of points in a segment (after adding intersections).
    min_width : float
        Minimum horizontal width (x_max - x_min) of a segment.

    Returns
    -------
    segments : list of (x_seg, y_seg)
        Each segment starts and ends exactly on the baseline.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    # Boolean mask: meaningfully above baseline
    above = y > (baseline + eps)
    trans = np.diff(above.astype(int))

    # Indices where we go from below -> above (entry) and above -> below (exit)
    start_idx = np.where(trans == 1)[0]    # crossing between i and i+1
    end_idx   = np.where(trans == -1)[0]   # crossing between i and i+1

    # If there is an exit before the first entry, drop that exit
    if len(end_idx) and len(start_idx) and end_idx[0] < start_idx[0]:
        end_idx = end_idx[1:]

    # If more entries than exits, drop the last entry (open segment at right edge)
    if len(start_idx) > len(end_idx):
        start_idx = start_idx[:len(end_idx)]

    def interpolate_cross(i):
        """Linear interpolation of x where line between i and i+1 hits baseline."""
        x0, x1 = x[i], x[i+1]
        y0, y1 = y[i], y[i+1]
        # assume y0 and y1 straddle the baseline
        t = (baseline - y0) / (y1 - y0)
        return x0 + t * (x1 - x0)

    segments = []

    for s_i, e_i in zip(start_idx, end_idx):
        # Intersections at entry and exit
        x_left  = interpolate_cross(s_i)
        x_right = interpolate_cross(e_i)

        # interior indices that are strictly above baseline
        # s_i+1 .. e_i inclusive
        x_mid = x[s_i+1 : e_i+1]
        y_mid = y[s_i+1 : e_i+1]

        # Build full segment with intersection points at ends
        x_seg = np.concatenate([[x_left], x_mid, [x_right]])
        y_seg = np.concatenate([[baseline], y_mid, [baseline]])

        # Size filters to reject tiny blips / noise
        if len(x_seg) < min_points:
            continue
        if x_seg[-1] - x_seg[0] < min_width:
            continue

        segments.append((x_seg - np.min(x_seg), y_seg - np.min(y_seg)))

    return segments

def trapezoidal_fit_segments(segments, top_fraction=0.9):
    """
    Approximate each segment as a trapezoid using a simple rule:

    - Define a top level: top_level = top_fraction * max(y_seg)
    - Find first two points on the LEFT with y >= top_level
    - Find first two points on the RIGHT with y >= top_level
    - Plateau height = min( those four y-values )
    - Left edge of plateau = x of first LEFT point above threshold
    - Right edge of plateau = x of first RIGHT point above threshold
    - Sides: straight lines from (x0,0) -> (x_left, plateau_h)
             and (x_right, plateau_h) -> (xN,0)

    Assumes each segment has already been leveled so that the baseline is ~0.

    Parameters
    ----------
    segments : list of (x_seg, y_seg)
        Each x_seg, y_seg is a 1D array of equal length.
    top_fraction : float in (0,1]
        Fraction of segment max used as the initial threshold.

    Returns
    -------
    trap_segments : list of (x_seg, y_trap_seg)
        Same x arrays, with y replaced by the trapezoidal fit.
    """

    trap_segments = []
    top_levels = []

    for xs, ys in segments:
        xs = np.asarray(xs)
        ys = np.asarray(ys)

        if len(xs) < 3:
            trap_segments.append((xs, ys.copy()))
            continue

        height = ys.max()
        if height <= 0:
            trap_segments.append((xs, np.zeros_like(ys)))
            continue

        top_level = top_fraction * height

        # --- find first point >= top_level on the LEFT ---
        left_indices = np.where(ys >= top_level)[0]
        if len(left_indices) < 2:
            # not enough points above threshold; just make a triangle
            peak_idx = int(np.argmax(ys))
            plateau_h = ys[peak_idx]
            x_left_plateau = xs[peak_idx]
            x_right_plateau = xs[peak_idx]
        else:
            L0 = left_indices[0]
            yL0 = ys[L0]
            x_left_plateau = xs[L0]

            # --- first points >= top_level on the RIGHT ---
            right_indices = left_indices[::-1]  # same set, reversed order
            R0 = right_indices[0]
            yR0 = ys[R0]
            x_right_plateau = xs[R0]

            plateau_h = min(yL0,yR0)

        # --- geometry for sides ---
        top_levels.append(plateau_h)
        x0 = xs[0]
        xN = xs[-1]

        left_dx = x_left_plateau - x0
        right_dx = xN - x_right_plateau

        m_left = plateau_h / left_dx
        m_right = plateau_h / right_dx

        # --- build trapezoid on original x grid ---
        y_trap = np.zeros_like(ys)

        for i, x in enumerate(xs):
            if x <= x_left_plateau:
                # left side
                y_trap[i] = m_left * (x - x0)
            elif x >= x_right_plateau:
                # right side
                y_trap[i] = m_right * (xN - x)
            else:
                # flat top
                y_trap[i] = plateau_h

            if y_trap[i] < 0:
                y_trap[i] = 0.0

        trap_segments.append((xs, y_trap))

    return trap_segments, top_levels

def cubic_fit_segments(segments, degree=3, return_coeffs=False):
    """
    Fit a polynomial of given degree (default 3) to each segment.

    Parameters
    ----------
    segments : list of (x_seg, y_seg)
        Each x_seg, y_seg is a 1D array of equal length.
    degree : int, optional
        Polynomial degree to fit (default = 3).
    return_coeffs : bool, optional
        If True, also return the list of coefficient arrays for each segment.

    Returns
    -------
    fitted_segments : list of (x_seg, y_fit_seg)
        Same x arrays, with y replaced by the polynomial fit.
    coeffs_list : list of np.ndarray (only if return_coeffs=True)
        Each entry is the polynomial coefficient array for that segment
        (as returned by np.polyfit, highest power first).
    """

    fitted_segments = []
    coeffs_list = []

    for xs, ys in segments:
        xs = np.asarray(xs)
        ys = np.asarray(ys)

        # Guard against degenerate segments
        if len(xs) <= degree:
            # Not enough points to fit the requested degree; just copy
            fitted_segments.append((xs, ys.copy()))
            coeffs_list.append(None)
            continue

        coeffs = np.polyfit(xs, ys, deg=degree)
        y_fit = np.polyval(coeffs, xs)

        fitted_segments.append((xs, y_fit))
        coeffs_list.append(coeffs)

    if return_coeffs:
        return fitted_segments, coeffs_list
    else:
        return fitted_segments
    
def increase_point_density(x, y, n):
    """
    Insert n linearly interpolated points between every pair of (x, y).

    Parameters
    ----------
    x : array-like
        Original x values (monotonic).
    y : array-like
        Original y values.
    n : int
        Number of points to add between every consecutive pair.

    Returns
    -------
    x_new : np.ndarray
        New x array with increased point density.
    y_new : np.ndarray
        New y array with increased point density.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if n < 1:
        raise ValueError("n must be >= 1")

    x_list = []
    y_list = []

    for i in range(len(x) - 1):
        # Original left point
        x0, x1 = x[i], x[i+1]
        y0, y1 = y[i], y[i+1]

        # Add original point
        x_list.append(x0)
        y_list.append(y0)

        # Interpolated points between x0 → x1
        for k in range(1, n+1):
            t = k / (n + 1)
            xi = x0 + t * (x1 - x0)
            yi = y0 + t * (y1 - y0)
            x_list.append(xi)
            y_list.append(yi)

    # Add final point
    x_list.append(x[-1])
    y_list.append(y[-1])

    return np.array(x_list), np.array(y_list)

def set_processing_params(data_index,
                        window_size_base,
                        blw_lvl_base,
                        tlw_lvl_base,
                        window_size_adjustments=np.array([]),
                        blw_lvl_adjustments=np.array([]),
                        tlw_lvl_adjustments=np.array([])):
    window_size = 0
    if window_size_adjustments.any():
        for adjustment in window_size_adjustments:
            if data_index == adjustment[0]:
                window_size = adjustment[1]
    if not window_size:
        window_size = window_size_base

    blw_lvl = 0
    if blw_lvl_adjustments.any():
        for adjustment in blw_lvl_adjustments:
            if data_index == adjustment[0]:
                blw_lvl = adjustment[1]
    if not blw_lvl:
        blw_lvl = blw_lvl_base

    tlw_lvl = 0
    if tlw_lvl_adjustments.any():
        for adjustment in tlw_lvl_adjustments:
            if data_index == adjustment[0]:
                tlw_lvl = adjustment[1]
    if not tlw_lvl:
        tlw_lvl = tlw_lvl_base

    print('data index = ' + str(data_index))
    print('tlw level = ' + str(tlw_lvl))
    print('blw level = ' + str(blw_lvl))
    print('window size = ' + str (window_size))

    return window_size, blw_lvl, tlw_lvl

def process_data(afm_data,
                baseline_lvl,
                min_segment_length,
                interp_points,
                window_size_base,
                blw_lvl_base,
                tlw_lvl_base,
                window_size_adjustments=np.array([]),
                blw_lvl_adjustments=np.array([]),
                tlw_lvl_adjustments=np.array([])):
    data_index = 0
    for data in afm_data:

        # ---- EXTRACT DATA ----
        x = np.array(data['AFMx'])
        y = np.array(data['AFMy'])
        x_dense, y_dense = increase_point_density(x, y, interp_points) # Interpolation
        sample = data['sample']
        grating = tag = os.path.splitext(os.path.basename(data['grating']))[0]

        # ---- DETERMINE GRATING PARAMETERS ----
        window_size, blw_lvl, tlw_lvl = set_processing_params(data_index,
                                                        window_size_base,
                                                        blw_lvl_base,
                                                        tlw_lvl_base, 
                                                        window_size_adjustments=window_size_adjustments,
                                                        blw_lvl_adjustments=blw_lvl_adjustments,
                                                        tlw_lvl_adjustments=tlw_lvl_adjustments)
        
        # ---- DATA SMOOTHING ----
        # Utilizes a moving average with a specified
        # windoe size to remove noise
        y_smooth = moving_average(y, window = window_size)
        _, y_smooth_dense = increase_point_density(x, y_smooth, interp_points) # Interpolation

        # ---- DATA LEVELING ----
        # Fits a first degree polynomial (a line) to a specified
        # percentile of data, then subtracts this fit from the data set
        baseline = estimate_baseline_AFM(x_dense, y_smooth_dense, low_percentile=baseline_lvl)
        y_smooth_level = y_smooth_dense - baseline
        y_raw_level = y_dense - baseline
        # plot_single_AFM(x_dense,
        #             y_dense,
        #             sample,
        #             grating + f": Grating {data_index}",
        #             label="Raw Data Dense",
        #             AFMy_smooth1=(y_smooth_dense, 'Smooth Data Dense'),
        #             AFMy_smooth2=(baseline, 'BLW Cutoff'))
        # plot_single_AFM(x_dense,
        #             y_raw_level,
        #             sample,
        #             grating + f": Grating {data_index}",
        #             label='Raw Data Leveled',
        #             AFMy_smooth1=(y_smooth_level,'Smooth Data Leveled'),
        #             level1=(blw_lvl, f'BLW Cutoff = {blw_lvl}'))


        # ---- DATA SEGMENTATION ----
        # divides each data set into a set of data points
        # for each grating groove in the set
        grooves_raw = segment_structures_with_intersections(x_dense, y_raw_level, blw_lvl * np.max(y_raw_level), min_points=min_segment_length)
        grooves_smooth = segment_structures_with_intersections(x_dense, y_smooth_level, blw_lvl * np.max(y_smooth_level), min_points=min_segment_length)

        print("Number of grooves (raw) = " + str(len(grooves_raw)))
        print("Number of grooves (smooth) = " + str(len(grooves_smooth)))

        data['grooves raw'] = grooves_raw
        data['grooves smooth'] = grooves_smooth
        data['groove count'] = len(grooves_smooth)

            # ---- DATA FITTING ----
        # here we cut the data off at some percentage of the maximum
        # value. The closest data point above this horizontal line determines
        # the height of the trapezoid. A horizontal line is drawn from this
        # point to the index of the closest point above the line on the opposite
        # side of the groove. The edges are created by linearly interporlating
        # from the first and last point to the line
        groove_fits, tlw_heights = trapezoidal_fit_segments(grooves_smooth, top_fraction=tlw_lvl)
        data['groove fits'] = groove_fits
        data['tlw heights'] = tlw_heights

        print("Fit Heights (um):", *[round(x, 3) for x in tlw_heights])
        print()

        data_index += 1