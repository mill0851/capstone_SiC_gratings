import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def import_data(
        root_dir: str) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:

    """
    This function is used to import comsol simulation data. It works with
    the specific directory structure:

    batch/
    |-- metadata.csv
    |-- runs/
    |   |-- run_001/
    |   |   |-- data.csv
    |   |-- run_002/
    |   |   |-- data.csv

    the metadata.csv is a metadata structure with 1 row per run.
    the data.csv files can be any relevant data but the first column of the csv
    should be "run_xxx" repeated in each row. The objective is to import data in
    a manner suitable for the beginning of a pytorch flow.
    
    :param root_dir: This should be the "batch" directory
    :type root_dir: str

    """

    # Read in metadata
    geom_path = os.path.join(root_dir, "metadata.csv")
    geom_table = pd.read_csv(geom_path)
    geom_table["run_id"] = geom_table["run_id"].astype(str)
    geom_table = geom_table.set_index("run_id").sort_index()
    geom_labels = geom_table.columns.to_numpy()

    # Read in background signal
    bg_path = os.path.join(root_dir, 'background.csv')
    bg_table = pd.read_csv(bg_path)
    bg_table['run_id'] = bg_table['run_id'].astype(str)
    bg_table['wavelength_um'] = bg_table['wavelength_um'] * 1e6
    
    refl_bg = bg_table.pivot(index='run_id', columns='wavelength_um', values='reflectance_0')
    refl_bg.columns = refl_bg.columns.astype(float)

    abs_bg = bg_table.pivot(index='run_id', columns='wavelength_um', values='Absorption')
    abs_bg.columns = abs_bg.columns.astype(float)

    # Loop through directory containing runs and import data
    runs_path = os.path.join(root_dir, "runs")
    df_list = []

    for item in os.listdir(runs_path):
        item_path = os.path.join(runs_path, item)

        if os.path.isdir(item_path):
            data_path = os.path.join(item_path, 'data.csv')
            data_table = pd.read_csv(data_path)
            data_table["run_id"] = data_table["run_id"].astype(str)
            df_list.append(data_table)

    all_data_long = pd.concat(df_list)
    all_data_long['wavelength_um'] = all_data_long['wavelength_um'] * 1e6

    all_data_pivot_refl = all_data_long.pivot(index='run_id', columns='wavelength_um', values='reflectance_0')
    all_data_pivot_refl.columns = all_data_pivot_refl.columns.astype(float)

    all_data_pivot_abs = all_data_long.pivot(index='run_id', columns='wavelength_um', values='Absorption')
    all_data_pivot_abs.columns = all_data_pivot_abs.columns.astype(float)

    wl_axis = df_list[0]['wavelength_um'].to_numpy() * 1e6


    return (geom_labels, geom_table, all_data_pivot_refl, all_data_pivot_abs, refl_bg, abs_bg, wl_axis)

def refl_vs_abs_plt(
        refl: pd.DataFrame,
        abs: pd.DataFrame,
        idx: int) -> None:
    """
    This function just does a quick plot on a chosen index of the absorption and
    reflection data. This is primarily useful for checking on the  data during processing
    or manually scanning through the data to see differeneces in abs vs refl or get a 
    general idea of what features there are.

    Args:
        refl (pd.DataFrame): pivoted table from import_data
        abs (pd.DataFrame): pivoted table from import_data
        idx (int): row you want to look at 
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(refl.columns, refl.iloc[idx], 'b-', linewidth=2)
    ax1.set_xlabel('Wavelength (μm)', fontsize=12)
    ax1.set_ylabel('Reflectance', fontsize=12)
    ax1.set_title('Reflectance vs Wavelength', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(abs.columns, abs.iloc[idx], 'r-', linewidth=2)
    ax2.set_xlabel('Wavelength (μm)', fontsize=12)
    ax2.set_ylabel('Absorption', fontsize=12)
    ax2.set_title('Absorption vs Wavelength', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def remove_background(
        data: pd.DataFrame,
        background: pd.DataFrame) -> None:
    """
    This function takes in one of the pivoted data table from import_data and removes a pivoted background
    signal from the same function.

    Args:
        data (pd.DataFrame): pivoted data table from import_data
        background (pd.DataFrame): pivoted background table from import_data
    """
    data[:] = data.values - background.iloc[0].values

def reduce_domain(
        bounds: tuple,
        data: pd.DataFrame) -> None:
    """
    Reduces the row width or domain of the pivoted data tables from import_data.

    Args:
        bounds (tuple): a tuple with an upper and lower bound
        data (pd.DataFrame): pivoted data table from import_data
    """
    data.sort_index(axis=1)
    xmin, xmax = bounds
    eps = 1e-12

    mask_data = (data.columns >= xmin - eps) & \
                (data.columns <= xmax + eps)
    
    data.drop(columns=data.columns[~mask_data],
                     inplace=True)

def normalize(
        data: pd.DataFrame) -> None:
    """
    Normalizes the data table my dividing my the max

    Args:
        data (pd.DataFrame): pivoted data table from import_data
    """
    max = data.max(axis=1)
    data[:] = data.div(max, axis=0)

def interp_linear(
        data: pd.DataFrame,
        n_samples: int) -> pd.DataFrame:
    """
    Linear interpolation of N point between each point. This is useful for function fitting.

    Args:
        data (pd.DataFrame): pivoted data table from import_data
        n_samples (int): how many samples to insert between each two data points

    Returns:
        pd.DataFrame: data except interpolated
    """
    wl_old = data.columns.to_numpy(dtype=float)
    num_orig = len(wl_old)

    wl_new = np.linspace(wl_old[0], wl_old[-1], (num_orig)*(n_samples+1)+1)

    interp_data = np.array([
        np.interp(wl_new, wl_old, row.values)
        for _, row in data.iterrows()
    ])

    data_interp = pd.DataFrame(interp_data, index=data.index, columns=wl_new)
    return data_interp

def lorentzian(
        x, A, x0, gamma):
    """Lorentzian with HWHM = gamma (no offset)"""
    return  (A / ((x - x0)**2 + gamma**2))

def highest_Q(
        df: pd.DataFrame,
        window: int,
        threshold: float) -> pd.DataFrame:
    """
    Here lorentzian fits are calculated for all peaks above
    a specific threshold and the highest Q resonance is extracted
    as the feature for that curve. The data stored is (lambda, Q)
    if no peak above the threshold is found both entries are 0
    """
    wl = df.columns.to_numpy(dtype=float)
    data = df.to_numpy()
    features = np.zeros((len(df),2))

    for rowIdx, row in enumerate(data):
        peaks, _ = find_peaks(row, height=threshold)
        best_res_q = [0,0]

        if len(peaks) == 0:
            features[rowIdx,:] = best_res_q
            continue

        for pIdx, p in enumerate(peaks):
            left_idx = max(0, p - window)
            right_idx = min(len(row), p + window)

            fit_region = row[left_idx:right_idx]
            fit_wl = wl[left_idx:right_idx]

            A0 = row[p]
            wl0 = wl[p]
            gamma0 = (fit_wl[-1] - fit_wl[0]) / 10
            p0 = [A0, wl0, gamma0]

            popt, pvoc = curve_fit(
                lorentzian,
                fit_wl,
                fit_region,
                p0=p0,
                bounds=(
                    [0, fit_wl[0], 0],
                    [np.inf, fit_wl[-1], np.inf]
                )
            )

            A = popt[0]
            wl_res = popt[1]
            gamma = popt[2]
            Q = wl_res / (2*gamma)

            if Q > best_res_q[1]:
                best_res_q[0] = wl_res
                best_res_q[1] = Q

        features[rowIdx,:] = best_res_q

    features = pd.DataFrame(features, index = df.index, columns=['lambda_res', 'Q'])
    return features

def multi_peak_extraction(
        df: pd.DataFrame,
        window: int,
        threshold: float,
        N: int) -> pd.DataFrame:
    """
    This function takes in a pivoted data table and does lorentzian fits on all peaks within the data.
    The Q factor for each peak is calculated and then the first N peaks (lower wl to higher wl) are kept.
    The data table stores: lambda1, lambda2, ..., lambdaN, Q1, Q2, ..., QN, M1, M2, ..., MN
    M represents a boolean mask that corresponds to whether a peak is present. This is included because
    it may be the case that some datasets will only have say 1 or 2 peaks and N may be something like 5

    Args:
        df (pd.DataFrame): pivoted data table from import_data
        window (int): how main points to fit on either side of the peak
        threshold (float): the amplitude above which a peak must reach to be considered
        N (int): the number of peaks to consider

    Returns:
        pd.DataFrame: datafram with columns as described above. One row for each curve.
    """
    
    wl = df.columns.to_numpy(dtype=float)
    data = df.to_numpy()
    rows = []

    for sampleIdx, row in enumerate(data):
        peaks, _ = find_peaks(row, height=threshold)

        if len(peaks) == 0:
            for i in range(N):
                rows.append({"sample_id": sampleIdx, "rank": i, "lambda": 0.0, "Q": 0.0, "mask": 0})
            continue

        pIdx = 0
        for p in peaks:

            if pIdx >= N:
                break

            left_idx = max(0, p - window)
            right_idx = min(len(row), p + window)

            fit_region = row[left_idx:right_idx]
            fit_wl = wl[left_idx:right_idx]

            A0 = row[p]
            wl0 = wl[p]
            gamma0 = (fit_wl[-1] - fit_wl[0]) / 10
            p0 = [A0, wl0, gamma0]

            try:
                popt, _ = curve_fit(
                    lorentzian,
                    fit_wl,
                    fit_region,
                    p0=p0,
                    bounds=(
                        [0, fit_wl[0], 0],
                        [np.inf, fit_wl[-1], np.inf]
                    )
                )

                A = popt[0]
                wl_res = popt[1]
                gamma = popt[2]
                Q = wl_res / (2 * gamma)

                rows.append({"sample_id": sampleIdx, "rank": pIdx, "lambda": wl_res, "Q": Q, "mask": 1})

            except RuntimeError:               # curve_fit failed to converge
                rows.append({"sample_id": sampleIdx, "rank": pIdx, "lambda": 0.0, "Q": 0.0, "mask": 0})

            pIdx += 1

        while pIdx < N:                        # pad remaining slots
            rows.append({"sample_id": sampleIdx, "rank": pIdx, "lambda": 0.0, "Q": 0.0, "mask": 0})
            pIdx += 1

    features = pd.DataFrame(rows)
    features_pivoted = features.pivot(index='sample_id', columns='rank', values=['lambda','Q','mask'])
    features_pivoted.columns = [f'{val}_{k}' for val, k in features_pivoted.columns]
    features_pivoted.index = df.index

    return features_pivoted
            