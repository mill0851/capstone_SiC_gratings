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
        np.interp(wl_new, wl_old, np.asarray(row.values,dtype=float))
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
        threshold: float,
        test_idx: np.ndarray | None = None) -> pd.DataFrame:
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
        fit_curves = []
        fit_wl_arr = []
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
            fit_wl_arr.append(fit_wl)

            A0 = row[p]
            wl0 = wl[p]
            gamma0 = (fit_wl[-1] - fit_wl[0]) / 10
            p0 = [A0, wl0, gamma0]

            try:
                popt, pvoc = curve_fit(
                    lorentzian,
                    fit_wl,
                    fit_region,
                    p0=p0,
                    bounds=(
                        [0, fit_wl[0], 0],
                        [np.inf, fit_wl[-1], np.inf]
                    ),
                    maxfev=10000
                )
            except RuntimeError:
                plt.plot(wl, row, 'o-')
                plt.title("Failed fit")
                plt.show()
                continue

            A = popt[0]
            wl_res = popt[1]
            gamma = popt[2]
            Q = wl_res / (2*gamma)

            fit_curve = lorentzian(fit_wl, *popt)
            fit_curves.append(fit_curve)

            if Q > best_res_q[1]:
                best_res_q[0] = wl_res
                best_res_q[1] = Q

        if test_idx is not None and rowIdx in test_idx:
            plt.figure(figsize=(10,6))
            plt.plot(wl, row, label="Data", color='k')
            for fitIdx, _ in enumerate(fit_curves):
                plt.plot(fit_wl_arr[fitIdx],
                         fit_curves[fitIdx],
                         label=f'Lorentzian fit {fitIdx}')
            plt.xlabel("Wavelength (um)", fontsize=16)
            plt.ylabel("a.u. Absorption/Reflection Proxy", fontsize=16)
            plt.title(f"Cruve {rowIdx} With Fits", fontsize=18)
            plt.grid(alpha=0.8)
            plt.legend(fontsize=16)
            plt.show()

        features[rowIdx,:] = best_res_q

    features = pd.DataFrame(features, index = df.index, columns=['lambda_res', 'Q'])
    return features

def multi_peak_extraction(
        df: pd.DataFrame,
        window: int,
        threshold: float,
        N: int,
        test_idx: np.ndarray | None = None) -> pd.DataFrame:
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
        test_idx (np.ndarray, optional): sample indices to plot with their Lorentzian fits

    Returns:
        pd.DataFrame: datafram with columns as described above. One row for each curve.
    """

    wl = df.columns.to_numpy(dtype=float)
    data = df.to_numpy()
    rows = []

    for sampleIdx, row in enumerate(data):
        peaks, _ = find_peaks(row, height=threshold)
        fit_curves = []
        fit_wl_arr = []

        if len(peaks) == 0:
            for i in range(N):
                rows.append({"sample_id": sampleIdx, "rank": i, "lambda": 0.0, "Q": 0.0, "mask": 0})

            if test_idx is not None and sampleIdx in test_idx:
                plt.figure(figsize=(10, 6))
                plt.plot(wl, row, label="Data", color='k')
                plt.xlabel("Wavelength (um)", fontsize=16)
                plt.ylabel("a.u. Absorption/Reflection Proxy", fontsize=16)
                plt.title(f"Curve {sampleIdx} (no peaks found)", fontsize=18)
                plt.grid(alpha=0.8)
                plt.legend(fontsize=16)
                plt.show()
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

                fit_curves.append(lorentzian(fit_wl, *popt))
                fit_wl_arr.append(fit_wl)

            except RuntimeError:               # curve_fit failed to converge
                rows.append({"sample_id": sampleIdx, "rank": pIdx, "lambda": 0.0, "Q": 0.0, "mask": 0})

            pIdx += 1

        while pIdx < N:                        # pad remaining slots
            rows.append({"sample_id": sampleIdx, "rank": pIdx, "lambda": 0.0, "Q": 0.0, "mask": 0})
            pIdx += 1

        if test_idx is not None and sampleIdx in test_idx:
            plt.figure(figsize=(10, 6))
            plt.plot(wl, row, label="Data", color='k')
            for fitIdx, fit_curve in enumerate(fit_curves):
                plt.plot(fit_wl_arr[fitIdx],
                         fit_curve,
                         label=f'Lorentzian fit {fitIdx}')
            plt.xlabel("Wavelength (um)", fontsize=16)
            plt.ylabel("a.u. Absorption/Reflection Proxy", fontsize=16)
            plt.title(f"Curve {sampleIdx} With Fits", fontsize=18)
            plt.grid(alpha=0.8)
            plt.legend(fontsize=16)
            plt.show()

    features = pd.DataFrame(rows)
    features_pivoted = features.pivot(index='sample_id', columns='rank', values=['lambda','Q','mask'])
    features_pivoted.columns = [f'{val}_{k}' for val, k in features_pivoted.columns]
    features_pivoted.index = df.index

    return features_pivoted

def extract_pca(
        df: pd.DataFrame,
        K: int,
        artifacts: dict | None = None,
        test_idx: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Extract PCA coefficients from a pivoted spectrum table. The feature per sample
    is a K-dimensional vector of principal-component scores. Unlike peak extraction,
    reconstruction requires the mean and components, so the fitted artifacts are
    returned alongside the feature table.

    If `artifacts` is None, PCA is fit on `df` (fit + transform). If `artifacts` is
    provided (mean, components), `df` is only projected onto those components. This
    supports a train-fit / val-transform split when needed.

    Args:
        df (pd.DataFrame): pivoted data table (rows=samples, cols=wavelengths)
        K (int): number of principal components to keep
        artifacts (dict, optional): prior fit output to reuse (keys: 'mean',
            'components_full'). If None, a fresh PCA is fit on df.
        test_idx (np.ndarray, optional): sample indices to plot with their
            K-component reconstruction overlaid on truth.

    Returns:
        tuple[pd.DataFrame, dict]:
            - features: (N, K) DataFrame, columns 'pc_0'...'pc_{K-1}'
            - artifacts: dict with keys 'mean', 'components' (top-K),
              'components_full', 'singular_values', 'explained_variance',
              'explained_variance_ratio', 'wl', 'K'
    """
    wl = df.columns.to_numpy(dtype=float)
    X = df.to_numpy()

    if artifacts is None:
        mean = X.mean(axis=0)
        Xc = X - mean
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        components_full = Vt
        singular_values = S
        ev = (S ** 2) / max(X.shape[0] - 1, 1)
        ev_ratio = ev / ev.sum()
    else:
        mean = artifacts['mean']
        components_full = artifacts['components_full']
        singular_values = artifacts.get('singular_values')
        ev = artifacts.get('explained_variance')
        ev_ratio = artifacts.get('explained_variance_ratio')
        Xc = X - mean

    components = components_full[:K]
    coeffs = Xc @ components.T

    features = pd.DataFrame(
        coeffs,
        index=df.index,
        columns=[f'pc_{i}' for i in range(K)]
    )

    new_artifacts = {
        'mean': mean,
        'components': components,
        'components_full': components_full,
        'singular_values': singular_values,
        'explained_variance': ev,
        'explained_variance_ratio': ev_ratio,
        'wl': wl,
        'K': K,
    }

    if test_idx is not None:
        for i in test_idx:
            if i >= len(X):
                continue
            recon = mean + coeffs[i] @ components
            plt.figure(figsize=(10, 6))
            plt.plot(wl, X[i], label='Data', color='k')
            plt.plot(wl, recon, label=f'PCA reconstruction (K={K})',
                     color='r', linestyle='--')
            plt.xlabel("Wavelength (um)", fontsize=16)
            plt.ylabel("a.u. Absorption/Reflection Proxy", fontsize=16)
            plt.title(f"Curve {i} PCA Reconstruction", fontsize=18)
            plt.grid(alpha=0.8)
            plt.legend(fontsize=16)
            plt.show()

    return features, new_artifacts

def pca_reconstruct(
        coeffs: np.ndarray,
        artifacts: dict) -> np.ndarray:
    """
    Reconstruct spectra from PCA coefficients using the fitted artifacts.
    Accepts a (K,) or (N, K) coefficient array; returns (L,) or (N, L).
    """
    mean = artifacts['mean']
    components = artifacts['components']
    coeffs = np.asarray(coeffs)
    return mean + coeffs @ components

def derive_geometry_features(
        geom_df: pd.DataFrame) -> pd.DataFrame:
    """
    Augments a 4D geometry table (tlw, blw, h, s) with 4 physically motivated
    derived features, returning an 8D table.

    Derived features:
        width_sum    = tlw + blw            (total ridge width; fill-factor proxy)
        width_diff   = tlw - blw            (taper sign and magnitude)
        aspect_ratio = h / s               (depth-to-period ratio; coupling strength)
        fill_factor  = (tlw + blw) / (2*s) (fraction of period occupied by material)

    Args:
        geom_df (pd.DataFrame): indexed by run_id, columns ['tlw', 'blw', 'h', 's']

    Returns:
        pd.DataFrame: same index, 8 columns (original 4 + 4 derived)
    """
    df = geom_df.copy()
    df['width_sum']    = df['tlw'] + df['blw']
    df['width_diff']   = df['tlw'] - df['blw']
    df['aspect_ratio'] = df['h'] / (df['s'] + 1e-8)
    df['fill_factor']  = (df['tlw'] + df['blw']) / (2.0 * df['s'] + 1e-8)
    return df

def normalize_geom(
        geom_df: pd.DataFrame) -> np.ndarray:
    geom_np = geom_df.values.astype(np.float32)
    geom_mean = geom_np.mean(axis=0)
    geom_std = geom_np.std(axis=0)
    geom_np = (geom_np - geom_mean) / geom_std
    return geom_np
