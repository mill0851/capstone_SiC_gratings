import pandas as pd
import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def import_data(root_dir: str) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:

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

    the geometries.csv is a metadata structure with 1 row per run. the data.csv
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
    bg_table = bg_table.pivot(index='run_id', columns='wavelength_um', values='reflectance_0')
    bg_table.columns = bg_table.columns.astype(float)

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
    all_data_pivot = all_data_long.pivot(index='run_id', columns='wavelength_um', values='reflectance_0')
    all_data_pivot.columns = all_data_pivot.columns.astype(float)
    wl_axis = df_list[0]['wavelength_um'].to_numpy() * 1e6


    return geom_labels, geom_table, all_data_pivot, wl_axis, bg_table

def remove_restrahlen(reflectance: pd.DataFrame, background: pd.DataFrame):

    reflectance[:] = reflectance.values - background.iloc[0].values

def reduce_domain(bounds: tuple, reflectance: pd.DataFrame, background: pd.DataFrame,):

    reflectance.sort_index(axis=1)
    background.sort_index(axis=1)

    xmin, xmax = bounds
    eps = 1e-12

    mask_refl = (reflectance.columns >= xmin - eps) & \
                (reflectance.columns <= xmax + eps)

    mask_bg = (background.columns >= xmin - eps) & \
              (background.columns <= xmax + eps)

    reflectance.drop(columns=reflectance.columns[~mask_refl],
                     inplace=True)
    
    background.drop(columns=background.columns[~mask_bg],
                    inplace=True)

    wl_reduced = reflectance.columns.to_numpy()

    return wl_reduced

def normalize_spectrum(refl: pd.DataFrame, background: pd.DataFrame):

    refl_max = refl.max(axis=1)
    refl[:] = refl.div(refl_max, axis=0)

    background_max = background.max(axis=1)
    background[:] = background.div(background_max, axis=0)

def flip_spectrum(refl: pd.DataFrame):
    refl *= -1.0

def interpolate_linear(refl: pd.DataFrame, n_samples: int):
    wl_old = refl.columns.to_numpy(dtype=float)
    num_orig = len(wl_old)

    wl_new = np.linspace(wl_old[0], wl_old[-1], (num_orig)*(n_samples+1)+1)

    interp_data = np.array([
        np.interp(wl_new, wl_old, row.values)
        for _, row in refl.iterrows()
    ])

    refl_interp = pd.DataFrame(interp_data, index=refl.index, columns=wl_new)
    return refl_interp

def lorentzian(x, A, x0, gamma):
    """Lorentzian with HWHM = gamma (no offset)"""
    return  (A / ((x - x0)**2 + gamma**2))

def extract_features_phase1(refl: pd.DataFrame, window_size: int, threshold: float):
    """
    Here lorentzian fits are calculated for all peaks above
    a specific threshold and the highest Q resonance is extracted
    as the feature for that curve. The data stored is (lambda, Q)
    if no peak above the threshold is found both entries are 0
    """
    wl_axis = refl.columns.to_numpy(dtype=float)
    data = refl.to_numpy()
    features = np.zeros((len(refl),2))

    for rowIdx, row in enumerate(data):
        peaks, _ = find_peaks(row, height=threshold)
        best_res_q = [0,0]

        if len(peaks) == 0:
            features[rowIdx,:] = best_res_q
            continue

        for pIdx, p in enumerate(peaks):
            left_idx = max(0, p - window_size)
            right_idx = min(len(row), p + window_size)

            fit_region = row[left_idx:right_idx]
            fit_wl = wl_axis[left_idx:right_idx]

            A0 = row[p]
            wl0 = wl_axis[p]
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

    features = pd.DataFrame(features, index = refl.index, columns=['lambda_res', 'Q'])
    return features

