import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from .classes.Phase1Dataset import Phase1Dataset

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

def create_dataloaders(
        dataset: Phase1Dataset,
        split_path: str,
        new_split: bool = False,
        train_ratio: float = 0.8,
        batch_size: int = 32,
        indices: dict = None):
    
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    
    # Case 1: Indices provided
    if indices:
        assert len(indices) == 3, "Must have indices for: train, val, test"
        train_indices = indices["train_indices"]
        val_indices = indices["val_indices"]
        test_indices = indices["test_indices"]

    # Case 2: Split already exists, load it
    elif os.path.exists(split_path) and not new_split:
        split = torch.load(split_path)
        train_indices = split["train_indices"]
        val_indices = split["val_indices"]
        test_indices = split['test_indices']
        print("Loaded existing data split.")

    # Case 3: Create new split and save it
    else:
        seed = np.random.randint(1, 1000)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_total, generator=generator)

        trainIdx = n_train + (len(dataset) - n_train)//2

        train_indices = indices[:n_train]
        val_indices = indices[n_train:trainIdx]
        test_indices = indices[trainIdx:]

        torch.save({
            "train_indices": train_indices,
            "val_indices": val_indices,
            'test_indices': test_indices
        }, split_path)

        print("Created and saved new data split.")

    # Build subsets + loaders
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader

def load_dataset(
        path: str,
        domain: tuple,
        upsample_rate: int,
        peak_threshold: float,
        window_samples: int,
        normalize_geom: bool = True,
        normalize_feat: bool = True,
        verbose: bool = True
) -> Phase1Dataset:
    
    """
    This function loads the Phase1Dataset object for this work. This
    class inhereits the pytorch Dataset class but has some features
    relevant to the data I will be dealing with.

    Returns:
        Phas1Dataset: Dataset object for model training and evaluation
    """

    geom_labels, geom_values, refl, wl, bg = import_data(path)
    wl = reduce_domain(domain, refl, bg)
    normalize_spectrum(refl, bg)
    remove_restrahlen(refl, bg)
    flip_spectrum(refl)
    refl = interpolate_linear(refl, upsample_rate)
    wl = refl.columns.astype(float).to_numpy()
    features = extract_features_phase1(refl, window_samples, peak_threshold)

    if verbose:
        print(f'\n geometry labels [names]: \n{geom_labels}')
        print(f'geometry labels type: {type(geom_labels)} \n')
        print(f'geometry table [um]: \n{geom_values.head()}')
        print(f'geometry table type: {type(geom_values)}\n')
        print(f'data table [% reflectance]: \n{refl.head()}')
        print(f'data table type: {type(refl)}\n')
        print(f'background table [% reflectance]: \n{bg.head()}')
        print(f'background table type: {type(bg)}\n')
        print(f'wavelength axis [um]: \n{wl[0:50]} \n {np.shape(wl)}')
        print(f'wavelength axis type: {type(wl)}\n')
        print(f'feature table:\n{features.head()}')
        print(f'feature table type: {type(features)}\n')

        for i in range(0,5):
            plt.plot(wl, refl.iloc[i,:], label = f'curve {i+1}')
        plt.xlabel('Wavelength [um]', fontsize=18)
        plt.ylabel('Absorption [a.u.]', fontsize=18)
        plt.title('Normalized Absorption - Background Removes', fontsize=20)
        plt.legend(fontsize=16)
        plt.grid(alpha=0.8)
        plt.show()

    dataset = Phase1Dataset(
        geom_df = geom_values,
        feat_df = features,
        normalize_geom = normalize_geom,
        normalize_feat = normalize_feat
    )

    return dataset
