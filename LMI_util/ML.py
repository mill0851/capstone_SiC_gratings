import pandas as pd
import os
from typing import List
import numpy as np
from ComsolDataset import ComsolDataset
import matplotlib.pyplot as plt
from scipy.stats import qmc

def import_comsol_data(root_dir: str, meta_data_filename: str, data_filenames: List[str], data_column_values: dict) -> ComsolDataset:
    """
    This function is used to import comsol simulation data. It works with
    the specific directory structure:

    batch/
    |-- geometries.csv
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
    meta_path = os.path.join(root_dir, meta_data_filename)
    meta_table = pd.read_csv(meta_path)
    meta_table["run_id"] = meta_table["run_id"].astype(str)
    meta_table = meta_table.set_index("run_id").sort_index()
    meta_labels = meta_table.columns.to_numpy()

    # Read in target data
    x_axes = []
    axes_ids = []
    data_dfs = []
    data_list_lengths = []
    runs_path = os.path.join(root_dir, "runs")

    for filename in data_filenames:
        data_list = []

        for item in os.listdir(runs_path):
            item_path = os.path.join(runs_path, item)

            if os.path.isdir(item_path):
                data_path = os.path.join(item_path, filename)
                data_table = pd.read_csv(data_path)
                data_table["run_id"] = data_table["run_id"].astype(str)
                data_list.append(data_table)

        data_list_lengths.append(len(data_list))
        data_df = pd.concat(data_list, ignore_index=True)
        data_dfs.append(data_df)

    # pivot data to one run per row
    for i, key in enumerate(data_column_values):
        x_axis = data_dfs[i][key][0: (len(data_dfs[i][key]) // data_list_lengths[i])]
        x_axes.append(x_axis.to_numpy())
        axes_ids.append(key)
        data_pivot = data_dfs[i].pivot(index="run_id", columns=key, values=data_column_values[key])
        data_pivot = data_pivot.sort_index()
        data_dfs[i] = data_pivot


    # Validate that the meta data and data is aligned
    for data in data_dfs:
       assert (data.index == meta_table.index).all()

    # Convert to numpy
    X_meta = meta_table.to_numpy(dtype=np.float32)
    X_data = []
    for data in data_dfs:
       X_data.append(data.to_numpy(dtype=np.float32))
    
    # Create dataset object
    data_final = ComsolDataset(X_meta, meta_labels,
                               X_data, data_filenames,
                               x_axes, axes_ids)
    return data_final

def generate_sample_space(tlw_bounds: List[float],
                          blw_bounds: List[float],
                          s_bounds: List[float],
                          h_bounds: List[float],
                          n_samples: int,
                          path: str = None):
    
    assert len(tlw_bounds) == 2 and tlw_bounds[0] < tlw_bounds[1], "tlw bounds must be two ordered numbers"
    assert len(blw_bounds) == 2 and blw_bounds[0] < blw_bounds[1], "blw bounds must be two ordered numbers"
    assert len(s_bounds) == 2 and s_bounds[0] < s_bounds[1], "s bounds must be two ordered numbers"
    assert len(h_bounds) == 2 and h_bounds[0] < h_bounds[1], "h bounds must be two ordered numbers"
    assert isinstance(n_samples, int), "n_samples must be an integer and a power of 2"

    # Here the sobol sampler object is created and used to generate a (500,4)
    # unit sample matrix. this is then processed below.
    sampler = qmc.Sobol(d=4, scramble=True)
    unit_samples = sampler.random(n=n_samples)

    # Here I want to enforce the criteria that the blw >= tlw so i sort
    # the first two columns, all smaller values in first column.
    sorted_pair = np.sort(unit_samples[:, :2], axis=1)
    unit_samples[:,0] = sorted_pair[:,0] # TLW
    unit_samples[:,1] = sorted_pair[:,1] # BLW

    # we now need to scale the unit samples to the ranges provided
    # as funciton arguments
    bounds = np.array([
        tlw_bounds, blw_bounds, h_bounds, s_bounds
    ])
    scaled_samples = np.round(qmc.scale(unit_samples,
                               bounds[:,0],
                               bounds[:,1]), 4)
    
    # Verify constraints
    TLW = scaled_samples[:,0]
    BLW = scaled_samples[:,1]
    print("Constraint satisfied:", np.all(BLW >= TLW))
    print("Min values:", scaled_samples.min(axis=0))
    print("Max values:", scaled_samples.max(axis=0))
    print("\nFirst 5 samples:")
    print(scaled_samples[:5])

    # save to file if needed
    if path:
        scaled_samples_df = pd.DataFrame(scaled_samples)
        scaled_samples_df.to_csv(path, header=['tlw','blw','h','s'], index=False)

    return scaled_samples




if __name__ == "__main__":
    root_dir = r"C:\Users\robert\Code\capstone\capstone_SiC_gratings\data\ML\blw_sweep_2"
    data = import_comsol_data(root_dir, "metadata.csv", ["data.csv"], {"wavelength_um":"reflectance_0"})

    data_np = data.X_data_np
    meta_np = data.X_meta_np
    data_axes = data.X_data_axis

    meta_labels = data.X_meta_labels
    data_ids = data.data_ids
    axes_ids = data.axes_ids

    for i in range(0, len(data_ids)):
        print(f"Meta Data Labels: {meta_labels}")
        print(f"Meta Data: \n {meta_np}")
        print(f"Data ID: {data_ids[i]}")
        print(f"X axis ID: {axes_ids[i]}")
        print(f"Data Table: \n {data_np[i]}")
        print(f"Data axis: \n {data_axes[i]}")

    plt.figure(figsize=(10,6))
    for i in range(0, len(data_np[0])):
        plt.plot(data_axes[0]*1e6, data_np[0][i]/np.max(data_np[0][i]), label=f"run {i+1}")
    plt.xlabel("Wavelength (um)", fontsize=18)
    plt.ylabel("Normalized Reflectance (%)", fontsize=18)
    plt.title(f"BLW Sweep [2.5-5.0 (um)]", fontsize=20)
    plt.grid(alpha=0.8)
    plt.legend(fontsize=18)
    plt.show()

    # # ---- BOUNDS/SAMPLE SPACE TESTING ----
    # tlw_bounds = [0.1, 5.0]     # um
    # blw_bounds = [0.5, 5.0]     # um
    # s_bounds = [2.0, 10.0]      # um
    # h_bounds = [0.25, 2.5]      # um
    # N = 1024

    # sample_grid = generate_sample_space(tlw_bounds,
    #                                     blw_bounds,
    #                                     s_bounds,
    #                                     h_bounds,
    #                                     N,
    #                                     path="batch1_geometries_1024")