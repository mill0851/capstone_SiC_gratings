import pandas as pd
import os
from typing import List
import numpy as np
from ComsolDataset import ComsolDataset

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
    print(f"meta table: {meta_table}")

    # Read in target data
    data_dfs = []
    runs_path = os.path.join(root_dir, "runs")
    print(f"runs path: {runs_path}")

    for filename in data_filenames:
        data_list = []

        for item in os.listdir(runs_path):
            item_path = os.path.join(runs_path, item)

            if os.path.isdir(item_path):
                data_path = os.path.join(item_path, filename)
                print(f"data path: {data_path}")
                data_table = pd.read_csv(data_path)
                data_table["run_id"] = data_table["run_id"].astype(str)
                data_list.append(data_table)

        data_df = pd.concat(data_list, ignore_index=True)
        print(f"dataframe: {data_df}")
        data_dfs.append(data_df)

    # pivot data to one run per row
    for i, key in enumerate(data_column_values):
       data_pivot = data_dfs[i].pivot(index="run_id", columns=key, values=data_column_values[key])
       data_pivot = data_pivot.sort_index()
       data_dfs[i] = data_pivot
       print(f"pivot df: {data_pivot}")

    # Validate that the meta data and data is aligned
    for data in data_dfs:
       assert (data.index == meta_table.index).all()

    # Convert to numpy
    X_meta = meta_table.to_numpy(dtype=np.float32)
    X_data = []
    for data in data_dfs:
       X_data.append(data.to_numpy(dtype=np.float32))
    
    # Create dataset object
    data_final = ComsolDataset(X_meta, X_data, data_filenames)
    return data_final

if __name__ == "__main__":
    root_dir = r"C:\Users\robert\Code\capstone\capstone_SiC_gratings\data\ML\test_batch"
    data = import_comsol_data(root_dir, "geometries.csv", ["reflectance.csv"], {"wavelength_um":"reflectance_0"})
    