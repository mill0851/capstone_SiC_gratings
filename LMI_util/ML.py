import pandas as pd
import torch
from torch.utils.data import Dataset

def import_comsol_data(root_dir: str, meta_data_filename: str):
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
    return 0

class SpectraDataset(Dataset):
 def __init__():
    return 0