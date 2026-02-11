import torch
from torch.utils.data import Dataset
from typing import List
import numpy as np

class ComsolDataset(Dataset):
 """
 This is a class for holding the COMSOL data I will use for my ML model.
 It will hold the original data in numpy arrays for plotting and exploration,
 as well as a torch.tensor() of each dataset for ML purposes
 """
 def __init__(self, X_meta_np: np.ndarray, X_data_np: List[np.ndarray], data_ids: List[str], run_ids=None):
    self.X_meta_np = X_meta_np
    self.X_meta = torch.tensor(X_meta_np, dtype=torch.float32)

    self.X_data_np = X_data_np
    self.X_data = []
    for data in X_data_np:
       self.X_data.append(torch.tensor(data, dtype=torch.float32))

    self.run_ids = run_ids
    self.data_ids = data_ids

def __len__(self):
   return self.X_geom.shape[0]

def __getitem__(self, idx):
   return self.X_geom[idx], self.X_spec[idx]

def get_numpy(self, idx):
   return self.X_geom_np[idx], self.X_spec_np[idx]