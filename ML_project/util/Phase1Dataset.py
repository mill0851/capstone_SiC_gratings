import torch
from torch.utils.data import Dataset
from typing import List
import numpy as np
import pandas as pd


class Phase1Dataset():

    def __init__(self,
                 background: pd.DataFrame,
                 wl_axis: np.ndarray,  spectrum_raw: torch.Tensor,
                 geom: torch.Tensor, geom_labels: np.ndarray,
                 feat: torch.Tensor, feat_labels: np.ndarray):
        
        self.wl_axis = wl_axis
        self.spectrum_raw = spectrum_raw
        self.background_raw = background
        self.spectrum_processed = 

        self.geom_labels = geom_labels
        self.feat_labels = feat_labels

        self.geom_units = ['um', 'um', 'um', 'um']
        self.feat_units = ['um', 'dimensionless']

        self.geom_raw = geom.clone()
        self.feat_raw = feat.clone()

        self.geom_norm = None
        self.feat_norm = None

        self.geom_scaler = None
        self.feat_scaler = None

