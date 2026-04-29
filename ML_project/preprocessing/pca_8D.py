import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from util.data_preprocessing import *
from util.pca_plotting import *
from preprocessing.pipelines import build_pca_pipeline


#### DATA PREPROCESSING PIPELINE - PCA FEATURES (8D GEOMETRY) ####
# Config
pd.set_option('display.max_columns', 10)
DATA_PATH = './data/batch2'
DOMAIN = (10.25, 11.0)
INTERP = 4
K = 26

# Import Data
geom_labels, geom_table, refl_data, abs_data, refl_bg, abs_bg, wl = import_data(DATA_PATH)

# Augment geometry table from 4D -> 8D
geom_table_8d = derive_geometry_features(geom_table)
geom_labels_8d = geom_table_8d.columns.to_numpy()

# Build and fit preprocessing pipelines
abs_pipe  = build_pca_pipeline(K, DOMAIN, INTERP, background=abs_bg)
abs_ft_pca  = abs_pipe.fit_transform(abs_data)
abs_pca  = abs_pipe.named_steps['pca'].artifacts_

# Processed spectra (all steps except PCA) — used by exploration plots below
abs_data  = abs_pipe[:-1].fit_transform(abs_data)
wl = abs_data.columns.to_numpy(dtype=float)

# Print for feature and data inspection
# print(f'Feature table (abs):\n{abs_ft_pca.head()}\n')
# print(f'\ngeometry labels (4D):\n{geom_labels}\n')
# print(f'geometry_table (4D):\n{geom_table.head()}\n')
# print(f'absorption data:\n{abs_data.head()}\n')
# print(f'absorption background:\n{abs_bg.head()}\n')

#### DATA EXPORT ####
# Create export dict (same format as pca_4D.py; geometry_table is now 8D)
pca_ft = {
    "wl": wl,
    "absorption": abs_data,
    "absorption_features": abs_ft_pca,
    "absorption_pca": abs_pca,
    "geometry_labels": geom_labels_8d,
    "geometry_table": geom_table_8d,
    "K": K,
}

data_config = {
    "data_path": DATA_PATH,
    "domain": DOMAIN,
    "interpolation_density": INTERP,
    "K": K,
    "geometry_dim": 8,
    "derived_features": ["width_sum", "width_diff", "aspect_ratio", "fill_factor"],
}

# Re-export under a distinct name so the optimization notebook can import the
# 4D and 8D feature dicts side-by-side without one shadowing the other.
pca_ft_8D = pca_ft
data_config_8D = data_config