import pandas as pd
import matplotlib.pyplot as plt
from util.data_preprocessing import *


#### DATA PREPROCESSING PIPELINE - MAX Q PEAK ####
# Config
pd.set_option('display.max_columns', 5)
DATA_PATH = './data/test_data'
DOMAIN = (10.0, 12.5)
TEST_IDX = 0
INTERP = 4
WINDOW = 20
THRESHOLD = 0.2

# Import Data
geom_labels, geom_table, refl_data, abs_data, refl_bg, abs_bg, wl = import_data(DATA_PATH)
print(f'\ngeometry labels:\n{geom_labels}\n')
print(f'geometry_table:\n{geom_table.head()}\n')
print(f'reflection data:\n{refl_data.head()}\n')
print(f'reflection background:\n{refl_bg.head()}\n')
print(f'absorption data:\n{abs_data.head()}\n')
print(f'absorption background:\n{abs_bg.head()}\n')
refl_vs_abs_plt(refl_data, abs_data, TEST_IDX)

# Normalize Spectrum
normalize(refl_bg)
normalize(refl_data)
normalize(abs_bg)
normalize(abs_data)
refl_vs_abs_plt(refl_data, abs_data, TEST_IDX)

# Remove Background signal
remove_background(refl_data, refl_bg)
remove_background(abs_data, abs_bg)
refl_vs_abs_plt(refl_data, abs_data, TEST_IDX)

# Reduce Domain
reduce_domain(DOMAIN, refl_data)
reduce_domain(DOMAIN, abs_data)
refl_vs_abs_plt(refl_data, abs_data, TEST_IDX)

# Flip reflectance data for feature extraction
refl_data *= -1.0
print(f'Pre Interpolation Length: {len(refl_data.iloc[0])}\n')

# Interpolate (Linear)
refl_data = interp_linear(refl_data, INTERP)
abs_data = interp_linear(abs_data, INTERP)
wl = refl_data.columns.to_numpy(dtype=float)
print(f'Post Interpolation Length: {len(refl_data.iloc[0])}\n')
refl_vs_abs_plt(refl_data, abs_data, TEST_IDX)

# Extract Feature Table
abs_ft_single = highest_Q(
    abs_data,
    WINDOW,
    THRESHOLD
)
refl_ft_single = highest_Q(
    refl_data,
    WINDOW,
    THRESHOLD
)
print(f'Feature table (abs):\n{abs_ft_single}\n')
print(f'Feature table (refl):\n{refl_ft_single}\n')\

# Create export dict
max_Q_ft = {
    "wl": wl,
    "reflection": refl_data,
    "absorption": abs_data,
    "reflection_features": refl_ft_single,
    "absorption_features": abs_ft_single,
    "geometry_labels": geom_labels,
    "geometry_table": geom_table
}

data_config = {
    "data_path": DATA_PATH,
    "domain": DOMAIN,
    "interpolation_density": INTERP,
    "window_size": WINDOW,
    "threshold": THRESHOLD
}

