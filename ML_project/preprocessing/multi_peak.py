import pandas as pd
import matplotlib.pyplot as plt
from util.data_preprocessing import *


#### DATA PREPROCESSING PIPELINE - MULTI PEAK ####
# Config
pd.set_option('display.max_columns', 10)
DATA_PATH = './data/batch2'
DOMAIN = (10.25, 11.0)
TEST_IDX = np.arange(0, 1024, 512)
INTERP = 4
WINDOW = 10
THRESHOLD = 0.2
N_PEAKS = 4

# Import Data
geom_labels, geom_table, refl_data, abs_data, refl_bg, abs_bg, wl = import_data(DATA_PATH)
print(f'\ngeometry labels:\n{geom_labels}\n')
print(f'geometry_table:\n{geom_table.head()}\n')
print(f'reflection data:\n{refl_data.head()}\n')
print(f'reflection background:\n{refl_bg.head()}\n')
print(f'absorption data:\n{abs_data.head()}\n')
print(f'absorption background:\n{abs_bg.head()}\n')
# for idx in TEST_IDX:
#     refl_vs_abs_plt(refl_data, abs_data, int(idx))

# Normalize Spectrum
# normalize(refl_bg)
normalize(abs_bg)
# normalize(refl_data)
normalize(abs_data)
# refl_vs_abs_plt(refl_data, abs_data, TEST_IDX)

# Remove Background signal
# remove_background(refl_data, refl_bg)
remove_background(abs_data, abs_bg)
# refl_vs_abs_plt(refl_data, abs_data, TEST_IDX)

# Reduce Domain
# reduce_domain(DOMAIN, refl_data)
reduce_domain(DOMAIN, abs_data)
# refl_vs_abs_plt(refl_data, abs_data, TEST_IDX)

# Flip reflectance data for feature extraction
# refl_data *= -1.0
print(f'Pre Interpolation Length: {len(abs_data.iloc[0])}\n')

# Interpolate (Linear)
# refl_data = interp_linear(refl_data, INTERP)
abs_data = interp_linear(abs_data, INTERP)
wl = abs_data.columns.to_numpy(dtype=float)
print(f'Post Interpolation Length: {len(abs_data.iloc[0])}\n')
# refl_vs_abs_plt(refl_data, abs_data, TEST_IDX)

# Extract Feature Table
abs_ft_multi = multi_peak_extraction(
    abs_data,
    WINDOW,
    THRESHOLD,
    N_PEAKS,
    TEST_IDX
)
print(f'Feature table (abs):\n{abs_ft_multi}\n')

# Create export dict
multi_peak_ft = {
    "wl": wl,
    "absorption": abs_data,
    "absorption_features": abs_ft_multi,
    "geometry_labels": geom_labels,
    "geometry_table": geom_table,
    "peak_count": N_PEAKS
}

# Save to pickle
import pickle
with open('multi_peak_ft.pkl', 'wb') as f:
    pickle.dump(multi_peak_ft, f)
print("Saved multi_peak_ft.pkl")
