from util.data_preprocessing import *
from util.Phase1Dataset import Phase1Dataset

# ---- Import ----
DATA_PATH = './data/batch2'
WL_DOMAIN = (11.25,12.5) # um
UPSAMPLE_RATE = 4
PEAK_THRESHOLD = 0.2
WINDOW_SAMPLES = 20

geom_labels, geom_values, refl, wl, bg = import_data(DATA_PATH)

wl = reduce_domain(WL_DOMAIN, refl, bg)
normalize_spectrum(refl, bg)
remove_restrahlen(refl, bg)
flip_spectrum(refl)
refl = interpolate_linear(refl, UPSAMPLE_RATE)
wl = refl.columns.astype(float).to_numpy()
features = extract_features_phase1(refl, WINDOW_SAMPLES, PEAK_THRESHOLD)

# Debug
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

# for i in range(0,5):
#     plt.plot(wl, refl.iloc[i,:], label = f'curve {i+1}')
# plt.xlabel('Wavelength [um]', fontsize=18)
# plt.ylabel('Absorption [a.u.]', fontsize=18)
# plt.title('Normalized Absorption - Background Removes', fontsize=20)
# plt.legend(fontsize=16)
# plt.grid(alpha=0.8)
# plt.show()

# for i in range(0,5):
#     plt.plot(refl.columns.to_numpy(), refl.iloc[i,:], label = f'curve {i+1}')
# plt.xlabel('Wavelength [um]', fontsize=18)
# plt.ylabel('Absorption [a.u.]', fontsize=18)
# plt.title('Normalized Absorption - Background Removes', fontsize=20)
# plt.legend(fontsize=16)
# plt.grid(alpha=0.8)
# plt.show()

dataset = Phase1Dataset(
    geom_df = geom_values,
    feat_df = features,
    normalize_geom = True,
    normalize_feat = True
)
train_loader, val_loader = create_dataloaders(dataset, train_ratio=0.8, batch_size=32)