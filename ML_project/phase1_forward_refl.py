from util.data_eval import*
from util.Phase1Dataset import *
from util.data_preprocessing import *
from util.MLPModel import *


# ---- PREPROCESSING ----
DATA_PATH = './data/ML_proj_refl_data'
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

# # Debug
# print(f'\n geometry labels [names]: \n{geom_labels}')
# print(f'geometry labels type: {type(geom_labels)} \n')
# print(f'geometry table [um]: \n{geom_values.head()}')
# print(f'geometry table type: {type(geom_values)}\n')
# print(f'data table [% reflectance]: \n{refl.head()}')
# print(f'data table type: {type(refl)}\n')
# print(f'background table [% reflectance]: \n{bg.head()}')
# print(f'background table type: {type(bg)}\n')
# print(f'wavelength axis [um]: \n{wl[0:50]} \n {np.shape(wl)}')
# print(f'wavelength axis type: {type(wl)}\n')
# print(f'feature table:\n{features.head()}')
# print(f'feature table type: {type(features)}\n')

# for i in range(0,5):
#     plt.plot(wl, refl.iloc[i,:], label = f'curve {i+1}')
# plt.xlabel('Wavelength [um]', fontsize=18)
# plt.ylabel('Absorption [a.u.]', fontsize=18)
# plt.title('Normalized Absorption - Background Removes', fontsize=20)
# plt.legend(fontsize=16)
# plt.grid(alpha=0.8)
# plt.show()


# ---- CONFIG ----
TRAIN_SPLIT = 0.75
BATCH_SIZE = 128
HIDDEN_DIM = 128
EPOCHS = 250
ALPHA = 4.0
LR = 1e-3
WD = 1e-4
TRAIN = True
NEW_SPLIT = False
NAME = "forward_refl_split-75-15-15"
PATH = f'C:/Users/robert/Code/capstone/capstone_SiC_gratings/ML_project/models_phase1/{NAME}'
os.makedirs(PATH, exist_ok=True)


# ---- SETUP ----
# Load in dataset
dataset = Phase1Dataset(
    geom_df = geom_values,
    feat_df = features,
    normalize_geom = True,
    normalize_feat = True
)

# Load in data loaders
train_loader, val_loader, test_loader = create_dataloaders(
    dataset,
    f"{PATH}/data_split.pt",
    new_split=NEW_SPLIT,
    train_ratio=TRAIN_SPLIT,
    batch_size=BATCH_SIZE
)


# ---- TRAINING ----
if TRAIN:
    model = MLPModel(hidden_dim=HIDDEN_DIM)
    train_mlp(
        model,
        train_loader,
        val_loader,
        PATH,
        epochs=EPOCHS,
        alpha=ALPHA,
        lr=LR,
        wd=WD
    )


# ---- EVALUATION ----
# Load Final Model
final_model = MLPModel(hidden_dim=HIDDEN_DIM)
final_checkpoint = torch.load(f'{PATH}/final_model.pt', map_location="cpu")
final_model.load_state_dict(final_checkpoint["model_state_dict"])
final_history = final_checkpoint["history"]
final_model.eval()

# Load Regression Model
reg_model = MLPModel(hidden_dim=HIDDEN_DIM)
reg_checkpoint = torch.load(f'{PATH}/best_reg.pt', map_location="cpu")
reg_model.load_state_dict(reg_checkpoint["model_state_dict"])
reg_history = reg_checkpoint["history"]
reg_model.eval()

# Load Classification Model
cls_model = MLPModel(hidden_dim=HIDDEN_DIM)
cls_checkpoint = torch.load(f'{PATH}/best_cls.pt', map_location="cpu")
cls_model.load_state_dict(cls_checkpoint["model_state_dict"])
cls_history = cls_checkpoint["history"]
cls_model.eval()

# Plot eval and training losses per epoch
plot_losses(final_history)
plot_losses(reg_history)
plot_losses(cls_history)

# Test regression performance with the test set
plot_lambda_predictions(reg_model, test_loader, dataset)
plot_Q_predictions(reg_model, test_loader)
compute_r2(reg_model, test_loader)

# Test classification performance with the test set
plot_confusion_matrix(cls_model, test_loader)
classification_metrics(cls_model, test_loader)