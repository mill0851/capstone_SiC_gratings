from util.model_eval import*
from util.classes.Phase1Dataset import *
from util.data_preprocessing import *
from util.classes.MLPModel import *
from util.model_optimization import *

# ---- IMPORT SETTINGS ----
DATA_PATH = './data/ML_proj_refl_data'
MODELS_PATH = 'C:/Users/robert/Code/capstone/capstone_SiC_gratings/ML_project/p1_models/kfold_models_1'
WL_DOMAIN = (11.25,12.5) # um
UPSAMPLE_RATE = 4
PEAK_THRESHOLD = 0.2
WINDOW_SAMPLES = 20
os.makedirs(MODELS_PATH, exist_ok=True)

# ---- MODEL/TRAINING SETTINGS ----
N_SPLITS = 4
SHUFFLE = True
SEED = 42
TEST_RATIO = 0.15
EPOCHS = 250
hidden_dim = [32, 64, 128, 256]
n_layers = [2, 3, 4, 5]
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
batch_sizes = [16, 32, 64, 128]
weight_decays = [0.0, 1e-4, 1e-3, 1e-2]
alphas = [1.0, 2.0, 3.0, 4.0]
hyper_grid = [
    {'hidden_dim': h, 'n_layers': l, 'lr': lr, 'batch_size': bs, 'wd': wd, 'alpha': a}
    for h in hidden_dim
    for l in n_layers
    for lr in learning_rates
    for bs in batch_sizes
    for wd in weight_decays
    for a in alphas
]

# ---- SETUP ----
dataset = load_dataset(
    DATA_PATH,
    WL_DOMAIN,
    UPSAMPLE_RATE,
    PEAK_THRESHOLD,
    WINDOW_SAMPLES,
    verbose=True
)

# Note here that kf_indices is an iterable
# of n tuples (train_indices, val_indices)
kf_indices, test_indices = generate_kfold(
    dataset,
    n_splits = N_SPLITS,
    shuffle = SHUFFLE,
    random_state = SEED,
    test_ratio = TEST_RATIO
)


# ---- TRAIN ALL N MODELS ----
all_r2_wl = []
all_rmse_wl = []

for i, (train_idx, val_idx) in enumerate(kf_indices):

    path = f'{MODELS_PATH}/split_{i}'
    model = MLPModel(hidden_dim=HIDDEN_DIM)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size = BATCH_SIZE,
        shuffle = True
    )

    val_loader = DataLoader(
        Subset(val_idx),
        batch_size = BATCH_SIZE,
        shuffle = False
    )

    train_mlp(
        model,
        train_loader,
        val_loader,
        path,
        epochs = EPOCHS,
        lr = LR,
        alpha = ALPHA,
        wd = WD
    )










