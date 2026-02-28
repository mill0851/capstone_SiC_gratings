from util.model_eval import*
from util.classes.Phase1Dataset import *
from util.data_preprocessing import *
from util.classes.MLPModel import *
from util.model_optimization import *
import itertools
import random

# ---- IMPORT SETTINGS ----
DATA_PATH = './data/ML_proj_refl_data'
MODELS_PATH = 'C:/Users/robert/Code/capstone/capstone_SiC_gratings/ML_project/p1_models/kfold_optimization_1'
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
kfold_config = {
    "n_splits": N_SPLITS,
    'shuffle': SHUFFLE,
    'seed': SEED,
    'test_ration': TEST_RATIO,
}

# ---- HYPER PARAM GRID ----
# hidden_dim = [64, 128]
# n_layers = [2, 3]
# learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
# batch_sizes = [32, 64]
# weight_decays = [0.0, 1e-4, 1e-3, 1e-2]
# alphas = [1.0, 2.0, 3.0, 4.0]

hidden_dim = [64]
n_layers = [2]
learning_rates = [1e-4, 5e-4]
batch_sizes = [32]
weight_decays = [1e-4]
alphas = [1.0]

hyper_grid = list(itertools.product(hidden_dim,
                                    n_layers,
                                    learning_rates,
                                    batch_sizes,
                                    weight_decays,
                                    alphas))

# ---- LOAD DATA ----
dataset = load_dataset(
    DATA_PATH,
    WL_DOMAIN,
    UPSAMPLE_RATE,
    PEAK_THRESHOLD,
    WINDOW_SAMPLES,
    verbose=False
)

# ---- OPTIMIZATION LOOP ----
results = []

for params in hyper_grid:
    print("Evaluating hyperparams:", params)

    hidden_dim, n_layers, lr, batch_size, wd, alpha = params
    fold_wl_rmses = []
    fold_Q_rmses = []

    kf_indices, test_indices = generate_kfold(
        dataset,
        n_splits = N_SPLITS,
        shuffle = SHUFFLE,
        random_state = SEED,
        test_ratio = TEST_RATIO
    )

    for i, (train_idx, val_idx) in enumerate(kf_indices):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        generator = torch.Generator()
        generator.manual_seed(SEED)

        print(f"Fold: {i}")
        print(f"train indices: {train_idx}")
        print(f"val indices: {val_idx}")

        model = MLPModel(hidden_dim=hidden_dim, n_layers=n_layers)

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            generator=generator
        )
        
        val_loader   = DataLoader(
            Subset(dataset, val_idx),
            batch_size=batch_size, 
            shuffle=False
        )

        best_reg, best_cls = train_mlp(
            model,
            train_loader,
            val_loader,
            MODELS_PATH,
            epochs=EPOCHS,
            lr=lr,
            alpha=alpha,
            wd=wd,
            save=False
        )

        rmses = compute_rmse(model, val_loader, dataset)
        fold_wl_rmses.append(rmses['wl_rmse'])
        fold_Q_rmses.append(rmses['Q_rmse'])

    wl_mean_rmse = np.mean(fold_wl_rmses)
    wl_std_rmse = np.std(fold_wl_rmses)
    Q_mean_rmse = np.mean(fold_Q_rmses)
    Q_std_rmse = np.std(fold_Q_rmses)

    results.append({
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": wd,
        "alpha": alpha,
        "wl_mean_rmse": wl_mean_rmse,
        "Q_mean_rmse": Q_mean_rmse,
        "wl_std_rmse": wl_std_rmse,
        "Q_std_rmse": Q_std_rmse
    })

results_df = pd.DataFrame(results)
results_df.to_csv("grid_search_results2.csv", index=False)

best_wl_row, best_Q_row = results_df.loc[results_df['wl_mean_rmse'].idxmin()], results_df.loc[results_df['Q_mean_rmse'].idxmin()]
print("Best wavelength hyperparam:\n", best_wl_row)
print("Best Q hyperparam:\n", best_Q_row)

