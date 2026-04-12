from util.model_eval import*
from util.classes.Phase1Dataset import *
from util.data_preprocessing import *
from util.classes.MLPModel import *
from util.model_optimization import *
import itertools
import json

# Path to save optimized model
MODELS_PATH = 'C:/Users/robert/Code/capstone/capstone_SiC_gratings/ML_project/p1_models/optimization_1'
os.makedirs(MODELS_PATH, exist_ok=True)


# Import optimized hyperparameters from optimziation resutlt
optim_df = pd.read_csv("./p1_optimization/grid_search_results_1.csv")
best_wl_row = optim_df.loc[optim_df['wl_mean_rmse'].idxmin()]

hidden_dim = int(best_wl_row['hidden_dim'].item())
n_layers = int(best_wl_row['n_layers'].item())
lr = best_wl_row['lr'].item()
batch_size = int(best_wl_row['batch_size'].item())
weight_decay = best_wl_row['weight_decay'].item()
alpha = best_wl_row['alpha'].item()

print('Optimized Hyperparameters: ')
print(f'Hidden Layer Dimensions: {hidden_dim}')
print(f'Number of Hidden Layers: {n_layers}')
print(f'Learning Rate: {lr}')
print(f'Batch Size: {batch_size}')
print(f'Weigth Decay: {weight_decay}')
print(f'Alpha: {alpha} \n')


# Import data processing, and index seeding configuration from
# the optimization loop (makes things controlled)
with open("./p1_optimization/data_config.json", "r") as f:
    data_config = json.load(f)

with open("./p1_optimization/kfold_config.json", "r") as f:
    kfold_config = json.load(f)

# print(f"Data configuration during optimization: \n {data_config} \n")
# print(f"kfold configuration during optimization: \n {kfold_config} \n")


# Import dataset
dataset = load_dataset(
    data_config['path'],
    data_config['wl_domain'],
    data_config['upsample_rate'],
    data_config['threshold'],
    data_config['window'],
    verbose=False
)


# Generate indices
kf_indices, test_indices = generate_kfold(
    dataset,
    n_splits=kfold_config['n_splits'],
    shuffle=kfold_config['shuffle'],
    random_state=kfold_config['seed'],
    test_ratio=kfold_config['test_ratio']
)


# Seed model and generators
torch.manual_seed(kfold_config['seed'])
np.random.seed(kfold_config['seed'])
generator = torch.Generator()
generator.manual_seed(kfold_config['seed'])


# Determine best epoch range for training
# best_epochs = []

# for i, (train_idx, val_idx) in enumerate(kf_indices):
#         torch.manual_seed(kfold_config['seed'])
#         np.random.seed(kfold_config['seed'])
#         generator = torch.Generator()
#         generator.manual_seed(kfold_config['seed'])

#         print(f"Fold: {i}")
#         print(f"train indices: {train_idx}")
#         print(f"val indices: {val_idx}")

#         model = MLPModel(hidden_dim=hidden_dim, n_layers=n_layers)

#         train_loader = DataLoader(
#             Subset(dataset, train_idx),
#             batch_size=batch_size,
#             shuffle=True,
#             generator=generator
#         )
        
#         val_loader   = DataLoader(
#             Subset(dataset, val_idx),
#             batch_size=batch_size, 
#             shuffle=False
#         )

#         best_reg, best_reg_epoch, best_cls, best_cls_epoch = train_mlp(
#             model,
#             train_loader,
#             val_loader,
#             MODELS_PATH,
#             epochs=250,
#             lr=lr,
#             alpha=alpha,
#             wd=weight_decay,
#             save=False
#         )

#         best_epochs.append(best_reg_epoch)

# print(f'Best epochs: {best_epochs}')


# Training loop, here early stopping is based on the 25th-75th epoch
# range of the optimization kfold early stops. The best result in this
# range is taken. A small percenteage of the non test data should be
# used for validation to prevent test data leaking.
epoch_range = (155, 190)

N = len(dataset)
indices = np.arange(N)

rng = np.random.RandomState(kfold_config['seed'])
rng.shuffle(indices)

non_test = [idx for idx in indices if idx not in test_indices]

val_split = 0.1
split = int(val_split * len(non_test))

val_indices = non_test[:split]
train_indices = non_test[split:]

print(f'Test samples: {len(test_indices)}')
print(f'Training samples: {len(train_indices)}')
print(f'Validation samples: {len(val_indices)}')
print(f'Total samples: {len(test_indices) + len(train_indices) + len(val_indices)}')

indices_dict = {
    'train_indices':train_indices,
    'val_indices':val_indices,
    'test_indices':test_indices
}

train_loader, val_loader, test_loader = create_dataloaders(
    dataset,
    "",
    batch_size=batch_size,
    indices=indices_dict
)

model = MLPModel(hidden_dim=hidden_dim, n_layers=n_layers)
reg_model, reg_epoch, cls_model, cls_epoch, final_model, history = train_mlp_optimal(
    model, train_loader, val_loader, MODELS_PATH,
    epochs=epoch_range[1],
    lr=lr,
    alpha=alpha,
    wd=weight_decay,
    save=True,
    min_epoch=epoch_range[0]
)

plot_losses(history)
plot_regression(reg_model, test_loader, dataset)
compute_r2(reg_model, test_loader)
plot_confusion_matrix(cls_model, test_loader)
classification_metrics(cls_model, test_loader)
rmse = compute_rmse(reg_model, test_loader, dataset)

print(f'wl_rmse: {rmse["wl_rmse"]}')
print(f'Q_rmse: {rmse["Q_rmse"]}')
print(f'reg_epoch: {reg_epoch}')
print(f'cls_epoch: {cls_epoch}')
