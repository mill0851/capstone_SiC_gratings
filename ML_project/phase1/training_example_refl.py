from util.classes.Phase1Dataset import *
from util.classes.RegMLP import *
from util.classes.ClsMLP import *
from preprocessing.max_peak import max_Q_ft, data_config
import json
import os


#### CREATE DIRECTORIES ####
# If a directory with the chosen name already exist an error will
# be thrown, this is to prevent overriding ones trained models.
# If you want to reuse the name just delete the directory
ROOT = "trained_models/phase1"
NAME = "refl_test_1"
DIR = f'{ROOT}/{NAME}'
os.makedirs(f'{DIR}', exist_ok=False)
os.makedirs(f'{DIR}/config', exist_ok=True)
os.makedirs(f'{DIR}/models', exist_ok=True)
os.makedirs(f'{DIR}/results', exist_ok=True)
os.makedirs(f'{DIR}/optimization_results', exist_ok=True)


#### SETUP/CONFIG ####
# Note that the data and data config comes from the chose preprocessing file
loader_config = {
    "seed": 1234,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "pre_split": {},
    "batch_size": 32
}

MLP_config = {
    "hidden_dim": 16,
    "n_layers": 2
}

train_config = {
    "epochs": 250,
    "lr": 1e-3,
    "wd": 1e-4,
    "patience": 30,
    "path": f'{DIR}/models'
}

# This is the settings used during preprocessing of the data
with open(f'{DIR}/config/data_preprocessing.json', "w") as f:
    json.dump(data_config, f, indent=4)

with open(f'{DIR}/config/data_loaders.json', "w") as f:
    json.dump(loader_config, f, indent=4)

with open(f'{DIR}/config/model.json', "w") as f:
    json.dump(MLP_config, f, indent=4)

with open(f'{DIR}/config/training.json', "w") as f:
    json.dump(train_config, f, indent=4)


#### CREATE DATASETS ####
dataset_refl = Phase1Dataset(
    max_Q_ft["geometry_table"],
    max_Q_ft["reflection_features"]
)

#### CREATE DATA LOADERS ####
train_loader_refl, val_loader_refl, test_loader_refl = create_dataloaders(
    dataset_refl,
    loader_config["seed"],
    loader_config['train_ratio'],
    loader_config['val_ratio'],
    loader_config['test_ratio'],
    loader_config['batch_size']
)

#### CREATE MLP ####
model_refl_reg = RegMLP(
    MLP_config['hidden_dim'],
    MLP_config['n_layers']
)

model_refl_cls = ClsMLP(
    MLP_config['hidden_dim'],
    MLP_config['n_layers']
)

#### TRAIN MODELS ####
reg_refl, reg_refl_hist = train_regression(
    model_refl_reg,
    train_loader_refl,
    val_loader_refl,
    train_config['epochs'],
    train_config['lr'],
    train_config['wd'],
    train_config['patience'],
    train_config['path']
)

cls_refl, cls_refl_hist = train_classification(
    model_refl_cls,
    train_loader_refl,
    val_loader_refl,
    train_config['epochs'],
    train_config['lr'],
    train_config['wd'],
    train_config['patience'],
    train_config['path']
)

