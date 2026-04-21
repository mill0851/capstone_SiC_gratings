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
NAME = "abs_test_1"
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
    "hidden_dim": 128,
    "n_layers": 4,
    "p": 0.15
}

train_config = {
    "epochs": 500,
    "lr": 5e-3,
    "wd": 3e-4,
    "patience": 100,
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
dataset_abs = Phase1Dataset(
    max_Q_ft["geometry_table"],
    max_Q_ft["absorption_features"]
)

#### CREATE DATA LOADERS ####
train_loader_abs, val_loader_abs, test_loader_abs = create_dataloaders(
    dataset_abs,
    loader_config["seed"],
    loader_config['train_ratio'],
    loader_config['val_ratio'],
    loader_config['test_ratio'],
    loader_config['batch_size']
)

#### CREATE MLP ####
model_abs_reg = RegMLP(
    MLP_config['hidden_dim'],
    MLP_config['n_layers'],
    MLP_config["p"]
)

model_abs_cls = ClsMLP(
    MLP_config['hidden_dim'],
    MLP_config['n_layers']
)

#### TRAIN MODELS ####
reg_abs, reg_abs_hist = train_regression(
    model_abs_reg,
    train_loader_abs,
    val_loader_abs,
    train_config['epochs'],
    train_config['lr'],
    train_config['wd'],
    train_config['patience'],
    train_config['path']
)

cls_abs, cls_abs_hist = train_classification(
    model_abs_cls,
    train_loader_abs,
    val_loader_abs,
    train_config['epochs'],
    train_config['lr'],
    train_config['wd'],
    train_config['patience'],
    train_config['path']
)