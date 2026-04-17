from util.model_eval import*
from util.classes.Phase1Dataset import *
from util.data_preprocessing import *
from util.classes.MLPModel import *
from preprocessing.max_peak import max_Q_ft, data_config
import json
import os


#### CREATE DIRECTORIES ####
# If a directory with the chosen name already exist an error will
# be thrown, this is to prevent overriding ones trained models.
# If you want to reuse the name just delete the directory
ROOT = "trained_models/phase1"
NAME = "test_run"
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
    "alpha": 4.0,
    "lr": 1e-3,
    "wd": 1e-4,
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

train_loader_refl, val_loader_refl, test_loader_refl = create_dataloaders(
    dataset_refl,
    loader_config["seed"],
    loader_config['train_ratio'],
    loader_config['val_ratio'],
    loader_config['test_ratio'],
    loader_config['batch_size']
)

#### CREATE MLP ####
model = MLPModel(
    MLP_config['hidden_dim'],
    MLP_config['n_layers']
)

#### TRAIN MODEL ####








