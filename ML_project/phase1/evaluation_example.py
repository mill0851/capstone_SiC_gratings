from util.model_eval import*
from util.classes.RegMLP import *
from util.classes.ClsMLP import *
import torch
import json


#### LOAD IN MODELS FOR EVALUATION ####
model_path = "trained_models/phase1/abs_test_1"

with open(f'{model_path}/config/model.json', 'r') as f:
    model_config = json.load(f)

checkpoint_reg = torch.load(f'{model_path}/models/best_reg.pt')
checkpoint_cls = torch.load(f'{model_path}/models/best_cls.pt')

reg_model = RegMLP(
    model_config['hidden_dim'],
    model_config['n_layers'],
    model_config['p']
)

cls_model = ClsMLP(
    model_config['hidden_dim'],
    model_config['n_layers']
)

reg_model.load_state_dict(checkpoint_reg['model_state_dict'])
cls_model.load_state_dict(checkpoint_cls['model_state_dict'])
reg_history = checkpoint_reg['history']
cls_history = checkpoint_cls['history']

#### PLOT LOSS FUNCTIONS ####
plot_losses(reg_history, "Regression Loss - Absorption")
plot_losses(cls_history, "Classification Loss - Absorption")
