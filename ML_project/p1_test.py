from util.model_eval import*
from util.classes.Phase1Dataset import *
from util.data_preprocessing import *
from util.classes.MLPModel import *

# ---- CONFIG ----
DATA_PATH = './data/ML_proj_refl_data'
WL_DOMAIN = (11.25,12.5) # um
UPSAMPLE_RATE = 4
PEAK_THRESHOLD = 0.2
WINDOW_SAMPLES = 20
TRAIN_SPLIT = 0.75
BATCH_SIZE = 128
HIDDEN_DIM = 128
EPOCHS = 250
ALPHA = 4.0
LR = 1e-3
WD = 1e-4
TRAIN = True
NEW_SPLIT = True
NAME = "forward_refl_split-75-15-15"
PATH = f'C:/Users/robert/Code/capstone/capstone_SiC_gratings/ML_project/p1_models/{NAME}'
os.makedirs(PATH, exist_ok=True)

# ---- SETUP ----
dataset = load_dataset(
    DATA_PATH,
    WL_DOMAIN,
    UPSAMPLE_RATE,
    PEAK_THRESHOLD,
    WINDOW_SAMPLES,
    verbose=True
)

train_loader, val_loader, test_loader = create_dataloaders(
    dataset,
    f"{PATH}/data_split.pt",
    new_split=NEW_SPLIT,
    train_ratio=TRAIN_SPLIT,
    batch_size=BATCH_SIZE
)

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