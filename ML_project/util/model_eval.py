import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
from sklearn.metrics import r2_score
from util.classes.RegMLP import RegMLP
from util.classes.ClsMLP import ClsMLP
from util.classes.Phase1Dataset import Phase1Dataset
from torch.utils.data import DataLoader


def plot_losses(history: dict, title: str):
    epochs = np.arange(1, history["stop_epoch"] + 2)

    # Train Classification
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="training loss", color='k')
    plt.plot(epochs, history["val_loss"], label="validation loss", color="cyan")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.title(title, fontsize=18)
    plt.grid(alpha=0.75)
    plt.legend(fontsize=16)
    plt.show()

def plot_losses_recon(history: dict, title: str):
    epochs = np.arange(1, history["stop_epoch"] + 2)

    # Train Classification
    plt.figure()
    plt.plot(epochs, history["train_loss_recon"], label="training loss", color='k')
    plt.plot(epochs, history["val_loss_recon"], label="validation loss", color="cyan")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.title(title, fontsize=18)
    plt.grid(alpha=0.75)
    plt.legend(fontsize=16)
    plt.show()

# def evaluate_model(
#         model: MLPModel,
#         data_loader: DataLoader,
#         dataset: Phase1Dataset,
#         device: str = "cpu"):
        
#     all_true = []
#     all_pred = []

#     model.eval()
#     with torch.no_grad():
#         for geom, feat, mask in data_loader:

#             geom = geom.to(device)
#             feat = feat.to(device)
#             mask = mask.to(device)

#             reg_pred, _ = model(geom)

#             mask_bool = mask == 1
#             if mask_bool.sum() > 0:

#                 # Select only valid resonances
#                 true_feat = feat[mask_bool]
#                 pred_feat = reg_pred[mask_bool]

#                 # Denormalize
#                 true_feat = dataset.denormalize_feature(true_feat)
#                 pred_feat = dataset.denormalize_feature(pred_feat)

#                 # Add to list
#                 all_true.append(true_feat.cpu())
#                 all_pred.append(pred_feat.cpu())

#     target = torch.cat(all_true).numpy()
#     pred = torch.cat(all_pred).numpy()

#     return target, pred

# def plot_regression(
#         model: MLPModel,
#         data_loader: DataLoader,
#         dataset: Phase1Dataset,
#         device="cpu"):
#     """
#     Plots regression predictions vs actual feature values.

#     Args:
#         model (MLPModel): Trained model to evaluate
#         data_loader (DataLoader): data loader to serve batches of
#         the relevant data
#         dataset (Phase1Dataset): dataset class used in this project 
#         device (str, optional): probably gonna always be cpu?. Defaults to "cpu".
#     """
    
#     target, pred = evaluate_model(model, data_loader, dataset, device=device)

#     plt.figure()
#     plt.scatter(target[:,0], pred[:,0], color='k', label="Regression")
#     plt.xlabel("Target Wavelength [um]", fontsize=18)
#     plt.ylabel("Model Prediction [um]", fontsize=18)
#     plt.title("Wavelength Regression", fontsize=20)
#     plt.plot([target[:,0].min(), target[:,0].max()],
#              [target[:,0].min(), target[:,0].max()], color="cyan", label="100% Accuracy")
#     plt.grid(alpha=0.8)
#     plt.legend(fontsize=14)
#     plt.show()

#     plt.figure()
#     plt.scatter(target[:,1], pred[:,1], color='k', label="Regression")
#     plt.xlabel("Target Q [unitless]", fontsize=18)
#     plt.ylabel("Model Prediction [unitless]", fontsize=18)
#     plt.title("Q Regression", fontsize=20)
#     plt.plot([target[:,1].min(), target[:,1].max()],
#              [target[:,1].min(), target[:,1].max()], color="cyan", label="100% Accuracy")
#     plt.grid(alpha=0.8)
#     plt.legend(fontsize=14)
#     plt.show()

# def plot_confusion_matrix(model, data_loader, device="cpu"):
#     model.eval()

#     all_true = []
#     all_pred = []

#     with torch.no_grad():
#         for geom, feat, mask in data_loader:
#             geom = geom.to(device)
#             mask = mask.to(device)

#             _, cls_logits = model(geom)
#             preds = (cls_logits > 0).float()

#             all_true.append(mask.cpu())
#             all_pred.append(preds.cpu())

#     y_true = torch.cat(all_true).numpy()
#     y_pred = torch.cat(all_pred).numpy()

#     cm = confusion_matrix(y_true, y_pred)

#     plt.figure()
#     sns.heatmap(cm, annot=True, fmt="d", cmap='coolwarm')
#     plt.xlabel("Predicted", fontsize=16)
#     plt.ylabel("True", fontsize=16)
#     plt.title("Confusion Matrix", fontsize=18)
#     plt.show()

# def compute_r2(model, data_loader, device="cpu"):
#     model.eval()

#     all_true = []
#     all_pred = []

#     with torch.no_grad():
#         for geom, feat, mask in data_loader:
#             geom = geom.to(device)
#             feat = feat.to(device)
#             mask = mask.to(device)

#             reg_pred, _ = model(geom)

#             mask_bool = mask == 1
#             if mask_bool.sum() > 0:
#                 all_true.append(feat[mask_bool].cpu())
#                 all_pred.append(reg_pred[mask_bool].cpu())

#     y_true = torch.cat(all_true).numpy()
#     y_pred = torch.cat(all_pred).numpy()

#     r2_lambda = r2_score(y_true[:,0], y_pred[:,0])
#     r2_Q = r2_score(y_true[:,1], y_pred[:,1])

#     print(f"R² Lambda: {r2_lambda:.4f}")
#     print(f"R² Q: {r2_Q:.4f}")

# def classification_metrics(model, data_loader, device="cpu"):
#     model.eval()

#     all_true = []
#     all_pred = []

#     with torch.no_grad():
#         for geom, feat, mask in data_loader:
#             geom = geom.to(device)
#             mask = mask.to(device)

#             _, cls_logits = model(geom)
#             preds = (cls_logits > 0).float()

#             all_true.append(mask.cpu())
#             all_pred.append(preds.cpu())

#     y_true = torch.cat(all_true).numpy()
#     y_pred = torch.cat(all_pred).numpy()

#     print("Accuracy:", accuracy_score(y_true, y_pred))
#     print("Precision:", precision_score(y_true, y_pred))
#     print("Recall:", recall_score(y_true, y_pred))

# def compute_rmse(
#         model: MLPModel,
#         data_loader: DataLoader,
#         dataset: Phase1Dataset,
#         device: str = 'cpu'):
#     """
#     Computes the RMSE value for the regression head in Non-normalized units. i.e. the
#     return values are in physical units (nm, Q is scaled properly)

#     parameters:
#         model: This is an MLPModel object which you have trained to desire
#         val_loader: data loader with validation (or whatever) data
#         device: device for computation (typically cpu, you would know otherwise)
#     """
#     target, pred = evaluate_model(model, data_loader, dataset, device=device)

#     wl_target = target[:,0]
#     wl_pred = pred[:,0]

#     Q_target = target[:,1]
#     Q_pred = pred[:,1]

#     wl_rmse = np.sqrt(np.mean((wl_pred - wl_target)**2))
#     Q_rmse = np.sqrt(np.mean((Q_pred - Q_target)**2))

#     return {"wl_rmse": wl_rmse, "Q_rmse": Q_rmse}



            