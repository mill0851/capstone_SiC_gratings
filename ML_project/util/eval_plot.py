import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
from sklearn.metrics import r2_score


def plot_losses(history: dict):
    epochs = np.arange(1, len(history["train_cls_loss"]) + 1)

    # Train Classification
    plt.figure()
    plt.plot(epochs, history["train_cls_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Classification Loss")
    plt.show()

    # Val Classification
    plt.figure()
    plt.plot(epochs, history["val_cls_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Classification Loss")
    plt.show()

    # Train Regression
    plt.figure()
    plt.plot(epochs, history["train_reg_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Regression Loss")
    plt.show()

    # Val Regression
    plt.figure()
    plt.plot(epochs, history["val_reg_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Regression Loss")
    plt.show()

def plot_lambda_predictions(model, data_loader, dataset, device="cpu"):
    model.eval()
    
    all_true = []
    all_pred = []

    with torch.no_grad():
        for geom, feat, mask in data_loader:
            geom = geom.to(device)
            feat = feat.to(device)
            mask = mask.to(device)

            reg_pred, cls_logits = model(geom)

            mask_bool = mask == 1
            if mask_bool.sum() > 0:
                # Select only valid resonances
                true_feat = feat[mask_bool][:, 0:1]     # lambda
                pred_feat = reg_pred[mask_bool][:, 0:1] # lambda

                # Denormalize
                true_feat = dataset.denormalize_feature(true_feat)
                pred_feat = dataset.denormalize_feature(pred_feat)

                true_feat = true_feat[:, 0]
                pred_feat = pred_feat[:, 0]

                all_true.append(true_feat.cpu())
                all_pred.append(pred_feat.cpu())

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()

    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Target Lambda")
    plt.ylabel("Predicted Lambda")
    plt.title("Lambda: Target vs Predicted")
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()])
    plt.show()

def plot_Q_predictions(model, data_loader, device="cpu"):
    model.eval()
    
    all_true = []
    all_pred = []

    with torch.no_grad():
        for geom, feat, mask in data_loader:
            geom = geom.to(device)
            feat = feat.to(device)
            mask = mask.to(device)

            reg_pred, cls_logits = model(geom)

            mask_bool = mask == 1
            if mask_bool.sum() > 0:
                all_true.append(feat[mask_bool][:, 1].cpu())
                all_pred.append(reg_pred[mask_bool][:, 1].cpu())

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()

    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Target Q")
    plt.ylabel("Predicted Q")
    plt.title("Q: Target vs Predicted")
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()])
    plt.show()

def plot_confusion_matrix(model, data_loader, device="cpu"):
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for geom, feat, mask in data_loader:
            geom = geom.to(device)
            mask = mask.to(device)

            _, cls_logits = model(geom)
            preds = (cls_logits > 0).float()

            all_true.append(mask.cpu())
            all_pred.append(preds.cpu())

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def compute_r2(model, data_loader, device="cpu"):
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for geom, feat, mask in data_loader:
            geom = geom.to(device)
            feat = feat.to(device)
            mask = mask.to(device)

            reg_pred, _ = model(geom)

            mask_bool = mask == 1
            if mask_bool.sum() > 0:
                all_true.append(feat[mask_bool].cpu())
                all_pred.append(reg_pred[mask_bool].cpu())

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()

    r2_lambda = r2_score(y_true[:,0], y_pred[:,0])
    r2_Q = r2_score(y_true[:,1], y_pred[:,1])

    print(f"R² Lambda: {r2_lambda:.4f}")
    print(f"R² Q: {r2_Q:.4f}")

def classification_metrics(model, data_loader, device="cpu"):
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for geom, feat, mask in data_loader:
            geom = geom.to(device)
            mask = mask.to(device)

            _, cls_logits = model(geom)
            preds = (cls_logits > 0).float()

            all_true.append(mask.cpu())
            all_pred.append(preds.cpu())

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))

