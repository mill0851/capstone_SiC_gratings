import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np


class MLPModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.regression_head = nn.Linear(hidden_dim, 2)
        self.classification_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        reg = self.regression_head(h)
        cls = torch.sigmoid(self.classification_head(h))
        return reg, cls.squeeze()

def train_mlp(model: MLPModel, train_loader: DataLoader, val_loader: DataLoader,
              epochs=150, lr=1e-3, alpha=1.0):
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    history = {
        "train_cls_loss": [],
        "val_cls_loss": [],
        "train_reg_loss": [],
        "val_reg_loss": []
    }

    for epoch in range(epochs):

        # Training Loop
        model.train()

        train_cls_loss = 0
        train_reg_loss = 0
        train_batches = 0

        for geom, feat, mask in train_loader:

            reg_pred, cls_logits = model(geom)

            loss_cls = bce(cls_logits, mask)

            mask_bool = mask == 1
            if mask_bool.sum() > 0:
                loss_reg = mse(reg_pred[mask_bool], feat[mask_bool])
            else:
                loss_reg = torch.tensor(0.0)

            loss = loss_reg + alpha*loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_cls_loss += loss_cls.item()
            train_reg_loss += loss_reg.item()
            train_batches += 1
        
        # Evaluation Loop
        model.eval()

        val_cls_loss = 0
        val_reg_loss = 0
        val_batches = 0

        with torch.no_grad():
            for geom, feat, mask in val_loader:
                
                reg_pred, cls_logits = model(geom)

                loss_cls = bce(cls_logits, mask)

                mask_bool = mask == 1
                if mask_bool.sum() > 0:
                    loss_reg = mse(reg_pred[mask_bool], feat[mask_bool])
                else:
                    loss_reg = torch.tensor(0.0)

                val_cls_loss += loss_cls.item()
                val_reg_loss += loss_reg.item()
                val_batches += 1

        # Average losses
        history["train_cls_loss"].append(train_cls_loss / train_batches)
        history["train_reg_loss"].append(train_reg_loss / train_batches)
        history["val_cls_loss"].append(val_cls_loss / val_batches)
        history["val_reg_loss"].append(val_reg_loss / val_batches)

        print(f"Epoch {epoch+1:03d} | "
              f"Train CLS: {history['train_cls_loss'][-1]:.4f} | "
              f"Train REG: {history['train_reg_loss'][-1]:.4f} | "
              f"Val CLS: {history['val_cls_loss'][-1]:.4f} | "
              f"Val REG: {history['val_reg_loss'][-1]:.4f}")
        
    return history

