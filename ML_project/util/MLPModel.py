import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import copy


class MLPModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.regression_head = nn.Linear(hidden_dim, 2)
        self.classification_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        reg = self.regression_head(h)
        cls = self.classification_head(h)
        return reg, cls.squeeze()

def train_mlp(
        model: MLPModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        PATH: str,
        epochs=150,
        lr=1e-3,
        alpha=1.0,
        wd=1e-4):
    
    # Config for training loop
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    # History dict for loss graphs
    history = {
        "train_cls_loss": [],
        "val_cls_loss": [],
        "train_reg_loss": [],
        "val_reg_loss": []
    }

    # keep track of best reg and cls states (minimum loss)
    best_cls_loss = float('inf')
    best_reg_loss = float('inf')
    best_cls_state = None
    best_reg_state = None

    for epoch in range(epochs):
        # Training Loop
        model.train()
        train_cls_loss = 0
        train_reg_loss = 0
        train_samples = 0

        # Forward pass each batch
        for geom, feat, mask in train_loader:

            # Forward
            reg_pred, cls_logits = model(geom)

            # Loss calculations
            loss_cls = bce(cls_logits, mask)
            mask_bool = mask == 1
            if mask_bool.sum() > 0:
                loss_reg = mse(reg_pred[mask_bool], feat[mask_bool])
            else:
                loss_reg = torch.tensor(0.0)

            # Backprop + optimization 
            loss = loss_reg + alpha*loss_cls
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            batch_size = geom.size(0)
            train_cls_loss += (loss_cls.item() * batch_size)
            train_reg_loss += (loss_reg.item() * batch_size)
            train_samples += batch_size
        

        # Evaluation Loop
        model.eval()
        val_cls_loss = 0
        val_reg_loss = 0
        val_samples = 0

        # Forward pass each batch
        with torch.no_grad():
            for geom, feat, mask in val_loader:
                
                # Forward
                reg_pred, cls_logits = model(geom)

                # Loss calculations
                loss_cls = bce(cls_logits, mask)
                mask_bool = mask == 1
                if mask_bool.sum() > 0:
                    loss_reg = mse(reg_pred[mask_bool], feat[mask_bool])
                else:
                    loss_reg = torch.zeros(1, device=geom.device)

                # Logging
                batch_size = geom.size(0)
                val_cls_loss += (loss_cls.item()*batch_size)
                val_reg_loss += (loss_reg.item()*batch_size)
                val_samples += batch_size

        # Convert to per-sample epoch-average losses
        val_cls_loss /= val_samples
        val_reg_loss /= val_samples
        train_cls_loss /= train_samples
        train_reg_loss /= train_samples

        # Update loss history
        history["train_cls_loss"].append(train_cls_loss)
        history["train_reg_loss"].append(train_reg_loss)
        history["val_cls_loss"].append(val_cls_loss)
        history["val_reg_loss"].append(val_reg_loss)

        # Update best loss states
        if val_cls_loss < best_cls_loss:
            best_cls_loss = val_cls_loss
            best_cls_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_cls_state,
                "history": history
            }, f"{PATH}/best_cls.pt")

        if val_reg_loss < best_reg_loss:
            best_reg_loss = val_reg_loss
            best_reg_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_reg_state,
                "history": history
            }, f"{PATH}/best_reg.pt")

        # Print out current epoch + losses
        print(f"Epoch {epoch+1:03d} | "
              f"Train CLS: {history['train_cls_loss'][-1]:.4f} | "
              f"Train REG: {history['train_reg_loss'][-1]:.4f} | "
              f"Val CLS: {history['val_cls_loss'][-1]:.4f} | "
              f"Val REG: {history['val_reg_loss'][-1]:.4f}")
    
    # Save final model
    torch.save({
    "model_state_dict": model.state_dict(),
    "history": history
    }, f"{PATH}/final_model.pt")
    

