import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import copy
from util.classes.Phase1Dataset import Phase1Dataset


class MLPModel(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super().__init__()

        layers = []

        # input layer
        layers.append(nn.Linear(4, hidden_dim))
        layers.append(nn.ReLU())

        # hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.shared = nn.Sequential(*layers)

        self.regression_head = nn.Linear(hidden_dim, 2)
        self.classification_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        reg = self.regression_head(h)
        cls = self.classification_head(h)
        return reg, cls.squeeze()

def train_mlp_phase1(
        model: MLPModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        alpha: float,
        wd: float,
        path: str):
    
    # Config for training loop
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    # History dict for loss graphs
    history = {
        "train_cls_loss": [],
        "val_cls_loss": [],
        "train_reg_loss": [],
        "val_reg_loss": [],
        "stop_epoch_reg": 0,
        "stop_epoch_cls": 0
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
        train_cls_samples = 0
        train_reg_samples = 0

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
                loss_reg = torch.tensor(0.0, device=geom.device)

            # Backprop + optimization 
            loss = loss_reg + alpha*loss_cls
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            batch_size = geom.size(0)
            train_cls_loss += (loss_cls.item() * batch_size)
            train_reg_loss += (loss_reg.item() * batch_size)
            train_cls_samples += batch_size
            train_reg_samples += mask.sum().item()

        # Evaluation Loop
        model.eval()
        val_cls_loss = 0
        val_reg_loss = 0
        val_cls_samples = 0
        val_reg_samples = 0

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
                    loss_reg = torch.tensor(0.0, device=geom.device)

                # Logging
                batch_size = geom.size(0)
                val_cls_loss += (loss_cls.item()*batch_size)
                val_reg_loss += (loss_reg.item()*batch_size)
                val_cls_samples += batch_size
                val_reg_samples += mask.sum().item()

        # Convert to per-sample epoch-average losses
        val_cls_loss /= val_cls_samples
        val_reg_loss /= val_reg_samples
        train_cls_loss /= train_cls_samples
        train_reg_loss /= train_reg_samples

        # Update loss history
        history["train_cls_loss"].append(train_cls_loss)
        history["train_reg_loss"].append(train_reg_loss)
        history["val_cls_loss"].append(val_cls_loss)
        history["val_reg_loss"].append(val_reg_loss)

        # Update best loss states
        if val_cls_loss < best_cls_loss:
            history['stop_epoch_cls'] = epoch
            best_cls_loss = val_cls_loss
            best_cls_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_cls_state,
                "history": copy.deepcopy(history)
            }, f"{path}/best_cls.pt")

        if val_reg_loss < best_reg_loss:
            history['stop_epoch_reg'] = epoch
            best_reg_loss = val_reg_loss
            best_reg_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_reg_state,
                "history": copy.deepcopy(history)
            }, f"{path}/best_reg.pt")

        # Print out current epoch + losses
        print(f"Epoch {epoch+1:03d} | "
              f"Train CLS: {history['train_cls_loss'][-1]:.4f} | "
              f"Train REG: {history['train_reg_loss'][-1]:.4f} | "
              f"Val CLS: {history['val_cls_loss'][-1]:.4f} | "
              f"Val REG: {history['val_reg_loss'][-1]:.4f}")
    
    # Save final model
    history['stop_epoch_cls'] = epochs
    history['stop_epoch_reg'] = epochs
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history
    }, f"{path}/final_model.pt")
    
    return best_reg_state, best_cls_state

def create_dataloaders(
    dataset: Phase1Dataset,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    batch_size: int,
    pre_split: dict | None = None
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "ratios must add to 1.0"

    if pre_split is not None:
        pass

    n_total = len(dataset)
    n_train = round(train_ratio * n_total)
    n_val = round(val_ratio * n_total)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_set = Subset(dataset, train_indices.tolist())
    val_set = Subset(dataset, val_indices.tolist())
    test_set = Subset(dataset, test_indices.tolist())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

    




