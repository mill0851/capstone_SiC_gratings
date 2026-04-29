import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
from util.classes.PCADataset import PCADataset
import numpy as np


class PcaMLP(nn.Module):
    """
    MLP surrogate for PCA-coefficient regression.
    Geometry (4D) -> K principal-component coefficients.

    K is an init argument so the same class supports sweeps
    over different PCA truncations (K=10, 20, 30, ...).
    """

    def __init__(self, K: int, hidden_dim: int, n_layers: int, p: float, input_dim: int = 4):
        super().__init__()

        self.K = K

        layers = []

        # input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p))

        # hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p))

        self.shared = nn.Sequential(*layers)
        self.regression_head = nn.Linear(hidden_dim, K)

    def forward(self, x):
        h = self.shared(x)
        reg = self.regression_head(h)
        return reg


def train_pca_regression(
        model: PcaMLP,
        dataset: PCADataset,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        wd: float,
        patience: int,
        path: str):
    """
    Train a PcaMLP to regress K PCA coefficients from 4D geometry.
    Mirrors RegMLP.train_regression but without the per-sample mask:
    every sample contributes to the loss since PCA coefficients are
    always defined.

    The dataset is passed in so reconstruction-space MSE can be logged
    each epoch alongside the coefficient-space training loss.
    """

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    mse = nn.MSELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_loss_recon": [],
        "val_loss_recon": [],
        "stop_epoch": 0,
        "K": model.K
    }

    best_recon_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_reg_loss = 0.0
        train_recon_loss = 0.0
        train_samples = 0

        for geom, feat in train_loader:
            optimizer.zero_grad()
            reg_pred = model(geom)
            loss_reg = mse(reg_pred, feat)

            loss_reg.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_n = geom.shape[0]
            train_reg_loss += loss_reg.item() * batch_n
            train_samples += batch_n

            with torch.no_grad():
                recon_pred = dataset.reconstruct_spectrum(reg_pred.detach())
                recon_true = dataset.reconstruct_spectrum(feat)
                loss_recon = ((recon_pred - recon_true) ** 2).mean()
                train_recon_loss += loss_recon.item() * batch_n

        # Validation loop
        model.eval()
        val_reg_loss = 0.0
        val_recon_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for geom, feat in val_loader:
                reg_pred = model(geom)
                loss_reg = mse(reg_pred, feat)

                recon_pred = dataset.reconstruct_spectrum(reg_pred)
                recon_true = dataset.reconstruct_spectrum(feat)
                loss_recon = ((recon_pred - recon_true) ** 2).mean()

                batch_n = geom.shape[0]
                val_reg_loss += loss_reg.item() * batch_n
                val_recon_loss += loss_recon.item() * batch_n
                val_samples += batch_n

        train_reg_loss /= max(train_samples, 1)
        val_reg_loss /= max(val_samples, 1)
        train_recon_loss /= max(train_samples, 1)
        val_recon_loss /= max(val_samples, 1)

        history["train_loss"].append(train_reg_loss)
        history["val_loss"].append(val_reg_loss)
        history["train_loss_recon"].append(train_recon_loss)
        history["val_loss_recon"].append(val_recon_loss)

        # print(f"Epoch {epoch+1:03d} | K={model.K:3d} | "
        #       f"Train REG: {train_reg_loss:.6f} | "
        #       f"Val REG: {val_reg_loss:.6f} | "
        #       f"Train RECON: {train_recon_loss:.3e} | "
        #       f"Val RECON: {val_recon_loss:.3e} | "
        #       f"Patience: {patience_counter}/{patience}")

        # Early stopping and checkpoint selection on reconstruction-space MSE.
        # Coeff-space MSE is a proxy; recon MSE is the metric we ultimately
        # report, so select the best epoch by it directly.
        if val_recon_loss < best_recon_loss:
            best_recon_loss = val_recon_loss
            patience_counter = 0
            history["stop_epoch"] = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "history": copy.deepcopy(history),
                "K": model.K
            }, f"{path}/best_pca_reg.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        scheduler.step(val_recon_loss)

    checkpoint = torch.load(f"{path}/best_pca_reg.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.state_dict(), checkpoint["history"]


def train_pca_regression_final(
        model: PcaMLP,
        dataset: PCADataset,
        train_loader: DataLoader,
        epochs: int,
        lr: float,
        wd: float,
        path: str):
    """
    Fixed-epoch final training: no val loop, no early stopping. Trains on the
    full (non-test) set so the held-out test split stays clean for a single
    post-training evaluation. `epochs` should be informed by the mean best
    epoch observed during k-fold CV.
    """

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # Same scheduler dynamics as the fold trainer, but keyed to a smoothed
    # train loss (no val set here). patience bumped 10 -> 15 and a small
    # rel threshold to absorb train-loss noise from dropout.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15,
        threshold=1e-3, threshold_mode='rel'
    )
    mse = nn.MSELoss()

    history = {
        "train_loss": [],
        "train_loss_recon": [],
        "epochs": epochs,
        "K": model.K,
    }

    smooth_window = 5

    for epoch in range(epochs):
        model.train()
        train_reg_loss = 0.0
        train_recon_loss = 0.0
        train_samples = 0

        for geom, feat in train_loader:
            optimizer.zero_grad()
            reg_pred = model(geom)
            loss_reg = mse(reg_pred, feat)

            loss_reg.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_n = geom.shape[0]
            train_reg_loss += loss_reg.item() * batch_n
            train_samples += batch_n

            with torch.no_grad():
                recon_pred = dataset.reconstruct_spectrum(reg_pred.detach())
                recon_true = dataset.reconstruct_spectrum(feat)
                loss_recon = ((recon_pred - recon_true) ** 2).mean()
                train_recon_loss += loss_recon.item() * batch_n

        train_reg_loss /= max(train_samples, 1)
        train_recon_loss /= max(train_samples, 1)

        history["train_loss"].append(train_reg_loss)
        history["train_loss_recon"].append(train_recon_loss)

        smoothed = float(np.mean(history["train_loss"][-smooth_window:]))
        scheduler.step(smoothed)

        print(f"Final | Epoch {epoch+1:03d}/{epochs} | K={model.K:3d} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"Train REG: {train_reg_loss:.6f} | "
              f"Train RECON: {train_recon_loss:.3e}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "history": copy.deepcopy(history),
        "K": model.K,
    }, f"{path}/final_pca.pt")

    return model.state_dict(), history


def create_dataloaders(
    dataset: PCADataset,
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


def create_dataloader_kfold(
        dataset: PCADataset,
        seed: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        batch_size: int,
):
    generator = torch.Generator().manual_seed(seed)
    
    train_set = Subset(dataset, train_idx.tolist())
    val_set = Subset(dataset, val_idx.tolist())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader