import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
from util.classes.PCAPeakDataset import PCAPeakDataset
import numpy as np


class PcaPeakMLP(nn.Module):
    """
    Multi-task MLP surrogate:
      Geometry (input_dim) --> shared trunk --> two heads:
        - pca_head:  (K,)  PCA coefficients of the spectrum
        - peak_head: (3N,) per-peak (lambda, gamma, A) targets, flattened
                     rank 0 -> rank N-1 (highest-amplitude peak first)

    K is an init argument so the same class supports sweeps over PCA
    truncations. n_peaks controls the auxiliary head size only; the
    training loop handles reshaping and per-peak masking.
    """

    def __init__(
            self,
            K: int,
            hidden_dim: int,
            n_layers: int,
            p: float,
            n_peaks: int,
            input_dim: int = 4):
        super().__init__()

        assert isinstance(n_peaks, int) and 1 <= n_peaks <= 4, \
            "n_peaks must be an integer in [1, 4]"

        self.K = K
        self.n_peaks = n_peaks

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
        self.pca_head  = nn.Linear(hidden_dim, K)
        self.peak_head = nn.Linear(hidden_dim, 3 * n_peaks)

    def forward(self, x):
        h = self.shared(x)
        pca = self.pca_head(h)
        peaks = self.peak_head(h)
        return (pca, peaks)


def train_pca_regression(
        model: PcaPeakMLP,
        dataset: PCAPeakDataset,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        wd: float,
        patience: int,
        path: str,
        peak_loss_weight: float = 0.1,
        on_epoch_end=None):
    """
    Train PcaPeakMLP to regress K PCA coefficients and max-amplitude peak
    targets from 4D geometry using multi-task learning.

    Loss = loss_pca + peak_loss_weight * masked_peak_loss

    The peak loss is masked: only samples with a detected peak (mask==1)
    contribute. Zero-padded no-peak rows are ignored.

    The dataset is passed in so reconstruction-space MSE can be logged
    each epoch alongside the coefficient-space training loss.
    """

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    mse = nn.MSELoss(reduction='none')

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_loss_pca": [],
        "train_loss_peak": [],
        "val_loss_pca": [],
        "val_loss_peak": [],
        "val_loss_recon": [],
        "stop_epoch": 0,
        "K": model.K
    }

    best_recon_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss_pca = 0.0
        train_loss_peak = 0.0
        train_samples = 0

        for geom, pca_feat, amp_feat, mask in train_loader:
            optimizer.zero_grad()
            pca_pred, peak_pred = model(geom)

            # PCA head loss: mean MSE over all K coefficients, all samples
            loss_pca = mse(pca_pred, pca_feat).mean()

            # Peak head loss: reshape (B, 3N) -> (B, N, 3); take mean over the
            # 3 channels (lambda/gamma/A) to get per-peak MSE (B, N); apply the
            # per-peak mask so absent peaks contribute nothing; average over
            # the number of present peaks across the batch.
            B = peak_pred.shape[0]
            peak_pred_r = peak_pred.view(B, model.n_peaks, 3)
            amp_feat_r  = amp_feat.view(B, model.n_peaks, 3)
            loss_peak_per_peak = ((peak_pred_r - amp_feat_r) ** 2).mean(dim=2)  # (B, N)
            loss_peak_masked = (loss_peak_per_peak * mask).sum() / mask.sum().clamp(min=1.0)

            # Combined loss
            loss_total = loss_pca + peak_loss_weight * loss_peak_masked
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_n = geom.shape[0]
            train_loss_pca += loss_pca.item() * batch_n
            train_loss_peak += loss_peak_masked.item() * batch_n
            train_samples += batch_n

        # Validation loop
        model.eval()
        val_loss_pca = 0.0
        val_loss_peak = 0.0
        val_recon_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for geom, pca_feat, amp_feat, mask in val_loader:
                pca_pred, peak_pred = model(geom)

                loss_pca = mse(pca_pred, pca_feat).mean()

                B = peak_pred.shape[0]
                peak_pred_r = peak_pred.view(B, model.n_peaks, 3)
                amp_feat_r  = amp_feat.view(B, model.n_peaks, 3)
                loss_peak_per_peak = ((peak_pred_r - amp_feat_r) ** 2).mean(dim=2)
                loss_peak_masked = (loss_peak_per_peak * mask).sum() / mask.sum().clamp(min=1.0)

                recon_pred = dataset.reconstruct_spectrum(pca_pred)
                recon_true = dataset.reconstruct_spectrum(pca_feat)
                loss_recon = ((recon_pred - recon_true) ** 2).mean()

                batch_n = geom.shape[0]
                val_loss_pca += loss_pca.item() * batch_n
                val_loss_peak += loss_peak_masked.item() * batch_n
                val_recon_loss += loss_recon.item() * batch_n
                val_samples += batch_n

        train_loss_pca /= max(train_samples, 1)
        val_loss_pca /= max(val_samples, 1)
        train_loss_peak /= max(train_samples, 1)
        val_loss_peak /= max(val_samples, 1)
        val_recon_loss /= max(val_samples, 1)

        # Total loss for early stopping
        train_loss_total = train_loss_pca + peak_loss_weight * train_loss_peak
        val_loss_total = val_loss_pca + peak_loss_weight * val_loss_peak

        history["train_loss"].append(train_loss_total)
        history["val_loss"].append(val_loss_total)
        history["train_loss_pca"].append(train_loss_pca)
        history["val_loss_pca"].append(val_loss_pca)
        history["train_loss_peak"].append(train_loss_peak)
        history["val_loss_peak"].append(val_loss_peak)
        history["val_loss_recon"].append(val_recon_loss)

        # Early stopping and checkpoint selection on reconstruction-space MSE
        # (the head we report on), not on the combined coeff+peak loss. This
        # makes the peak-aware model directly comparable to the pure PCA model
        # in HPO/ablation: both are selected by val recon MSE.
        if val_recon_loss < best_recon_loss:
            best_recon_loss = val_recon_loss
            patience_counter = 0
            history["stop_epoch"] = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "history": copy.deepcopy(history),
                "K": model.K
            }, f"{path}/best_pca_peak.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        scheduler.step(val_recon_loss)

        if on_epoch_end is not None:
            stop = on_epoch_end(epoch, val_recon_loss, best_recon_loss, val_loss_pca)
            if stop:
                break

    checkpoint = torch.load(f"{path}/best_pca_peak.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.state_dict(), checkpoint["history"]


def train_pca_regression_final(
        model: PcaPeakMLP,
        dataset: PCAPeakDataset,
        train_loader: DataLoader,
        epochs: int,
        lr: float,
        wd: float,
        path: str,
        peak_loss_weight: float = 0.1):
    """
    Fixed-epoch final training: no val loop, no early stopping. Trains on the
    full (non-test) set so the held-out test split stays clean for a single
    post-training evaluation. `epochs` should be informed by the mean best
    epoch observed during k-fold CV.

    Loss = loss_pca + peak_loss_weight * masked_peak_loss
    """

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # Same scheduler dynamics as the fold trainer, but keyed to a smoothed
    # train loss (no val set here). patience bumped 10 -> 15 and a small
    # rel threshold to absorb train-loss noise from dropout.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15,
        threshold=1e-3, threshold_mode='rel'
    )
    mse = nn.MSELoss(reduction='none')

    history = {
        "train_loss": [],
        "train_loss_pca": [],
        "train_loss_peak": [],
        "train_loss_recon": [],
        "epochs": epochs,
        "K": model.K,
    }

    smooth_window = 5

    for epoch in range(epochs):
        model.train()
        train_loss_pca = 0.0
        train_loss_peak = 0.0
        train_recon_loss = 0.0
        train_samples = 0

        for geom, pca_feat, amp_feat, mask in train_loader:
            optimizer.zero_grad()
            pca_pred, peak_pred = model(geom)

            loss_pca = mse(pca_pred, pca_feat).mean()

            B = peak_pred.shape[0]
            peak_pred_r = peak_pred.view(B, model.n_peaks, 3)
            amp_feat_r  = amp_feat.view(B, model.n_peaks, 3)
            loss_peak_per_peak = ((peak_pred_r - amp_feat_r) ** 2).mean(dim=2)
            loss_peak_masked = (loss_peak_per_peak * mask).sum() / mask.sum().clamp(min=1.0)

            loss_total = loss_pca + peak_loss_weight * loss_peak_masked
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_n = geom.shape[0]
            train_loss_pca += loss_pca.item() * batch_n
            train_loss_peak += loss_peak_masked.item() * batch_n
            train_samples += batch_n

            with torch.no_grad():
                recon_pred = dataset.reconstruct_spectrum(pca_pred.detach())
                recon_true = dataset.reconstruct_spectrum(pca_feat)
                loss_recon = ((recon_pred - recon_true) ** 2).mean()
                train_recon_loss += loss_recon.item() * batch_n

        train_loss_pca /= max(train_samples, 1)
        train_loss_peak /= max(train_samples, 1)
        train_recon_loss /= max(train_samples, 1)
        train_loss_total = train_loss_pca + peak_loss_weight * train_loss_peak

        history["train_loss"].append(train_loss_total)
        history["train_loss_pca"].append(train_loss_pca)
        history["train_loss_peak"].append(train_loss_peak)
        history["train_loss_recon"].append(train_recon_loss)

        smoothed = float(np.mean(history["train_loss"][-smooth_window:]))
        scheduler.step(smoothed)

        print(f"Final | Epoch {epoch+1:03d}/{epochs} | K={model.K:3d} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"Loss PCA: {train_loss_pca:.6f} | "
              f"Loss Peak: {train_loss_peak:.6f} | "
              f"RECON: {train_recon_loss:.3e}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "history": copy.deepcopy(history),
        "K": model.K,
    }, f"{path}/final_pca_peak.pt")

    return model.state_dict(), history


def train_pca_regression_final_patience(
        model: PcaPeakMLP,
        dataset: PCAPeakDataset,
        train_loader: DataLoader,
        max_epochs: int,
        lr: float,
        wd: float,
        path: str,
        patience: int = 30,
        peak_loss_weight: float = 0.1):
    """
    Patience-based final training on smoothed training loss. No val set.
    Stops when the 5-epoch smoothed train loss has not improved for `patience`
    consecutive epochs and saves the best-seen checkpoint.

    Loss = loss_pca + peak_loss_weight * masked_peak_loss
    """

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15,
        threshold=1e-3, threshold_mode='rel'
    )
    mse = nn.MSELoss(reduction='none')

    history = {
        "train_loss": [],
        "train_loss_pca": [],
        "train_loss_peak": [],
        "train_loss_recon": [],
        "stop_epoch": 0,
        "K": model.K,
    }

    smooth_window = 5
    best_smoothed = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss_pca = 0.0
        train_loss_peak = 0.0
        train_recon_loss = 0.0
        train_samples = 0

        for geom, pca_feat, amp_feat, mask in train_loader:
            optimizer.zero_grad()
            pca_pred, peak_pred = model(geom)

            loss_pca = mse(pca_pred, pca_feat).mean()

            B = peak_pred.shape[0]
            peak_pred_r = peak_pred.view(B, model.n_peaks, 3)
            amp_feat_r  = amp_feat.view(B, model.n_peaks, 3)
            loss_peak_per_peak = ((peak_pred_r - amp_feat_r) ** 2).mean(dim=2)
            loss_peak_masked = (loss_peak_per_peak * mask).sum() / mask.sum().clamp(min=1.0)

            loss_total = loss_pca + peak_loss_weight * loss_peak_masked
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_n = geom.shape[0]
            train_loss_pca += loss_pca.item() * batch_n
            train_loss_peak += loss_peak_masked.item() * batch_n
            train_samples += batch_n

            with torch.no_grad():
                recon_pred = dataset.reconstruct_spectrum(pca_pred.detach())
                recon_true = dataset.reconstruct_spectrum(pca_feat)
                loss_recon = ((recon_pred - recon_true) ** 2).mean()
                train_recon_loss += loss_recon.item() * batch_n

        train_loss_pca /= max(train_samples, 1)
        train_loss_peak /= max(train_samples, 1)
        train_recon_loss /= max(train_samples, 1)
        train_loss_total = train_loss_pca + peak_loss_weight * train_loss_peak

        history["train_loss"].append(train_loss_total)
        history["train_loss_pca"].append(train_loss_pca)
        history["train_loss_peak"].append(train_loss_peak)
        history["train_loss_recon"].append(train_recon_loss)

        smoothed = float(np.mean(history["train_loss"][-smooth_window:]))
        scheduler.step(smoothed)

        if smoothed < best_smoothed:
            best_smoothed = smoothed
            patience_counter = 0
            history["stop_epoch"] = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "history": copy.deepcopy(history),
                "K": model.K,
            }, f"{path}/final_pca_peak_patience.pt")
        else:
            patience_counter += 1

        print(f"Patience | Epoch {epoch+1:04d} | K={model.K:3d} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"Loss PCA: {train_loss_pca:.6f} | "
              f"Loss Peak: {train_loss_peak:.6f} | "
              f"RECON: {train_recon_loss:.3e} | "
              f"patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping on train loss at epoch {epoch+1}")
            break

    checkpoint = torch.load(f"{path}/final_pca_peak_patience.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.state_dict(), checkpoint["history"]


def create_dataloaders(
    dataset: PCAPeakDataset,
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
        dataset: PCAPeakDataset,
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