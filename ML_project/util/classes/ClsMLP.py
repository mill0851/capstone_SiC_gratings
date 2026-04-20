import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
from util.classes.Phase1Dataset import Phase1Dataset


class ClsMLP(nn.Module):
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
        self.classification_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        cls = self.classification_head(h)
        return cls.squeeze()

def train_classification(
        model: ClsMLP,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        wd: float,
        patience: int,
        path: str):
    
    # Config for training loop
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()

    # History dict for loss graphs
    history = {
        "train_loss": [],
        "val_loss": [],
        "stop_epoch": 0
    }

    best_cls_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training Loop
        model.train()
        train_cls_loss = 0
        train_cls_samples = 0

        # Forward pass each batch
        for geom, feat, mask in train_loader:

            # Zero + Forward
            optimizer.zero_grad()
            cls_logits = model(geom)

            # Loss + Backprop
            loss_cls = bce(cls_logits, mask)
            loss_cls.backward()
            optimizer.step()

            # Logging
            batch_size = geom.size(0)
            train_cls_loss += (loss_cls.item() * batch_size)
            train_cls_samples += batch_size

        # Evaluation Loop
        model.eval()
        val_cls_loss = 0
        val_cls_samples = 0

        with torch.no_grad():
            for geom, feat, mask in val_loader:

                # Forward
                cls_logits = model(geom)

                # Loss
                loss_cls = bce(cls_logits, mask)

                # Logging
                batch_size = geom.size(0)
                val_cls_loss += (loss_cls.item() * batch_size)
                val_cls_samples += batch_size

        # Convert to per-sample epoch-average losses
        val_cls_loss /= val_cls_samples
        train_cls_loss /= train_cls_samples

        # Update loss history
        history["train_loss"].append(train_cls_loss)
        history["val_loss"].append(val_cls_loss)

        # Print current status
        print(f"Epoch {epoch+1:03d} | "
              f"Train CLS: {history['train_loss'][-1]:.4f} | "
              f"Val CLS: {history['val_loss'][-1]:.4f} | "
              f"Patience: {patience_counter}/{patience}")

        # Update best loss states
        if val_cls_loss < best_cls_loss:
            best_cls_loss = val_cls_loss
            patience_counter = 0
            history["stop_epoch"] = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "history": copy.deepcopy(history)
            }, f"{path}/best_cls.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    checkpoint = torch.load(f"{path}/best_cls.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.state_dict(), checkpoint["history"]

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

