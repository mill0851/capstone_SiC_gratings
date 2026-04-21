"""
Two-phase hyperparameter search for the PcaMLP surrogate.

Phase 1 — Coarse axis sweeps
    Vary K, hidden_dim, n_layers independently from a fixed baseline; keep
    the best value on each axis by score = mean(val_recon_mse) + STD_PENALTY*std.

Phase 2 — Optuna TPE joint search
    Architecture locked from Phase 1. Search lr, wd, p, batch_size jointly.

Every trial uses the same k-fold CV split as the rest of the project; the
held-out test indices are never touched. Per-trial kfold_metrics.json is
saved under trained_models/pca/{name}/config/ for inspection. No model
weights are saved. Final winner + full sweep log is written as JSON in the
same directory as this script.
"""

from util.classes.PCADataset import *
from util.classes.PcaMLP import *
from util.model_eval import *
from util.model_optimization import *
from preprocessing.pca import pca_ft, data_config

import json
import os
import math
import tempfile
import numpy as np
import torch
import optuna


#### KNOBS ####
SEED = 1234
TEST_RATIO = 0.15
SPLITS = 4
EPOCHS = 500
PATIENCE = 100

PHASE1_FIXED = {
    "hidden_dim": 128,
    "n_layers": 4,
    "k": 20,
    "p": 0.15,
    "lr": 5e-3,
    "wd": 1e-3,
    "batch_size": 32,
}
K_VALUES = [10, 15, 20, 30, 40]
HIDDEN_DIM_VALUES = [32, 64, 128, 256]
N_LAYERS_VALUES = [2, 3, 4, 5]

PHASE2_TRIALS = 30
PHASE2_RANGES = {
    "lr": (1e-4, 1e-2),          # log-uniform
    "wd": (1e-5, 1e-2),          # log-uniform
    "p":  (0.0, 0.3),            # uniform
    "batch_size": [16, 32, 64],  # categorical
}

STD_PENALTY = 0.5
TRAINED_ROOT = "trained_models/pca"
OUTPUT_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pca_mlp_optimization.json"
)


#### HELPERS ####
def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    raise TypeError(f"Not JSON serializable: {type(o).__name__}")


def fmt_sci(x: float) -> str:
    mantissa, exp = f"{x:.2e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def config_name(cfg: dict) -> str:
    return (
        f"dim{cfg['hidden_dim']}_layers{cfg['n_layers']}_k{cfg['k']}"
        f"_p{int(round(cfg['p']*100)):03d}"
        f"_lr{fmt_sci(cfg['lr'])}_wd{fmt_sci(cfg['wd'])}"
        f"_bs{cfg['batch_size']}"
    )


def build_dataset(K: int) -> PCADataset:
    """Build a PCADataset truncated to the first K PCs."""
    feat = pca_ft['absorption_features'].iloc[:, :K]
    return PCADataset(
        pca_ft['geometry_table'],
        feat,
        pca_ft['absorption_pca'],
        normalize_geom=True,
        normalize_feat=False,
    )


#### DATA + FIXED SPLITS ####
assert pca_ft['absorption_features'].shape[1] >= max(K_VALUES), (
    f"pca_ft only has {pca_ft['absorption_features'].shape[1]} PCs; "
    f"sweep requires at least {max(K_VALUES)}. Refit PCA with K>={max(K_VALUES)}."
)

# Splits depend only on sample count + seed, so any dataset works here.
_base_data = build_dataset(max(K_VALUES))
kf_indices, test_indices = generate_kfold(_base_data, SPLITS, SEED, TEST_RATIO)
print(f"Splits: {SPLITS}-fold CV on {len(_base_data) - len(test_indices)} samples, "
      f"{len(test_indices)} held out as test (untouched).")


#### PER-CONFIG K-FOLD EVALUATION ####
def evaluate_config(cfg: dict):
    """
    Run k-fold CV for one config. Returns (fold_metrics, summary, score).
    train_pca_regression writes a checkpoint per fold into a tempdir that
    is cleaned up automatically — nothing persists on disk.
    """
    data = build_dataset(cfg['k'])
    wl_tensor = torch.tensor(np.asarray(data.wl), dtype=torch.float32)

    fold_metrics = {
        "best_val_coeff_mse": [],
        "best_val_recon_mse": [],
        "best_val_recon_rmse": [],
        "peak_mae": [],
        "peak_loc_error_um": [],
        "train_val_gap_recon": [],
        "best_epoch": [],
        "epochs_trained": [],
    }

    for i, (train_idx, val_idx) in enumerate(kf_indices):
        model = PcaMLP(cfg['k'], cfg['hidden_dim'], cfg['n_layers'], cfg['p'])

        train_loader, val_loader = create_dataloader_kfold(
            data, SEED, train_idx, val_idx, cfg['batch_size']
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            _, history = train_pca_regression(
                model,
                data,
                train_loader,
                val_loader,
                EPOCHS,
                cfg['lr'],
                cfg['wd'],
                PATIENCE,
                tmpdir,
            )

        best_ep = history["stop_epoch"]
        best_val_coeff = history["val_loss"][best_ep]
        best_val_recon = history["val_loss_recon"][best_ep]
        best_train_recon = history["train_loss_recon"][best_ep]

        model.eval()
        peak_abs_errs = []
        peak_loc_errs = []
        with torch.no_grad():
            for geom, feat in val_loader:
                pred_spec = data.reconstruct_spectrum(model(geom))
                true_spec = data.reconstruct_spectrum(feat)
                pred_peak = pred_spec.max(dim=-1)
                true_peak = true_spec.max(dim=-1)
                peak_abs_errs.append((pred_peak.values - true_peak.values).abs())
                peak_loc_errs.append(
                    (wl_tensor[pred_peak.indices] - wl_tensor[true_peak.indices]).abs()
                )
        peak_mae = torch.cat(peak_abs_errs).mean().item()
        peak_loc_err = torch.cat(peak_loc_errs).mean().item()

        fold_metrics["best_val_coeff_mse"].append(float(best_val_coeff))
        fold_metrics["best_val_recon_mse"].append(float(best_val_recon))
        fold_metrics["best_val_recon_rmse"].append(float(math.sqrt(best_val_recon)))
        fold_metrics["peak_mae"].append(float(peak_mae))
        fold_metrics["peak_loc_error_um"].append(float(peak_loc_err))
        fold_metrics["train_val_gap_recon"].append(float(best_val_recon - best_train_recon))
        fold_metrics["best_epoch"].append(int(best_ep))
        fold_metrics["epochs_trained"].append(int(len(history["val_loss"])))

    summary = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in fold_metrics.items()
    }
    score = (
        summary["best_val_recon_mse"]["mean"]
        + STD_PENALTY * summary["best_val_recon_mse"]["std"]
    )
    return fold_metrics, summary, float(score)


def save_trial_artifacts(cfg, fold_metrics, summary, score) -> str:
    name = config_name(cfg)
    run_dir = f"{TRAINED_ROOT}/{name}/config"
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "config": cfg,
        "per_fold": fold_metrics,
        "summary": summary,
        "score": score,
    }
    with open(f"{run_dir}/kfold_metrics.json", "w") as f:
        json.dump(payload, f, indent=4, default=_json_default)
    return name


# Cache so repeated configs (e.g. baseline hit by multiple axis sweeps) only run once.
trial_cache: dict = {}


def run_trial(cfg: dict) -> dict:
    name = config_name(cfg)
    if name in trial_cache:
        print(f">>> Trial (cached): {name}")
        return trial_cache[name]

    print(f"\n>>> Trial: {name}")
    fold_metrics, summary, score = evaluate_config(cfg)
    save_trial_artifacts(cfg, fold_metrics, summary, score)
    record = {
        "name": name,
        "config": dict(cfg),
        "score": score,
        "summary": summary,
    }
    trial_cache[name] = record
    print(f"    score = {score:.6g}  "
          f"(mean={summary['best_val_recon_mse']['mean']:.6g}, "
          f"std={summary['best_val_recon_mse']['std']:.6g})")
    return record


#### PHASE 1 — AXIS SWEEPS ####
print("\n" + "=" * 72 + "\nPhase 1: axis sweeps (K, hidden_dim, n_layers)\n" + "=" * 72)

phase1 = {"K": [], "hidden_dim": [], "n_layers": []}

for K in K_VALUES:
    phase1["K"].append(run_trial({**PHASE1_FIXED, "k": K}))

for hd in HIDDEN_DIM_VALUES:
    phase1["hidden_dim"].append(run_trial({**PHASE1_FIXED, "hidden_dim": hd}))

for nl in N_LAYERS_VALUES:
    phase1["n_layers"].append(run_trial({**PHASE1_FIXED, "n_layers": nl}))

best_k  = min(phase1["K"],          key=lambda r: r["score"])["config"]["k"]
best_hd = min(phase1["hidden_dim"], key=lambda r: r["score"])["config"]["hidden_dim"]
best_nl = min(phase1["n_layers"],   key=lambda r: r["score"])["config"]["n_layers"]
print(f"\nPhase 1 axis winners: K={best_k}, hidden_dim={best_hd}, n_layers={best_nl}")


#### PHASE 2 — OPTUNA JOINT SEARCH ####
print("\n" + "=" * 72 + f"\nPhase 2: Optuna TPE — {PHASE2_TRIALS} trials\n" + "=" * 72)

phase2: list = []


def objective(trial: optuna.trial.Trial) -> float:
    cfg = {
        "hidden_dim": best_hd,
        "n_layers": best_nl,
        "k": best_k,
        "p":  trial.suggest_float("p",  *PHASE2_RANGES["p"]),
        "lr": trial.suggest_float("lr", *PHASE2_RANGES["lr"], log=True),
        "wd": trial.suggest_float("wd", *PHASE2_RANGES["wd"], log=True),
        "batch_size": trial.suggest_categorical("batch_size", PHASE2_RANGES["batch_size"]),
    }
    try:
        rec = run_trial(cfg)
    except Exception as e:
        print(f"    trial failed: {e}")
        return float("inf")
    phase2.append(rec)
    return rec["score"]


sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=PHASE2_TRIALS)


#### WINNER + SAVE ####
all_trials = [*phase1["K"], *phase1["hidden_dim"], *phase1["n_layers"], *phase2]
best_trial = min(all_trials, key=lambda r: r["score"])

result = {
    "best_config": best_trial["config"],
    "best_score": best_trial["score"],
    "best_name": best_trial["name"],
    "phase1": phase1,
    "phase2": phase2,
    "meta": {
        "seed": SEED,
        "test_ratio": TEST_RATIO,
        "splits": SPLITS,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "std_penalty": STD_PENALTY,
        "phase1_fixed": PHASE1_FIXED,
        "phase1_ranges": {
            "k": K_VALUES,
            "hidden_dim": HIDDEN_DIM_VALUES,
            "n_layers": N_LAYERS_VALUES,
        },
        "phase2_ranges": PHASE2_RANGES,
        "phase2_trials": PHASE2_TRIALS,
    },
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=4, default=_json_default)

print("\n" + "=" * 72)
print(f"Best config: {best_trial['name']}")
print(f"Best score:  {best_trial['score']:.6g}")
print(f"Saved:       {OUTPUT_JSON}")
print("=" * 72)
