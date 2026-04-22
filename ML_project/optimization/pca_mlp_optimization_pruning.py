"""
Hyperparameter search for the PcaMLP surrogate (4D or 8D pipeline).

Single Optuna TPE study over all hyperparameters jointly:
    hidden_dim, n_layers, K, p, lr, wd, batch_size

The study is seeded with a known-good config (BEST_KNOWN[PIPELINE]) as
trial 0 so TPE starts from a non-random baseline. A MedianPruner running
at fold-level granularity short-circuits trials whose running CV mean
trails the median of completed trials.

Loss used for training is coefficient-space MSE (inside PcaMLP); trial
score is reconstruction-space RMSE (what we actually care about):
    score = mean(val_recon_rmse) + STD_PENALTY * std(val_recon_rmse)

Every trial uses the same k-fold CV split as the rest of the project; the
held-out test indices are never touched so the test set stays clean for
a separate final evaluation (ensemble-of-folds and/or full-train model)
the user runs outside this loop.

Per-trial fold checkpoints live in a tempdir that's cleaned up after the
trial; nothing persists on disk per trial. The only output is a single
JSON in the same directory as this script containing the winning config,
its per-fold performance summary, and a compact log of every completed
trial's score.

Switch between pipelines via the PIPELINE constant below.
"""

PIPELINE = "8D"  # "4D" or "8D"
NAME = "optimization1"

from util.classes.PCADataset import *
from util.classes.PcaMLP import *
from util.model_eval import *
from util.model_optimization import *

if PIPELINE == "4D":
    from preprocessing.pca_4D import pca_ft, data_config
elif PIPELINE == "8D":
    from preprocessing.pca_8D import pca_ft, data_config
else:
    raise ValueError(f"PIPELINE must be '4D' or '8D'; got {PIPELINE!r}")

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
TRIALS = 100
STD_PENALTY = 0.3

# Known-best configs per pipeline. Enqueued as trial 0 so TPE starts from
# a non-random baseline instead of rediscovering what we already know.
BEST_KNOWN = {
    "4D": {"hidden_dim": 256, "n_layers": 3, "k": 10, "p": 0.036,
           "lr": 7.5e-3, "wd": 4.3e-4, "batch_size": 16},
    "8D": {"hidden_dim": 256, "n_layers": 3, "k": 14, "p": 0.07,
           "lr": 5e-3,   "wd": 8e-4,   "batch_size": 16},
}

# Joint search space. Discrete params are categorical lists so enqueued
# known-best values land cleanly on grid points.
SEARCH_RANGES = {
    "hidden_dim": [128, 192, 256, 384],
    "n_layers":   [2, 3, 4],
    "k":          [10, 12, 14, 16, 18, 20, 25],
    "p":          (0.0, 0.3),           # uniform
    "lr":         (1e-4, 1e-2),         # log-uniform
    "wd":         (1e-5, 1e-2),         # log-uniform
    "batch_size": [16, 32, 64],
}

OUTPUT_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"pca_mlp_{NAME}_pruning_{PIPELINE}.json"
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
    feat = pca_ft['absorption_features'].iloc[:, :K]
    return PCADataset(
        pca_ft['geometry_table'],
        feat,
        pca_ft['absorption_pca'],
        normalize_geom=True,
        normalize_feat=False,
    )


#### DATA + FIXED SPLITS ####
INPUT_DIM = pca_ft['geometry_table'].shape[1]
max_k = max(SEARCH_RANGES["k"])
assert pca_ft['absorption_features'].shape[1] >= max_k, (
    f"pca_ft only has {pca_ft['absorption_features'].shape[1]} PCs; "
    f"sweep requires at least {max_k}. Refit PCA with K>={max_k}."
)

_base_data = build_dataset(max_k)
kf_indices, test_indices = generate_kfold(_base_data, SPLITS, SEED, TEST_RATIO)
print(f"Pipeline: {PIPELINE}  (input_dim={INPUT_DIM})")
print(f"Splits:   {SPLITS}-fold CV on {len(_base_data) - len(test_indices)} samples, "
      f"{len(test_indices)} held out as test (untouched).")


#### PER-CONFIG K-FOLD EVALUATION ####
def evaluate_config(cfg: dict, trial: optuna.trial.Trial | None = None):
    """
    Run k-fold CV for one config. After each fold, reports the running
    mean of val_recon_rmse to `trial` so Optuna's MedianPruner can kill
    trials whose running CV score trails the median of completed trials.
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
        model = PcaMLP(cfg['k'], cfg['hidden_dim'], cfg['n_layers'], cfg['p'],
                       input_dim=INPUT_DIM)

        train_loader, val_loader = create_dataloader_kfold(
            data, SEED, train_idx, val_idx, cfg['batch_size']
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            _, history = train_pca_regression(
                model, data, train_loader, val_loader,
                EPOCHS, cfg['lr'], cfg['wd'], PATIENCE, tmpdir,
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

        # Fold-level pruning: report running mean recon RMSE so far.
        if trial is not None:
            running_mean = float(np.mean(fold_metrics["best_val_recon_rmse"]))
            trial.report(running_mean, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    summary = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in fold_metrics.items()
    }
    score = (
        summary["best_val_recon_rmse"]["mean"]
        + STD_PENALTY * summary["best_val_recon_rmse"]["std"]
    )
    return fold_metrics, summary, float(score)


# Cache completed trials so repeated configs (e.g. the enqueued best
# reappearing via TPE) don't re-run.
trial_cache: dict = {}


def run_trial(cfg: dict, trial: optuna.trial.Trial | None = None) -> dict:
    name = config_name(cfg)
    if name in trial_cache:
        print(f">>> Trial (cached): {name}")
        return trial_cache[name]

    print(f"\n>>> Trial: {name}")
    _, summary, score = evaluate_config(cfg, trial=trial)
    record = {
        "name": name,
        "config": dict(cfg),
        "score": score,
        "summary": summary,
    }
    trial_cache[name] = record
    print(f"    score = {score:.6g}  "
          f"(mean_rmse={summary['best_val_recon_rmse']['mean']:.6g}, "
          f"std={summary['best_val_recon_rmse']['std']:.6g}, "
          f"peak_loc_mae={summary['peak_loc_error_um']['mean']:.6g} µm)")
    return record


#### OPTUNA JOINT SEARCH ####
print("\n" + "=" * 72 + f"\nOptuna TPE — {TRIALS} trials (fold-level MedianPruner)\n" + "=" * 72)


def objective(trial: optuna.trial.Trial) -> float:
    cfg = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", SEARCH_RANGES["hidden_dim"]),
        "n_layers":   trial.suggest_categorical("n_layers",   SEARCH_RANGES["n_layers"]),
        "k":          trial.suggest_categorical("k",          SEARCH_RANGES["k"]),
        "p":          trial.suggest_float("p",  *SEARCH_RANGES["p"]),
        "lr":         trial.suggest_float("lr", *SEARCH_RANGES["lr"], log=True),
        "wd":         trial.suggest_float("wd", *SEARCH_RANGES["wd"], log=True),
        "batch_size": trial.suggest_categorical("batch_size", SEARCH_RANGES["batch_size"]),
    }
    try:
        rec = run_trial(cfg, trial=trial)
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"    trial failed: {e}")
        return float("inf")
    return rec["score"]


sampler = optuna.samplers.TPESampler(seed=SEED)
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

if PIPELINE in BEST_KNOWN:
    study.enqueue_trial(BEST_KNOWN[PIPELINE])

study.optimize(objective, n_trials=TRIALS)


#### WINNER + SAVE ####
n_pruned   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)

all_trials = list(trial_cache.values())
if not all_trials:
    raise RuntimeError("No completed trials — every trial was pruned or failed.")
best_trial = min(all_trials, key=lambda r: r["score"])

trial_log = [
    {"name": r["name"], "config": r["config"], "score": r["score"]}
    for r in all_trials
]

result = {
    "pipeline": PIPELINE,
    "best_config": best_trial["config"],
    "best_name": best_trial["name"],
    "best_score": best_trial["score"],
    "best_summary": best_trial["summary"],
    "meta": {
        "seed": SEED,
        "test_ratio": TEST_RATIO,
        "splits": SPLITS,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "trials": TRIALS,
        "trials_completed": n_complete,
        "trials_pruned": n_pruned,
        "std_penalty": STD_PENALTY,
        "input_dim": INPUT_DIM,
        "search_ranges": SEARCH_RANGES,
        "best_known_enqueued": BEST_KNOWN.get(PIPELINE),
    },
    "trial_log": trial_log,
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=4, default=_json_default)

print("\n" + "=" * 72)
print(f"Pipeline:         {PIPELINE}")
print(f"Trials completed: {n_complete}")
print(f"Trials pruned:    {n_pruned}")
print(f"Best config:      {best_trial['name']}")
print(f"Best score:       {best_trial['score']:.6g}")
print(f"Saved:            {OUTPUT_JSON}")
print("=" * 72)
