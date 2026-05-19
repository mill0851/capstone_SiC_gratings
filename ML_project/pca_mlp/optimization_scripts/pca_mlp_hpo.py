"""
PCA Surrogate — Hyperparameter Optimization (script form).

Optuna TPE + Hyperband over three model variants:

    pca_4D        PcaMLP        4D geom    coeff-MSE (uniform)
    pca_n_peak_4D PcaMLP        4D geom    coeff-MSE, per-sample peak-count weighting
    pca_phys_4D   PcaPeakMLP    4D geom    coeff-MSE + pw * masked-peak-MSE

K is fixed at 25 (no sweep). TPE then uses all 60 trials per variant purely
for architecture/regularisation tuning.

Per-trial scoring (clean cross-validated estimate):
  1. Train K_FOLDS fold models with the proposed config.
  2. Each fold model is evaluated only on the val set it withheld (true OOF).
  3. Trial score = mean(per_fold_coeff_rmse) + STD_PENALTY * std(...).

Ensemble metrics (all fold models on each val set) are also logged for
reference but are not used for scoring — they are contaminated because
K-1 of K models trained on each val fold.

Reported metrics (per-fold, OOF): coeff MSE, coeff RMSE, recon MSE, recon RMSE,
peak MAE, peak location MAE.

Pruning: per-epoch trial.report inside fold 0 (on recon RMSE) lets
HyperbandPruner kill bad trials before they finish a single fold.
"""

import os
import sys
import json
import math
import gc
import time
import pickle
import tempfile

import numpy as np
import torch
import optuna

_ML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ML_ROOT not in sys.path:
    sys.path.insert(0, _ML_ROOT)
os.chdir(_ML_ROOT)

from util.classes.PCADataset import PCADataset
from util.classes.PCAPeakDataset import PCAPeakDataset
from util.classes.PcaMLP import (
    PcaMLP,
    train_pca_regression as train_pca_only,
    create_dataloader_kfold as kfold_loaders_pca,
)
from util.classes.PcaPeakMLP import (
    PcaPeakMLP,
    train_pca_regression as train_pca_peak,
    create_dataloader_kfold as kfold_loaders_pca_peak,
)
from util.model_optimization import generate_kfold

from preprocessing.pca_4D import pca_ft_4D, data_config_4D  # noqa: F401


# ============================== CONFIG ==============================

# WHAT TO RUN
MODEL_VARIANTS = ['pca_4D', 'pca_n_peak_4D', 'pca_phys_4D']
K              = 25          # fixed — no sweep
NAME           = 'optimization_v3'

# PEAK-COUNT WEIGHTING (pca_n_peak_4D)
PEAK_PROMINENCE = 0.05       # fraction of per-spectrum max for find_peaks

# REPRODUCIBILITY
SEED           = 1234

# CV + TRAINING
TEST_RATIO     = 0.15
K_FOLDS        = 4
EPOCHS         = 300
PATIENCE       = 30

# OPTUNA
TRIALS         = 40
STD_PENALTY    = 0.1

# SEARCH RANGES
SEARCH_RANGES = {
    'hidden_dim':        [128, 256, 512],
    'n_layers':          [3, 4],
    'p':                 (0.0, 0.3),
    'lr':                (1e-4, 1e-2),
    'wd':                (1e-5, 1e-2),
    'batch_size':        [32, 64],
    'peak_weight_alpha': [1.0, 1.5, 2.0, 3.0],   # pca_n_peak_4D only
    'peak_loss_weight':  (1e-3, 1.0),             # pca_phys_4D only
}

# SEED CONFIGS (enqueued as trial 0 per variant)
BEST_KNOWN = {
    'pca_4D':        {'hidden_dim': 256, 'n_layers': 3, 'p': 0.036,    'lr': 7.5e-3,  'wd': 4.3e-4,    'batch_size': 32},
    'pca_n_peak_4D': {'hidden_dim': 512, 'n_layers': 3, 'p': 0.008894, 'lr': 1.541e-3,'wd': 2.1989e-5, 'batch_size': 32, 'peak_weight_alpha': 2.0},
    'pca_phys_4D':   {'hidden_dim': 256, 'n_layers': 3, 'p': 0.05,     'lr': 5e-3,    'wd': 5e-4,      'batch_size': 32, 'peak_loss_weight': 0.1},
}


# ============================== HELPERS ==============================

def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    raise TypeError(f"Not JSON serializable: {type(o).__name__}")


def fmt_sci(x: float) -> str:
    m, e = f"{x:.2e}".split('e')
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"


def config_name(cfg: dict) -> str:
    base = (f"dim{cfg['hidden_dim']}_layers{cfg['n_layers']}"
            f"_p{int(round(cfg['p']*100)):03d}"
            f"_lr{fmt_sci(cfg['lr'])}_wd{fmt_sci(cfg['wd'])}_bs{cfg['batch_size']}")
    if 'peak_weight_alpha' in cfg:
        base += f"_alpha{cfg['peak_weight_alpha']:.1f}"
    if 'peak_loss_weight' in cfg:
        base += f"_pw{fmt_sci(cfg['peak_loss_weight'])}"
    return base


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(variant: str, variant_info: dict, multi_peak_ft: dict):
    """Construct the right dataset for a variant at the fixed K."""
    info = variant_info[variant]
    feat = info['feat']
    geom_df = feat['geometry_table']
    pca_coeffs = feat['absorption_features'].iloc[:, :K]
    pca_arts = feat['absorption_pca']

    if not info['has_peak']:
        return PCADataset(
            geom_df, pca_coeffs, pca_arts,
            normalize_geom=True, normalize_feat=False,
        )
    amp_df = multi_peak_ft['absorption_features'].loc[geom_df.index]
    return PCAPeakDataset(
        geom_df, pca_coeffs, amp_df, pca_arts,
        normalize_geom=True, normalize_pca=False, normalize_amp=True,
    )


def suggest_config(trial: optuna.trial.Trial, has_peak: bool, n_peak_weighted: bool) -> dict:
    cfg = {
        'hidden_dim': trial.suggest_categorical('hidden_dim', SEARCH_RANGES['hidden_dim']),
        'n_layers':   trial.suggest_categorical('n_layers',   SEARCH_RANGES['n_layers']),
        'p':          trial.suggest_float('p',  *SEARCH_RANGES['p']),
        'lr':         trial.suggest_float('lr', *SEARCH_RANGES['lr'], log=True),
        'wd':         trial.suggest_float('wd', *SEARCH_RANGES['wd'], log=True),
        'batch_size': trial.suggest_categorical('batch_size', SEARCH_RANGES['batch_size']),
    }
    if n_peak_weighted:
        cfg['peak_weight_alpha'] = trial.suggest_categorical(
            'peak_weight_alpha', SEARCH_RANGES['peak_weight_alpha']
        )
    if has_peak:
        cfg['peak_loss_weight'] = trial.suggest_float(
            'peak_loss_weight', *SEARCH_RANGES['peak_loss_weight'], log=True
        )
    return cfg


# ============================== EVAL ==============================

def evaluate_config(
        cfg: dict,
        variant: str,
        kf_indices: list,
        variant_info: dict,
        multi_peak_ft: dict,
        trial: optuna.trial.Trial | None = None,
):
    """Train K_FOLDS fold models and return a clean cross-validated score.

    Trial score = mean(per-fold val coeff RMSE) + STD_PENALTY * std(...),
    where each fold's RMSE comes only from the model that withheld that fold
    (true out-of-fold estimate, no contamination).

    Ensemble metrics (all fold models predicting each val set) are also
    computed and logged for reference but are NOT used for scoring.
    """
    info = variant_info[variant]
    has_peak = info['has_peak']
    n_peak_weighted = info.get('n_peak_weighted', False)
    input_dim = info['input_dim']

    data = build_dataset(variant, variant_info, multi_peak_ft)

    if n_peak_weighted:
        alpha = cfg['peak_weight_alpha']
        raw_counts = info['peak_counts']
        weights = raw_counts ** alpha
        weights = weights / weights.mean()
        data.set_sample_weights(weights)

    wl_tensor = torch.tensor(np.asarray(data.wl), dtype=torch.float32)

    per_fold_single = {
        'best_val_coeff_mse':  [],
        'best_val_coeff_rmse': [],
        'best_val_recon_mse':  [],
        'best_val_recon_rmse': [],
        'peak_mae':            [],
        'peak_loc_um':         [],
        'best_epoch':          [],
        'epochs_trained':      [],
    }

    for fold_i, (train_idx, val_idx) in enumerate(kf_indices):
        seed_everything(SEED + fold_i)

        if has_peak:
            model = PcaPeakMLP(
                K, cfg['hidden_dim'], cfg['n_layers'], cfg['p'],
                n_peaks=multi_peak_ft['peak_count'], input_dim=input_dim,
            )
            train_loader, val_loader = kfold_loaders_pca_peak(
                data, SEED, train_idx, val_idx, cfg['batch_size'])
        else:
            model = PcaMLP(
                K, cfg['hidden_dim'], cfg['n_layers'], cfg['p'],
                input_dim=input_dim,
                peak_weighted_loss=n_peak_weighted,
            )
            train_loader, val_loader = kfold_loaders_pca(
                data, SEED, train_idx, val_idx, cfg['batch_size'])

        on_epoch_end = None
        if trial is not None and fold_i == 0:
            def _cb(*args):
                epoch, val_coeff_loss = args[0], args[3]
                trial.report(float(math.sqrt(val_coeff_loss)), step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                return False
            on_epoch_end = _cb

        with tempfile.TemporaryDirectory() as tmpdir:
            if has_peak:
                _, history = train_pca_peak(
                    model, data, train_loader, val_loader,
                    EPOCHS, cfg['lr'], cfg['wd'], PATIENCE, tmpdir,
                    peak_loss_weight=cfg['peak_loss_weight'],
                    on_epoch_end=on_epoch_end,
                )
            else:
                _, history = train_pca_only(
                    model, data, train_loader, val_loader,
                    EPOCHS, cfg['lr'], cfg['wd'], PATIENCE, tmpdir,
                    on_epoch_end=on_epoch_end,
                )

        best_ep = history['stop_epoch']
        coeff_mse_fold = history.get('val_loss_pca', history.get('val_loss'))[best_ep]
        recon_mse_fold = history['val_loss_recon'][best_ep]

        per_fold_single['best_val_coeff_mse'].append(float(coeff_mse_fold))
        per_fold_single['best_val_coeff_rmse'].append(float(math.sqrt(coeff_mse_fold)))
        per_fold_single['best_val_recon_mse'].append(float(recon_mse_fold))
        per_fold_single['best_val_recon_rmse'].append(float(math.sqrt(recon_mse_fold)))
        per_fold_single['best_epoch'].append(int(best_ep))
        per_fold_single['epochs_trained'].append(int(len(history['val_loss_recon'])))

        # Per-fold peak metrics: this fold's model on its own withheld val set
        model.eval()
        with torch.no_grad():
            geom_val    = data.geom[val_idx]
            true_coeffs = data.pca[val_idx] if has_peak else data.feat[val_idx]
            out         = model(geom_val)
            pred_coeffs = out[0] if has_peak else out

            pred_spec = data.reconstruct_spectrum(pred_coeffs)
            true_spec = data.reconstruct_spectrum(true_coeffs)
            pred_peak = pred_spec.max(dim=-1)
            true_peak = true_spec.max(dim=-1)
            peak_mae  = (pred_peak.values - true_peak.values).abs().mean().item()
            peak_loc  = (wl_tensor[pred_peak.indices] - wl_tensor[true_peak.indices]).abs().mean().item()

        per_fold_single['peak_mae'].append(float(peak_mae))
        per_fold_single['peak_loc_um'].append(float(peak_loc))

    summary = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
               for k, v in per_fold_single.items()}
    score = (summary['best_val_coeff_rmse']['mean']
             + STD_PENALTY * summary['best_val_coeff_rmse']['std'])
    return summary, float(score)


# ============================== MAIN ==============================

def main():
    from scipy.signal import find_peaks as _find_peaks

    print(f"cwd -> {os.getcwd()}")

    with open('multi_peak_ft.pkl', 'rb') as f:
        multi_peak_ft = pickle.load(f)

    print(f"4D PCA: {pca_ft_4D['absorption_features'].shape[0]} samples x "
          f"{pca_ft_4D['absorption_features'].shape[1]} PCs "
          f"(geom dim {pca_ft_4D['geometry_table'].shape[1]})")
    print(f"Multi-peak amp table: {multi_peak_ft['absorption_features'].shape}  "
          f"(peak_count={multi_peak_ft['peak_count']})")

    have = pca_ft_4D['absorption_features'].shape[1]
    assert have >= K, \
        f"pca_4D only has {have} PCs but K={K} is required. Re-run preprocessing."

    # Peak counts for pca_n_peak_4D — sorted in the same order PCADataset uses
    geom_sorted = pca_ft_4D['geometry_table'].sort_index()
    abs_sorted  = pca_ft_4D['absorption'].loc[geom_sorted.index].to_numpy()
    peak_counts_arr = np.array([
        len(_find_peaks(s, prominence=PEAK_PROMINENCE * s.max())[0])
        for s in abs_sorted
    ], dtype=np.float32)
    peak_counts_arr = np.maximum(peak_counts_arr, 1.0)
    print(f"Peak counts (n_peak variant): min={int(peak_counts_arr.min())}  "
          f"max={int(peak_counts_arr.max())}  mean={peak_counts_arr.mean():.2f}")

    input_dim = pca_ft_4D['geometry_table'].shape[1]
    variant_info = {
        'pca_4D':        {'feat': pca_ft_4D, 'has_peak': False, 'n_peak_weighted': False,
                          'input_dim': input_dim},
        'pca_n_peak_4D': {'feat': pca_ft_4D, 'has_peak': False, 'n_peak_weighted': True,
                          'input_dim': input_dim, 'peak_counts': peak_counts_arr},
        'pca_phys_4D':   {'feat': pca_ft_4D, 'has_peak': True,  'n_peak_weighted': False,
                          'input_dim': input_dim},
    }
    for v in MODEL_VARIANTS:
        assert v in variant_info, \
            f"Unknown variant {v!r}. Choose from {list(variant_info)}."

    results_root = os.path.join('optimization_results', NAME)
    os.makedirs(results_root, exist_ok=True)
    print(f"\nResults root: {results_root}")
    print(f"K fixed at {K}  (no sweep)")

    all_results = {}
    start_wall = time.time()

    for variant in MODEL_VARIANTS:
        info = variant_info[variant]
        has_peak        = info['has_peak']
        n_peak_weighted = info.get('n_peak_weighted', False)
        variant_dir = os.path.join(results_root, variant)
        os.makedirs(variant_dir, exist_ok=True)

        base_data = build_dataset(variant, variant_info, multi_peak_ft)
        kf_indices, test_indices = generate_kfold(base_data, K_FOLDS, SEED, TEST_RATIO)

        print('\n' + '#' * 72)
        print(f"# VARIANT: {variant}  (input_dim={info['input_dim']}, "
              f"has_peak={has_peak}, n_peak_weighted={n_peak_weighted})")
        print(f"# {len(base_data) - len(test_indices)} train+val samples / "
              f"{len(test_indices)} held-out test (untouched)")
        print(f"# {TRIALS} trials  |  K={K}")
        print('#' * 72)

        trial_cache: dict = {}

        def objective(trial: optuna.trial.Trial) -> float:
            cfg = suggest_config(trial, has_peak, n_peak_weighted)
            name = config_name(cfg)
            if name in trial_cache:
                return trial_cache[name]['score']
            try:
                summary, score = evaluate_config(
                    cfg, variant, kf_indices, variant_info, multi_peak_ft, trial=trial,
                )
            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"    trial failed: {e}")
                return float('inf')
            trial_cache[name] = {'name': name, 'config': cfg, 'score': score, 'summary': summary}
            print(f"    [{trial.number:03d}] {name}\n"
                  f"      score={score:.6g}"
                  f"  coeff_rmse={summary['best_val_coeff_rmse']['mean']:.6g}"
                  f"  recon_rmse={summary['best_val_recon_rmse']['mean']:.6g}"
                  f"  peak_loc_mae={summary['peak_loc_um']['mean']:.6g} um")
            gc.collect()
            return score

        sampler = optuna.samplers.TPESampler(seed=SEED)
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=10, max_resource=EPOCHS, reduction_factor=3,
        )
        study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)

        if variant in BEST_KNOWN:
            seed_cfg = dict(BEST_KNOWN[variant])
            allowed = {'hidden_dim', 'n_layers', 'p', 'lr', 'wd', 'batch_size',
                       'peak_loss_weight', 'peak_weight_alpha'}
            seed_cfg = {k: v for k, v in seed_cfg.items() if k in allowed}
            study.enqueue_trial(seed_cfg)

        study.optimize(objective, n_trials=TRIALS)

        completed = list(trial_cache.values())
        if not completed:
            print(f"  !! No completed trials for {variant}; skipping.")
            continue

        best = min(completed, key=lambda r: r['score'])
        n_pruned   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)

        record = {
            'variant':      variant,
            'K':            K,
            'best_config':  best['config'],
            'best_name':    best['name'],
            'best_score':   best['score'],
            'best_summary': best['summary'],
            'meta': {
                'seed': SEED, 'test_ratio': TEST_RATIO, 'k_folds': K_FOLDS,
                'epochs': EPOCHS, 'patience': PATIENCE, 'trials': TRIALS,
                'trials_completed': n_complete, 'trials_pruned': n_pruned,
                'std_penalty': STD_PENALTY, 'input_dim': info['input_dim'],
                'K': K, 'search_ranges': SEARCH_RANGES,
                'has_peak': has_peak, 'n_peak_weighted': n_peak_weighted,
                'best_known_enqueued': BEST_KNOWN.get(variant),
                'metric': 'per_fold_single_coeff_rmse_mean + std_penalty * std',
                'pruner': 'HyperbandPruner(min_resource=10, max_resource=epochs, rf=3)',
            },
            'trial_log': [
                {'name': r['name'], 'config': r['config'], 'score': r['score'],
                 'metrics': r['summary']}
                for r in completed
            ],
        }
        out_path = os.path.join(variant_dir, 'study.json')
        with open(out_path, 'w') as f:
            json.dump(record, f, indent=4, default=_json_default)
        all_results[variant] = record
        print(f"  -> best score {best['score']:.6g}  "
              f"({n_complete} complete / {n_pruned} pruned)")
        print(f"     saved {out_path}")

        gc.collect()

    elapsed_min = (time.time() - start_wall) / 60.0
    print(f"\nDONE. Total wall time: {elapsed_min:.1f} min")

    try:
        import pandas as pd
        rows = []
        for variant, rec in all_results.items():
            best_r = min(rec['trial_log'], key=lambda r: r['score'])
            m   = best_r['metrics']
            cfg = best_r['config']
            rows.append({
                'variant':           variant,
                'score':             best_r['score'],
                'coeff_rmse':        m['best_val_coeff_rmse']['mean'],
                'coeff_rmse_std':    m['best_val_coeff_rmse']['std'],
                'recon_rmse':        m['best_val_recon_rmse']['mean'],
                'recon_rmse_std':    m['best_val_recon_rmse']['std'],
                'peak_mae':          m['peak_mae']['mean'],
                'peak_loc_um':       m['peak_loc_um']['mean'],
                'hidden_dim':         cfg['hidden_dim'],
                'n_layers':           cfg['n_layers'],
                'p':                  round(cfg['p'], 4),
                'lr':                 cfg['lr'],
                'wd':                 cfg['wd'],
                'batch_size':         cfg['batch_size'],
                'peak_weight_alpha':  cfg.get('peak_weight_alpha'),
                'peak_loss_weight':   cfg.get('peak_loss_weight'),
            })
        if rows:
            summary_df = pd.DataFrame(rows).sort_values('score').reset_index(drop=True)
            with pd.option_context('display.max_columns', None, 'display.width', 220,
                                   'display.float_format', '{:.6g}'.format):
                print('\n' + summary_df.to_string(index=False))
            summary_path = os.path.join(results_root, 'summary_table.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSaved {summary_path}")
    except Exception as e:
        print(f"(summary table skipped: {e})")


if __name__ == '__main__':
    main()
