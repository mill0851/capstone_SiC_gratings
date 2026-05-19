"""
PCA Surrogate — Hyperparameter Optimization (split BLW/TLW regimes).

Same TPE + Hyperband loop as pca_mlp_hpo.py, but the dataset is partitioned
by the BLW/TLW ratio into two regimes and a separate study is run for each:

    high  ratio >= BLW_TLW_BOUNDARY
    low   1 <= ratio <  BLW_TLW_BOUNDARY

Three variants are optimised per regime:

    pca_4D        PcaMLP        uniform coeff-MSE
    pca_n_peak_4D PcaMLP        peak-count-weighted coeff-MSE
    pca_phys_4D   PcaPeakMLP    coeff-MSE + pw * masked-peak-MSE

K is fixed at 25.  Score = ensemble coeff RMSE mean + STD_PENALTY * std.
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
import pandas as pd
import torch
import optuna

_ML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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

MODEL_VARIANTS   = ['pca_4D', 'pca_n_peak_4D', 'pca_phys_4D']
K                = 25
NAME             = 'optimization_split_v2'

BLW_TLW_BOUNDARY = 1.5

PEAK_PROMINENCE  = 0.05

SEED             = 1234
TEST_RATIO       = 0.15
K_FOLDS          = 4
EPOCHS           = 300
PATIENCE         = 30

TRIALS           = 60
STD_PENALTY      = 0.3

SEARCH_RANGES = {
    'hidden_dim':        [128, 192, 256, 384, 512],
    'n_layers':          [2, 3, 4],
    'p':                 (0.0, 0.3),
    'lr':                (1e-4, 1e-2),
    'wd':                (1e-5, 1e-2),
    'batch_size':        [16, 32, 64],
    'peak_weight_alpha': [1.0, 1.5, 2.0, 3.0],
    'peak_loss_weight':  (1e-3, 1.0),
}

BEST_KNOWN = {
    'pca_4D':        {'hidden_dim': 256, 'n_layers': 3, 'p': 0.036,    'lr': 7.5e-3,  'wd': 4.3e-4,    'batch_size': 16},
    'pca_n_peak_4D': {'hidden_dim': 512, 'n_layers': 3, 'p': 0.008894, 'lr': 1.541e-3,'wd': 2.1989e-5, 'batch_size': 16, 'peak_weight_alpha': 2.0},
    'pca_phys_4D':   {'hidden_dim': 256, 'n_layers': 3, 'p': 0.05,     'lr': 5e-3,    'wd': 5e-4,      'batch_size': 16, 'peak_loss_weight': 0.1},
}


# ============================== REGIME FILTER ==============================

def regime_mask(geom_df: pd.DataFrame, regime: str, boundary: float) -> pd.Series:
    ratio = geom_df['blw'] / geom_df['tlw']
    if regime == 'high':
        return ratio >= boundary
    if regime == 'low':
        return (ratio >= 1.0) & (ratio < boundary)
    raise ValueError(f"Unknown regime {regime!r}; expected 'high' or 'low'.")


def filter_pca_feat(feat: dict, mask: pd.Series) -> dict:
    geom_df  = feat['geometry_table']
    keep_idx = geom_df.index[mask.reindex(geom_df.index, fill_value=False)]
    out = dict(feat)
    out['geometry_table']      = feat['geometry_table'].loc[keep_idx].reset_index(drop=True)
    out['absorption_features'] = feat['absorption_features'].loc[keep_idx].reset_index(drop=True)
    if 'absorption' in feat and feat['absorption'] is not None:
        out['absorption']      = feat['absorption'].loc[keep_idx].reset_index(drop=True)
    out['_kept_ids'] = list(keep_idx)
    return out


def filter_multi_peak_ft(multi_peak_ft: dict, kept_ids: list) -> dict:
    out = dict(multi_peak_ft)
    out['absorption_features'] = (
        multi_peak_ft['absorption_features'].loc[kept_ids].reset_index(drop=True)
    )
    return out


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
    info = variant_info[variant]
    feat = info['feat']
    geom_df    = feat['geometry_table']
    pca_coeffs = feat['absorption_features'].iloc[:, :K]
    pca_arts   = feat['absorption_pca']

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
    info = variant_info[variant]
    has_peak        = info['has_peak']
    n_peak_weighted = info.get('n_peak_weighted', False)
    input_dim       = info['input_dim']

    data = build_dataset(variant, variant_info, multi_peak_ft)

    if n_peak_weighted:
        alpha      = cfg['peak_weight_alpha']
        raw_counts = info['peak_counts']
        weights    = raw_counts ** alpha
        weights    = weights / weights.mean()
        data.set_sample_weights(weights)

    wl_tensor = torch.tensor(np.asarray(data.wl), dtype=torch.float32)

    fold_models = []
    per_fold_single = {
        'best_val_coeff_mse':  [],
        'best_val_coeff_rmse': [],
        'best_val_recon_mse':  [],
        'best_val_recon_rmse': [],
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
            def _cb(epoch, val_recon_loss, best_recon_loss):
                trial.report(float(math.sqrt(val_recon_loss)), step=epoch)
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

        best_ep        = history['stop_epoch']
        coeff_mse_fold = history.get('val_loss_pca', history.get('val_loss'))[best_ep]
        recon_mse_fold = history['val_loss_recon'][best_ep]

        per_fold_single['best_val_coeff_mse'].append(float(coeff_mse_fold))
        per_fold_single['best_val_coeff_rmse'].append(float(math.sqrt(coeff_mse_fold)))
        per_fold_single['best_val_recon_mse'].append(float(recon_mse_fold))
        per_fold_single['best_val_recon_rmse'].append(float(math.sqrt(recon_mse_fold)))
        per_fold_single['best_epoch'].append(int(best_ep))
        per_fold_single['epochs_trained'].append(int(len(history['val_loss_recon'])))

        model.eval()
        fold_models.append(model)

    ensemble_per_fold = {
        'coeff_mse':  [], 'coeff_rmse':  [],
        'recon_mse':  [], 'recon_rmse':  [],
        'peak_mae':   [], 'peak_loc_um': [],
    }
    with torch.no_grad():
        for fold_i, (_train_idx, val_idx) in enumerate(kf_indices):
            geom_val    = data.geom[val_idx]
            true_coeffs = data.pca[val_idx] if has_peak else data.feat[val_idx]

            preds = []
            for m in fold_models:
                out = m(geom_val)
                preds.append(out[0] if has_peak else out)
            ensemble_coeffs = torch.stack(preds, dim=0).mean(dim=0)

            coeff_mse = ((ensemble_coeffs - true_coeffs) ** 2).mean().item()

            pred_spec = data.reconstruct_spectrum(ensemble_coeffs)
            true_spec = data.reconstruct_spectrum(true_coeffs)
            recon_mse = ((pred_spec - true_spec) ** 2).mean().item()

            pred_peak = pred_spec.max(dim=-1)
            true_peak = true_spec.max(dim=-1)
            peak_mae  = (pred_peak.values - true_peak.values).abs().mean().item()
            peak_loc  = (wl_tensor[pred_peak.indices] - wl_tensor[true_peak.indices]).abs().mean().item()

            ensemble_per_fold['coeff_mse'].append(float(coeff_mse))
            ensemble_per_fold['coeff_rmse'].append(float(math.sqrt(coeff_mse)))
            ensemble_per_fold['recon_mse'].append(float(recon_mse))
            ensemble_per_fold['recon_rmse'].append(float(math.sqrt(recon_mse)))
            ensemble_per_fold['peak_mae'].append(float(peak_mae))
            ensemble_per_fold['peak_loc_um'].append(float(peak_loc))

    summary = {
        'ensemble':    {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                        for k, v in ensemble_per_fold.items()},
        'single_fold': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                        for k, v in per_fold_single.items()},
    }
    score = (summary['ensemble']['coeff_rmse']['mean']
             + STD_PENALTY * summary['ensemble']['coeff_rmse']['std'])
    return summary, float(score)


# ============================== STUDY DRIVER ==============================

def run_study_for_variant(
        variant: str,
        variant_info: dict,
        multi_peak_ft: dict,
        regime: str,
        boundary: float,
        regime_dir: str,
):
    info            = variant_info[variant]
    has_peak        = info['has_peak']
    n_peak_weighted = info.get('n_peak_weighted', False)
    variant_dir = os.path.join(regime_dir, variant)
    os.makedirs(variant_dir, exist_ok=True)

    base_data = build_dataset(variant, variant_info, multi_peak_ft)
    kf_indices, test_indices = generate_kfold(base_data, K_FOLDS, SEED, TEST_RATIO)

    print('\n' + '#' * 72)
    print(f"# REGIME: {regime}  (BLW/TLW boundary={boundary})")
    print(f"# VARIANT: {variant}  (input_dim={info['input_dim']}, "
          f"has_peak={has_peak}, n_peak_weighted={n_peak_weighted})")
    print(f"# {len(base_data) - len(test_indices)} train+val samples / "
          f"{len(test_indices)} held-out test  |  K={K}  |  {TRIALS} trials")
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
        ens = summary['ensemble']
        print(f"    [{trial.number:03d}] {name}\n"
              f"      score={score:.6g}"
              f"  ens_coeff_rmse={ens['coeff_rmse']['mean']:.6g}"
              f"  ens_recon_rmse={ens['recon_rmse']['mean']:.6g}"
              f"  peak_loc_mae={ens['peak_loc_um']['mean']:.6g} um")
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
        print(f"  !! No completed trials for {regime}/{variant}; skipping.")
        return None

    best = min(completed, key=lambda r: r['score'])
    n_pruned   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)

    record = {
        'variant':           variant,
        'regime':            regime,
        'blw_tlw_boundary':  boundary,
        'K':                 K,
        'n_train_val':       int(len(base_data) - len(test_indices)),
        'n_test':            int(len(test_indices)),
        'best_config':       best['config'],
        'best_name':         best['name'],
        'best_score':        best['score'],
        'best_summary':      best['summary'],
        'meta': {
            'seed': SEED, 'test_ratio': TEST_RATIO, 'k_folds': K_FOLDS,
            'epochs': EPOCHS, 'patience': PATIENCE, 'trials': TRIALS,
            'trials_completed': n_complete, 'trials_pruned': n_pruned,
            'std_penalty': STD_PENALTY, 'input_dim': info['input_dim'],
            'K': K, 'search_ranges': SEARCH_RANGES,
            'has_peak': has_peak, 'n_peak_weighted': n_peak_weighted,
            'best_known_enqueued': BEST_KNOWN.get(variant),
            'metric': 'ensemble_coeff_rmse_mean + std_penalty * std',
            'pruner': 'HyperbandPruner(min_resource=10, max_resource=epochs, rf=3)',
            'regime_filter': {
                'regime': regime, 'boundary': boundary,
                'rule': ('ratio >= boundary' if regime == 'high'
                         else '1 <= ratio < boundary'),
            },
        },
        'trial_log': [
            {'name': r['name'], 'config': r['config'], 'score': r['score'],
             'ensemble': r['summary']['ensemble']}
            for r in completed
        ],
    }
    out_path = os.path.join(variant_dir, 'study.json')
    with open(out_path, 'w') as f:
        json.dump(record, f, indent=4, default=_json_default)
    print(f"  -> best score {best['score']:.6g}  "
          f"({n_complete} complete / {n_pruned} pruned)")
    print(f"     saved {out_path}")
    return record


# ============================== MAIN ==============================

def main():
    from scipy.signal import find_peaks as _find_peaks

    print(f"cwd -> {os.getcwd()}")
    print(f"BLW/TLW boundary: {BLW_TLW_BOUNDARY}  "
          f"(high: ratio >= {BLW_TLW_BOUNDARY}; low: 1 <= ratio < {BLW_TLW_BOUNDARY})")

    with open('multi_peak_ft.pkl', 'rb') as f:
        multi_peak_ft = pickle.load(f)

    print(f"4D PCA: {pca_ft_4D['absorption_features'].shape[0]} samples x "
          f"{pca_ft_4D['absorption_features'].shape[1]} PCs "
          f"(geom dim {pca_ft_4D['geometry_table'].shape[1]})")
    print(f"Multi-peak amp table: {multi_peak_ft['absorption_features'].shape}  "
          f"(peak_count={multi_peak_ft['peak_count']})")

    have = pca_ft_4D['absorption_features'].shape[1]
    assert have >= K, \
        f"pca_4D only has {have} PCs but K={K} required. Re-run preprocessing."
    for col in ('blw', 'tlw'):
        assert col in pca_ft_4D['geometry_table'].columns, \
            f"geometry_table missing required column {col!r}."

    for name, src in [('pca_4D', pca_ft_4D)]:
        m_high = regime_mask(src['geometry_table'], 'high', BLW_TLW_BOUNDARY)
        m_low  = regime_mask(src['geometry_table'], 'low',  BLW_TLW_BOUNDARY)
        n = len(src['geometry_table'])
        print(f"  {name}: high={int(m_high.sum())}/{n}, low={int(m_low.sum())}/{n}, "
              f"discarded={n - int(m_high.sum()) - int(m_low.sum())} (ratio < 1)")

    results_root = os.path.join('optimization_results', NAME)
    os.makedirs(results_root, exist_ok=True)
    print(f"\nResults root: {results_root}")
    print(f"K fixed at {K}  (no sweep)")

    all_results: dict = {}
    start_wall = time.time()

    for regime in ('high', 'low'):
        regime_dir = os.path.join(results_root, regime)
        os.makedirs(regime_dir, exist_ok=True)

        feat_4D_r      = filter_pca_feat(pca_ft_4D, regime_mask(
            pca_ft_4D['geometry_table'], regime, BLW_TLW_BOUNDARY))
        multi_peak_ft_r = filter_multi_peak_ft(multi_peak_ft, feat_4D_r['_kept_ids'])

        # Peak counts for n_peak variant, ordered to match the filtered PCADataset
        abs_r = feat_4D_r['absorption'].to_numpy()
        peak_counts_r = np.array([
            len(_find_peaks(s, prominence=PEAK_PROMINENCE * s.max())[0])
            for s in abs_r
        ], dtype=np.float32)
        peak_counts_r = np.maximum(peak_counts_r, 1.0)

        input_dim = feat_4D_r['geometry_table'].shape[1]
        variant_info = {
            'pca_4D':        {'feat': feat_4D_r, 'has_peak': False, 'n_peak_weighted': False,
                              'input_dim': input_dim},
            'pca_n_peak_4D': {'feat': feat_4D_r, 'has_peak': False, 'n_peak_weighted': True,
                              'input_dim': input_dim, 'peak_counts': peak_counts_r},
            'pca_phys_4D':   {'feat': feat_4D_r, 'has_peak': True,  'n_peak_weighted': False,
                              'input_dim': input_dim},
        }
        for v in MODEL_VARIANTS:
            assert v in variant_info, \
                f"Unknown variant {v!r}. Choose from {list(variant_info)}."

        all_results[regime] = {}
        for variant in MODEL_VARIANTS:
            record = run_study_for_variant(
                variant, variant_info, multi_peak_ft_r,
                regime, BLW_TLW_BOUNDARY, regime_dir,
            )
            if record is not None:
                all_results[regime][variant] = record
            gc.collect()

    elapsed_min = (time.time() - start_wall) / 60.0
    print(f"\nDONE. Total wall time: {elapsed_min:.1f} min")

    try:
        rows = []
        for regime, regime_results in all_results.items():
            for variant, rec in regime_results.items():
                best_r = min(rec['trial_log'], key=lambda r: r['score'])
                ens = best_r['ensemble']
                cfg = best_r['config']
                rows.append({
                    'regime':             regime,
                    'variant':            variant,
                    'score':              best_r['score'],
                    'ens_coeff_rmse':     ens['coeff_rmse']['mean'],
                    'ens_coeff_rmse_std': ens['coeff_rmse']['std'],
                    'ens_recon_rmse':     ens['recon_rmse']['mean'],
                    'ens_recon_rmse_std': ens['recon_rmse']['std'],
                    'ens_peak_mae':       ens['peak_mae']['mean'],
                    'ens_peak_loc_um':    ens['peak_loc_um']['mean'],
                    'hidden_dim':         cfg['hidden_dim'],
                    'n_layers':           cfg['n_layers'],
                    'p':                  round(cfg['p'], 4),
                    'lr':                 cfg['lr'],
                    'wd':                 cfg['wd'],
                    'batch_size':         cfg['batch_size'],
                    'peak_weight_alpha':  cfg.get('peak_weight_alpha'),
                    'peak_loss_weight':   cfg.get('peak_loss_weight'),
                    'n_train_val':        rec['n_train_val'],
                    'n_test':             rec['n_test'],
                })
        if rows:
            summary_df = pd.DataFrame(rows).sort_values(['regime', 'score']).reset_index(drop=True)
            with pd.option_context('display.max_columns', None, 'display.width', 220,
                                   'display.float_format', '{:.6g}'.format):
                print('\n' + summary_df.to_string(index=False))
            summary_path = os.path.join(results_root, 'summary_table.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSaved {summary_path}")

        best_rows = []
        for regime, regime_results in all_results.items():
            for variant, rec in regime_results.items():
                cfg = rec['best_config']
                ens = rec['best_summary']['ensemble']
                best_rows.append({
                    'variant':            variant,
                    'regime':             regime,
                    'score':              rec['best_score'],
                    'ens_coeff_rmse':     ens['coeff_rmse']['mean'],
                    'ens_recon_rmse':     ens['recon_rmse']['mean'],
                    'ens_peak_mae':       ens['peak_mae']['mean'],
                    'ens_peak_loc_um':    ens['peak_loc_um']['mean'],
                    'hidden_dim':         cfg['hidden_dim'],
                    'n_layers':           cfg['n_layers'],
                    'p':                  round(cfg['p'], 4),
                    'lr':                 cfg['lr'],
                    'wd':                 cfg['wd'],
                    'batch_size':         cfg['batch_size'],
                    'peak_weight_alpha':  cfg.get('peak_weight_alpha'),
                    'peak_loss_weight':   cfg.get('peak_loss_weight'),
                    'n_train_val':        rec['n_train_val'],
                    'n_test':             rec['n_test'],
                })
        if best_rows:
            best_df = pd.DataFrame(best_rows).sort_values(['variant', 'regime']).reset_index(drop=True)
            print('\n=== Best per regime per variant ===')
            with pd.option_context('display.max_columns', None, 'display.width', 220,
                                   'display.float_format', '{:.6g}'.format):
                print(best_df.to_string(index=False))
            best_path = os.path.join(results_root, 'best_per_regime.csv')
            best_df.to_csv(best_path, index=False)
            print(f"\nSaved {best_path}")
    except Exception as e:
        print(f"(summary table skipped: {e})")


if __name__ == '__main__':
    main()
