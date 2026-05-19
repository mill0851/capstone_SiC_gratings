"""
Generate a publication-quality table figure from summary_table.csv.
Run from any directory — paths are resolved relative to this script.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, 'summary_table.csv'))

# ── display names ──────────────────────────────────────────────────────────────
VARIANT_LABELS = {
    'pca_4D':        'PCA Baseline',
    'pca_n_peak_4D': 'PCA + Peak Weight',
    'pca_phys_4D':   'PCA + Peak Head',
}
df['label'] = df['variant'].map(VARIANT_LABELS)
df = df.sort_values('score').reset_index(drop=True)

# ── metric table data ──────────────────────────────────────────────────────────
def pm(mean, std, decimals=4):
    fmt = f'{{:.{decimals}f}}'
    return f"{fmt.format(mean)} ± {fmt.format(std)}"

def sci(x):
    if pd.isna(x):
        return '—'
    return f'{x:.3e}'

def f4(x):
    if pd.isna(x):
        return '—'
    return f'{x:.4f}'

metric_rows = []
for _, row in df.iterrows():
    metric_rows.append([
        row['label'],
        f"{row['score']:.4f}",
        pm(row['coeff_rmse'], row['coeff_rmse_std']),
        pm(row['recon_rmse'], row['recon_rmse_std']),
        f"{row['peak_mae']:.4f}",
        f"{row['peak_loc_um']:.4f}",
    ])

metric_cols = ['Model', 'Score', 'Coeff RMSE (mean ± std)',
               'Recon RMSE (mean ± std)', 'Peak MAE', 'Peak Loc MAE (μm)']

# ── hyperparameter table data ──────────────────────────────────────────────────
hp_rows = []
for _, row in df.iterrows():
    hp_rows.append([
        row['label'],
        str(int(row['hidden_dim'])),
        str(int(row['n_layers'])),
        f"{row['p']:.3f}",
        sci(row['lr']),
        sci(row['wd']),
        str(int(row['batch_size'])),
        f"{row['peak_weight_alpha']:.1f}" if not pd.isna(row.get('peak_weight_alpha')) else '—',
        sci(row.get('peak_loss_weight')) if not pd.isna(row.get('peak_loss_weight')) else '—',
    ])

hp_cols = ['Model', 'Hidden Dim', 'Layers', 'Dropout',
           'LR', 'Weight Decay', 'Batch', 'Peak α', 'Peak LW']

# ── figure ─────────────────────────────────────────────────────────────────────
HEADER_COLOR  = '#2c3e50'
HEADER_TEXT   = 'white'
ROW_COLORS    = ['#f2f3f4', '#ffffff']
BEST_COLOR    = '#d5f5e3'
FONT_FAMILY   = 'monospace'

fig = plt.figure(figsize=(18, 7))
gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.55)

def draw_table(ax, rows, cols, title, best_row=0):
    ax.set_axis_off()
    ax.set_title(title, fontsize=14, fontweight='bold', pad=8, loc='left')

    n_rows = len(rows)
    n_cols = len(cols)

    cell_colors = []
    for i in range(n_rows):
        base = ROW_COLORS[i % 2]
        row_c = [BEST_COLOR if i == best_row else base] * n_cols
        cell_colors.append(row_c)

    tbl = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.auto_set_column_width(col=list(range(n_cols)))
    tbl.scale(1, 2.2)

    # Style header
    for col_i in range(n_cols):
        cell = tbl[0, col_i]
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color=HEADER_TEXT, fontweight='bold', fontfamily=FONT_FAMILY)

    # Style body
    for row_i in range(1, n_rows + 1):
        for col_i in range(n_cols):
            cell = tbl[row_i, col_i]
            cell.set_text_props(fontfamily=FONT_FAMILY, fontsize=12)
            if col_i == 0:
                cell.set_text_props(fontweight='bold', fontfamily=FONT_FAMILY)

    return tbl

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

draw_table(ax1, metric_rows, metric_cols, 'Performance Metrics  (4-fold OOF, sorted by score ↑ best)', best_row=0)
draw_table(ax2, hp_rows,     hp_cols,     'Best Hyperparameters',                                        best_row=0)

fig.suptitle('HPO Summary — PCA Surrogate Models  (K=25, 4D Geometry)',
             fontsize=18, fontweight='bold', y=1.01)

out_path = os.path.join(HERE, 'summary_table.png')
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
print(f"Saved → {out_path}")
plt.show()
