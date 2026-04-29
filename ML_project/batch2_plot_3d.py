import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import pickle

# Read the geometry CSV file (geometry now lives in metadata.csv)
geom_df = pd.read_csv('data/batch2/metadata.csv')

# Load the multi-peak features from pickle (or other saved format)
# Assuming the features are saved in a pkl file from multi_peak.py
try:
    with open('multi_peak_ft.pkl', 'rb') as f:
        multi_peak_ft = pickle.load(f)
    # refl_features = multi_peak_ft['reflection_features']
    abs_features = multi_peak_ft['absorption_features']
except FileNotFoundError:
    print("Warning: multi_peak_ft.pkl not found. Trying alternative paths...")
    # Try looking in common locations
    try:
        with open('data/batch2/multi_peak_ft.pkl', 'rb') as f:
            multi_peak_ft = pickle.load(f)
        # refl_features = multi_peak_ft['reflection_features']
        abs_features = multi_peak_ft['absorption_features']
    except FileNotFoundError:
        print("Error: Could not find multi_peak features file")
        raise

# Calculate average spacing between peaks from reflection features
# For peaks at wavelengths l1, l2, l3, ..., spacing = (l_max - l_min) / (n_peaks - 1)
N_PEAKS = 4
avg_peak_spacing = []

for idx in range(len(abs_features)):
    lambdas = []
    for i in range(N_PEAKS):
        lambda_col = f'lambda_{i}'
        mask_col = f'mask_{i}'
        if lambda_col in abs_features.columns and mask_col in abs_features.columns:
            if abs_features.iloc[idx][mask_col] == 1:  # Only include valid peaks
                lambdas.append(abs_features.iloc[idx][lambda_col])

    if len(lambdas) > 1:
        # Average spacing = (max - min) / (num_peaks - 1)
        avg_spacing = (max(lambdas) - min(lambdas)) / (len(lambdas) - 1)
    elif len(lambdas) == 1:
        avg_spacing = 0.0  # Single peak has no spacing
    else:
        avg_spacing = 0.0  # No peaks
    avg_peak_spacing.append(avg_spacing)

avg_peak_spacing = np.array(avg_peak_spacing)

# Calculate average lambda and min peak metrics (only for samples with at least one peak)
avg_lambda = []
min_peak_lambda = []
min_peak_lambda_div_Q = []
has_peak = []

for idx in range(len(abs_features)):
    lambdas = []
    Qs = []
    for i in range(N_PEAKS):
        lambda_col = f'lambda_{i}'
        Q_col = f'Q_{i}'
        mask_col = f'mask_{i}'
        if lambda_col in abs_features.columns and mask_col in abs_features.columns:
            if abs_features.iloc[idx][mask_col] == 1:
                lambdas.append(abs_features.iloc[idx][lambda_col])
                if Q_col in abs_features.columns:
                    Qs.append(abs_features.iloc[idx][Q_col])

    if len(lambdas) > 0:
        avg_lambda.append(np.mean(lambdas))
        has_peak.append(True)

        # Find minimum peak lambda
        min_idx = np.argmin(lambdas)
        min_lambda = lambdas[min_idx]
        min_peak_lambda.append(min_lambda)

        # Calculate min lambda / Q
        if len(Qs) > 0 and Qs[min_idx] > 0:
            min_peak_lambda_div_Q.append(min_lambda / Qs[min_idx])
        else:
            min_peak_lambda_div_Q.append(np.nan)
    else:
        avg_lambda.append(np.nan)
        has_peak.append(False)
        min_peak_lambda.append(np.nan)
        min_peak_lambda_div_Q.append(np.nan)

avg_lambda = np.array(avg_lambda)
min_peak_lambda = np.array(min_peak_lambda)
min_peak_lambda_div_Q = np.array(min_peak_lambda_div_Q)
has_peak = np.array(has_peak)

# BLW/TLW ratio used for slider filtering. ratio >= 1 means BLW >= TLW.
ratio = geom_df['blw'].to_numpy() / geom_df['tlw'].to_numpy()
finite_ratio = ratio[np.isfinite(ratio)]
RATIO_LO = float(np.floor(finite_ratio.min() * 20) / 20)
RATIO_HI = float(np.ceil(finite_ratio.max() * 20) / 20)
RATIO_INIT = max(1.0, RATIO_LO)

tlw_range = geom_df['tlw'].min(), geom_df['tlw'].max()
s_range = geom_df['s'].min(), geom_df['s'].max()
_s_plane = np.linspace(s_range[0], s_range[1], 10)
_tlw_plane = np.linspace(tlw_range[0], tlw_range[1], 10)
_S_MESH, _TLW_MESH = np.meshgrid(_s_plane, _tlw_plane)

def draw_plane(ax):
    ax.plot_surface(_TLW_MESH, _TLW_MESH, _S_MESH, alpha=0.2, color='red')

def make_filtered_figure(panels, fig_title):
    """panels: list of dicts with keys: title, cmap, cbar_label, color_array, extra_mask (optional)."""
    fig = plt.figure(figsize=(16, 7))
    fig.subplots_adjust(bottom=0.18)
    n = len(panels)
    axes_and_cbars = []
    for i, p in enumerate(panels):
        ax = fig.add_subplot(1, n, i + 1, projection='3d')
        axes_and_cbars.append({'ax': ax, 'cbar': None, 'panel': p})

    def render(min_ratio):
        base = ratio >= min_ratio
        for entry in axes_and_cbars:
            ax = entry['ax']
            p = entry['panel']
            extra = p.get('extra_mask')
            mask = base if extra is None else (base & extra)
            ax.clear()
            n_pts = int(mask.sum())
            if n_pts > 0:
                sc = ax.scatter(
                    geom_df.loc[mask, 'tlw'],
                    geom_df.loc[mask, 'blw'],
                    geom_df.loc[mask, 's'],
                    c=p['color_array'][mask],
                    cmap=p['cmap'],
                    s=30,
                    alpha=0.6,
                )
                if entry['cbar'] is None:
                    entry['cbar'] = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
                    entry['cbar'].set_label(p['cbar_label'])
                else:
                    entry['cbar'].update_normal(sc)
            ax.set_xlabel('TLW')
            ax.set_ylabel('BLW')
            ax.set_zlabel('S')
            ax.set_title(f"{p['title']}\n(min BLW/TLW = {min_ratio:.2f}, n={n_pts})")
            draw_plane(ax)
        fig.canvas.draw_idle()

    slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        slider_ax,
        'Min BLW/TLW',
        RATIO_LO,
        RATIO_HI,
        valinit=RATIO_INIT,
        valstep=0.05,
    )
    slider.on_changed(render)
    render(RATIO_INIT)
    fig.suptitle(fig_title)
    return fig, slider

valid_div_q = ~np.isnan(min_peak_lambda_div_Q)

fig1, slider1 = make_filtered_figure(
    panels=[
        {
            'title': 'Colored by Average Peak Spacing',
            'cmap': 'viridis',
            'cbar_label': 'Average Peak Spacing (μm)',
            'color_array': avg_peak_spacing,
        },
        {
            'title': 'Colored by Average Lambda (samples with peaks)',
            'cmap': 'plasma',
            'cbar_label': 'Average Lambda (μm)',
            'color_array': avg_lambda,
            'extra_mask': has_peak,
        },
    ],
    fig_title='Geometry colored by peak metrics',
)

fig2, slider2 = make_filtered_figure(
    panels=[
        {
            'title': 'Colored by Minimum Peak Lambda (samples with peaks)',
            'cmap': 'cool',
            'cbar_label': 'Min Peak Lambda (μm)',
            'color_array': min_peak_lambda,
            'extra_mask': has_peak,
        },
        {
            'title': 'Colored by Min Peak Lambda / Q (valid samples)',
            'cmap': 'RdYlBu_r',
            'cbar_label': 'Min Lambda / Q',
            'color_array': min_peak_lambda_div_Q,
            'extra_mask': has_peak & valid_div_q,
        },
    ],
    fig_title='Geometry colored by min-peak metrics',
)

plt.show()
