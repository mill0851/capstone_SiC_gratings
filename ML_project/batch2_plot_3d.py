import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

# Read the geometry CSV file
geom_df = pd.read_csv('data/batch2/batch2_geometries_1024.csv')

# Load the multi-peak features from pickle (or other saved format)
# Assuming the features are saved in a pkl file from multi_peak.py
try:
    with open('multi_peak_ft.pkl', 'rb') as f:
        multi_peak_ft = pickle.load(f)
    refl_features = multi_peak_ft['reflection_features']
    abs_features = multi_peak_ft['absorption_features']
except FileNotFoundError:
    print("Warning: multi_peak_ft.pkl not found. Trying alternative paths...")
    # Try looking in common locations
    try:
        with open('data/batch2/multi_peak_ft.pkl', 'rb') as f:
            multi_peak_ft = pickle.load(f)
        refl_features = multi_peak_ft['reflection_features']
        abs_features = multi_peak_ft['absorption_features']
    except FileNotFoundError:
        print("Error: Could not find multi_peak features file")
        raise

# Calculate average spacing between peaks from reflection features
# For peaks at wavelengths l1, l2, l3, ..., spacing = (l_max - l_min) / (n_peaks - 1)
N_PEAKS = 4
avg_peak_spacing = []

for idx in range(len(refl_features)):
    lambdas = []
    for i in range(N_PEAKS):
        lambda_col = f'lambda_{i}'
        mask_col = f'mask_{i}'
        if lambda_col in refl_features.columns and mask_col in refl_features.columns:
            if refl_features.iloc[idx][mask_col] == 1:  # Only include valid peaks
                lambdas.append(refl_features.iloc[idx][lambda_col])

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

for idx in range(len(refl_features)):
    lambdas = []
    Qs = []
    for i in range(N_PEAKS):
        lambda_col = f'lambda_{i}'
        Q_col = f'Q_{i}'
        mask_col = f'mask_{i}'
        if lambda_col in refl_features.columns and mask_col in refl_features.columns:
            if refl_features.iloc[idx][mask_col] == 1:
                lambdas.append(refl_features.iloc[idx][lambda_col])
                if Q_col in refl_features.columns:
                    Qs.append(refl_features.iloc[idx][Q_col])

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

# Create first figure with plots 1 and 2
fig1 = plt.figure(figsize=(16, 6))

# Plot 1: Colored by average peak spacing
ax1 = fig1.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(
    geom_df['tlw'],
    geom_df['blw'],
    geom_df['s'],
    c=avg_peak_spacing,
    cmap='viridis',
    s=30,
    alpha=0.6
)
ax1.set_xlabel('TLW')
ax1.set_ylabel('BLW')
ax1.set_zlabel('S')
ax1.set_title('Colored by Average Peak Spacing')
colorbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=5)
colorbar1.set_label('Average Peak Spacing (μm)')

# Plot 2: Colored by average lambda (only samples with peaks)
ax2 = fig1.add_subplot(122, projection='3d')
mask = has_peak
scatter2 = ax2.scatter(
    geom_df.loc[mask, 'tlw'],
    geom_df.loc[mask, 'blw'],
    geom_df.loc[mask, 's'],
    c=avg_lambda[mask],
    cmap='plasma',
    s=30,
    alpha=0.6
)
ax2.set_xlabel('TLW')
ax2.set_ylabel('BLW')
ax2.set_zlabel('S')
ax2.set_title('Colored by Average Lambda\n(Only samples with peaks)')
colorbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=5)
colorbar2.set_label('Average Lambda (μm)')

plt.tight_layout()

# Add semi-transparent planes at tlw=blw for first figure
for ax in [ax1, ax2]:
    # Get axis limits
    tlw_range = geom_df['tlw'].min(), geom_df['tlw'].max()
    s_range = geom_df['s'].min(), geom_df['s'].max()

    # Create mesh for the plane where tlw=blw
    s_plane = np.linspace(s_range[0], s_range[1], 10)
    tlw_plane = np.linspace(tlw_range[0], tlw_range[1], 10)
    S, TLW = np.meshgrid(s_plane, tlw_plane)
    BLW = TLW  # tlw = blw

    # Plot the plane
    ax.plot_surface(TLW, BLW, S, alpha=0.2, color='red')

plt.show()

# Create second figure with plots 3 and 4
fig2 = plt.figure(figsize=(16, 6))

# Plot 3: Colored by minimum peak lambda (only samples with peaks)
ax3 = fig2.add_subplot(121, projection='3d')
scatter3 = ax3.scatter(
    geom_df.loc[mask, 'tlw'],
    geom_df.loc[mask, 'blw'],
    geom_df.loc[mask, 's'],
    c=min_peak_lambda[mask],
    cmap='cool',
    s=30,
    alpha=0.6
)
ax3.set_xlabel('TLW')
ax3.set_ylabel('BLW')
ax3.set_zlabel('S')
ax3.set_title('Colored by Minimum Peak Lambda\n(Only samples with peaks)')
colorbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.5, aspect=5)
colorbar3.set_label('Min Peak Lambda (μm)')

# Plot 4: Colored by minimum peak lambda / Q (only samples with peaks)
ax4 = fig2.add_subplot(122, projection='3d')
# Filter out NaN values for this plot
valid_mask = mask & ~np.isnan(min_peak_lambda_div_Q)
scatter4 = ax4.scatter(
    geom_df.loc[valid_mask, 'tlw'],
    geom_df.loc[valid_mask, 'blw'],
    geom_df.loc[valid_mask, 's'],
    c=min_peak_lambda_div_Q[valid_mask],
    cmap='RdYlBu_r',
    s=30,
    alpha=0.6
)
ax4.set_xlabel('TLW')
ax4.set_ylabel('BLW')
ax4.set_zlabel('S')
ax4.set_title('Colored by Min Peak Lambda / Q\n(Only valid samples)')
colorbar4 = plt.colorbar(scatter4, ax=ax4, shrink=0.5, aspect=5)
colorbar4.set_label('Min Lambda / Q')

plt.tight_layout()

# Add semi-transparent planes at tlw=blw for second figure
for ax in [ax3, ax4]:
    # Get axis limits
    tlw_range = geom_df['tlw'].min(), geom_df['tlw'].max()
    s_range = geom_df['s'].min(), geom_df['s'].max()

    # Create mesh for the plane where tlw=blw
    s_plane = np.linspace(s_range[0], s_range[1], 10)
    tlw_plane = np.linspace(tlw_range[0], tlw_range[1], 10)
    S, TLW = np.meshgrid(s_plane, tlw_plane)
    BLW = TLW  # tlw = blw

    # Plot the plane
    ax.plot_surface(TLW, BLW, S, alpha=0.2, color='red')

plt.show()
