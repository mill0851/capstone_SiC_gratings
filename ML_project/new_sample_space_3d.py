import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import *

# Read batch 2 coordinates
df_original = pd.read_csv('data/batch2/batch2_geometries_1024.csv')

# Create new dataframe with blw shifted down by 1.75 um
df_shifted = df_original.copy()
df_shifted['blw'] = df_shifted['blw'] - 1.75

# Remove points where (tlw, blw) is above the blw = tlw + 1.75 line
# A point is above the line if blw > tlw + 1.75
df_filtered = df_shifted[df_shifted['blw'] <= df_shifted['tlw'] + 1.75].copy()

# Create third dataframe: mirror original across blw = tlw + 1.75 line
# Mirroring formula: (tlw', blw') = (blw - 1.75, tlw + 1.75)
df_mirrored = df_original.copy()
df_mirrored['tlw'] = df_original['blw'] - 1.75
df_mirrored['blw'] = df_original['tlw'] + 1.75

# Remove points outside the region bounded by:
# 1) blw = tlw + 1.75 (upper bound)
# 2) blw = 3.25 (lower bound)
# 3) tlw = 3.5 (right bound)
df_mirrored_filtered = df_mirrored[
    (df_mirrored['blw'] <= df_mirrored['tlw'] + 1.75) &
    (df_mirrored['blw'] >= 3.25) &
    (df_mirrored['tlw'] <= 3.5)
].copy()

# Create dataframe of new points (shifted + mirrored) before filtering
df_new_points = pd.concat([df_filtered, df_mirrored_filtered], ignore_index=True)

# Concatenate all 3 dataframes
df_combined = pd.concat([df_original, df_new_points], ignore_index=True)

# Remove all points where tlw > blw
df_combined = df_combined[df_combined['tlw'] <= df_combined['blw']].copy()

# Also filter new points with tlw > blw
df_new_points_filtered = df_new_points[df_new_points['tlw'] <= df_new_points['blw']].copy()

# Plot all points in 3D
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot original points in blue
scatter1 = ax.scatter(
    df_original['tlw'],
    df_original['blw'],
    df_original['s'],
    c='blue',
    s=30,
    alpha=0.6,
    label='Original'
)

# Plot new points (shifted + mirrored) in purple
scatter2 = ax.scatter(
    df_new_points_filtered['tlw'],
    df_new_points_filtered['blw'],
    df_new_points_filtered['s'],
    c='purple',
    s=30,
    alpha=0.6,
    label='New Points'
)

# Add semi-transparent red plane at tlw=blw
tlw_range = df_combined['tlw'].min(), df_combined['tlw'].max()
s_range = df_combined['s'].min(), df_combined['s'].max()

s_plane = np.linspace(s_range[0], s_range[1], 10)
tlw_plane = np.linspace(tlw_range[0], tlw_range[1], 10)
S, TLW = np.meshgrid(s_plane, tlw_plane)
BLW = TLW  # tlw = blw
ax.plot_surface(TLW, BLW, S, alpha=0.1, color='red')

# Add semi-transparent green plane at blw = tlw + 1.75
BLW_offset = TLW + 1.75
ax.plot_surface(TLW, BLW_offset, S, alpha=0.1, color='green')

ax.set_xlabel('TLW')
ax.set_ylabel('BLW')
ax.set_zlabel('S')
ax.set_title('Original and Shifted Sample Space')
ax.legend()

plt.tight_layout()
plt.show()

# Plot full sample space colored by height (h)
fig2 = plt.figure(figsize=(12, 9))
ax2 = fig2.add_subplot(111, projection='3d')

scatter_h = ax2.scatter(
    df_combined['tlw'],
    df_combined['blw'],
    df_combined['s'],
    c=df_combined['h'],
    cmap='viridis',
    s=30,
    alpha=0.7,
)

cbar = fig2.colorbar(scatter_h, ax=ax2, shrink=0.6, pad=0.1)
cbar.set_label('H', fontsize=18)

# Add semi-transparent red plane at tlw=blw
ax2.plot_surface(TLW, BLW, S, alpha=0.1, color='red')

ax2.set_xlabel('TLW', fontsize=18)
ax2.set_ylabel('BLW', fontsize=18)
ax2.set_zlabel('S', fontsize=18)
ax2.set_title('Full Sample Space Colored by Height')

plt.tight_layout()
plt.show()

# Set df_final to df_combined (no additional transformations)
df_final = df_combined.copy()

# Create dataframe of all new points from all transformations
df_all_new_points = df_final[len(df_original):].reset_index(drop=True)

# Round all values to 3 decimal places
df_combined_rounded = df_combined.round(3)
df_all_new_points_rounded = df_all_new_points.round(3)
df_final_rounded = df_final.round(3)

# Save all dataframes
df_combined_rounded.to_csv('new_sample_space.csv', index=False)
df_all_new_points_rounded.to_csv('new_sample_space_new_points_only.csv', index=False)
df_final_rounded.to_csv('new_sample_space_final.csv', index=False)

print(f"\n--- Initial Processing ---")
print(f"Original points: {len(df_original)}")
print(f"Filtered shifted points: {len(df_filtered)}")
print(f"Filtered mirrored points: {len(df_mirrored_filtered)}")
print(f"Total new points (after tlw > blw filter): {len(df_new_points_filtered)}")

print(f"\n--- Final Results ---")
print(f"Original points: {len(df_original)}")
print(f"Total non-original points: {len(df_final) - len(df_original)}")
print(f"Final total points: {len(df_final)}")

print(f"\n--- Files Saved ---")
print("Saved to new_sample_space.csv")
print("Saved to new_sample_space_new_points_only.csv")
print("Saved to new_sample_space_final.csv")
