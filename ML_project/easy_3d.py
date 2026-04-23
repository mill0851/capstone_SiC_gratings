import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import *

# Read the CSV file
df = pd.read_csv('new_sample_space_new_points_only.csv')

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(df['tlw'], df['blw'], df['s'], s=30, alpha=0.6)

# Set labels
ax.set_xlabel('TLW')
ax.set_ylabel('BLW')
ax.set_zlabel('S')

plt.tight_layout()
plt.show()
