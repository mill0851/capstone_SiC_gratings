import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('data2.csv')

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Reflectance vs Wavelength
ax1.plot(df['wavelength_um'] * 1e6, df['reflectance_0'], 'b-', linewidth=2)
ax1.set_xlabel('Wavelength (μm)', fontsize=12)
ax1.set_ylabel('Reflectance', fontsize=12)
ax1.set_title('Reflectance vs Wavelength', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Absorption vs Wavelength
ax2.plot(df['wavelength_um'] * 1e6, df['Absorption'], 'r-', linewidth=2)
ax2.set_xlabel('Wavelength (μm)', fontsize=12)
ax2.set_ylabel('Absorption', fontsize=12)
ax2.set_title('Absorption vs Wavelength', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Adjust layout and show
plt.tight_layout()
plt.savefig('plots.png', dpi=300, bbox_inches='tight')
plt.show()