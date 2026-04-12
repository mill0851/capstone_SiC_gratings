import pandas as pd
import matplotlib.pyplot as plt

# Read in data and 
df = pd.read_csv('data1.csv')

# PLOT
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(df['wavelength_um'] * 1e6, df['reflectance_0'], 'b-', linewidth=2)
ax1.set_xlabel('Wavelength (μm)', fontsize=12)
ax1.set_ylabel('Reflectance', fontsize=12)
ax1.set_title('Reflectance vs Wavelength', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(df['wavelength_um'] * 1e6, df['Absorption'], 'r-', linewidth=2)
ax2.set_xlabel('Wavelength (μm)', fontsize=12)
ax2.set_ylabel('Absorption', fontsize=12)
ax2.set_title('Absorption vs Wavelength', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()