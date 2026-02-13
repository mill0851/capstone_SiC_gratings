import pandas as pd
import numpy as np

# Load your original CSV
df = pd.read_csv(r"C:\Users\robert\Code\capstone\capstone_SiC_gratings\data\ML\tlw_debug_sweep\runs\run_001\data.csv")   # <-- replace with your actual filename

# Ensure wavelength is numeric
df["wavelength_um"] = pd.to_numeric(df["wavelength_um"])

# Compute cosine reflectance across full wavelength span
lam_min = df["wavelength_um"].min()
lam_max = df["wavelength_um"].max()

df["reflectance_0"] = (
    0.5
    + np.cos(
        2 * np.pi *
        (df["wavelength_um"] - lam_min) /
        (lam_max - lam_min)
    )
)

# Save full corrected file
df.to_csv("data_cosine_full.csv", index=False)

print("Done. Saved as data_cosine_full.csv")