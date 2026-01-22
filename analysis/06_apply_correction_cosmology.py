# Converted from 06_apply_correction_cosmology.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import os

# 1. Define Model Parameters
cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # Reference cosmology for age calculations
SLOPE_PAPER = -0.030  # mag/Gyr

def get_paper_delta_age(z):
    """
    Calculates the change in median progenitor age relative to z=0.
    Matches Son et al. Fig 2: Delta_Age ~ -5.3 Gyr at z=1.
    """
    t0 = cosmo.age(0).value
    tz = cosmo.age(z).value
    delta_cosmic = tz - t0
    # Scale factor (5.3 / 7.7 ≈ 0.69-0.71)
    return delta_cosmic * 0.71

# Verification plot
zs = np.linspace(0, 1.5, 100)
d_ages = get_paper_delta_age(zs)
print(f"Delta Age at z=1: {get_paper_delta_age(1.0):.2f} Gyr (Target: -5.3)")

plt.figure(figsize=(6,4))
plt.plot(zs, d_ages, 'b-', label='Model: $0.71 \times \Delta t_{cosmic}$')
plt.axhline(-5.3, color='r', linestyle='--', label='Target at z=1')
plt.axvline(1.0, color='r', linestyle='--')
plt.xlabel('Redshift z')
plt.ylabel('$\Delta$ Progenitor Age (Gyr)')
plt.title('Calibrated Age Evolution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('output/fig_correction_validation.png')
plt.show()

# 2. Load and Apply
DATA_DIR = r'../data/external/PantheonPlus/Pantheon+_Data/4_DISTANCES_AND_COVAR'
PATH_DAT = os.path.join(DATA_DIR, 'Pantheon+SH0ES.dat')

df = pd.read_csv(PATH_DAT, sep='\s+', skiprows=0 if 'CID' in open(PATH_DAT).read(100) else 1)
print(f"Loaded {len(df)} SNe.")

# Calculate Delta Age and Bias
df['delta_age'] = df['zHD'].apply(get_paper_delta_age)
df['bias'] = SLOPE_PAPER * df['delta_age']
df['MU_CORRECTED'] = df['MU_SH0ES'] - df['bias']

print(f"Max Correction: {df['bias'].max():.3f} mag at z={df['zHD'].max():.2f}")

# 3. Save Corrected Dataset
OUT_DIR = r'../data/external/PantheonPlus_Corrected/Pantheon+_Data/4_DISTANCES_AND_COVAR'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

df_out = df.copy()
df_out['MU_SH0ES'] = df_out['MU_CORRECTED']
df_out['m_b_corr'] = df_out['m_b_corr'] - df_out['bias']

# Final Save
cols_to_save = [c for c in df_out.columns if c not in ['delta_age', 'bias', 'MU_CORRECTED']]
df_out[cols_to_save].to_csv(os.path.join(OUT_DIR, 'Pantheon+SH0ES.dat'), sep=' ', index=False, float_format='%.5f')
print(f"Corrected data saved to {OUT_DIR}/Pantheon+SH0ES.dat")

