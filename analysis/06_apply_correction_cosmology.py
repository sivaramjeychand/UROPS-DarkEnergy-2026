# Converted from 06_apply_correction_cosmology.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import os

# 1. Define Model Parameters
cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # Reference cosmology for age calculations
cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # Reference cosmology for age calculations
SLOPE_PAPER = -0.030  # mag/Gyr

# --- POLYNOMIAL CORRECTION SETUP ---
# Load Rose+19 Data for calibration
rose_file = 'Rose19_corrected.csv' # Assumes running in analysis/ dir
if not os.path.exists(rose_file):
    rose_file = os.path.join(os.path.dirname(__file__) if '__file__' in locals() else '.', 'Rose19_corrected.csv')

poly_func = None
log_age_at_today = None
age_at_today_gyr = None

if os.path.exists(rose_file):
    print(f"Loading calibration data from {rose_file}...")
    df_rose = pd.read_csv(rose_file)
    # Clean data
    mask = ~np.isnan(df_rose['HR']) & ~np.isnan(df_rose['logAl']) & ~np.isnan(df_rose['e_HR'])
    df_rose = df_rose[mask].copy()
    
    # Weighted Polynomial Fit (Degree 2) on LOG AGE (consistent with Fig 1)
    weights = 1.0 / (df_rose['e_HR']**2)
    # logAl is typically Log10(Age [yr])
    coeffs = np.polyfit(df_rose['logAl'], df_rose['HR'], 2, w=weights)
    poly_func = np.poly1d(coeffs)
    
    # Reference Age (Mean Progenitor Age at z=0)
    # We use the scaled cosmic age as the baseline for "Today"
    age_at_today_gyr = cosmo.age(0).value * 0.71
    # Convert to log10(years)
    log_age_at_today = np.log10(age_at_today_gyr * 1e9)
    
    print(f"Polynomial Fit Coefficients (High to Low): {coeffs}")
    print(f"Reference Age (Today): {age_at_today_gyr:.3f} Gyr -> LogAge: {log_age_at_today:.3f}")
else:
    print("WARNING: Rose19_corrected.csv not found! Fallback to Linear Slope.")
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

# Calculate Delta Age
df['delta_age'] = df['zHD'].apply(get_paper_delta_age)

# --- 1. Linear Correction ---
df['bias_linear'] = SLOPE_PAPER * df['delta_age']
print(f"Linear Correction: Max bias = {df['bias_linear'].max():.3f} mag")

# --- 2. Polynomial Correction ---
if poly_func is not None:
    # Polynomial Correction in Log Space
    # 1. Calculate Linear Age at z (Gyr)
    df['age_at_z_gyr'] = age_at_today_gyr + df['delta_age']
    
    # 2. Convert to Log10(Years)
    df['log_age_at_z'] = np.log10(df['age_at_z_gyr'] * 1e9)
    
    # 3. Correction = P(LogAge_z) - P(LogAge_today)
    df['bias_poly'] = poly_func(df['log_age_at_z']) - poly_func(log_age_at_today)
    print(f"Polynomial Correction: Max bias = {df['bias_poly'].max():.3f} mag")
else:
    print("Polynomial fit failed/missing. Setting bias_poly = 0.")
    df['bias_poly'] = 0.0

# Apply to MU (We will save separate files)
# df['MU_CORRECTED'] = df['MU_SH0ES'] - df['bias'] # Old way

# print(f"Max Correction: {df['bias'].max():.3f} mag at z={df['zHD'].max():.2f}")

# 3. Save Corrected Dataset
OUT_DIR = r'../data/external/PantheonPlus_Corrected/Pantheon+_Data/4_DISTANCES_AND_COVAR'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Save Linear Corrected
df_linear = df.copy()
df_linear['MU_SH0ES'] = df_linear['MU_SH0ES'] - df_linear['bias_linear']
df_linear['m_b_corr'] = df_linear['m_b_corr'] - df_linear['bias_linear']
cols_save = [c for c in df_linear.columns if c not in ['delta_age', 'bias_linear', 'bias_poly', 'age_at_z_gyr', 'log_age_at_z', 'MU_CORRECTED', 'bias']]
df_linear[cols_save].to_csv(os.path.join(OUT_DIR, 'Pantheon+SH0ES_Linear.dat'), sep=' ', index=False, float_format='%.5f')
print(f"Saved Linear Corrected to {OUT_DIR}/Pantheon+SH0ES_Linear.dat")

# Save Polynomial Corrected
df_poly = df.copy()
df_poly['MU_SH0ES'] = df_poly['MU_SH0ES'] - df_poly['bias_poly']
df_poly['m_b_corr'] = df_poly['m_b_corr'] - df_poly['bias_poly']
df_poly[cols_save].to_csv(os.path.join(OUT_DIR, 'Pantheon+SH0ES_Poly.dat'), sep=' ', index=False, float_format='%.5f')
print(f"Saved Polynomial Corrected to {OUT_DIR}/Pantheon+SH0ES_Poly.dat")

