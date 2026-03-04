# Converted from 06_apply_correction_cosmology.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
import os

# 1. Define Model Parameters
cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # Reference cosmology for age calculations
SLOPE_PAPER = 0.030  # mag/Gyr

# --- NEW LOOKUP TABLE: AGE EVOLUTION (Solid Line) ---
z_points = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
age_points_solid = np.array([0, 1.0, 1.8, 2.6, 3.3, 3.9, 4.4, 4.8, 5.1, 5.3, 5.5, 5.6, 5.7, 5.8, 5.85, 5.9, 5.95, 6.0])

graph1_curve = interp1d(z_points, age_points_solid, kind='quadratic', fill_value="extrapolate")

# --- POLYNOMIAL CORRECTION SETUP (Rose et al. 2019 Calibration) ---
rose_file = 'Rose19_corrected.csv'
if not os.path.exists(rose_file):
    rose_file = os.path.join(os.path.dirname(__file__) if '__file__' in locals() else '.', 'Rose19_corrected.csv')

poly_func = None
log_age_at_today = None
age_at_today_gyr = None

if os.path.exists(rose_file):
    print(f"Loading calibration data from {rose_file}...")
    df_rose = pd.read_csv(rose_file)
    mask = ~np.isnan(df_rose['HR']) & ~np.isnan(df_rose['logAl']) & ~np.isnan(df_rose['e_HR'])
    df_rose = df_rose[mask].copy()
    weights = 1.0 / (df_rose['e_HR']**2)
    coeffs = np.polyfit(df_rose['logAl'], df_rose['HR'], 2, w=weights)
    poly_func = np.poly1d(coeffs)
    age_at_today_gyr = cosmo.age(0).value * 0.71
    log_age_at_today = np.log10(age_at_today_gyr * 1e9)
    print(f"Polynomial Fit Coefficients: {coeffs}")
    print(f"Reference Age (Today): {age_at_today_gyr:.3f} Gyr")
else:
    print("WARNING: Rose19_corrected.csv not found! Polynomial correction disabled.")

def get_paper_delta_age(z):
    """
    Returns the difference in mean stellar age (Gyr) relative to z=0.
    Using the 'Solid Line' lookup table provided.
    """
    return graph1_curve(z)

# Verification plot
zs = np.linspace(0, 1.7, 100)
d_ages = get_paper_delta_age(zs)
print(f"Delta Age at z=1 (Table): {get_paper_delta_age(1.0):.2f} Gyr")

plt.figure(figsize=(6,4))
plt.plot(zs, d_ages, 'k-', label='Table-based (Solid Line)')
plt.scatter(z_points, age_points_solid, color='red', s=15, alpha=0.5, label='Table Points')
plt.xlabel('Redshift z')
plt.ylabel('Delta Age (Gyr)')
plt.title('Updated Age Evolution: Table-Based')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('output/fig_correction_validation_table.png')
plt.close()


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
    # Since Delta_Age is now positive (Age(0) - Age(z)), we subtract it
    df['age_at_z_gyr'] = age_at_today_gyr - df['delta_age']
    
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

