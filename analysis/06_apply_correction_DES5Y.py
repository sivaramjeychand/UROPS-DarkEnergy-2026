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

def get_paper_delta_age(z):
    """
    Returns the difference in mean stellar age (Gyr) relative to z=0.
    Using the 'Solid Line' lookup table provided.
    """
    return graph1_curve(z)

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
    print("WARNING: Rose19_corrected.csv not found!")

# 2. Load DES-SN5YR Data
DATA_PATH = r'../data/external/DES-SN5YR/4_DISTANCES_COVMAT/DES-Dovekie_HD.csv'

if not os.path.exists(DATA_PATH):
    print(f"Error: {DATA_PATH} not found.")
    exit()

# DES files often have a header line starting with VARNAMES:
# We'll read it manually first
header = None
skip_rows = 0
with open(DATA_PATH, 'r') as f:
    for i, line in enumerate(f):
        if line.startswith('VARNAMES:'):
            header = line.strip().replace('VARNAMES:', '').split()
            skip_rows = i
            break

if header:
    # Read with explicit names, skipping the VARNAMES line and any metadata before it is handled by skiprows ?? 
    # Actually pd.read_csv with skiprows=skip_rows might include the VARNAMES line as the first data line if not careful.
    # We should use names=header and skiprows=skip_rows+1
    df = pd.read_csv(DATA_PATH, delim_whitespace=True, names=header, skiprows=skip_rows+1, comment='#')
else:
    # Fallback to standard read (Pandas might infer)
    df = pd.read_csv(DATA_PATH, delim_whitespace=True, comment='#')

print(f"Loaded {len(df)} DES SNe.")
# Ensure zHD is numeric
df['zHD'] = pd.to_numeric(df['zHD'], errors='coerce')
df = df.dropna(subset=['zHD'])

# 3. Calculate Delta Age and Corrections
df['delta_age'] = df['zHD'].apply(get_paper_delta_age)

# --- Linear Correction ---
df['bias_linear'] = SLOPE_PAPER * df['delta_age']
print(f"Linear Correction: Max bias = {df['bias_linear'].max():.3f} mag")

# --- Polynomial Correction ---
if poly_func is not None:
    df['age_at_z_gyr'] = age_at_today_gyr - df['delta_age']
    df['log_age_at_z'] = np.log10(df['age_at_z_gyr'] * 1e9)
    df['bias_poly'] = poly_func(df['log_age_at_z']) - poly_func(log_age_at_today)
    print(f"Polynomial Correction: Max bias = {df['bias_poly'].max():.3f} mag")
else:
    df['bias_poly'] = 0.0

# 4. Save Outputs
OUT_DIR = r'../data/external/DES-SN5YR_Corrected/4_DISTANCES_COVMAT'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Save Linear Corrected
df_linear = df.copy()
# DES usually has 'MU' and 'MUERR'. Pantheon+ had 'MU_SH0ES'.
# We update 'MU'.
if 'MU' in df_linear.columns:
    df_linear['MU'] = df_linear['MU'] - df_linear['bias_linear']
elif 'MU_SH0ES' in df_linear.columns:
    df_linear['MU_SH0ES'] = df_linear['MU_SH0ES'] - df_linear['bias_linear']

cols_save_lin = [c for c in df_linear.columns if c not in ['delta_age', 'bias_linear', 'bias_poly', 'age_at_z_gyr', 'log_age_at_z']]
df_linear[cols_save_lin].to_csv(os.path.join(OUT_DIR, 'DES-Dovekie_HD_Linear.csv'), sep=' ', index=False)
print(f"Saved Linear Corrected to {OUT_DIR}/DES-Dovekie_HD_Linear.csv")

# Save Poly Corrected
df_poly = df.copy()
if 'MU' in df_poly.columns:
    df_poly['MU'] = df_poly['MU'] - df_poly['bias_poly']
elif 'MU_SH0ES' in df_poly.columns:
    df_poly['MU_SH0ES'] = df_poly['MU_SH0ES'] - df_poly['bias_poly']

cols_save_poly = [c for c in df_poly.columns if c not in ['delta_age', 'bias_linear', 'bias_poly', 'age_at_z_gyr', 'log_age_at_z']]
df_poly[cols_save_poly].to_csv(os.path.join(OUT_DIR, 'DES-Dovekie_HD_Poly.csv'), sep=' ', index=False)
print(f"Saved Linear Corrected to {OUT_DIR}/DES-Dovekie_HD_Poly.csv")

# Also save uncorrected (copy) for plotting ease if needed, or assume original
