# Converted from 04_figure3_replication.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, w0waCDM, FlatwCDM
import os


# 2. Define Cosmologies
# H0 should match the data derivation roughly, but cancels in residuals if M is fitted. We use 73.04 typical.
H0 = 73.04

# Baseline: Omega_m = 0.30, Omega_de = 0.00 (Open Universe)
cosmo_baseline = LambdaCDM(H0=H0, Om0=0.30, Ode0=0.0)

# 1. Load Data
# We now use the processed files from Step 6 for consistency
path_sn_orig = r'../data/external/PantheonPlus/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat'
path_sn_lin = r'../data/external/PantheonPlus_Corrected/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_Linear.dat'
path_sn_poly = r'../data/external/PantheonPlus_Corrected/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_Poly.dat'

# Load files
if os.path.exists(path_sn_orig):
    sn_orig = pd.read_csv(path_sn_orig, sep='\s+', skiprows=0 if 'CID' in open(path_sn_orig).read(100) else 1)
else:
    print("Error: Original SN file not found.")
    exit()

if os.path.exists(path_sn_lin):
    sn_lin = pd.read_csv(path_sn_lin, sep='\s+', skiprows=0 if 'CID' in open(path_sn_lin).read(100) else 1)
else:
    sn_lin = sn_orig.copy(); print("Linear missing.")

if os.path.exists(path_sn_poly):
    sn_poly = pd.read_csv(path_sn_poly, sep='\s+', skiprows=0 if 'CID' in open(path_sn_poly).read(100) else 1)
else:
    sn_poly = sn_orig.copy(); print("Poly missing.")


# Baseline: Omega_m = 0.30, Omega_de = 0.00 (Open Universe)
# Note: LambdaCDM with Ode0=0 is effectively open if Om0 < 1
cosmo_baseline = LambdaCDM(H0=H0, Om0=0.30, Ode0=0.0)

# Old Reference (Implicit in CSV HR): Flat LambdaCDM, Om=0.3
cosmo_old_ref = FlatLambdaCDM(H0=H0, Om0=0.30)

# Model 1 (Red): LambdaCDM (Om=0.30, Ode=0.70)
cosmo_red = FlatLambdaCDM(H0=H0, Om0=0.30)

# Model 2 (Blue): w0waCDM (Om=0.35, Ode=0.65, w0=-0.42, wa=-1.75)
cosmo_blue = w0waCDM(H0=H0, Om0=0.35, Ode0=0.65, w0=-0.42, wa=-1.75)

# Model 3 (Green): w0waCDM (Om=0.32, Ode=0.68, w0=-0.75, wa=-0.86)
cosmo_green = w0waCDM(H0=H0, Om0=0.32, Ode0=0.68, w0=-0.75, wa=-0.86)

# 3. Calculate Curves relative to Baseline
z_grid = np.logspace(np.log10(0.01), np.log10(1.3), 100)

def get_diff(cosmo, z):
    return cosmo.distmod(z).value - cosmo_baseline.distmod(z).value

diff_red = get_diff(cosmo_red, z_grid)
diff_blue = get_diff(cosmo_blue, z_grid)
diff_green = get_diff(cosmo_green, z_grid)

# Load Best Fits
best_fits = {}
mcmc_file = r'chains_emcee_all/best_fits.csv'
if os.path.exists(mcmc_file):
    try:
        df_fits = pd.read_csv(mcmc_file)
        # Columns: Dataset, Om, w, H0
        for _, row in df_fits.iterrows():
            best_fits[row['Dataset']] = row
        print("Loaded MCMC Best Fits.")
    except Exception as e:
        print(f"Failed to load MCMC fits: {e}")

# Transform Data Distances
# Current HR = mu_obs - mu_old_ref
# We want HR_new = mu_obs - mu_baseline
# so HR_new = (HR + mu_old_ref) - mu_baseline
#           = HR + (mu_old_ref - mu_baseline)

# 4. Calculate Residuals for Plotting
# HR = MU_SH0ES - MU_Model_Baseline
# Note: Pantheon+ MU_SH0ES includes H0 ~ 73.04 scaling.
# Baseline cosmo should match this.

base_dist_orig = cosmo_baseline.distmod(sn_orig['zHD'].values).value
sn_orig['HR_plot'] = sn_orig['MU_SH0ES'] - base_dist_orig

base_dist_lin = cosmo_baseline.distmod(sn_lin['zHD'].values).value
sn_lin['HR_plot'] = sn_lin['MU_SH0ES'] - base_dist_lin

base_dist_poly = cosmo_baseline.distmod(sn_poly['zHD'].values).value
sn_poly['HR_plot'] = sn_poly['MU_SH0ES'] - base_dist_poly

print("Cosmologies defined and data transformed.")

# 5. Plotting
fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True, gridspec_kw={'hspace': 0.1})

# Loop for Panels
datasets = [
    ('Before Correction', sn_orig, 'gray', axes[0]),
    ('Linear Correction', sn_lin, 'gray', axes[1]),
    ('Polynomial Correction', sn_poly, 'black', axes[2])
]


for label, df_sn, color, ax in datasets:
    data_y = df_sn['HR_plot']
    z_vals = df_sn['zHD']
    
    # Plot Baseline
    ax.axhline(0, color='black', linestyle=':', label='Baseline ($\Omega_m=0.3, \Omega_{de}=0.0$)')
    
    # Plot Model Curves
    # Note: Curves are identical in both panels as they represent models
    ax.plot(z_grid, diff_red, 'r-', linewidth=2, label='$\Lambda$CDM ($\Omega_m=0.30$)')
    ax.plot(z_grid, diff_blue, 'b-', linewidth=2, alpha=0.8, label='$w_0w_a$CDM (Blue Params)')
    ax.plot(z_grid, diff_green, 'g--', linewidth=2, alpha=0.8, label='$w_0w_a$CDM (Green Params)')
    
    # Plot MCMC Best Fit
    fit_name = None
    if label == 'Before Correction': fit_name = 'Pantheon_Uncorr'
    elif label == 'Linear Correction': fit_name = 'Pantheon_Lin'
    elif label == 'Polynomial Correction': fit_name = 'Pantheon_Poly'
    
    if fit_name and fit_name in best_fits:
        p = best_fits[fit_name]
        # Create Cosmo
        # Note: MCMC w is w0. wa=0.
        cosmo_fit = FlatwCDM(H0=p['H0'], Om0=p['Om'], w0=p['w'])
        diff_fit = get_diff(cosmo_fit, z_grid)
        # Apply the same offset as data: None for Pantheon as we didn't shift it.
        diff_fit_shifted = diff_fit 
        
        # Add label with params
        lbl_fit = f"MCMC Best Fit"
        ax.plot(z_grid, diff_fit_shifted, color='cyan', linestyle='--', linewidth=3, label=lbl_fit)

    # Plot Individual Points
    ax.scatter(z_vals, data_y, color=color, alpha=0.05, s=10, zorder=1)
    
    # Binned Statistics
    # Binned Statistics (50 SNe per bin)
    # Sort by redshift
    df_sorted = pd.DataFrame({'z': z_vals, 'HR': data_y}).sort_values('z')
    df_sorted['bin_id'] = np.arange(len(df_sorted)) // 50
    
    # Aggregation
    group = df_sorted.groupby('bin_id')
    m = group['z'].mean()
    v = group['HR'].mean()
    e = group['HR'].std() / np.sqrt(group['HR'].count())
    
    # Use light blue for points as in the image (or similar)
    ax.errorbar(m, v, yerr=e, fmt='o', color='dodgerblue', markersize=6, 
                markeredgecolor='blue', ecolor='blue', capsize=2, label='Binned Data', zorder=10)

    ax.set_ylabel('HR (mag)', fontsize=12)
    ax.text(0.05, 0.85, label, transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.grid(True, which='both', linestyle=':', alpha=0.3)

    # Legend only on top or bottom? User image has it on both or specific.
    ax.text(0.05, 0.85, label, transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.grid(True, which='both', linestyle=':', alpha=0.3)

    if ax == axes[2]:
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=2)

axes[2].set_xlabel('Redshift (z)', fontsize=12)
axes[2].set_xlim(0, 1.3)

# Determine Y-limits to match image roughly
axes[0].set_ylim(-0.3, 0.4)
axes[1].set_ylim(-0.3, 0.4)
axes[2].set_ylim(-0.3, 0.4)

plt.tight_layout()
plt.savefig('output/fig3_replication_comparison.png')
plt.show()

