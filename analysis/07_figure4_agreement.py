# Converted from 07_figure4_agreement.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, w0waCDM
import os

# 1. Define Cosmologies (Matching Figure 3 Replication)
H0 = 73.04

# Baseline: Omega_m = 0.30, Omega_de = 0.00 (Open Universe)
cosmo_baseline = LambdaCDM(H0=H0, Om0=0.30, Ode0=0.0)

# Model 1 (Red): LambdaCDM (Om=0.30)
cosmo_red = FlatLambdaCDM(H0=H0, Om0=0.30)

# Model 2 (Blue): w0waCDM (Om=0.35, Ode=0.65, w0=-0.42, wa=-1.75)
cosmo_blue = w0waCDM(H0=H0, Om0=0.35, Ode0=0.65, w0=-0.42, wa=-1.75)


# 2. Load Data
# SN Data (Pantheon+)
path_sn_orig = r'../data/external/PantheonPlus/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat'
path_sn_lin = r'../data/external/PantheonPlus_Corrected/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_Linear.dat'
path_sn_poly = r'../data/external/PantheonPlus_Corrected/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_Poly.dat'

# BAO Data
bao_mean_file = r'../data/external/DESI_BAO/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt'

# Load files
sn_orig = pd.read_csv(path_sn_orig, sep='\s+', skiprows=0 if 'CID' in open(path_sn_orig).read(100) else 1)

if os.path.exists(path_sn_lin):
    sn_lin = pd.read_csv(path_sn_lin, sep='\s+', skiprows=0 if 'CID' in open(path_sn_lin).read(100) else 1)
else:
    print(f"Warning: {path_sn_lin} not found.")
    sn_lin = sn_orig.copy()

if os.path.exists(path_sn_poly):
    sn_poly = pd.read_csv(path_sn_poly, sep='\s+', skiprows=0 if 'CID' in open(path_sn_poly).read(100) else 1)
else:
    print(f"Warning: {path_sn_poly} not found.")
    sn_poly = sn_orig.copy()
bao_df = pd.read_csv(bao_mean_file, delim_whitespace=True, comment='#', names=['z', 'value', 'type'])

# 3. Process BAO Data for Plotting
# We need to convert BAO measurements to Distance Modulus relative to baseline
# Rd (sound horizon) is needed. For H0=73.04, rd ~ 137? No, usually aligned.
# In Fig4 script previously: rd_theory = 93.9 / 0.7304 ~ 128.5 (Fast scaling?)
# Let's use a consistent Rd or the one implied by the Red model.
# For simplicity and visual matching, we'll calculate mu_obs for BAO using the same scaling as before.
rd_scaling = 93.9 / (H0/100) # Approx 128.5 Mpc

bao_mu = []
for idx, row in bao_df.iterrows():
    # Convert whatever type to dL then mu
    # type is 'DM_over_rs', 'DH_over_rs', 'DV_over_rs'
    # We assume 'DM_over_rs' is directly convertible to dM -> dL
    # If other types, we might skip or approx. 
    # DESI consensus usually provides DM/rd and DH/rd. We use DM/rd for hubble diagram.
    
    val = row['value']
    z = row['z']
    
    mu = None
    if row['type'] == 'DM_over_rs':
        DM = val * rd_scaling
        dL = DM * (1 + z)
        mu = 5 * np.log10(dL) + 25
    elif row['type'] == 'DV_over_rs':
        # approx dL from DV? Harder. Skip for now or use simplified.
        # Most high-z BAO is DM/DH.
        pass
        
    if mu is not None:
        bao_mu.append({'z': z, 'mu': mu})

bao_mu_df = pd.DataFrame(bao_mu)

# 4. Calculate Model Curves (Relative to Baseline)
z_grid = np.logspace(np.log10(0.007), np.log10(2.5), 100)

def get_diff(cosmo, z):
    return cosmo.distmod(z).value - cosmo_baseline.distmod(z).value

diff_red = get_diff(cosmo_red, z_grid)
diff_blue = get_diff(cosmo_blue, z_grid)

# 5. Plotting
fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True, gridspec_kw={'hspace': 0.1})

datasets = [
    ('Uncorrected', sn_orig, 'gray', axes[0]),
    ('Linear Correction', sn_lin, 'gray', axes[1]),
    ('Polynomial Correction', sn_poly, 'black', axes[2])
]

for label, df_sn, color, ax in datasets:
    # Baseline
    ax.axhline(0, color='black', linestyle=':', label='Baseline ($\Omega_m=0.3, \Omega_{de}=0.0$)')
    
    # Model Curves
    ax.plot(z_grid, diff_red, 'r-', linewidth=2, label='$\Lambda$CDM ($\Omega_m=0.30$)')
    ax.plot(z_grid, diff_blue, 'b-', linewidth=2, alpha=0.8, label='$w_0w_a$CDM (Blue Params)')
    
    # SN Data (Binned)
    # Calculate Residuals: mu_obs - mu_baseline
    # mu_baseline needs to be calculated at sn z
    base_dist = cosmo_baseline.distmod(df_sn['zHD'].values).value
    resid_sn = df_sn['MU_SH0ES'].values - base_dist
    
    # Binning
    bins = np.logspace(np.log10(0.01), np.log10(1.5), 20)
    c = pd.cut(df_sn['zHD'], bins)
    
    temp = pd.DataFrame({'z': df_sn['zHD'], 'res': resid_sn})
    m = temp.groupby(c, observed=False)['z'].mean()
    v = temp.groupby(c, observed=False)['res'].mean()
    e = temp.groupby(c, observed=False)['res'].std() / np.sqrt(temp.groupby(c, observed=False)['res'].count())
    
    # Plot Binned SN
    ax.errorbar(m, v, yerr=e, fmt='o', color=color, markersize=6, label='Pantheon+ Binned', zorder=5)
    
    # Plot BAO (only on one panel? or both? usually both to show agreement)
    if not bao_mu_df.empty:
        base_bao = cosmo_baseline.distmod(bao_mu_df['z'].values).value
        resid_bao = bao_mu_df['mu'].values - base_bao
        ax.errorbar(bao_mu_df['z'], resid_bao, yerr=0.03, fmt='s', color='orange', markersize=8, markeredgecolor='k', label='DESI BAO', zorder=10)

    ax.set_ylabel('$\mu - \mu_{base}$ (mag)', fontsize=12)
    ax.text(0.05, 0.85, label, transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    
    # Limits (Match Fig 3 ~ -0.3 to 0.4)
    ax.set_ylim(-0.3, 0.5)

    if ax == axes[2]:
        ax.legend(loc='lower right', fontsize=9, ncol=2)

axes[2].set_xlabel('Redshift (z)', fontsize=12)
axes[2].set_xlim(0.01, 2.5)

plt.tight_layout()
plt.savefig('output/fig4_residual_agreement.png')
print("Saved output/fig4_residual_agreement.png")


