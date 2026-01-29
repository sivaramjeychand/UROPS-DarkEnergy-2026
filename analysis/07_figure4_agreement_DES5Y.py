
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, w0waCDM
import os

# 1. Define Cosmologies
H0 = 73.04
cosmo_baseline = LambdaCDM(H0=H0, Om0=0.30, Ode0=0.0)
cosmo_red = FlatLambdaCDM(H0=H0, Om0=0.30)
cosmo_blue = w0waCDM(H0=H0, Om0=0.35, Ode0=0.65, w0=-0.42, wa=-1.75)

# 2. Load Data (DES)
path_orig = r'../data/external/DES-SN5YR/4_DISTANCES_COVMAT/DES-Dovekie_HD.csv'
path_lin = r'../data/external/DES-SN5YR_Corrected/4_DISTANCES_COVMAT/DES-Dovekie_HD_Linear.csv'
path_poly = r'../data/external/DES-SN5YR_Corrected/4_DISTANCES_COVMAT/DES-Dovekie_HD_Poly.csv'

# BAO Data
bao_mean_file = r'../data/external/DESI_BAO/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt'
bao_df = pd.read_csv(bao_mean_file, delim_whitespace=True, comment='#', names=['z', 'value', 'type'])

# Process BAO to mu
rd_scaling = 93.9 / (H0/100)
bao_mu = []
for idx, row in bao_df.iterrows():
    val = row['value']; z = row['z']
    mu = None
    if row['type'] == 'DM_over_rs':
        DM = val * rd_scaling
        dL = DM * (1 + z)
        mu = 5 * np.log10(dL) + 25
    if mu is not None:
        bao_mu.append({'z': z, 'mu': mu})
bao_mu_df = pd.DataFrame(bao_mu)

# Function to load DES
def load_des(path, is_raw=False):
    if not os.path.exists(path):
        print(f"File not found: {path}"); return None
    if is_raw:
        header = None; skip = 0
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('VARNAMES:'):
                    header = line.strip().replace('VARNAMES:', '').split()
                    skip = i + 1; break
        if header:
            return pd.read_csv(path, delim_whitespace=True, names=header, skiprows=skip, comment='#')
        return pd.read_csv(path, delim_whitespace=True, comment='#')
    else:
        return pd.read_csv(path, sep=' ')

sn_orig = load_des(path_orig, is_raw=True)
sn_lin = load_des(path_lin)
sn_poly = load_des(path_poly)

if sn_orig is None: exit()
if sn_lin is None: sn_lin = sn_orig.copy()
if sn_poly is None: sn_poly = sn_orig.copy()

# Calculate Residuals
for df in [sn_orig, sn_lin, sn_poly]:
    df['zHD'] = pd.to_numeric(df['zHD'], errors='coerce')
    df['MU'] = pd.to_numeric(df['MU'], errors='coerce')
    df.dropna(subset=['zHD', 'MU'], inplace=True)
    base_dist = cosmo_baseline.distmod(df['zHD'].values).value
    df['HR_plot'] = df['MU'] - base_dist

# Shift Residuals (Center low-z around 0)
# print("Anchoring each dataset individually to 0 at low-z...")
for df in [sn_orig, sn_lin, sn_poly]:
    # 1. Find low-z stars IN THIS SPECIFIC DATAFRAME
    mask_anchor = df['zHD'] < 0.1
    
    # 2. Calculate the median offset specific to THIS dataset
    if mask_anchor.sum() > 0:
        hr_offset = df.loc[mask_anchor, 'HR_plot'].median()
    else:
        hr_offset = df['HR_plot'].median()
        
    # 3. Apply the specific offset to the specific dataframe
    df['HR_plot'] = df['HR_plot'] - hr_offset

    print(f"Shifted by {hr_offset:.3f} mag")

# 3. Plotting
z_grid = np.logspace(np.log10(0.01), np.log10(2.5), 100)

def get_diff(cosmo, z):
    return cosmo.distmod(z).value - cosmo_baseline.distmod(z).value

diff_red = get_diff(cosmo_red, z_grid)
diff_blue = get_diff(cosmo_blue, z_grid)

fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True, gridspec_kw={'hspace': 0.1})

datasets = [
    ('Uncorrected (DES5Y)', sn_orig, 'gray', axes[0]),
    ('Linear Correction (DES5Y)', sn_lin, 'gray', axes[1]),
    ('Polynomial Correction (DES5Y)', sn_poly, 'black', axes[2])
]

for label, df_sn, color, ax in datasets:
    # Baseline
    ax.axhline(0, color='black', linestyle=':', label='Baseline')
    
    # Models
    ax.plot(z_grid, diff_red, 'r-', linewidth=2, label='$\Lambda$CDM')
    ax.plot(z_grid, diff_blue, 'b-', linewidth=2, alpha=0.8, label='$w_0w_a$CDM')
    
    # Binning SN
    # Binning SN (50 SNe per bin)
    z_vals = df_sn['zHD']
    df_sorted = pd.DataFrame({'z': z_vals, 'HR': df_sn['HR_plot']}).sort_values('z')
    df_sorted['bin_id'] = np.arange(len(df_sorted)) // 50
    
    group = df_sorted.groupby('bin_id')
    m = group['z'].mean()
    v = group['HR'].mean()
    e = group['HR'].std() / np.sqrt(group['HR'].count())
    
    ax.errorbar(m, v, yerr=e, fmt='o', color=color, markersize=6, label='Binned DES', zorder=5)
    
    # Plot BAO
    if not bao_mu_df.empty:
        base_bao = cosmo_baseline.distmod(bao_mu_df['z'].values).value
        resid_bao = bao_mu_df['mu'].values - base_bao
        ax.errorbar(bao_mu_df['z'], resid_bao, yerr=0.03, fmt='s', color='orange', markersize=8, markeredgecolor='k', label='DESI BAO', zorder=10)

    ax.text(0.05, 0.85, label, transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_ylim(-0.3, 0.5)
    
    if ax == axes[2]:
        ax.legend(loc='lower right', fontsize=9, ncol=2)

axes[2].set_xlabel('Redshift (z)', fontsize=12)
axes[2].set_xlim(0.01, 2.5)

plt.tight_layout()
plt.savefig('output/fig4_residual_agreement_DES5Y.png')
print("Saved output/fig4_residual_agreement_DES5Y.png")
