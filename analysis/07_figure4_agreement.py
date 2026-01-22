# Converted from 07_figure4_agreement.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import os

# 1. Setup Cosmologies
cosmo_ref = FlatLambdaCDM(H0=73.04, Om0=0.30)  # Fiducial baseline (y=0)
cosmo_orig = FlatLambdaCDM(H0=73.28, Om0=0.35) # Approx best-fit (Original)
cosmo_corr = FlatLambdaCDM(H0=73.62, Om0=0.43) # Approx best-fit (Corrected)

# 2. Load Data
path_sn_orig = r'../data/external/PantheonPlus/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat'
path_sn_corr = r'../data/external/PantheonPlus_Corrected/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat'
bao_mean_file = r'../data/external/DESI_BAO/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt'

sn_orig = pd.read_csv(path_sn_orig, sep='\s+', skiprows=0 if 'CID' in open(path_sn_orig).read(100) else 1)
sn_corr = pd.read_csv(path_sn_corr, sep='\s+', skiprows=0 if 'CID' in open(path_sn_corr).read(100) else 1)
bao_df = pd.read_csv(bao_mean_file, delim_whitespace=True, comment='#', names=['z', 'value', 'type'])

# 3. Calculate Scale-Aligned BAO
rd_theory = 93.9 / 0.7304 # Scale alignment fix
bao_mu = []
for idx, row in bao_df.iterrows():
    if row['type'] == 'DM_over_rs':
        DM = row['value'] * rd_theory
        dL = DM * (1 + row['z'])
        mu = 5 * np.log10(dL) + 25
        bao_mu.append({'z': row['z'], 'mu': mu})
bao_mu_df = pd.DataFrame(bao_mu)

# 4. Calculate Residuals relative to cosmo_ref
z_grid = np.logspace(-2.5, 0.4, 100)
resid_orig_model = cosmo_orig.distmod(z_grid).value - cosmo_ref.distmod(z_grid).value
resid_corr_model = cosmo_corr.distmod(z_grid).value - cosmo_ref.distmod(z_grid).value

# 5. Plotting
plt.figure(figsize=(10, 6))
plt.axhline(0, color='k', linestyle='--', label='Fiducial LCDM ($\Omega_m=0.30$)')

# Models (The Curves)
plt.plot(z_grid, resid_orig_model, 'k--', alpha=0.6, label='Best-fit LCDM (Original)')
plt.plot(z_grid, resid_corr_model, 'r-', linewidth=2, label='Best-fit LCDM (Corrected)')

# Binned SN Data
bins = np.logspace(-2, 0.2, 12)
for df_loop, color, label in zip([sn_orig, sn_corr], ['gray', 'red'], ['Orig Data', 'Corrected Data']):
    resid = df_loop['MU_SH0ES'] - cosmo_ref.distmod(df_loop['zHD'].values).value
    c = pd.cut(df_loop['zHD'], bins)
    m = df_loop.groupby(c, observed=False)[['zHD']].mean()
    v = pd.Series(resid).groupby(c, observed=False).mean()
    plt.plot(m['zHD'], v, 'o', color=color, markersize=8, markeredgecolor='k', label=label)

# BAO Data
resid_bao_pts = bao_mu_df['mu'] - cosmo_ref.distmod(bao_mu_df['z'].values).value
plt.errorbar(bao_mu_df['z'], resid_bao_pts, yerr=0.03, fmt='s', color='blue', label='DESI BAO (Scaled)', markersize=8, markeredgecolor='k')

plt.xscale('log')
plt.xlabel('Redshift z')
plt.ylabel('$\Delta \mu$ (mag) relative to $\Omega_m=0.3$')
plt.title('Figure 4 Replication: Visualizing the New Concordance')
plt.legend(ncol=2, loc='upper left')
plt.grid(True, which='both', alpha=0.3)
plt.ylim(-0.4, 0.4)
plt.savefig('output/fig4_residual_agreement.png')
plt.show()

