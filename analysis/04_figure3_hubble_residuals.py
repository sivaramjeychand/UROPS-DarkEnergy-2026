# Converted from 04_figure3_hubble_residuals.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM

# 1. Load Data
DATA_FILE = 'Rose19_corrected.csv'
df = pd.read_csv(DATA_FILE)

# 2. Setup Reference Cosmology
cosmo_ref = FlatLambdaCDM(H0=73.04, Om0=0.30) # Fiducial baseline
cosmo_fit = FlatLambdaCDM(H0=73.04, Om0=0.35) # Best-fit curve

z_grid = np.logspace(-2, 0.4, 100)
model_curve = cosmo_fit.distmod(z_grid).value - cosmo_ref.distmod(z_grid).value

print(f"Loaded {len(df)} SNe.")

fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'hspace': 0.1})

for i, (col, label, color) in enumerate(zip(['HR', 'HR_corr'], ['Original', 'Age-Corrected'], ['gray', 'red'])):
    ax = axes[i]
    ax.axhline(0, color='k', linestyle='--', alpha=0.5, label='Fiducial (Om=0.3)')
    
    if i == 1: # Highlight the trend in the corrected plot
        ax.plot(z_grid, model_curve, 'r-', linewidth=1.5, alpha=0.8, label='Best-fit Trend')
    
    ax.scatter(df['z'], df[col], color=color, alpha=0.3, s=20, label=f'Individual {label}')
    
    bins = np.logspace(-2, 0.4, 8)
    c = pd.cut(df['z'], bins)
    m = df.groupby(c, observed=False)[['z']].mean()
    v = df.groupby(c, observed=False)[[col]].mean()
    e = df.groupby(c, observed=False)[[col]].std() / np.sqrt(df.groupby(c, observed=False)[[col]].count())
    
    ax.errorbar(m['z'], v[col], yerr=e[col], fmt='o', color='black', markersize=8, label=f'Binned {label}', zorder=5)
    
    ax.set_ylabel('Residual (mag)', fontsize=12)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(loc='lower left', ncol=2, fontsize=10)
    
    rms = np.std(df[col])
    ax.text(0.95, 0.05, f"RMS = {rms:.4f} mag", transform=ax.transAxes, ha='right', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
    ax.set_title(f'{label} Hubble Residuals', fontsize=14, pad=10)

axes[1].set_xlabel('Redshift z', fontsize=12)
axes[1].set_xscale('log')
axes[1].set_xlim(0.01, 1.5)

plt.tight_layout()
plt.savefig('output/fig3_replication_premium.png')
plt.show()

