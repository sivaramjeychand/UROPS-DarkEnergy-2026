# Converted from 04_figure3_replication.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, w0waCDM

# 1. Load Data
DATA_FILE = 'Rose19_corrected.csv'
df = pd.read_csv(DATA_FILE)

# 2. Define Cosmologies
# H0 should match the data derivation roughly, but cancels in residuals if M is fitted. We use 73.04 typical.
H0 = 73.04

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

# 4. Transform Data Distances
# Current HR = mu_obs - mu_old_ref
# We want HR_new = mu_obs - mu_baseline
# so HR_new = (HR + mu_old_ref) - mu_baseline
#           = HR + (mu_old_ref - mu_baseline)
data_offset = cosmo_old_ref.distmod(df['z']).value - cosmo_baseline.distmod(df['z']).value

df['HR_new'] = df['HR'] + data_offset
df['HR_corr_new'] = df['HR_corr'] + data_offset

print("Cosmologies defined and data transformed.")

# 5. Plotting
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'hspace': 0.1})

# Loop for Top (Before) and Bottom (After)
datasets = [
    ('Before Correction', df['HR_new'], 'gray', axes[0]),
    ('After Correction', df['HR_corr_new'], 'gray', axes[1])
]

for label, data_y, color, ax in datasets:
    # Plot Baseline
    ax.axhline(0, color='black', linestyle=':', label='Baseline ($\Omega_m=0.3, \Omega_{de}=0.0$)')
    
    # Plot Model Curves
    # Note: Curves are identical in both panels as they represent models
    ax.plot(z_grid, diff_red, 'r-', linewidth=2, label='$\Lambda$CDM ($\Omega_m=0.30$)')
    ax.plot(z_grid, diff_blue, 'b-', linewidth=2, alpha=0.8, label='$w_0w_a$CDM (Blue Params)')
    # Green is dashed in the original image example roughly, or solid. We'll use dashed for distinction as per some conventions or just solid.
    # User image has Blue solid, Red solid, Green dashed.
    ax.plot(z_grid, diff_green, 'g--', linewidth=2, alpha=0.8, label='$w_0w_a$CDM (Green Params)')
    
    # Plot Individual Points
    # We skip plotting 1000s of points to keep it clean, or plot them very faintly?
    # The user image shows binned points mainly. We will plot faint scatter + binned.
    ax.scatter(df['z'], data_y, color=color, alpha=0.1, s=10, zorder=1)
    
    # Binned Statistics
    # ~50 bins or strictly 50 per bin. We'll use logspace bins.
    bins = np.logspace(np.log10(df['z'].min()), np.log10(df['z'].max()), 15)
    c = pd.cut(df['z'], bins)
    
    # Recalculate means in the new HR space
    df_temp = pd.DataFrame({'z': df['z'], 'HR': data_y})
    m = df_temp.groupby(c, observed=False)['z'].mean()
    v = df_temp.groupby(c, observed=False)['HR'].mean()
    e = df_temp.groupby(c, observed=False)['HR'].std() / np.sqrt(df_temp.groupby(c, observed=False)['HR'].count())
    
    # Use light blue for points as in the image (or similar)
    ax.errorbar(m, v, yerr=e, fmt='o', color='dodgerblue', markersize=6, 
                markeredgecolor='blue', ecolor='blue', capsize=2, label='Binned Data', zorder=10)

    ax.set_ylabel('HR (mag)', fontsize=12)
    ax.text(0.05, 0.85, label, transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.grid(True, which='both', linestyle=':', alpha=0.3)

    # Legend only on top or bottom? User image has it on both or specific.
    if ax == axes[1]:
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

axes[1].set_xlabel('Redshift (z)', fontsize=12)
axes[1].set_xlim(0, 1.3)

# Determine Y-limits to match image roughly
# Image goes from ~ -0.2 to 0.4
axes[0].set_ylim(-0.3, 0.4)
axes[1].set_ylim(-0.3, 0.4)

plt.tight_layout()
plt.savefig('output/fig3_replication_custom.png')
plt.show()

