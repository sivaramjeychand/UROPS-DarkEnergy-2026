# Converted from 03_figure2_age_evolution.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Data
DATA_FILE = 'Rose19_corrected.csv'
df = pd.read_csv(DATA_FILE)

# 2. Physical Model (from Son et al. 2025)
z_grid = np.linspace(0.01, 1.5, 100)
age_model = 6.0 / (1 + 1.5 * z_grid) # Progenitor age evolution proxy

print(f"Loaded {len(df)} SNe.")

plt.figure(figsize=(10, 6))

# Individual Points
scatter = plt.scatter(df['z'], 10**(df['logAl'])/1e9, alpha=0.4, c=df['HR'], cmap='RdBu_r', 
                      s=40, edgecolors='k', linewidth=0.5, label='Rose+19 Sample')
cbar = plt.colorbar(scatter)
cbar.set_label('Hubble Residual (mag)', fontsize=12)

# Physical Model Curve
plt.plot(z_grid, age_model, 'k-', linewidth=3, label='Median Progenitor Age (Son et al.)')
plt.plot(z_grid, age_model * 1.5, 'k--', alpha=0.3)
plt.plot(z_grid, age_model * 0.5, 'k--', alpha=0.3)
plt.fill_between(z_grid, age_model*0.5, age_model*1.5, color='gray', alpha=0.1, label='Model Uncertainty')

plt.xlabel('Redshift z', fontsize=12)
plt.ylabel('Host Galaxy Age (Gyr)', fontsize=12)
plt.title('Progenitor Age Evolution vs. Redshift', fontsize=14, pad=15)
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.legend(loc='upper right', fontsize=10)

plt.xscale('log')
plt.xlim(0.01, 1.5)
plt.ylim(0, 10)

plt.tight_layout()
plt.savefig('output/fig2_replication_premium.png')
plt.show()

