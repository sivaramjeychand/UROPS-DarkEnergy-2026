import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os

# 1. Configuration
OUT_DIR = "output"
if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

# 2. Table Points from User Image
z_points = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
age_solid = np.array([0, 1.0, 1.8, 2.6, 3.3, 3.9, 4.4, 4.8, 5.1, 5.3, 5.5, 5.6, 5.7, 5.8, 5.85, 5.9, 5.95, 6.0])
age_dashed = np.array([0, 1.0, 1.9, 2.8, 3.6, 4.3, 4.8, 5.3, 5.7, 6.0, 6.3, 6.5, 6.7, 6.9, 7.0, 7.1, 7.2, 7.3])

# Create Interpolation for smoothness
f_solid = interp1d(z_points, age_solid, kind='quadratic')
f_dashed = interp1d(z_points, age_dashed, kind='quadratic')

z_grid = np.linspace(0, 1.7, 100)

plt.figure(figsize=(7, 6))

# Plot High-Precision Curves
plt.plot(z_grid, f_solid(z_grid), 'k-', linewidth=2.5, label='Solid Line (Model A)')
plt.plot(z_grid, f_dashed(z_grid), 'k--', linewidth=2.0, label='Dashed Line (Model B)')

# Scatter points for reference
plt.scatter(z_points, age_solid, color='black', s=20, alpha=0.3)

# Aesthetics
plt.xlabel('Redshift (z)', fontsize=13)
plt.ylabel(r'$\Delta$ Age (Gyr)', fontsize=13)
plt.title('Figure 2 Replication: Progenitor Age Evolution', fontsize=15, pad=15)
plt.xlim(0, 1.75)
plt.ylim(0, 8)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(frameon=True, fontsize=11)
plt.tick_params(direction='in', top=True, right=True)

# Save
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig2_reproduction_table.png'), dpi=300)
plt.close()
print("Saved output/fig2_reproduction_table.png")

