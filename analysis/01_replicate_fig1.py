# Converted from 01_replicate_fig1.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 1. Load Data
DATA_FILE = 'Rose19_corrected.csv'
df = pd.read_csv(DATA_FILE)

print(f"Loaded {len(df)} SNe.")

# --- 1. SETUP & WEIGHTED REGRESSION ---
# valid_mask ensures we don't hit NaNs
mask = ~np.isnan(df['HR']) & ~np.isnan(df['logAl']) & ~np.isnan(df['e_HR'])
df_clean = df[mask].copy()

# We assume 'logAl' in the CSV is actually linear Age in Gyr (unit correction/mismatch in naming)
# because the values range from 1 to 10.
df_clean['Age'] = df_clean['logAl']

# Weights
weights = 1.0 / (df_clean['e_HR']**2)

# Perform Weighted Fits
# 1. Polynomial Fit (Degree 2)
p_poly, _ = np.polyfit(df_clean['Age'], df_clean['HR'], 2, w=weights, cov=True)
a_poly, b_poly, c_poly = p_poly

# 2. Linear Fit (Degree 1)
p_lin, _ = np.polyfit(df_clean['Age'], df_clean['HR'], 1, w=weights, cov=True)
m_lin, b_lin = p_lin

# Calculate correlation
r_value, _ = stats.pearsonr(df_clean['Age'], df_clean['HR'])

# Common Plotting Data
x_range = np.linspace(df_clean['Age'].min(), df_clean['Age'].max(), 100)
bins = np.linspace(df_clean['Age'].min(), df_clean['Age'].max(), 7)
c = pd.cut(df_clean['Age'], bins)
m_bins = df_clean.groupby(c, observed=False)[['Age']].mean()
v_bins = df_clean.groupby(c, observed=False)[['HR']].mean()
e_bins = df_clean.groupby(c, observed=False)[['HR']].std() / np.sqrt(df_clean.groupby(c, observed=False)[['HR']].count())

def create_plot(fit_p, fit_label, fit_color, fit_type, stats_text, filename):
    plt.figure(figsize=(9, 7))
    
    # Individual Points
    plt.errorbar(df_clean['Age'], df_clean['HR'], yerr=df_clean['e_HR'], fmt='o', 
                 color='gray', alpha=0.3, markersize=5, label='Rose+19 Individual SNe', zorder=1)
    
    # Fit Line
    y_fit = np.polyval(fit_p, x_range)
    plt.plot(x_range, y_fit, color=fit_color, linewidth=2.5, label=fit_label, zorder=6)
    
    # Binned Averages
    plt.errorbar(m_bins['Age'], v_bins['HR'], yerr=e_bins['HR'], fmt='o', color='red', markersize=10, 
                 markeredgecolor='black', label='Binned Averages', zorder=10)
    
    # Aesthetics
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Age (Local) [Gyr]', fontsize=12)
    plt.ylabel('Hubble Residual (mag)', fontsize=12)
    plt.title(f'Figure 1: Hubble Residual vs. Local Age ({fit_type})', fontsize=14, pad=15)
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # Stats Box
    plt.text(0.05, 0.05, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))
    
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.show()

# 1. Plot Polynomial Fit
stats_poly = (f"Polynomial Fit: $y = ax^2 + bx + c$\n"
              f"a = {a_poly:.3f}, b = {b_poly:.3f}, c = {c_poly:.3f}\n"
              f"Pearson r = {r_value:.3f}")
create_plot(p_poly, 'Weighted Poly Fit (deg=2)', 'black', 'Polynomial Fit', stats_poly, 'output/fig1_replication_poly.png')

# 2. Plot Linear Fit
stats_lin = (f"Linear Fit: $y = mx + c$\n"
             f"m = {m_lin:.3f}, c = {b_lin:.3f}\n"
             f"Pearson r = {r_value:.3f}")
create_plot(p_lin, f'Weighted Linear Fit (slope={m_lin:.3f})', 'blue', 'Linear Fit', stats_lin, 'output/fig1_replication_linear.png')
