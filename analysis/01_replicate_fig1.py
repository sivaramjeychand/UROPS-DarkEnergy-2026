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
# We use Weighted Least Squares (WLS) because some SNe have large errors.
# Weight = 1 / (error^2). This allows precise points to pull the line more.
# valid_mask ensures we don't hit NaNs
mask = ~np.isnan(df['HR']) & ~np.isnan(df['logAl']) & ~np.isnan(df['e_HR'])
df_clean = df[mask].copy()

# Weights
weights = 1.0 / (df_clean['e_HR']**2)

# Perform Weighted Polynomial Fit (Degree 2)
# p returns coefficients [a, b, c] for ax^2 + bx + c
# cov returns covariance matrix
p, cov = np.polyfit(df_clean['logAl'], df_clean['HR'], 2, w=weights, cov=True)

# Extract coefficients
a_curr = p[0] # Curvature
b_curr = p[1] # Slope-like
c_curr = p[2] # Intercept


# Calculate standard stats for the text box (using Pearson from scipy)
# Note: Pearson r is technically unweighted, but standard for reporting correlation strength
r_value, p_value = stats.pearsonr(df_clean['logAl'], df_clean['HR'])

# --- 2. PLOTTING ---
plt.figure(figsize=(9, 7))

# Individual Points (Grey Cloud)
plt.errorbar(df_clean['logAl'], df_clean['HR'], yerr=df_clean['e_HR'], fmt='o', 
             color='gray', alpha=0.3, markersize=5, label='Rose+19 Individual SNe', zorder=1)

# The Weighted Best-Fit Curve (Black Line)
x_fit = np.linspace(df_clean['logAl'].min(), df_clean['logAl'].max(), 100)
y_fit = np.polyval(p, x_fit)
plt.plot(x_fit, y_fit, 'k-', linewidth=2, 
         label=f'Weighted Poly Fit (deg=2)', zorder=5)

# Binned Averages (Red Dots) - Helps visualize the trend
bins = np.linspace(df_clean['logAl'].min(), df_clean['logAl'].max(), 7)
c = pd.cut(df_clean['logAl'], bins)
m = df_clean.groupby(c, observed=False)[['logAl']].mean()
v = df_clean.groupby(c, observed=False)[['HR']].mean()
# Error of mean = std / sqrt(N)
e = df_clean.groupby(c, observed=False)[['HR']].std() / np.sqrt(df_clean.groupby(c, observed=False)[['HR']].count())

plt.errorbar(m['logAl'], v['HR'], yerr=e['HR'], fmt='o', color='red', markersize=10, 
             markeredgecolor='black', label='Binned Averages', zorder=10)

# --- 3. STYLING ---
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Log Age (Local) [yr]', fontsize=12)
plt.ylabel('Hubble Residual (mag)', fontsize=12)
plt.title('Figure 1 Replication: Age-Bias Correlation (Polynomial Fit)', fontsize=14, pad=15)
plt.grid(True, linestyle=':', alpha=0.5)

# Statistics Box
stats_text = (f"Fit: $y = ax^2 + bx + c$\n"
              f"a = {a_curr:.3f}\n"
              f"b = {b_curr:.3f}\n"
              f"c = {c_curr:.3f}\n"
              f"Pearson r = {r_value:.3f}")

plt.text(0.05, 0.05, stats_text, transform=plt.gca().transAxes, 
         fontsize=12, fontweight='bold', 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()

# Save and Show
plt.savefig('output/fig1_replication_polynomial.png')
plt.show()

