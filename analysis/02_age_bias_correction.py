# Converted from 02_age_bias_correction.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats

# Define paths
ROSE19_TABLE1 = os.path.join('..', 'data', 'external', 'Rose19', 'J_ApJ_874_32_table1.csv')
ROSE19_TABLE7 = os.path.join('..', 'data', 'external', 'Rose19', 'J_ApJ_874_32_table7.csv')
OUTPUT_DIR = os.path.join('..', 'analysis')

# Load data
t1 = pd.read_csv(ROSE19_TABLE1)
t7 = pd.read_csv(ROSE19_TABLE7)

# Merge
t1['SNID'] = t1['SNID'].astype(int)
t7['SNID'] = t7['SNID'].astype(int)
df = pd.merge(t1, t7, on='SNID', how='inner')

# Drop NaNs
df_clean = df.dropna(subset=['logAl', 'HR']).copy()
# Assume logAl is linear Age in Gyr
df_clean['Age'] = df_clean['logAl']
print(f"Using {len(df_clean)} SNe for fitting.")

slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['Age'], df_clean['HR'])
print(f"Best fit (Linear): HR = ({slope:.3f} +/- {std_err:.3f}) * Age + ({intercept:.3f})")

# Plot Fit
x_range = np.linspace(df_clean['Age'].min(), df_clean['Age'].max(), 100)
y_fit = slope * x_range + intercept

plt.figure(figsize=(8, 6))
plt.errorbar(df_clean['Age'], df_clean['HR'], yerr=df_clean['e_HR'], fmt='o', alpha=0.5, label='Data (Rose+19 Individual SNe)')
plt.plot(x_range, y_fit, 'r-', linewidth=2, label=f'Linear Fit (slope={slope:.3f})')
plt.xlabel('Age (Local) [Gyr]')
plt.ylabel('Hubble Residual (mag)')
plt.title('Figure 1 Linear Fit: Age-Bias Correlation')
plt.axhline(0, color='k', linestyle=':', alpha=0.5)
plt.legend()
plt.savefig('output/fig_age_bias_check_linear.png')
plt.show()

mean_age = df_clean['Age'].mean()
print(f"Mean Age: {mean_age:.3f} Gyr")

# Calculate Correction Term
# HR_corr = HR - slope * (Age - Age_ref)
correction = slope * (df_clean['Age'] - mean_age)

df_clean['HR_corr'] = df_clean['HR'] - correction

print("Correction applied based on Linear Fit.")

# Check correlation after correction
new_slope, _, _, new_p, _ = stats.linregress(df_clean['Age'], df_clean['HR_corr'])
print(f"New Slope: {new_slope:.3e} (should be 0)")

rms_old = np.std(df_clean['HR'])
rms_new = np.std(df_clean['HR_corr'])

print(f"RMS (Original):  {rms_old:.4f} mag")
print(f"RMS (Corrected): {rms_new:.4f} mag")
print(f"Improvement:     {rms_old - rms_new:.4f} mag")

# Save corrected dataset
output_file = os.path.join(OUTPUT_DIR, 'Rose19_corrected.csv')
df_clean.to_csv(output_file, index=False)
print(f"Saved corrected data to {output_file}")

