
import pandas as pd
import scipy.stats as stats
import os

# Define paths
ROSE19_TABLE1 = os.path.join('data', 'external', 'Rose19', 'J_ApJ_874_32_table1.csv')
ROSE19_TABLE7 = os.path.join('data', 'external', 'Rose19', 'J_ApJ_874_32_table7.csv')

# Load data
try:
    t1 = pd.read_csv(ROSE19_TABLE1)
    t7 = pd.read_csv(ROSE19_TABLE7)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Merge
t1['SNID'] = t1['SNID'].astype(int)
t7['SNID'] = t7['SNID'].astype(int)
df = pd.merge(t1, t7, on='SNID', how='inner')

print(f"Merged Data: {len(df)} supernovae")

# Clean
df_clean = df.dropna(subset=['logAl', 'HR'])

# Statistics
pearson_r, p_val = stats.pearsonr(df_clean['logAl'], df_clean['HR'])
slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['logAl'], df_clean['HR'])

print("-" * 30)
print("Replication Results (Rose+19)")
print("-" * 30)
print(f"Sample Size: {len(df_clean)}")
print(f"Pearson r:   {pearson_r:.3f}")
print(f"P-value:     {p_val:.3e}")
print(f"Slope:       {slope:.3f} +/- {std_err:.3f}")
print("-" * 30)

if p_val < 0.05:
    print("SUCCESS: Detection of significant correlation (Age Bias).")
else:
    print("WARNING: No significant correlation found. Check data units or column choice.")
