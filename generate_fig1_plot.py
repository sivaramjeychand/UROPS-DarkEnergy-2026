
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths
ROSE19_TABLE1 = os.path.join('data', 'external', 'Rose19', 'J_ApJ_874_32_table1.csv')
ROSE19_TABLE7 = os.path.join('data', 'external', 'Rose19', 'J_ApJ_874_32_table7.csv')
OUTPUT_DIR = 'analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
t1 = pd.read_csv(ROSE19_TABLE1)
t7 = pd.read_csv(ROSE19_TABLE7)

# Merge
t1['SNID'] = t1['SNID'].astype(int)
t7['SNID'] = t7['SNID'].astype(int)
df = pd.merge(t1, t7, on='SNID', how='inner')
df_clean = df.dropna(subset=['logAl', 'HR'])

# Plot
plt.figure(figsize=(8, 6))
plt.errorbar(df_clean['logAl'], df_clean['HR'], yerr=df_clean['e_HR'], fmt='o', alpha=0.7, color='blue', label='Rose+19 Data')

plt.xlabel('Log Age (Local Environment)')
plt.ylabel('Hubble Residual (mag)')
plt.title('Replication of Figure 1: HR vs Age')
plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.legend()

output_path = os.path.join(OUTPUT_DIR, 'fig1_replication.png')
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
