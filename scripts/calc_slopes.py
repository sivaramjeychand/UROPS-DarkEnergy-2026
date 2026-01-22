
import pandas as pd
import numpy as np
from scipy.stats import linregress

# Load Corrected Data (Rose19 merged)
try:
    df = pd.read_csv('analysis/Rose19_corrected.csv')
    # Fit Age vs Redshift
    # We want LogAge = m * z + c
    slope, intercept, r, p, err = linregress(df['z'], df['logAl'])
    
    print(f"Age-Redshift Slope (dex/z): {slope:.4f}")
    print(f"Age-Redshift Intercept: {intercept:.4f}")
    
    # Also get the HR vs Age slope again to be sure
    # HR = alpha + beta * Age
    # We need beta.
    # Note: verify_fig1_stats.py might print correlation, not slope. Let's calc here.
    slope_hr_age, int_hr_age, _, _, _ = linregress(df['logAl'], df['HR'])
    print(f"HR-Age Slope (beta) [mag/dex]: {slope_hr_age:.4f}")

except Exception as e:
    print(f"Error: {e}")
