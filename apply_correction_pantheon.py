
import pandas as pd
import numpy as np
import os
import shutil

# Constants derived from previous steps
BETA_HR_AGE = -0.0424 # mag/dex
SLOPE_AGE_Z = -2.1466 # dex/z
COMBINED_SLOPE = BETA_HR_AGE * SLOPE_AGE_Z # mag/z -> ~ 0.091

print(f"Applying correction with slope: {COMBINED_SLOPE:.4f} mag/z")

# Paths
BASE_DIR = r'data/external/PantheonPlus/Pantheon+_Data/4_DISTANCES_AND_COVAR'
SOURCE_FILE = os.path.join(BASE_DIR, 'Pantheon+SH0ES.dat')
DEST_DIR = r'data/external/PantheonPlus_Corrected/Pantheon+_Data/4_DISTANCES_AND_COVAR'
DEST_FILE = os.path.join(DEST_DIR, 'Pantheon+SH0ES.dat')

# 1. Create Destination Directory and Copy ALL files (covariance etc needed)
# We copy the whole folder structure to be safe
print(f"Copying files from {BASE_DIR} to {DEST_DIR}...")
if os.path.exists(DEST_DIR):
    shutil.rmtree(DEST_DIR)
os.makedirs(DEST_DIR)

# Copy all files
for item in os.listdir(BASE_DIR):
    s = os.path.join(BASE_DIR, item)
    d = os.path.join(DEST_DIR, item)
    if os.path.isdir(s):
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

# 2. Read Data
print("Reading Pantheon+ data...")
with open(SOURCE_FILE, 'r') as f:
    header_line = f.readline().strip()
    # Pantheon header starts with VARNAMES: ...
    # But checking file view, first line was just column names?
    # Ah, view_file showed Line 1: CID... but read_csv needs to handle it.
    # We will read carefully.
    
# Use pandas with space delimiter
df = pd.read_csv(SOURCE_FILE, delim_whitespace=True)

# Check columns
if 'MU_SH0ES' not in df.columns:
    # Maybe first line is special?
    # Try skipping first line if it starts with VARNAMES
    df = pd.read_csv(SOURCE_FILE, delim_whitespace=True, skiprows=1)

print(f"Loaded {len(df)} rows.")

# 3. Apply Correction
# Correction = Bias. We want Corrected = Observed - Bias.
# Bias(z) = Slope * z
# We use zHD (Hubble Diagram redshift)
bias = COMBINED_SLOPE * df['zHD']
df['MU_SH0ES'] = df['MU_SH0ES'] - bias
df['m_b_corr'] = df['m_b_corr'] - bias # Also correct apparent mag if used

# 4. Save
print(f"Saving corrected data to {DEST_FILE}...")
# We must preserve the exact format for Cobaya to read it?
# Cobaya's sn.pantheonplus reader expects specific header.
# We'll stick to simple csv write with space, but check if header needs 'VARNAMES:' prefix
# The original file might have had it. Let's check first line of original again.
with open(SOURCE_FILE, 'r') as f:
    first_line = f.readline()

# If first line has VARNAMES, we need to match it.
# Based on view_file earlier: "1: CID IDSURVEY..." -> It seems it DOES NOT have VARNAMES: prefix in the view?
# Wait, view_file output:
# 1: CID IDSURVEY ...
# Oh, checking my logs... Step 215 view_file of DES-Dovekie has VARNAMES.
# Step 279 of Pantheon+SH0ES.dat:
# 1: CID IDSURVEY ...
# It seems it's just raw columns.
# But wait, looking closer at Step 279 output:
# "1: CID IDSURVEY zHD ..."
# So simple to_csv with sep=' ' should work.

df.to_csv(DEST_FILE, sep=' ', index=False, float_format='%.5f')

print("Done. Corrected dataset ready.")
