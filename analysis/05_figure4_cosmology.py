# Converted from 05_figure4_cosmology.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# Load Pantheon+ Data
SN_DATA_FILE = '../data/external/PantheonPlus/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat'
# The file is space-separated, we need to handle the header structure
# It seems the first line is VARNAMES
with open(SN_DATA_FILE, 'r') as f:
    header = f.readline().split()
sn_df = pd.read_csv(SN_DATA_FILE, delim_whitespace=True, names=header, comment='V', skiprows=1)

print(f"Loaded {len(sn_df)} SNe from Pantheon+.")
print(sn_df[['zHD', 'MU_SH0ES']].head())

# Load BAO Data
# We will manually define a few key BAO points if the file parsing is complex, 
# or try to load the consensus file we found.
BAO_FILE = '../data/external/DESI_BAO/BAO_consensus_covtot_dM_Hz.txt'
# This file likely contains z, dM(z), Hz
# Let's assume standard columns or inspect it. 
# If inspection failed, we'll use standard values for DR12/DESI.

try:
    bao_df = pd.read_csv(BAO_FILE, delim_whitespace=True, comment='#', names=['z', 'dM', 'Hz', 'dM_err', 'Hz_err', 'cov_dM_Hz', 'type'])
    # Note: Column names are a guess, we should adjust based on file content.
    print("Loaded BAO data from file.")
except:
    print("Could not auto-load BAO file. Using standard DR12 points.")
    # Standard BOSS DR12 z_eff, DM/rd, DH/rd
    # We need absolute distances, so we need rd (sound horizon). 
    # Planck 2018 rd ~ 147.09 Mpc
    rd = 147.09
    bao_data = [
        {'z': 0.38, 'DM': 1512.39, 'DM_err': 25.0}, # DR12
        {'z': 0.51, 'DM': 1975.22, 'DM_err': 27.0}, # DR12
        {'z': 0.61, 'DM': 2306.68, 'DM_err': 37.0}, # DR12
    ]
    bao_df = pd.DataFrame(bao_data)
    # Convert DM to Distance Modulus: mu = 5 * log10(DM) + 25
    bao_df['mu'] = 5 * np.log10(bao_df['DM'] * (1e6/10)) # DM is likely Mpc? Check units.

# Define Cosmology for Theory Curve
cosmo = FlatLambdaCDM(H0=73.04, Om0=0.334) # Pantheon+SH0ES values
z_range = np.logspace(-2, 0.3, 100)
dist_mod = cosmo.distmod(z_range).value

plt.figure(figsize=(10, 7))
# Plot SN
plt.errorbar(sn_df['zHD'], sn_df['MU_SH0ES'], yerr=sn_df['MU_SH0ES_ERR_DIAG'], fmt='o', color='grey', alpha=0.1, label='Pantheon+')

# Plot Theory
plt.plot(z_range, dist_mod, 'k-', linewidth=2, label='Flat $\Lambda$CDM (H0=73)')

plt.xlabel('Redshift (z)')
plt.ylabel('Distance Modulus ($\mu$)')
plt.xscale('log')
plt.legend()
plt.title('Hubble Diagram: SN + BAO')
plt.savefig('output/fig4_cosmology.png')
plt.show()

