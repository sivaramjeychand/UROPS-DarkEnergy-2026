
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
from getdist import plots, MCSamples

# Setup
CHAIN_DIR = "chains_emcee_w0wa"
OUT_DIR = "output"
if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

datasets = [
    ("DES_Uncorr", "DES Uncorrected", "red"),
    ("DES_Lin", "DES Linear", "blue"),
    ("DES_Poly", "DES Polynomial", "black")
]

samples_list = []

print("Loading chains...")
for name, label, color in datasets:
    h5_path = os.path.join(CHAIN_DIR, f"{name}.h5")
    if os.path.exists(h5_path):
        reader = emcee.backends.HDFBackend(h5_path)
        # Discard burn-in (30%)
        # Shape: (nsteps, nwalkers, ndim)
        chain = reader.get_chain(discard=int(reader.iteration * 0.3), flat=True)
        
        # Params: Om, w0, wa, H0
        # We focus on w0, wa
        # chain columns: 0=Om, 1=w0, 2=wa, 3=H0
        
        # Create MCSamples for GetDist
        names = ['Om', 'w0', 'wa', 'H0']
        labels = [r'\Omega_m', r'w_0', r'w_a', r'H_0']
        
        s = MCSamples(samples=chain, names=names, labels=labels, label=label)
        samples_list.append(s)
    else:
        print(f"Missing chain: {name}")

# Plot w0-wa contours
g = plots.get_subplot_plotter()
g.settings.num_plot_contours = 2
g.triangle_plot(samples_list, ['w0', 'wa'], filled=True, 
                legend_loc='upper right',
                colors=['red', 'blue', 'black'])

# Add LCDM marker
# w0=-1, wa=0
plt.plot(-1, 0, 'k*', markersize=15, label='$\Lambda$CDM', markeredgecolor='white', markeredgewidth=1.5)

# Save
out_path = os.path.join(OUT_DIR, "fig_w0wa_contours.png")
plt.savefig(out_path, dpi=300)
print(f"Saved {out_path}")
