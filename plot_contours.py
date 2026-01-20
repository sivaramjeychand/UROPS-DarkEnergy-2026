
import os
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import pandas as pd
import emcee

# Define paths
CHAIN_DIR = "analysis/chains_emcee"
OUT_DIR = "analysis"

# Load chains
def load_chain(name, label):
    filename = os.path.join(CHAIN_DIR, f"chain_{name}.h5")
    try:
        reader = emcee.backends.HDFBackend(filename)
        # Check convergence / burnin
        # For now, just discard first 30% as burnin
        tau = reader.get_autocorr_time(quiet=True)
        burnin = int(2 * np.max(tau)) if np.any(np.isfinite(tau)) else 100
        burnin = max(burnin, 100)
        
        flat_samples = reader.get_chain(discard=burnin, flat=True)
        # Params: Om, w, H0
        samples = MCSamples(samples=flat_samples, names=['omm', 'w', 'H0'], labels=[r'\Omega_m', 'w', 'H_0'], label=label)
        return samples
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

print("Loading samples...")
# Names match run_emcee_cosmology.py: 'bao_only', 'uncorr', 'corr'
samples_bao = load_chain('bao_only', 'BAO Only')
samples_uncorr = load_chain('uncorr', 'BAO + SN (Uncorrected)')
samples_corr = load_chain('corr', 'BAO + SN (Corrected)')

# Filter None
samples_list = [s for s in [samples_bao, samples_uncorr, samples_corr] if s is not None]

# --- TABLE GENERATION ---
print("\n--- Parameter Constraints ---")
for s in samples_list:
    print(f"\ndataset: {s.label}")
    stats = s.getMargeStats()
    # Print mean +/- std
    for par in ['omm', 'w', 'H0']:
        val = stats.parWithName(par)
        print(f"{par}: {val.mean:.3f} +/- {val.err:.3f}")

# --- PLOTTING ---
# Figure 8 equivalent: w - Om plane (and full triangle)
g = plots.get_subplot_plotter()
g.triangle_plot(samples_list, ['omm', 'w', 'H0'], filled=True, title_limit=1)

# Save
out_file = os.path.join(OUT_DIR, 'fig8_contours.png')
g.export(out_file)
print(f"Saved contours to {out_file}")

# Zoom in on w-Om if desired (Figure 5/6 style)
g = plots.get_subplot_plotter()
g.plot_2d(samples_list, 'omm', 'w', filled=True)
g.add_legend([s.label for s in samples_list], legend_loc='upper right')
out_file_2d = os.path.join(OUT_DIR, 'fig5_w_omm.png')
g.export(out_file_2d)
print(f"Saved 2D plot to {out_file_2d}")
