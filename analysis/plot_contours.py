
import getdist
from getdist import plots, MCSamples
import matplotlib.pyplot as plt
import numpy as np
import os
import emcee

# Create output dir if not exists
OUT_DIR = 'output'
os.makedirs(OUT_DIR, exist_ok=True)

def load_chain(name, label):
    filename = f'chains_emcee/chain_{name}.h5'
    print(f"Loading {filename}...")
    try:
        reader = emcee.backends.HDFBackend(filename)
        # Discard burn-in
        tau = reader.get_autocorr_time(quiet=True)
        burnin = int(2 * np.max(tau)) if np.any(np.isfinite(tau)) else 50
        flat_samples = reader.get_chain(discard=burnin, flat=True)
        # Params: Om, w, H0
        samples = MCSamples(samples=flat_samples, names=['omm', 'w', 'H0'], 
                           labels=[r'\Omega_m', 'w', 'H_0'], label=label)
        return samples
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

print("Loading samples...")
samples_bao = load_chain('bao_only', 'BAO Only')
samples_uncorr = load_chain('uncorr', 'BAO + SN (Original)')
samples_corr = load_chain('corr', 'BAO + SN (Corrected)')

roots = [s for s in [samples_bao, samples_uncorr, samples_corr] if s is not None]

# --- Figure 8: Triangle Plot (Full Posterior) ---
g = plots.get_subplot_plotter(width_inch=10)
g.settings.axes_fontsize = 12
g.settings.lab_fontsize = 14
g.settings.legend_fontsize = 14
g.triangle_plot(roots, ['omm', 'w', 'H0'], 
                filled=True, 
                colors=['#3498db', '#95a5a6', '#e74c3c'], # Blue, Grey, Red
                legend_labels=['BAO Only', 'BAO + SN (Original)', 'BAO + SN (Corrected)'],
                line_args=[{'ls':'-', 'color':'#3498db'}, {'ls':'-', 'color':'#95a5a6'}, {'ls':'-', 'color':'#e74c3c'}])
g.export('output/fig8_contours.png')
print("Saved contours to analysis/fig8_contours.png")

# --- Figure 5: w vs Omm (2D Focused) ---
g = plots.get_single_plotter(width_inch=7)
g.settings.axes_fontsize = 12
g.settings.lab_fontsize = 14
g.plot_2d(roots, 'omm', 'w', filled=True, colors=['#3498db', '#95a5a6', '#e74c3c'])
g.add_legend(['BAO Only', 'BAO + SN (Original)', 'BAO + SN (Corrected)'], legend_loc='upper right')
plt.title('Equation of State $w$ vs $\Omega_m$', fontsize=16, pad=20)
plt.savefig('output/fig5_w_omm.png', bbox_inches='tight')
print("Saved 2D plot to analysis/fig5_w_omm.png")

# --- Export Precision Table ---
print("\n--- Parameter Constraints ---")
for s in roots:
    print(f"\ndataset: {s.label}")
    for p in ['omm', 'w', 'H0']:
        m = s.mean(p)
        std = s.std(p)
        print(f"{p}: {m:.3f} +/- {std:.3f}")
