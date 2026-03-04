
import os
import numpy as np
import matplotlib.pyplot as plt
import emcee
from getdist import MCSamples, plots

CHAIN_DIR = "chains_comparisons"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

def load_mcmc(name, label):
    path = os.path.join(CHAIN_DIR, f"{name}.h5")
    if not os.path.exists(path): return None
    reader = emcee.backends.HDFBackend(path)
    n = reader.iteration
    if n < 20: return None
    samples = reader.get_chain(discard=int(n*0.4), flat=True)
    # Params order: Om, w0, wa, H0, ob
    mc_samples = MCSamples(samples=samples, 
                          names=['omm', 'w0', 'wa', 'H0', 'ob'],
                          labels=[r'\Omega_m', 'w_0', 'w_a', 'H_0', r'\Omega_b h^2'],
                          label=label)
    return mc_samples

print("Generating Comparison Plots...")

# --- 1. Model Parametrization Comparison (Uncorrected Panth) ---
roots_models = []
for m in ['CPL', 'JBP', 'LOG']:
    s = load_mcmc(f"{m}_Panth_Uncorr", f"{m} (Uncorr)")
    if s: roots_models.append(s)

if roots_models:
    g = plots.get_subplot_plotter(width_inch=10)
    g.settings.axes_fontsize = 12
    g.settings.lab_fontsize = 14
    g.triangle_plot(roots_models, ['omm', 'w0', 'wa'], 
                    filled=True, 
                    colors=['#e74c3c', '#3498db', '#2ecc71'], # Red, Blue, Green
                    legend_labels=['CPL', 'JBP', 'LOG'])
    plt.suptitle("Impact of Dark Energy Parametrization (Uncorrected Pantheon+)", fontsize=18, y=1.02)
    plt.savefig(os.path.join(OUT_DIR, "comparison_models_uncorr.png"), bbox_inches='tight')
    print("Saved comparison_models_uncorr.png")

# --- 2. Shift Consistency Across Models (Uncorr vs Linear) ---
# We show CPL-Uncorr vs all models with Linear correction
roots_shift = []
roots_shift.append(load_mcmc("CPL_Panth_Uncorr", "CPL (Uncorr)"))
roots_shift.append(load_mcmc("CPL_Panth_Lin", "CPL (Linear)"))
roots_shift.append(load_mcmc("JBP_Panth_Lin", "JBP (Linear)"))
roots_shift.append(load_mcmc("LOG_Panth_Lin", "LOG (Linear)"))

roots_shift = [r for r in roots_shift if r is not None]

if len(roots_shift) > 1:
    g = plots.get_single_plotter(width_inch=8)
    g.settings.axes_fontsize = 12
    g.settings.lab_fontsize = 14
    g.plot_2d(roots_shift, 'omm', 'w0', filled=True, 
              colors=['#95a5a6', '#e74c3c', '#3498db', '#2ecc71'])
    g.add_legend(['CPL Uncorr', 'CPL Linear', 'JBP Linear', 'LOG Linear'], legend_loc='upper right')
    plt.title("Constraint Shift vs. Parametrization Choice", fontsize=16, pad=20)
    plt.savefig(os.path.join(OUT_DIR, "comparison_shift_robustness.png"), bbox_inches='tight')
    print("Saved comparison_shift_robustness.png")

# --- 3. Full Comparison (CPL Uncorr vs LOG Linear) ---
# Highlight the most significant difference
roots_ext = []
roots_ext.append(load_mcmc("CPL_Panth_Uncorr", "Standard (CPL Uncorr)"))
roots_ext.append(load_mcmc("LOG_Panth_Lin", "Corrected (LOG Linear)"))

if all(roots_ext):
    g = plots.get_subplot_plotter(width_inch=8)
    g.triangle_plot(roots_ext, ['omm', 'w0', 'wa'], filled=True, 
                    colors=['#95a5a6', '#e67e22']) # Grey vs Orange
    plt.suptitle("The Dark Energy Shift: Standard vs. Corrected (Alternative Model)", fontsize=16, y=1.02)
    plt.savefig(os.path.join(OUT_DIR, "comparison_extreme_shift.png"), bbox_inches='tight')
    print("Saved comparison_extreme_shift.png")

print("All contour plots generated.")
