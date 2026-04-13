import os
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import emcee

# --- CONFIGURATION ---
CHAIN_DIR = "chains_emcee_w0wa"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# Define labels for GetDist (w0-wa model)
# Om, w0, wa, H0, ombh2
names = ['Om', 'w0', 'wa', 'H0', 'ombh2']
labels = [r'\Omega_m', r'w_0', r'w_a', r'H_0', r'\Omega_b h^2']

def load_emcee_chain(name):
    path = os.path.join(CHAIN_DIR, f"{name}.h5")
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    
    reader = emcee.backends.HDFBackend(path)
    # Burn-in 50 steps
    burnin = 50
    samples = reader.get_chain(discard=burnin, flat=True)
    
    print(f"Loaded {name}: {len(samples)} samples.")
    return MCSamples(samples=samples, names=names, labels=labels, label=name)

# 1. Load Samples
samples_uncorr = load_emcee_chain('DES_Uncorr')
samples_poly = load_emcee_chain('DES_Poly')

# Mocking BAO for w0-wa (Centered near w0=-1, wa=0)
bao_samps = np.random.multivariate_normal(
    [0.315, -0.9, -0.1, 68, 0.0224], 
    [[0.0001, 0, 0, 0, 0], [0, 0.01, 0, 0, 0], [0, 0, 0.1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0.0001]], 
    5000
)
samples_bao = MCSamples(samples=bao_samps, names=names, labels=labels, label='BAO')

if samples_uncorr and samples_poly:
    # --- PLOT FIGURE 8: w0 vs wa ---
    
    # 1. Panel Before
    g_before = plots.get_subplot_plotter(subplot_size=5)
    g_before.plot_2d([samples_bao, samples_uncorr], 'w0', 'wa', filled=True, colors=['green', 'blue'], lims=[-1.8, 0.2, -4.0, 2.0])
    g_before.add_legend(['BAO (Mock)', 'DES5Y (Uncorr)'], legend_loc='upper right')
    plt.title('Figure 8: Before Correction', fontsize=16, fontweight='bold', pad=20)
    plt.axvline(-1.0, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(0.0, color='gray', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, 'fig8_before.png'), bbox_inches='tight')
    plt.close()

    # 2. Panel After
    g_after = plots.get_subplot_plotter(subplot_size=5)
    g_after.plot_2d([samples_bao, samples_poly], 'w0', 'wa', filled=True, colors=['green', 'blue'], lims=[-1.8, 0.2, -4.0, 2.0])
    g_after.add_legend(['BAO (Mock)', 'DES5Y (Poly)'], legend_loc='upper right')
    plt.title('Figure 8: After Correction (Poly)', fontsize=16, fontweight='bold', pad=20)
    plt.axvline(-1.0, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(0.0, color='gray', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, 'fig8_after.png'), bbox_inches='tight')
    plt.close()

    # 3. Combine them
    from matplotlib.image import imread
    img1 = imread(os.path.join(OUT_DIR, 'fig8_before.png'))
    img2 = imread(os.path.join(OUT_DIR, 'fig8_after.png'))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig8_reproduction_contours.png'), dpi=300)
    print("Saved output/fig8_reproduction_contours.png")

else:
    print("Could not load chains for plotting.")
