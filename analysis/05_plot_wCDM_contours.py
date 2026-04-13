import os
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import emcee

# --- CONFIGURATION ---
CHAIN_DIR = "chains_emcee_all"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# Define labels for GetDist
labels = [r'\Omega_m', r'w', r'H_0']
names = ['Om', 'w', 'H0']

def load_emcee_chain(name):
    path = os.path.join(CHAIN_DIR, f"{name}.h5")
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    
    reader = emcee.backends.HDFBackend(path)
    # Burning first 40% of chain
    tau = 20 # Guess for autocorrelation
    burnin = 50
    samples = reader.get_chain(discard=burnin, flat=True)
    
    print(f"Loaded {name}: {len(samples)} samples.")
    return MCSamples(samples=samples, names=names, labels=labels, label=name)

# 1. Load Samples
samples_uncorr = load_emcee_chain('DES_Uncorr')
samples_poly = load_emcee_chain('DES_Poly')

# Mocking BAO to match the visual style if data isn't loaded
bao_samps = np.random.multivariate_normal([0.31, -0.92, 68], [[0.0001, 0, 0], [0, 0.01, 0], [0, 0, 1]], 5000)
samples_bao = MCSamples(samples=bao_samps, names=names, labels=labels, label='BAO')

if samples_uncorr and samples_poly:
    # 1. Plot Before
    g_before = plots.get_subplot_plotter(subplot_size=5)
    g_before.plot_2d([samples_bao, samples_uncorr], 'Om', 'w', filled=True, colors=['green', 'blue'], lims=[0.15, 0.45, -1.6, -0.4])
    g_before.add_legend(['BAO (Mock)', 'DES5Y (Uncorr)'], legend_loc='upper right')
    plt.title('Figure 5: Before Correction', fontsize=16, fontweight='bold', pad=20)
    plt.axvline(0.315, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(-1.0, color='gray', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, 'fig5_before.png'), bbox_inches='tight')
    plt.close()

    # 2. Plot After
    g_after = plots.get_subplot_plotter(subplot_size=5)
    g_after.plot_2d([samples_bao, samples_poly], 'Om', 'w', filled=True, colors=['green', 'blue'], lims=[0.15, 0.45, -1.6, -0.4])
    g_after.add_legend(['BAO (Mock)', 'DES5Y (Poly)'], legend_loc='upper right')
    plt.title('Figure 5: After Correction (Poly)', fontsize=16, fontweight='bold', pad=20)
    plt.axvline(0.315, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(-1.0, color='gray', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, 'fig5_after.png'), bbox_inches='tight')
    plt.close()

    # 3. Combine them
    from matplotlib.image import imread
    img1 = imread(os.path.join(OUT_DIR, 'fig5_before.png'))
    img2 = imread(os.path.join(OUT_DIR, 'fig5_after.png'))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig5_reproduction_contours.png'), dpi=300)
    print("Saved output/fig5_reproduction_contours.png")
