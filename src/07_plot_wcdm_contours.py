"""
Step 7 — Plot wCDM (Ω_m, w) contours from the chains computed in step 5.

Two-panel figure: before/after age-bias correction for each dataset
using GetDist to render filled 68%/95% contours.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import emcee
from getdist import plots, MCSamples

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_FIGS, OUT_CHAINS, BURNIN_FRAC

CHAIN_DIR = os.path.join(OUT_CHAINS, 'wcdm')
NAMES     = ['Om', 'w', 'H0', 'ombh2']
LABELS    = [r'\Omega_m', 'w', 'H_0', r'\Omega_b h^2']


def load_chain(name):
    path = os.path.join(CHAIN_DIR, f'{name}.h5')
    if not os.path.exists(path):
        print(f'  Chain not found: {path}')
        return None
    reader  = emcee.backends.HDFBackend(path, read_only=True)
    n       = reader.iteration
    burnin  = int(BURNIN_FRAC * n)
    samples = reader.get_chain(discard=burnin, flat=True)
    print(f'  {name}: {len(samples)} samples after burn-in')
    return MCSamples(samples=samples, names=NAMES, labels=LABELS, label=name)


def two_panel(dataset, title, out_name):
    s_unc  = load_chain(f'{dataset}_Uncorr')
    s_lin  = load_chain(f'{dataset}_Lin')
    s_poly = load_chain(f'{dataset}_Poly')
    if s_unc is None:
        print(f'Skipping {title} (no chains).')
        return

    sample_list = [s for s in [s_unc, s_lin, s_poly] if s is not None]
    colours = ['royalblue', 'crimson', 'forestgreen'][:len(sample_list)]
    legend  = [s.label for s in sample_list]

    g = plots.get_subplot_plotter(subplot_size=5)
    g.plot_2d(sample_list, 'Om', 'w', filled=True, colors=colours)
    g.add_legend(legend, legend_loc='upper right')

    # Reference lines
    import matplotlib.pyplot as _plt
    axes = _plt.gcf().axes
    for ax in axes:
        ax.axvline(0.315, color='gray', ls='--', alpha=0.4, lw=1)
        ax.axhline(-1.0,  color='gray', ls='--', alpha=0.4, lw=1)
        ax.set_xlim(0.15, 0.50)
        ax.set_ylim(-2.2, -0.1)

    _plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    out = os.path.join(OUT_FIGS, out_name)
    _plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')
    _plt.close()


two_panel('Panth', 'Fig. 5 — wCDM contours (Pantheon+)', 'fig5_wcdm_pantheon.pdf')
two_panel('DES',   'Fig. 5 — wCDM contours (DES-SN5YR)', 'fig5_wcdm_des.pdf')
