"""
Step 8 — Plot w0–wa contours for CPL, JBP, and LOG models.

One figure per model showing before/after correction contours in the
(w0, wa) plane, with the no-acceleration boundary q0=0 overlaid.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import emcee
from getdist import plots, MCSamples

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_FIGS, OUT_CHAINS, BURNIN_FRAC

CHAIN_DIR = os.path.join(OUT_CHAINS, 'w0wa')
NAMES     = ['Om', 'w0', 'wa', 'H0', 'ombh2']
LABELS    = [r'\Omega_m', 'w_0', 'w_a', 'H_0', r'\Omega_b h^2']


def load_chain(name):
    path = os.path.join(CHAIN_DIR, f'{name}.h5')
    if not os.path.exists(path):
        return None
    reader  = emcee.backends.HDFBackend(path, read_only=True)
    n       = reader.iteration
    burnin  = int(BURNIN_FRAC * n)
    samples = reader.get_chain(discard=burnin, flat=True)
    print(f'  {name}: {len(samples)} samples')
    return MCSamples(samples=samples, names=NAMES, labels=LABELS, label=name)


def plot_model(model, dataset='Panth'):
    """
    Three-way contour plot: Uncorr / Lin / Poly for a given model + dataset.
    """
    chains = {}
    for corr in ['Uncorr', 'Lin', 'Poly']:
        name = f'{model}_{dataset}_{corr}'
        s    = load_chain(name)
        if s is not None:
            chains[corr] = s

    if not chains:
        print(f'  No chains found for {model}/{dataset}')
        return

    sample_list = list(chains.values())
    colours = ['royalblue', 'crimson', 'forestgreen'][:len(sample_list)]

    g = plots.get_subplot_plotter(subplot_size=5)
    g.plot_2d(sample_list, 'w0', 'wa',
              filled=True, colors=colours,
              lims=[-2.5, 0.5, -5.0, 3.0])
    g.add_legend([f'{corr}' for corr in chains.keys()], legend_loc='upper right')

    ax = plt.gcf().axes[0]
    # ΛCDM point
    ax.plot(-1, 0, 'k*', ms=12, zorder=10, label=r'$\Lambda$CDM')
    # No-acceleration boundary: q0 = 0 → w0 = -(1 + 3*Om*(w0+1)/(1-Om))/3 ≈ -0.5 for Om~0.3
    w0_line = np.linspace(-2.5, 0.5, 200)
    # q0 = Om/2 + (1-Om)/2 * (1 + 3w0) = 0 → w0 = -(1 + Om/(1-Om)) / 3
    # With wa: q0 = Om/2 + (1-Om)/2*(1+3w0) at z=0 (wa doesn't enter q0(z=0) for CPL)
    # q0 = 0 line: w0 = -1/3 - Om/(3*(1-Om))  (independent of wa)
    Om_ref = 0.315
    w0_crit = -1.0/3.0 - Om_ref / (3.0 * (1.0 - Om_ref))
    ax.axvline(w0_crit, color='k', ls=':', lw=1.5, label=f'$q_0=0$ ($w_0={w0_crit:.2f}$)')
    ax.legend(fontsize=8)
    ax.set_title(f'{model} — {dataset}', fontsize=12)

    out = os.path.join(OUT_FIGS, f'fig8_{model}_{dataset}_w0wa.pdf')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'  Saved {out}')
    plt.close()


for model in ['CPL', 'JBP', 'LOG']:
    for dataset in ['Panth', 'DES']:
        print(f'\n=== {model} / {dataset} ===')
        plot_model(model, dataset)
