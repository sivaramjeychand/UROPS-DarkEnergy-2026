"""
Step 4 — Figure 3: Hubble diagram residuals (3-panel).

Three panels showing Hubble residuals relative to an open baseline
(Ω_m=0.30, Ω_Λ=0.00) for:
  (a) Uncorrected Pantheon+
  (b) Linear-corrected Pantheon+
  (c) Polynomial-corrected Pantheon+

Overlaid model curves follow the paper: flat ΛCDM and the paper's
best-fit corrected cosmology.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM, FlatLambdaCDM, w0waCDM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_FIGS, PANTH_DAT, PANTH_LIN, PANTH_POLY

H0_REF    = 73.04
cosmo_base = LambdaCDM(H0=H0_REF, Om0=0.30, Ode0=0.0)   # open baseline

def load_panth(path):
    df = pd.read_csv(path, sep=r'\s+')
    if 'zHD' not in df.columns:
        df = pd.read_csv(path, sep=r'\s+', skiprows=1)
    df = df[df['zHD'] > 0.005].copy()
    return df

# Load datasets
datasets = {}
for key, path in [('Uncorrected', PANTH_DAT),
                  ('Linear corr.', PANTH_LIN),
                  ('Poly corr.',   PANTH_POLY)]:
    if os.path.exists(path):
        datasets[key] = load_panth(path)
    else:
        print(f'  Warning: {path} not found — skipping panel.')

if not datasets:
    print('No Pantheon+ data found. Run 02_apply_age_correction.py first.')
    sys.exit(0)

# Model curves
z_grid   = np.logspace(np.log10(0.01), np.log10(1.3), 200)
def diff(cosmo): return cosmo.distmod(z_grid).value - cosmo_base.distmod(z_grid).value

c_lcdm   = FlatLambdaCDM(H0=H0_REF, Om0=0.30)
c_corr   = w0waCDM(H0=H0_REF, Om0=0.353, Ode0=0.647, w0=-0.447, wa=-1.585)
c_des    = w0waCDM(H0=H0_REF, Om0=0.363, Ode0=0.637, w0=-0.337, wa=-1.902)

fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 4.5 * len(datasets)),
                         sharex=True, gridspec_kw={'hspace': 0.05})
if len(datasets) == 1:
    axes = [axes]

for ax, (label, df) in zip(axes, datasets.items()):
    base = cosmo_base.distmod(df['zHD'].values).value
    res  = df['MU_SH0ES'].values - base

    # Individual scatter
    ax.scatter(df['zHD'], res, color='lightgray', s=3, alpha=0.4, zorder=1)

    # Binned averages (50 SNe per bin)
    df_s = pd.DataFrame({'z': df['zHD'].values, 'r': res}).sort_values('z')
    df_s['bid'] = np.arange(len(df_s)) // 50
    grp = df_s.groupby('bid')
    bz  = grp['z'].mean(); br = grp['r'].mean()
    be  = grp['r'].std() / np.sqrt(grp['r'].count())
    ax.errorbar(bz, br, yerr=be, fmt='o', color='dodgerblue',
                ms=6, markeredgecolor='steelblue', capsize=2,
                label='Binned (50 SNe/bin)', zorder=8)

    # Model curves
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.plot(z_grid, diff(c_lcdm), 'r-',  lw=2.0, label=r'Flat $\Lambda$CDM ($\Omega_m=0.30$)')
    ax.plot(z_grid, diff(c_corr), 'b--', lw=2.0, label='Best fit (Lin-corr Panth, CPL)')
    ax.plot(z_grid, diff(c_des),  'g:',  lw=2.0, label='Best fit (Lin-corr DES, CPL)')

    ax.set_ylabel(r'$\mu - \mu_{\rm base}$ (mag)', fontsize=11)
    ax.text(0.03, 0.88, label, transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.set_ylim(-0.35, 0.45)
    ax.grid(True, ls=':', alpha=0.3)
    if ax is axes[-1]:
        ax.legend(fontsize=8, loc='lower right', ncol=2)

axes[-1].set_xlabel('Redshift z', fontsize=12)
axes[-1].set_xlim(0, 1.3)

fig.suptitle('Fig. 3 — Hubble diagram residuals (Pantheon+)', fontsize=14, y=1.01)
fig.tight_layout()
out = os.path.join(OUT_FIGS, 'fig3_hubble_diagram.pdf')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved {out}')
plt.show()
