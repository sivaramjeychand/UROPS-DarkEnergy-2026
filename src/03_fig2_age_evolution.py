"""
Step 3 — Figure 2: Progenitor age evolution with redshift.

Shows Δage(z) = −α × t_L(z) for CPL, JBP, and LOG dark energy models
(both uncorrected best-fit and corrected best-fit cosmologies), replicating
Figure 2 of the paper.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_FIGS, PAPER_VALUES, ALPHA_SFH
from cosmology import lookback_time

z_arr = np.linspace(0.0, 1.8, 300)

# Cosmologies to display
cosmologies = [
    # label,                   Om,     w0,     wa,   H0,   model,  ls,    colour
    ('Uncorr (Panth, CPL)',    0.311, -0.838, -0.612, 70.0, 'CPL', '-',   'tab:blue'),
    ('Lin corr (Panth, CPL)',  0.353, -0.447, -1.585, 70.0, 'CPL', '--',  'tab:blue'),
    ('Uncorr (DES, CPL)',      0.319, -0.754, -0.848, 70.0, 'CPL', '-',   'tab:orange'),
    ('Lin corr (DES, CPL)',    0.363, -0.337, -1.902, 70.0, 'CPL', '--',  'tab:orange'),
    ('ΛCDM (Planck)',          0.315, -1.000,  0.000, 67.4, 'CPL', ':',   'dimgray'),
    ('Uncorr JBP',             0.31,  -0.84,  -0.61,  70.0, 'JBP', '-.',  'tab:green'),
    ('Uncorr LOG',             0.31,  -0.84,  -0.61,  70.0, 'LOG', '-.',  'tab:red'),
]

fig, ax = plt.subplots(figsize=(9, 6))

for label, Om, w0, wa, H0, model, ls, colour in cosmologies:
    tL       = lookback_time(z_arr, Om, w0, wa, H0, model=model)
    delta_age = -ALPHA_SFH * tL
    ax.plot(z_arr, delta_age, ls=ls, color=colour, lw=1.8, label=label)

ax.axhline(0, color='k', lw=0.8, ls=':')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel(r'$\Delta$age (Gyr)  [=  $-\alpha\,t_L(z)$]', fontsize=12)
ax.set_title('Fig. 2 — Progenitor age shift vs. redshift', fontsize=13)
ax.set_xlim(0, 1.8)
ax.legend(fontsize=8, ncol=2, loc='lower left')
ax.grid(True, ls=':', alpha=0.4)
fig.tight_layout()

out = os.path.join(OUT_FIGS, 'fig2_age_evolution.pdf')
fig.savefig(out, dpi=200)
print(f'Saved {out}')
plt.show()
