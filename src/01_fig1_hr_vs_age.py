"""
Step 1 — Replicate Figure 1: Hubble Residual vs. Local Stellar Age.

Reproduces the HR–Age scatter plot from Rose et al. (2019) with both a
weighted linear fit and a weighted polynomial (degree-2) fit, along with
binned averages.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_FIGS
from data_io import load_rose19
from age_correction import fit_linear, fit_poly

# ── Load & fit ────────────────────────────────────────────────────────────────
df           = load_rose19()
lin_slope, lin_int = fit_linear(df)
poly_coeffs  = fit_poly(df, deg=2)

ages = df['Age'].values
hrs  = df['HR'].values
errs = df['e_HR'].values

r_val, p_val = pearsonr(ages, hrs)
print(f'N = {len(df)} SNe')
print(f'Linear slope   = {lin_slope:.4f} ± (see fit) mag/Gyr')
print(f'Pearson r      = {r_val:.3f}  (p = {p_val:.3e})')

x_rng  = np.linspace(ages.min(), ages.max(), 300)
y_lin  = lin_slope * x_rng + lin_int
y_poly = np.polyval(poly_coeffs, x_rng)

# Binned averages (6 equal-width bins)
bins   = np.linspace(ages.min(), ages.max(), 7)
cat    = np.digitize(ages, bins)
b_age, b_hr, b_err = [], [], []
for k in range(1, 7):
    sel = cat == k
    if sel.sum() < 2:
        continue
    b_age.append(ages[sel].mean())
    b_hr.append(hrs[sel].mean())
    b_err.append(hrs[sel].std() / np.sqrt(sel.sum()))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, (label, y_fit, colour) in zip(axes, [
    ('Linear fit', y_lin,  'royalblue'),
    ('Poly fit',   y_poly, 'black'),
]):
    ax.errorbar(ages, hrs, yerr=errs, fmt='o',
                color='dimgray', alpha=0.35, markersize=4,
                label='Rose+19 SNe', zorder=1)
    ax.plot(x_rng, y_fit, color=colour, lw=2.5, label=label, zorder=5)
    ax.errorbar(b_age, b_hr, yerr=b_err, fmt='o',
                color='crimson', markersize=9, markeredgecolor='k',
                label='Binned average', zorder=10)
    ax.axhline(0, color='k', ls='--', alpha=0.4)
    ax.set_xlabel('Local stellar age [Gyr]', fontsize=12)
    ax.set_ylabel('Hubble Residual (mag)', fontsize=12)
    ax.set_title(f'Fig. 1 — HR vs Age  ({label})', fontsize=13)
    ax.grid(True, ls=':', alpha=0.4)
    info = (f'slope = {lin_slope:.4f} mag/Gyr\n'
            f'Pearson r = {r_val:.3f}')
    ax.text(0.05, 0.05, info, transform=ax.transAxes,
            fontsize=10, bbox=dict(fc='white', alpha=0.85, ec='lightgray'))
    ax.legend(fontsize=9)

fig.tight_layout()
out = os.path.join(OUT_FIGS, 'fig1_hr_vs_age.pdf')
fig.savefig(out, dpi=200)
print(f'Saved {out}')
plt.show()
