"""
Step 2 — Fit the HR–Age relation and apply static corrections to Pantheon+ and DES.

Outputs:
  - src/output/tables/age_correction_summary.txt
  - data/external/PantheonPlus_Corrected/.../Pantheon+SH0ES_Linear.dat
  - data/external/PantheonPlus_Corrected/.../Pantheon+SH0ES_Poly.dat
  - data/external/DES-SN5YR_Corrected/.../DES-Dovekie_HD_Linear.csv
  - data/external/DES-SN5YR_Corrected/.../DES-Dovekie_HD_Poly.csv

The MC-age catalogue (../data/external/mc-age/) provides photometric ages
matched to Pantheon+ and DES SNe by SNID.  If not present, only the Rose+19
calibration fit is performed and the corrected data files are skipped.

Note: the *dynamic* correction (inside MCMC) uses only the fit coefficients
computed here — it does not require the pre-corrected data files.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (ROOT, OUT_FIGS, OUT_TABLES,
                    PANTH_DAT, PANTH_COV, PANTH_CORR_DIR,
                    DES_DAT, DES_CORR_DIR)
from data_io import load_rose19
from age_correction import fit_linear, fit_poly

# ── Fit on Rose+19 ────────────────────────────────────────────────────────────
df = load_rose19()
lin_slope, lin_int = fit_linear(df)
poly_coeffs        = fit_poly(df, deg=2)
mean_age           = df['Age'].mean()
p_poly             = np.poly1d(poly_coeffs)

print('=== Age–HR fit on Rose+19 ===')
print(f'  Linear: HR = {lin_slope:.5f} * Age + {lin_int:.5f}  (mag/Gyr)')
print(f'  Poly:   coeffs = {poly_coeffs}')
print(f'  Mean age: {mean_age:.3f} Gyr')

# ── Diagnostic plot ───────────────────────────────────────────────────────────
x_rng = np.linspace(df['Age'].min(), df['Age'].max(), 200)
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(df['Age'], df['HR'], yerr=df['e_HR'], fmt='o',
            color='dimgray', alpha=0.4, ms=4, label='Rose+19')
ax.plot(x_rng, lin_slope * x_rng + lin_int,  'b-',  lw=2, label='Linear fit')
ax.plot(x_rng, p_poly(x_rng),               'k--', lw=2, label='Poly fit (deg 2)')
ax.axhline(0, color='k', ls=':', alpha=0.4)
ax.set_xlabel('Local stellar age [Gyr]')
ax.set_ylabel('Hubble Residual (mag)')
ax.set_title('Age–HR calibration (Rose+19)')
ax.legend(); ax.grid(True, ls=':', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_FIGS, 'fig2a_age_bias_fit.pdf'), dpi=150)
plt.show()

# ── Save summary ──────────────────────────────────────────────────────────────
os.makedirs(OUT_TABLES, exist_ok=True)
with open(os.path.join(OUT_TABLES, 'age_correction_summary.txt'), 'w') as f:
    f.write('Age-bias fit on Rose et al. 2019\n')
    f.write(f'  N_SNe      = {len(df)}\n')
    f.write(f'  Mean age   = {mean_age:.4f} Gyr\n')
    f.write(f'  Lin slope  = {lin_slope:.6f} mag/Gyr\n')
    f.write(f'  Lin int    = {lin_int:.6f} mag\n')
    f.write(f'  Poly [a,b,c] = {poly_coeffs}\n')
print(f'Summary written to {OUT_TABLES}/age_correction_summary.txt')


# ── Apply static correction to Pantheon+ (if mc-age file is available) ───────

MC_AGE_DIR = os.path.join(ROOT, 'data', 'external', 'mc-age')
MC_AGE_PANTH = os.path.join(MC_AGE_DIR, 'mc_ages_pantheon.csv')

def apply_and_save_pantheon():
    if not os.path.exists(MC_AGE_PANTH):
        print(f'  [Pantheon] mc-age file not found: {MC_AGE_PANTH}')
        print('  → Skipping static correction for Pantheon+.')
        print('    (Dynamic correction inside MCMC will be used instead.)')
        return

    # Load raw Pantheon+
    sn = pd.read_csv(PANTH_DAT, sep=r'\s+')
    if 'zHD' not in sn.columns:
        sn = pd.read_csv(PANTH_DAT, sep=r'\s+', skiprows=1)

    # Load matched ages
    ages = pd.read_csv(MC_AGE_PANTH)
    merged = sn.merge(ages[['CID', 'Age']], on='CID', how='left')
    merged = merged.dropna(subset=['Age', 'MU_SH0ES'])

    # Corrections
    merged['MU_Lin']  = (merged['MU_SH0ES']
                         - lin_slope * (merged['Age'] - mean_age))
    merged['MU_Poly'] = (merged['MU_SH0ES']
                         - (p_poly(merged['Age']) - p_poly(mean_age)))

    os.makedirs(PANTH_CORR_DIR, exist_ok=True)
    # Write Linear file (same format as original but MU_SH0ES replaced)
    for tag, col in [('Linear', 'MU_Lin'), ('Poly', 'MU_Poly')]:
        out = merged.copy()
        out['MU_SH0ES'] = out[col]
        path = os.path.join(PANTH_CORR_DIR, f'Pantheon+SH0ES_{tag}.dat')
        out.drop(columns=['Age', 'MU_Lin', 'MU_Poly'], errors='ignore')\
           .to_csv(path, sep=' ', index=False)
        print(f'  [Pantheon] Saved {path}')


def apply_and_save_des():
    MC_AGE_DES = os.path.join(MC_AGE_DIR, 'mc_ages_des.csv')
    if not os.path.exists(MC_AGE_DES):
        print(f'  [DES] mc-age file not found: {MC_AGE_DES}')
        print('  → Skipping static correction for DES.')
        return

    # Load raw DES
    import io as _io
    with open(DES_DAT) as f:
        lines = f.readlines()
    names, start = None, 0
    for i, line in enumerate(lines):
        if line.strip().startswith('VARNAMES:'):
            names = line.replace('VARNAMES:', '').split(); start = i + 1; break
    if names:
        sn = pd.read_csv(_io.StringIO(''.join(lines[start:])),
                         sep=r'\s+', names=names, comment='#')
    else:
        sn = pd.read_csv(DES_DAT, sep=r'\s+', comment='#')

    ages   = pd.read_csv(MC_AGE_DES)
    merged = sn.merge(ages[['CID', 'Age']], on='CID', how='left')
    merged = merged.dropna(subset=['Age', 'MU'])

    merged['MU_Lin']  = (merged['MU']
                         - lin_slope * (merged['Age'] - mean_age))
    merged['MU_Poly'] = (merged['MU']
                         - (p_poly(merged['Age']) - p_poly(mean_age)))

    os.makedirs(DES_CORR_DIR, exist_ok=True)
    for tag, col in [('Linear', 'MU_Lin'), ('Poly', 'MU_Poly')]:
        out = merged.copy()
        out['MU'] = out[col]
        path = os.path.join(DES_CORR_DIR, f'DES-Dovekie_HD_{tag}.csv')
        out.drop(columns=['Age', 'MU_Lin', 'MU_Poly'], errors='ignore')\
           .to_csv(path, index=False)
        print(f'  [DES] Saved {path}')


print('\n=== Applying static corrections ===')
apply_and_save_pantheon()
apply_and_save_des()
print('\nStep 2 complete.')
