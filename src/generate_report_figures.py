"""
Generate all figures and extract all table values for report/main.tex.

Run from: the src/ directory  (cd src && python generate_report_figures.py)

Outputs all files to ../report/  so LaTeX can find them without path changes.

Figures generated:
  fig1_replication_linear.png   -- HR vs linear Age + linear fit
  fig1_replication_age.png      -- HR vs log10(Age/yr) + polynomial fit
  fig2_reproduction_table.png   -- Static age evolution spline (from paper digitisation)
  fig3_replication_comparison.png -- Pantheon+ Hubble residuals (3 panels)
  fig3_replication_DES5Y.png    -- DES Hubble residuals (3 panels)
  fig4_residual_agreement.png   -- BAO + SN agreement (Pantheon+)
  fig4_residual_agreement_DES5Y.png -- BAO + SN agreement (DES)
  fig5_w_omm.png                -- w-Omega_m contours (CPL, DES)
  fig8_contours.png             -- w0-wa contours (CPL, DES)
  fig_w0wa_contours.png         -- w0-wa contours (CPL/JBP/LOG, DES)

Table values printed to stdout in LaTeX-pasteable format.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.stats import linregress
import emcee

warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SRC_DIR)
REPORT_DIR = os.path.join(ROOT, 'report')
CHAIN_DIR  = os.path.join(ROOT, 'analysis', 'chains_comparisons')
CHAIN_WCDM = os.path.join(ROOT, 'analysis', 'chains_emcee_all')
os.makedirs(REPORT_DIR, exist_ok=True)

sys.path.insert(0, SRC_DIR)
from data_io   import load_rose19, load_bao, load_sn
from cosmology import distance_modulus, lookback_time

PANTH_DIR  = os.path.join(ROOT, 'data', 'external', 'PantheonPlus',
                           'Pantheon+_Data', '4_DISTANCES_AND_COVAR')
PANTH_DAT  = os.path.join(PANTH_DIR, 'Pantheon+SH0ES.dat')
PANTH_CORR = os.path.join(ROOT, 'data', 'external', 'PantheonPlus_Corrected',
                           'Pantheon+_Data', '4_DISTANCES_AND_COVAR')
DES_DIR    = os.path.join(ROOT, 'data', 'external', 'DES-SN5YR', '4_DISTANCES_COVMAT')
DES_DAT    = os.path.join(DES_DIR, 'DES-Dovekie_HD.csv')
DES_CORR   = os.path.join(ROOT, 'data', 'external', 'DES-SN5YR_Corrected', '4_DISTANCES_COVMAT')

STYLE = dict(dpi=180, bbox_inches='tight')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — HR vs Age (linear) + HR vs log10(Age/yr) (polynomial)
# ══════════════════════════════════════════════════════════════════════════════

def fig1():
    df = load_rose19()
    ages  = df['Age'].values          # linear Gyr
    hrs   = df['HR'].values
    errs  = df['e_HR'].values
    w     = 1.0 / errs**2

    # --- linear fit (Gyr) ---
    p_lin  = np.polyfit(ages, hrs, 1, w=w)
    slope, intercept = p_lin
    r_lin  = np.corrcoef(ages, hrs)[0, 1]

    x_lin  = np.linspace(ages.min(), ages.max(), 300)
    y_lin  = np.polyval(p_lin, x_lin)

    bins = np.linspace(ages.min(), ages.max(), 7)
    cat  = np.digitize(ages, bins)
    bx, by, be = [], [], []
    for k in range(1, 7):
        sel = cat == k
        if sel.sum() < 2: continue
        bx.append(ages[sel].mean())
        by.append(hrs[sel].mean())
        be.append(hrs[sel].std() / np.sqrt(sel.sum()))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.errorbar(ages, hrs, yerr=errs, fmt='o', color='steelblue',
                alpha=0.35, ms=4, elinewidth=0.8, label='Rose+19 SNe')
    ax.plot(x_lin, y_lin, 'k-', lw=2.5,
            label=f'Linear fit  $s={slope:.3f}$ mag/Gyr')
    ax.errorbar(bx, by, yerr=be, fmt='s', color='crimson',
                ms=7, markeredgecolor='k', lw=1.5, label='Binned avg.')
    ax.axhline(0, color='k', ls=':', alpha=0.4, lw=1)
    ax.set_xlabel('Local stellar age [Gyr]', fontsize=12)
    ax.set_ylabel('Hubble Residual (mag)', fontsize=12)
    ax.set_title('Age-Bias Calibration — Linear Fit', fontsize=12)
    ax.text(0.05, 0.06,
            f'$r={r_lin:.3f}$\n$s={slope:.4f}$ mag/Gyr',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(fc='white', alpha=0.85, ec='lightgray'))
    ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, 'fig1_replication_linear.png'), **STYLE)
    plt.close(); print('  fig1_replication_linear.png')

    # --- linear + polynomial fits in log10(Age/yr) space ---
    log_ages = np.log10(ages * 1e9)          # convert Gyr → yr then log10
    p_poly   = np.polyfit(log_ages, hrs, 2, w=w)
    p_lin_log = np.polyfit(log_ages, hrs, 1, w=w)   # linear fit in log-age space
    s_log    = p_lin_log[0]
    r_log    = np.corrcoef(log_ages, hrs)[0, 1]

    x_log   = np.linspace(log_ages.min(), log_ages.max(), 300)
    y_poly  = np.polyval(p_poly,    x_log)
    y_lin_log = np.polyval(p_lin_log, x_log)

    cat_l = np.digitize(log_ages, np.linspace(log_ages.min(), log_ages.max(), 7))
    bxl, byl, bel = [], [], []
    for k in range(1, 7):
        sel = cat_l == k
        if sel.sum() < 2: continue
        bxl.append(log_ages[sel].mean())
        byl.append(hrs[sel].mean())
        bel.append(hrs[sel].std() / np.sqrt(sel.sum()))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.errorbar(log_ages, hrs, yerr=errs, fmt='o', color='steelblue',
                alpha=0.35, ms=4, elinewidth=0.8, label='Rose+19 SNe')
    ax.plot(x_log, y_lin_log, 'r--', lw=2.0,
            label=f'Linear fit  $s={s_log:.3f}$ mag/dex  ($r={r_log:.3f}$)')
    ax.plot(x_log, y_poly, 'k-', lw=2.5, label='Poly fit (deg 2)')
    ax.errorbar(bxl, byl, yerr=bel, fmt='s', color='crimson',
                ms=7, markeredgecolor='k', lw=1.5, label='Binned avg.')
    ax.axhline(0, color='k', ls=':', alpha=0.4, lw=1)
    ax.set_xlabel(r'$\log_{10}(\mathrm{Age\,/\,yr})$', fontsize=12)
    ax.set_ylabel('Hubble Residual (mag)', fontsize=12)
    ax.set_title('Age-Bias Calibration — Log Age (Linear + Poly fits)', fontsize=12)
    ax.text(0.05, 0.06,
            f'Poly: $c_2={p_poly[0]:.4f}$, $c_1={p_poly[1]:.4f}$, $c_0={p_poly[2]:.4f}$',
            transform=ax.transAxes, fontsize=8,
            bbox=dict(fc='white', alpha=0.85, ec='lightgray'))
    ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, 'fig1_replication_age.png'), **STYLE)
    plt.close(); print('  fig1_replication_age.png')

    return slope, p_poly   # return for use in other figures


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Age evolution spline (digitised from paper)
# ══════════════════════════════════════════════════════════════════════════════

def fig2():
    from config import ALPHA_SFH

    # ── Son+2025 digitised curves (flipped to positive) ──────────────────────
    z_pts     = np.array([0,   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                          0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    age_solid = np.array([0,   1.0, 1.8, 2.6, 3.3, 3.9, 4.4, 4.8, 5.1,
                          5.3, 5.5, 5.6, 5.7, 5.8, 5.85,5.9, 5.95,6.0])
    age_dash  = np.array([0,   1.0, 1.9, 2.8, 3.6, 4.3, 4.8, 5.3, 5.7,
                          6.0, 6.3, 6.5, 6.7, 6.9, 7.0, 7.1, 7.2, 7.3])

    f_s = interp1d(z_pts, age_solid, kind='quadratic')
    f_d = interp1d(z_pts, age_dash,  kind='quadratic')
    z_g = np.linspace(0, 1.7, 300)

    # ── Lee+2021 digitised data (extracted_data (1).csv) ─────────────────────
    # Lee Fig. 6 plots |ΔAge| as positive Gyr under their ΛCDM cosmology.
    # Dynamic scaling converts these values to Son+2025's CPL cosmology by
    # reweighting with the ratio of lookback times at each redshift:
    #   Lee_scaled(z) = Lee_raw(z) × t_L(z, Son_CPL) / t_L(z, Lee_ΛCDM)
    lee_csv = os.path.join(ROOT, 'extracted_data (1).csv')
    lee_df  = pd.read_csv(lee_csv)
    lee_z = lee_df['Redshift (z)'].values

    # Convention note: Lee+2021 Fig. 6 uses the *opposite* line-style convention
    # to Son+2025.  What Lee labels "Solid Line" corresponds to Son's dashed
    # (alternative) model, and what Lee labels "Dashed Line" corresponds to
    # Son's solid (CPL best-fit) curve.  We swap here so that lee_solid_raw
    # aligns with age_solid (Son's CPL best-fit) for a like-for-like comparison.
    lee_solid_raw = lee_df['Dashed Line'].values   # Lee dashed ↔ Son solid
    lee_dash_raw  = lee_df['Solid Line'].values    # Lee solid  ↔ Son dashed

    # Cosmological parameters
    OM_LEE, W0_LEE, WA_LEE, H0_LEE = 0.27, -1.0,   0.0,    70.0   # Lee: flat ΛCDM
    OM_SON, W0_SON, WA_SON, H0_SON = 0.353, -0.447, -1.75, 73.0   # Son: CPL best-fit

    # Lookback-time ratio at each Lee z-node (skip z=0 to avoid 0/0)
    mask   = lee_z > 0
    z_mask = lee_z[mask]
    tL_lee_pts = lookback_time(z_mask, OM_LEE, W0_LEE, WA_LEE, H0_LEE, model='CPL')
    tL_son_pts = lookback_time(z_mask, OM_SON, W0_SON, WA_SON, H0_SON, model='CPL')
    ratio_pts  = tL_son_pts / tL_lee_pts

    # Apply dynamic scaling; z=0 anchor stays at 0
    lee_solid_scaled = np.zeros_like(lee_solid_raw)
    lee_dash_scaled  = np.zeros_like(lee_dash_raw)
    lee_solid_scaled[mask] = lee_solid_raw[mask] * ratio_pts
    lee_dash_scaled[mask]  = lee_dash_raw[mask]  * ratio_pts

    print(f'  t_L ratio (Son/Lee) range: {ratio_pts.min():.4f} - {ratio_pts.max():.4f}')
    idx1 = np.argmin(np.abs(lee_z - 1.0))
    print(f'  Lee solid at z=1: raw={lee_solid_raw[idx1]:.3f}  '
          f'dyn-scaled={lee_solid_scaled[idx1]:.3f}  '
          f'Son+2025={interp1d(z_pts, age_solid, kind="quadratic")(1.0):.3f} Gyr')

    f_lee_s = interp1d(lee_z, lee_solid_scaled, kind='quadratic')
    f_lee_d = interp1d(lee_z, lee_dash_scaled,  kind='quadratic')

    # ── ALPHA_SFH approximation evaluated under Son's cosmology ───────────────
    tL_son_fine = lookback_time(z_g, OM_SON, W0_SON, WA_SON, H0_SON, model='CPL')
    delta_dyn   = ALPHA_SFH * tL_son_fine

    # ── Residuals vs Son+2025 digitised solid at Lee z-nodes ─────────────────
    son_at_lee_z  = interp1d(z_pts, age_solid, kind='quadratic')(z_mask)
    dyn_at_z_mask = ALPHA_SFH * lookback_time(z_mask, OM_SON, W0_SON, WA_SON, H0_SON, model='CPL')
    rms_lee_dyn   = np.sqrt(np.mean((lee_solid_scaled[mask] - son_at_lee_z) ** 2))
    rms_alpha_dyn = np.sqrt(np.mean((dyn_at_z_mask - son_at_lee_z) ** 2))
    print(f'  RMS vs Son+2025 solid:  Lee dyn-scaled={rms_lee_dyn:.3f} Gyr,  '
          f'alpha scaling={rms_alpha_dyn:.3f} Gyr')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={'width_ratios': [1.8, 1]})

    ax = axes[0]
    ax.plot(z_g, f_s(z_g),      'k-',  lw=2.5, label='Son+2025 solid / CPL best-fit (digitised)')
    ax.plot(z_g, f_d(z_g),      'k--', lw=2.0, label='Son+2025 dashed / alternative (digitised)')
    ax.plot(z_g, f_lee_s(z_g),  color='tab:orange', lw=2.0, ls='-',
            label='Lee+2021 (conv.-corrected solid) dyn-scaled to Son cosm.')
    ax.plot(z_g, f_lee_d(z_g),  color='tab:orange', lw=1.5, ls='--',
            label='Lee+2021 (conv.-corrected dashed) dyn-scaled to Son cosm.')
    ax.plot(z_g, delta_dyn,     'b-',  lw=2.0, alpha=0.85,
            label=r'$\alpha_{\rm SFH}\,t_L(z,\,\rm Son)$'
                  f'  ($\\alpha={ALPHA_SFH}$)')
    ax.scatter(z_pts, age_solid, color='k', s=18, alpha=0.4, zorder=5)
    ax.axhline(0, color='k', ls=':', alpha=0.4, lw=1)
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel(r'Age change $|\Delta\mathrm{Age}|$ (Gyr)', fontsize=12)
    ax.set_title('Mean Stellar Age Evolution vs. Redshift', fontsize=12)
    ax.set_xlim(0, 1.75); ax.set_ylim(-0.5, 8)
    ax.grid(True, ls=':', alpha=0.35)
    ax.legend(fontsize=8, loc='upper left')

    # ── Right panel: residuals ─────────────────────────────────────────────────
    ax2 = axes[1]
    res_lee = lee_solid_scaled[mask] - son_at_lee_z
    res_alpha = dyn_at_z_mask - son_at_lee_z
    ax2.plot(z_mask, res_alpha, 'o-', color='royalblue', lw=1.5, ms=4, 
             label=f'Alpha scaling  (RMS={rms_alpha_dyn:.2f} Gyr)')
    ax2.plot(z_mask, res_lee, 's-', color='tab:orange', lw=1.5, ms=4,
             label=f'Lee dyn-scaled (RMS={rms_lee_dyn:.2f} Gyr)')
    ax2.axhline(0, color='k', ls=':', alpha=0.5, lw=1)
    ax2.set_xlabel('Redshift $z$', fontsize=12)
    ax2.set_ylabel(r'Residual vs Son+2025 solid (Gyr)', fontsize=10)
    ax2.set_title('Accuracy check', fontsize=12)
    ax2.set_xlim(0, 1.75)
    ax2.grid(True, ls=':', alpha=0.35)
    ax2.legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, 'fig2_reproduction_table.png'), **STYLE)
    plt.close(); print('  fig2_reproduction_table.png')

    # Return a spline with the original sign convention (ΔAge < 0) for the
    # downstream Hubble-diagram pipeline which expects negative age offsets.
    age_solid_neg = -age_solid
    return interp1d(z_pts, age_solid_neg, kind='quadratic')


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — load corrected/uncorrected SN distances
# ══════════════════════════════════════════════════════════════════════════════

def _load_panth_df(tag):
    """Load a Pantheon+ .dat file, return DataFrame."""
    if tag == 'Uncorr':
        path = PANTH_DAT
    else:
        path = os.path.join(PANTH_CORR, f'Pantheon+SH0ES_{tag}.dat')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=r'\s+')
    if 'zHD' not in df.columns:
        df = pd.read_csv(path, sep=r'\s+', skiprows=1)
    return df[df['zHD'] > 0.005].copy()


def _load_des_df(tag):
    """Load a DES CSV file, return DataFrame."""
    import io as _io
    if tag == 'Uncorr':
        path = DES_DAT
    else:
        path = os.path.join(DES_CORR, f'DES-Dovekie_HD_{tag}.csv')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = f.readlines()
    names, start = None, 0
    for i, l in enumerate(lines):
        if l.strip().startswith('VARNAMES:'):
            names = l.replace('VARNAMES:', '').split(); start = i+1; break
    if names:
        # Data rows begin with "SN:" token before the actual columns
        data_lines = [l for l in lines[start:] if not l.strip().startswith('#') and l.strip()]
        data_lines_stripped = [l.replace('SN:', '', 1) for l in data_lines]
        df = pd.read_csv(_io.StringIO(''.join(data_lines_stripped)),
                         sep=r'\s+', names=names)
    else:
        df = pd.read_csv(path, sep=r'\s+', engine='python')
    df['zHD'] = pd.to_numeric(df['zHD'], errors='coerce')
    mu_col = 'MU' if 'MU' in df.columns else 'MU_SH0ES'
    df[mu_col] = pd.to_numeric(df[mu_col], errors='coerce')
    df = df.dropna(subset=['zHD', mu_col]).copy()
    df['MU_SH0ES'] = df[mu_col]
    return df


def _bin_sn(z_arr, hr_arr, n_per_bin=50):
    idx = np.argsort(z_arr)
    zs  = z_arr[idx]; hs = hr_arr[idx]
    bins = len(zs) // n_per_bin
    bz, bh, be = [], [], []
    for i in range(bins):
        sl  = slice(i*n_per_bin, (i+1)*n_per_bin)
        bz.append(zs[sl].mean())
        bh.append(hs[sl].mean())
        be.append(hs[sl].std() / np.sqrt(n_per_bin))
    return np.array(bz), np.array(bh), np.array(be)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Hubble diagram residuals (3 panels)
# ══════════════════════════════════════════════════════════════════════════════

def fig3(dataset='Pantheon', out_name='fig3_replication_comparison.png',
         title='Pantheon+'):
    from astropy.cosmology import LambdaCDM, FlatLambdaCDM, w0waCDM

    H0     = 73.04
    cosmo_base = LambdaCDM(H0=H0, Om0=0.30, Ode0=0.0)   # open baseline
    c_lcdm = FlatLambdaCDM(H0=H0, Om0=0.30)

    # Paper best-fit corrected cosmologies
    if dataset == 'Pantheon':
        c_lin  = w0waCDM(H0=H0, Om0=0.353, Ode0=0.647, w0=-0.447, wa=-1.585)
    else:
        c_lin  = w0waCDM(H0=H0, Om0=0.363, Ode0=0.637, w0=-0.337, wa=-1.902)

    z_g   = np.logspace(np.log10(0.01), np.log10(1.3), 200)
    def diff(c): return c.distmod(z_g).value - cosmo_base.distmod(z_g).value

    d_lcdm = diff(c_lcdm)
    d_lin  = diff(c_lin)

    loaders = {'Uncorrected': ('Uncorr', 'dimgray'),
               'Linear corr.': ('Linear', 'steelblue'),
               'Poly. corr.': ('Poly', 'black')}

    fig, axes = plt.subplots(3, 1, figsize=(9, 13), sharex=True,
                              gridspec_kw={'hspace': 0.04})

    for ax, (label, (tag, col)) in zip(axes, loaders.items()):
        df = _load_panth_df(tag) if dataset == 'Pantheon' else _load_des_df(tag)
        if df is None:
            ax.text(0.5, 0.5, f'{tag} file not found', transform=ax.transAxes,
                    ha='center', fontsize=11, color='red')
            ax.set_ylabel(r'$\mu - \mu_{\rm base}$ [mag]', fontsize=11)
            ax.text(0.03, 0.88, label, transform=ax.transAxes,
                    fontsize=12, fontweight='bold')
            continue

        base = cosmo_base.distmod(df['zHD'].values).value
        res  = df['MU_SH0ES'].values - base

        # Centre low-z SNe around 0 (same approach as 04_figure3_replication_DES5Y.py)
        mask_lz = df['zHD'].values < 0.1
        if mask_lz.sum() > 0:
            res = res - np.median(res[mask_lz])

        bz, bh, be = _bin_sn(df['zHD'].values, res)

        ax.scatter(df['zHD'], res, c=col, s=2, alpha=0.15, zorder=1)
        ax.errorbar(bz, bh, yerr=be, fmt='o', color='royalblue',
                    ms=5.5, markeredgecolor='navy', capsize=2,
                    label='Binned (50/bin)', zorder=8)
        ax.axhline(0, color='k', ls=':', lw=0.8)
        ax.plot(z_g, d_lcdm, 'r-', lw=2,
                label=r'$\Lambda$CDM ($\Omega_m=0.30$)')
        ax.plot(z_g, d_lin, 'b--', lw=2,
                label='Best-fit (lin. corr.)')
        ax.set_ylabel(r'$\mu - \mu_{\rm base}$ [mag]', fontsize=11)
        ax.set_ylim(-0.38, 0.48)
        ax.text(0.03, 0.88, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold')
        ax.grid(True, ls=':', alpha=0.3)
        if ax is axes[-1]:
            ax.legend(fontsize=9, loc='lower right', ncol=2)

    axes[-1].set_xlabel('Redshift $z$', fontsize=12)
    axes[-1].set_xlim(0, 1.3)
    axes[0].set_title(f'Hubble Diagram Residuals — {title}', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, out_name), **STYLE)
    plt.close(); print(f'  {out_name}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — BAO + SN agreement
# ══════════════════════════════════════════════════════════════════════════════

def fig4(dataset='Pantheon', out_name='fig4_residual_agreement.png', title='Pantheon+'):
    from astropy.cosmology import LambdaCDM, FlatLambdaCDM, w0waCDM

    H0 = 73.04
    cosmo_base = LambdaCDM(H0=H0, Om0=0.30, Ode0=0.0)
    c_lcdm = FlatLambdaCDM(H0=H0, Om0=0.30)

    if dataset == 'Pantheon':
        c_corr = w0waCDM(H0=H0, Om0=0.353, Ode0=0.647, w0=-0.447, wa=-1.585)
    else:
        c_corr = w0waCDM(H0=H0, Om0=0.363, Ode0=0.637, w0=-0.337, wa=-1.902)

    z_g = np.linspace(0.01, 2.5, 300)
    def diff(c): return c.distmod(z_g).value - cosmo_base.distmod(z_g).value

    bao_df, _, _ = load_bao()
    # rd calibrated to match analysis/07_figure4_agreement*.py reference scripts
    rd_fid = 93.9 / (H0 / 100)   # ≈ 128.5 Mpc for H0=73.04
    bao_mu = []
    for _, row in bao_df.iterrows():
        if row['type'] == 'DM_over_rs':
            DM = row['value'] * rd_fid
            dL = DM * (1 + row['z'])
            if dL > 0:
                bao_mu.append({'z': row['z'],
                               'mu': 5*np.log10(dL)+25})
    bao_mu_df = pd.DataFrame(bao_mu)

    loaders = {'Uncorrected': 'Uncorr', 'Linear corr.': 'Linear',
               'Poly. corr.': 'Poly'}

    fig, axes = plt.subplots(3, 1, figsize=(9, 13), sharex=True,
                              gridspec_kw={'hspace': 0.04})

    for ax, (label, tag) in zip(axes, loaders.items()):
        df = _load_panth_df(tag) if dataset == 'Pantheon' else _load_des_df(tag)

        ax.axhline(0, color='k', ls=':', lw=0.8)
        ax.plot(z_g, diff(c_lcdm), 'r-', lw=2, label=r'$\Lambda$CDM')
        ax.plot(z_g, diff(c_corr), 'b--', lw=2, label='Best-fit (lin. corr.)')

        if df is not None:
            base = cosmo_base.distmod(df['zHD'].values).value
            res  = df['MU_SH0ES'].values - base
            # Centre low-z SNe around 0 (same approach as fig3)
            mask_lz = df['zHD'].values < 0.1
            if mask_lz.sum() > 0:
                res = res - np.median(res[mask_lz])
            bz, bh, be = _bin_sn(df['zHD'].values, res)
            ax.errorbar(bz, bh, yerr=be, fmt='o', color='dimgray',
                        ms=5, capsize=2, label='SN binned', zorder=8)

        if not bao_mu_df.empty:
            base_b = cosmo_base.distmod(bao_mu_df['z'].values).value
            res_b  = bao_mu_df['mu'].values - base_b
            ax.errorbar(bao_mu_df['z'], res_b, yerr=0.03, fmt='D',
                        color='darkorange', ms=7, markeredgecolor='k',
                        label='DESI BAO', zorder=10)

        ax.set_ylabel(r'$\mu - \mu_{\rm base}$ [mag]', fontsize=11)
        ax.set_ylim(-0.38, 0.6)
        ax.text(0.03, 0.88, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold')
        ax.grid(True, ls=':', alpha=0.3)
        if ax is axes[-1]:
            ax.legend(fontsize=9, loc='lower right', ncol=2)

    axes[-1].set_xlabel('Redshift $z$', fontsize=12)
    axes[-1].set_xlim(0, 2.5)
    axes[0].set_title(f'BAO + SN Ia Agreement — {title}', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, out_name), **STYLE)
    plt.close(); print(f'  {out_name}')


# ══════════════════════════════════════════════════════════════════════════════
# CHAIN UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _load_chain(path, burnin_frac=0.4):
    """Return (flat_chain, lnprob) after burn-in, or None."""
    if not os.path.exists(path):
        return None, None
    try:
        r = emcee.backends.HDFBackend(path, read_only=True)
        n = r.iteration
        if n < 20:
            return None, None
        bi = max(1, int(burnin_frac * n))
        chain  = r.get_chain(discard=bi, flat=True)
        lnprob = r.get_log_prob(discard=bi, flat=True)
        return chain, lnprob
    except Exception:
        return None, None


def _percentiles(x):
    p = np.percentile(x, [16, 50, 84])
    return p[1], p[2]-p[1], p[1]-p[0]


def _q0(Om, w0):
    """Deceleration parameter at z=0 for w0wa model (wa doesn't enter at z=0)."""
    return Om/2.0 + (1.0-Om)/2.0*(1.0+3.0*w0)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — w-Omega_m contours (CPL, wCDM from chains_emcee_all)
# ══════════════════════════════════════════════════════════════════════════════

def fig5():
    """Plot w vs Omega_m from wCDM MCMC (3D chains: Om, w, H0)."""
    try:
        from getdist import plots as gdp, MCSamples
    except ImportError:
        print('  [fig5] getdist not available — skipping contour plot')
        return

    names  = ['Om', 'w', 'H0']
    labels = [r'\Omega_m', 'w', 'H_0']
    cols   = ['royalblue', 'crimson', 'forestgreen']
    slist, llist = [], []

    for tag, label in [('DES_Uncorr', 'DES Uncorr.'),
                       ('DES_Lin',    'DES Linear'),
                       ('DES_Poly',   'DES Poly.')]:
        chain, _ = _load_chain(os.path.join(CHAIN_WCDM, f'{tag}.h5'))
        if chain is not None and chain.shape[1] == 3:
            slist.append(MCSamples(samples=chain, names=names,
                                   labels=labels, label=label))
            llist.append(label)

    if len(slist) < 1:
        print('  [fig5] No wCDM chains available — skipping')
        return

    g = gdp.get_subplot_plotter(subplot_size=5)
    g.plot_2d(slist, 'Om', 'w', filled=True,
              colors=cols[:len(slist)])
    g.add_legend(llist, legend_loc='upper right')
    import matplotlib.pyplot as _p
    ax = _p.gcf().axes[0]
    ax.axvline(0.315, color='gray', ls='--', alpha=0.4, lw=1)
    ax.axhline(-1.0,  color='gray', ls='--', alpha=0.4, lw=1)
    ax.set_xlim(0.15, 0.50); ax.set_ylim(-2.3, -0.1)
    _p.suptitle(r'wCDM: $\Omega_m$–$w$ Contours (DES-SN5YR)',
                fontsize=13, y=1.02)
    _p.savefig(os.path.join(REPORT_DIR, 'fig5_w_omm.png'), **STYLE)
    _p.close(); print('  fig5_w_omm.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — w0-wa contours (CPL, DES)
# ══════════════════════════════════════════════════════════════════════════════

def fig8():
    """Plot w0 vs wa from CPL MCMC (DES, Uncorr/Lin/Poly)."""
    try:
        from getdist import plots as gdp, MCSamples
    except ImportError:
        print('  [fig8] getdist not available — skipping'); return

    names  = ['Om', 'w0', 'wa', 'H0', 'ombh2']
    labels = [r'\Omega_m', 'w_0', 'w_a', 'H_0', r'\Omega_b h^2']
    cols   = ['royalblue', 'crimson', 'forestgreen']
    slist, llist = [], []

    for tag, label in [('CPL_DES_Uncorr', 'Uncorrected'),
                       ('CPL_DES_Lin',    'Linear corr.'),
                       ('CPL_DES_Poly',   'Poly corr.')]:
        chain, _ = _load_chain(os.path.join(CHAIN_DIR, f'{tag}.h5'))
        if chain is not None and chain.shape[1] == 5:
            slist.append(MCSamples(samples=chain, names=names,
                                   labels=labels, label=label))
            llist.append(label)

    if not slist:
        print('  [fig8] No CPL DES chains — skipping'); return

    g = gdp.get_subplot_plotter(subplot_size=5)
    g.plot_2d(slist, 'w0', 'wa', filled=True, colors=cols[:len(slist)],
              lims=[-2.5, 0.5, -5.0, 2.5])
    g.add_legend(llist, legend_loc='upper right')
    import matplotlib.pyplot as _p
    ax = _p.gcf().axes[0]
    ax.plot(-1, 0, 'k*', ms=12, zorder=10, label=r'$\Lambda$CDM')
    # q0=0 boundary: w0 = -1/3 - Om_ref/(3*(1-Om_ref))
    Om_ref = 0.32
    w0_crit = -1./3. - Om_ref/(3.*(1.-Om_ref))
    ax.axvline(w0_crit, color='k', ls=':', lw=1.5,
               label=f'$q_0=0$  ($w_0\\approx{w0_crit:.2f}$)')
    ax.legend(fontsize=8)
    _p.suptitle(r'CPL: $w_0$–$w_a$ Contours (DES-SN5YR)',
                fontsize=13, y=1.02)
    _p.savefig(os.path.join(REPORT_DIR, 'fig8_contours.png'), **STYLE)
    _p.close(); print('  fig8_contours.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — w0-wa extension (CPL / JBP / LOG, DES Lin)
# ══════════════════════════════════════════════════════════════════════════════

def fig_w0wa_ext():
    try:
        from getdist import plots as gdp, MCSamples
    except ImportError:
        print('  [fig_w0wa] getdist not available — skipping'); return

    names  = ['Om', 'w0', 'wa', 'H0', 'ombh2']
    labels = [r'\Omega_m', 'w_0', 'w_a', 'H_0', r'\Omega_b h^2']
    slist, llist = [], []

    for model, colour in [('CPL', 'royalblue'),
                          ('JBP', 'crimson'),
                          ('LOG', 'forestgreen')]:
        # Use Linear correction for each model (shows the corrected result)
        chain, _ = _load_chain(os.path.join(CHAIN_DIR, f'{model}_DES_Lin.h5'))
        if chain is not None and chain.shape[1] == 5:
            slist.append(MCSamples(samples=chain, names=names,
                                   labels=labels, label=f'{model} (Lin)'))
            llist.append(f'{model} (Lin.)')

    if not slist:
        print('  [fig_w0wa_ext] No chains — skipping'); return

    cols = ['royalblue', 'crimson', 'forestgreen']
    g = gdp.get_subplot_plotter(subplot_size=5)
    g.plot_2d(slist, 'w0', 'wa', filled=True, colors=cols[:len(slist)],
              lims=[-2.5, 0.5, -5.0, 2.5])
    g.add_legend(llist, legend_loc='upper right')
    import matplotlib.pyplot as _p
    ax = _p.gcf().axes[0]
    ax.plot(-1, 0, 'k*', ms=12, zorder=10)
    _p.suptitle(r'$w_0$–$w_a$ Contours: CPL vs JBP vs LOG (DES, Linear corr.)',
                fontsize=12, y=1.02)
    _p.savefig(os.path.join(REPORT_DIR, 'fig_w0wa_contours.png'), **STYLE)
    _p.close(); print('  fig_w0wa_contours.png')


# ══════════════════════════════════════════════════════════════════════════════
# TABLE — extract parameter estimates from all chains
# ══════════════════════════════════════════════════════════════════════════════

def extract_tables():
    """Extract median + 68% CI from all w0wa chains and print LaTeX rows."""
    rows = {}
    for model in ['CPL', 'JBP', 'LOG']:
        for ds in ['DES', 'Panth']:
            for corr in ['Uncorr', 'Lin', 'Poly']:
                key  = f'{model}_{ds}_{corr}'
                path = os.path.join(CHAIN_DIR, f'{key}.h5')
                chain, lnp = _load_chain(path)
                if chain is None:
                    continue
                Om_m, Om_u, Om_l = _percentiles(chain[:,0])
                w0_m, w0_u, w0_l = _percentiles(chain[:,1])
                wa_m, wa_u, wa_l = _percentiles(chain[:,2])
                q0 = _q0(chain[:,0], chain[:,1])
                q0_m, q0_u, q0_l = _percentiles(q0)
                state = 'Decelerating' if q0_m > 0 else 'Accelerating'
                rows[key] = dict(Om=Om_m, Om_u=Om_u, Om_l=Om_l,
                                 w0=w0_m, w0_u=w0_u, w0_l=w0_l,
                                 wa=wa_m, wa_u=wa_u, wa_l=wa_l,
                                 q0=q0_m, q0_u=q0_u, q0_l=q0_l,
                                 state=state)

    def pm(m, u, l):
        if abs(u-l) < 0.005:
            return f'${m:.3f} \\pm {(u+l)/2:.3f}$'
        return f'${m:.3f}^{{+{u:.3f}}}_{{-{l:.3f}}}$'

    print('\n\n%%  ===== TABLE 1: CPL Replication (DES + Pantheon) =====')
    paper = {
        'CPL_DES_Uncorr':  (0.319, 0.006, -0.754, 0.056, -0.848, 0.217, -0.271),
        'CPL_DES_Lin':     (0.363, 0.006, -0.337, 0.062, -1.902, 0.246,  0.178),
        'CPL_Panth_Uncorr':(0.311, 0.006, -0.838, 0.054, -0.612, 0.201, -0.366),
        'CPL_Panth_Lin':   (0.353, 0.006, -0.447, 0.059, -1.585, 0.227,  0.065),
    }
    ds_label = {'DES': 'DES-SN5YR', 'Panth': 'Pantheon+'}
    corr_label = {'Uncorr': 'Uncorr.', 'Lin': 'Linear', 'Poly': 'Poly.'}

    for ds in ['DES', 'Panth']:
        for corr in ['Uncorr', 'Lin', 'Poly']:
            key = f'CPL_{ds}_{corr}'
            dl  = ds_label[ds]; cl = corr_label[corr]
            if key in paper and corr != 'Poly':
                p = paper[key]
                state = 'Decel.' if p[6] > 0 else 'Accel.'
                s = '+' if p[6] > 0 else ''
                print(f'{dl} & {cl} (Paper) & '
                      f'${p[0]:.3f}\\pm{p[1]:.3f}$ & '
                      f'${p[2]:.3f}\\pm{p[3]:.3f}$ & '
                      f'${p[4]:.3f}\\pm{p[5]:.3f}$ & '
                      f'${s}{p[6]:.3f}$ & {state} \\\\')
            if key in rows:
                r = rows[key]; s = '+' if r['q0'] > 0 else ''
                print(f'{dl} & {cl} (This work) & '
                      f'{pm(r["Om"], r["Om_u"], r["Om_l"])} & '
                      f'{pm(r["w0"], r["w0_u"], r["w0_l"])} & '
                      f'{pm(r["wa"], r["wa_u"], r["wa_l"])} & '
                      f'${s}{r["q0"]:.3f}$ & {r["state"]} \\\\')
        print('\\midrule')

    print('\n\n%%  ===== TABLE 2: All models, DES =====')
    for model in ['CPL', 'JBP', 'LOG']:
        print(f'\\textbf{{{model}}}')
        for corr in ['Uncorr', 'Lin', 'Poly']:
            key = f'{model}_DES_{corr}'
            cl  = corr_label[corr]
            if key in rows:
                r = rows[key]; s = '+' if r['q0'] > 0 else ''
                print(f'  & {cl} & '
                      f'{pm(r["Om"], r["Om_u"], r["Om_l"])} & '
                      f'{pm(r["w0"], r["w0_u"], r["w0_l"])} & '
                      f'{pm(r["wa"], r["wa_u"], r["wa_l"])} & '
                      f'${s}{r["q0"]:.3f}$ & {r["state"]} \\\\')
        print('\\midrule')

    print('\n\n%%  ===== TABLE 3: All models, Pantheon+ =====')
    for model in ['CPL', 'JBP', 'LOG']:
        print(f'\\textbf{{{model}}}')
        for corr in ['Uncorr', 'Lin', 'Poly']:
            key = f'{model}_Panth_{corr}'
            cl  = corr_label[corr]
            if key in rows:
                r = rows[key]; s = '+' if r['q0'] > 0 else ''
                print(f'  & {cl} & '
                      f'{pm(r["Om"], r["Om_u"], r["Om_l"])} & '
                      f'{pm(r["w0"], r["w0_u"], r["w0_l"])} & '
                      f'{pm(r["wa"], r["wa_u"], r["wa_l"])} & '
                      f'${s}{r["q0"]:.3f}$ & {r["state"]} \\\\')
        print('\\midrule')

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Generating report figures...')

    print('\n[1] Figure 1 — Age-bias fits')
    lin_slope, poly_coeffs_logyr = fig1()

    print('\n[2] Figure 2 — Age evolution spline')
    fig2()

    print('\n[3] Figure 3 — Hubble residuals')
    fig3('Pantheon', 'fig3_replication_comparison.png', 'Pantheon+')
    fig3('DES',      'fig3_replication_DES5Y.png',      'DES-SN5YR')

    print('\n[4] Figure 4 — BAO + SN agreement')
    fig4('Pantheon', 'fig4_residual_agreement.png',      'Pantheon+')
    fig4('DES',      'fig4_residual_agreement_DES5Y.png','DES-SN5YR')

    print('\n[5] Figure 5 — wCDM contours')
    fig5()

    print('\n[8] Figure 8 — w0wa contours (CPL, DES)')
    fig8()

    print('\n[ext] Figure w0wa extension — CPL/JBP/LOG comparison')
    fig_w0wa_ext()

    print('\n\n===== TABLE VALUES =====')
    rows = extract_tables()

    print(f'\nAll figures saved to: {REPORT_DIR}')
