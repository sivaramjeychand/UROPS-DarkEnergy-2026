"""
Central configuration: data paths, physical constants, and MCMC settings.
All scripts import from here to avoid hardcoded paths.
"""
import os

# ── Root of the repository (one level above src/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data paths ──────────────────────────────────────────────────────────────
ROSE19_T1 = os.path.join(ROOT, 'data', 'external', 'Rose19', 'J_ApJ_874_32_table1.csv')
ROSE19_T7 = os.path.join(ROOT, 'data', 'external', 'Rose19', 'J_ApJ_874_32_table7.csv')

PANTH_DIR   = os.path.join(ROOT, 'data', 'external', 'PantheonPlus',
                           'Pantheon+_Data', '4_DISTANCES_AND_COVAR')
PANTH_DAT   = os.path.join(PANTH_DIR, 'Pantheon+SH0ES.dat')
PANTH_COV   = os.path.join(PANTH_DIR, 'Pantheon+SH0ES_STAT+SYS.cov')

PANTH_CORR_DIR = os.path.join(ROOT, 'data', 'external', 'PantheonPlus_Corrected',
                               'Pantheon+_Data', '4_DISTANCES_AND_COVAR')
PANTH_LIN   = os.path.join(PANTH_CORR_DIR, 'Pantheon+SH0ES_Linear.dat')
PANTH_POLY  = os.path.join(PANTH_CORR_DIR, 'Pantheon+SH0ES_Poly.dat')

DES_DIR     = os.path.join(ROOT, 'data', 'external', 'DES-SN5YR', '4_DISTANCES_COVMAT')
DES_DAT     = os.path.join(DES_DIR, 'DES-Dovekie_HD.csv')
DES_COV     = os.path.join(DES_DIR, 'STAT+SYS.npz')

DES_CORR_DIR = os.path.join(ROOT, 'data', 'external', 'DES-SN5YR_Corrected',
                             '4_DISTANCES_COVMAT')
DES_LIN     = os.path.join(DES_CORR_DIR, 'DES-Dovekie_HD_Linear.csv')
DES_POLY    = os.path.join(DES_CORR_DIR, 'DES-Dovekie_HD_Poly.csv')

BAO_MEAN    = os.path.join(ROOT, 'data', 'external', 'DESI_BAO',
                           'desi_2024_gaussian_bao_ALL_GCcomb_mean.txt')
BAO_COV     = os.path.join(ROOT, 'data', 'external', 'DESI_BAO',
                           'desi_2024_gaussian_bao_ALL_GCcomb_cov.txt')

# ── Output directories ────────────────────────────────────────────────────
OUT_FIGS    = os.path.join(ROOT, 'src', 'output', 'figures')
OUT_CHAINS  = os.path.join(ROOT, 'src', 'output', 'chains')
OUT_TABLES  = os.path.join(ROOT, 'src', 'output', 'tables')

for _d in [OUT_FIGS, OUT_CHAINS, OUT_TABLES]:
    os.makedirs(_d, exist_ok=True)

# ── Physical constants ────────────────────────────────────────────────────
C_LIGHT_KMS = 299792.458        # km s^-1
T_H_FACTOR  = 977.792           # Gyr · (km/s/Mpc) → gives T_H = T_H_FACTOR / H0 in Gyr
TCMB        = 2.7255            # CMB temperature [K]

# ── Age correction ────────────────────────────────────────────────────────
# Ratio of mean SFH delay time to Hubble time at z=0:
#   mean_age_z0 ≈ 5.3 Gyr (Rose+19)  /  T_H(H0=70) ≈ 7.73 Gyr  = 0.685
ALPHA_SFH   = 0.685

# ── BBN prior on ω_b (PDG 2022) ──────────────────────────────────────────
BBN_OMBH2_MEAN = 0.02226
BBN_OMBH2_SIG  = 0.00016        # 1-sigma uncertainty

# ── MCMC settings ─────────────────────────────────────────────────────────
NWALKERS_WCDM  = 64
NSTEPS_WCDM    = 3000
NWALKERS_W0WA  = 100
NSTEPS_W0WA    = 5000
BURNIN_FRAC    = 0.35           # discard first 35% of chain as burn-in

# ── Paper reference values (Son et al. 2024 Table, BAO+CMB+SN, CPL) ──────
PAPER_VALUES = {
    'Panth_Uncorr': dict(Om=0.311, Om_err=0.006, w0=-0.838, w0_err=0.054,
                         wa=-0.612, wa_err=0.201, q0=-0.366, q0_err=0.061),
    'Panth_Lin':    dict(Om=0.353, Om_err=0.006, w0=-0.447, w0_err=0.059,
                         wa=-1.585, wa_err=0.227, q0= 0.065, q0_err=0.060),
    'DES_Uncorr':   dict(Om=0.319, Om_err=0.006, w0=-0.754, w0_err=0.056,
                         wa=-0.848, wa_err=0.217, q0=-0.271, q0_err=0.061),
    'DES_Lin':      dict(Om=0.363, Om_err=0.006, w0=-0.337, w0_err=0.062,
                         wa=-1.902, wa_err=0.246, q0= 0.178, q0_err=0.061),
}
