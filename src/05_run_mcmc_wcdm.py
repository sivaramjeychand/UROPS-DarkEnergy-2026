"""
Step 5 — Run wCDM MCMC.

Model: flat wCDM with constant dark energy equation of state w.
Free parameters (4-D):  Om, w, H0, ombh2

Datasets × corrections:
  Pantheon+ : Uncorr, Lin, Poly
  DES-SN5YR : Uncorr, Lin, Poly

Correction approach: *static* (pre-corrected μ values from data files).
The static correction replicates the paper's primary analysis.

Key convergence improvements over the original analysis scripts:
  - rdrag via Eisenstein & Hu (1998) analytical formula (no CAMB per step)
  - 64 walkers, 3000 steps (up from 16–32 walkers, 100–120 steps)
  - Initial walkers drawn from a tight Gaussian ball around (0.3, -1, 70, 0.022)
  - Convergence reported via integrated autocorrelation time τ
  - Results saved as emcee HDF5 backends in src/output/chains/wcdm/
"""
import os
import sys
import numpy as np
import emcee

# Allow imports from src/ when running as a standalone script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (OUT_CHAINS, BBN_OMBH2_MEAN, BBN_OMBH2_SIG,
                    NWALKERS_WCDM, NSTEPS_WCDM, BURNIN_FRAC)
from data_io import load_bao, load_sn
from cosmology import bao_theory, distance_modulus

OUT_DIR = os.path.join(OUT_CHAINS, 'wcdm')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load BAO once (shared across all chains) ──────────────────────────────────
bao_df, bao_data, bao_invcov = load_bao()


# ── Log-probability ───────────────────────────────────────────────────────────

def log_prob(theta, z_sn, mu_sn, invcov_sn):
    """
    Log-posterior for wCDM:  ln P ∝ ln π(θ) − ½ χ²_BAO − ½ χ²_SN(marg)

    Parameters are  θ = (Om, w, H0, ombh2).

    The SN absolute magnitude M is analytically marginalised:
        χ²_SN(marg) = χ²_SN − (Σ_ij C^{-1}_{ij} δ_j)² / (Σ_ij C^{-1}_{ij})
    """
    Om, w, H0, ombh2 = theta

    # ── Hard priors (DESI DR2 standard) ──────────────────────────────────────
    if not (0.10 < Om    < 0.70): return -np.inf
    if not (-3.0 < w     < 1.0 ): return -np.inf   # DESI: U[-3, 1]
    if not (50.0 < H0    < 100.): return -np.inf
    if not (0.017 < ombh2 < 0.028): return -np.inf

    # ── Gaussian BBN prior on ω_b ─────────────────────────────────────────
    ln_prior = -0.5 * ((ombh2 - BBN_OMBH2_MEAN) / BBN_OMBH2_SIG) ** 2

    # ── BAO likelihood ────────────────────────────────────────────────────
    # wCDM is CPL with wa=0; pass w as w0
    bao_th = bao_theory(bao_df, Om, w, 0.0, H0, ombh2, model='wCDM')
    if bao_th is None:
        return -np.inf

    d_bao   = bao_data - bao_th
    chi2_bao = d_bao @ bao_invcov @ d_bao

    # ── SN likelihood (marginalised over M) ──────────────────────────────
    mu_th = distance_modulus(z_sn, Om, w, 0.0, H0, model='wCDM')
    if mu_th is None:
        return -np.inf

    d_sn = mu_sn - mu_th
    Cd   = invcov_sn @ d_sn
    chi2_sn_marg = d_sn @ Cd - (np.sum(Cd)) ** 2 / np.sum(invcov_sn)

    return ln_prior - 0.5 * (chi2_bao + chi2_sn_marg)


# ── Run one chain ─────────────────────────────────────────────────────────────

def run_chain(name, z, mu, invcov, nwalkers=NWALKERS_WCDM, nsteps=NSTEPS_WCDM):
    """
    Run (or resume) an emcee chain and save to an HDF5 backend.

    Parameters: θ = (Om, w, H0, ombh2)
    """
    backend_path = os.path.join(OUT_DIR, f'{name}.h5')

    # Resume if a completed chain already exists
    if os.path.exists(backend_path):
        reader = emcee.backends.HDFBackend(backend_path, read_only=True)
        if reader.iteration >= nsteps:
            print(f'  [{name}] Already complete ({reader.iteration} steps). Skipping.')
            return

    ndim = 4
    # Tight Gaussian ball around the prior centre
    p0_centre = np.array([0.30, -1.00, 70.0, BBN_OMBH2_MEAN])
    p0_sigma  = np.array([0.02,  0.05,  1.0,  BBN_OMBH2_SIG])
    pos = p0_centre + p0_sigma * np.random.randn(nwalkers, ndim)

    backend = emcee.backends.HDFBackend(backend_path)
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob,
        args=(z, mu, invcov),
        backend=backend
    )
    print(f'  [{name}] Running {nsteps} steps with {nwalkers} walkers …')
    sampler.run_mcmc(pos, nsteps, progress=True)

    # ── Convergence report ────────────────────────────────────────────────
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        burnin = int(BURNIN_FRAC * nsteps)
        eff    = (nsteps - burnin) * nwalkers / np.max(tau)
        print(f'  [{name}] tau = {tau.round(1)}  |  eff. samples ~ {eff:.0f}')
        if np.max(tau) * 50 > nsteps:
            print(f'  [{name}] WARNING: chain may not be fully converged '
                  f'(need >= {int(np.max(tau)*50)} steps, ran {nsteps}).')
    except emcee.autocorr.AutocorrError:
        print(f'  [{name}] Autocorrelation estimate unavailable (chain too short).')


# ── Main ──────────────────────────────────────────────────────────────────────

TASKS = {
    # name              dataset      correction
    'Panth_Uncorr': ('Pantheon',  'Uncorr'),
    'Panth_Lin':    ('Pantheon',  'Lin'),
    'Panth_Poly':   ('Pantheon',  'Poly'),
    'DES_Uncorr':   ('DES',       'Uncorr'),
    'DES_Lin':      ('DES',       'Lin'),
    'DES_Poly':     ('DES',       'Poly'),
}

if __name__ == '__main__':
    np.random.seed(42)
    for name, (dataset, corr) in TASKS.items():
        print(f'\n=== {name} ===')
        try:
            z, mu, invcov = load_sn(dataset, corr)
        except FileNotFoundError as e:
            print(f'  Skipping (file not found): {e}')
            continue
        run_chain(name, z, mu, invcov)
    print('\nDone. Chains saved to', OUT_DIR)
