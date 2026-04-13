"""
Step 6 — Run w0wa MCMC for CPL, JBP, and LOG dark energy models.

Free parameters (5-D):  Om, w0, wa, H0, ombh2

Correction approach (model-dependent):
  CPL   — *static pre-correction*: for 'Lin'/'Poly', age-bias has already
           been applied to μ_SN in the pre-corrected data files
           (DES-SN5YR_Corrected/, PantheonPlus_Corrected/), matching the
           methodology of Son et al. (2025).
  JBP / LOG — *dynamic correction*: no pre-corrected files exist for these
           parametrizations, so the age-bias correction is computed inside
           log_prob at every MCMC step using the trial cosmology.
           Correction = slope × δ(age)(z) [linear] or poly form [poly].

Datasets × corrections × models:
  {Pantheon, DES} × {Uncorr, Lin, Poly} × {CPL, JBP, LOG}  =  18 chains

Key convergence improvements:
  - rdrag via Alam et al. (2017) power-law — no CAMB per likelihood call
  - 100 walkers, 5000 steps (≫ previous 32 walkers, 200–250 steps)
  - Priors match DESI DR2 (Abdul-Karim et al. 2025): w0∈[-3,1], wa∈[-3,2],
    w0+wa < 0 (early matter domination)
  - Multiprocessing pool for parallel chain execution
"""
import os
import sys
import numpy as np
import emcee
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (OUT_CHAINS, BBN_OMBH2_MEAN, BBN_OMBH2_SIG,
                    NWALKERS_W0WA, NSTEPS_W0WA, BURNIN_FRAC)
from data_io import load_bao, load_sn, load_rose19
from cosmology import bao_theory, distance_modulus
from age_correction import (fit_linear, fit_poly,
                             dynamic_linear_bias, dynamic_poly_bias)

OUT_DIR = os.path.join(OUT_CHAINS, 'w0wa')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load BAO once ─────────────────────────────────────────────────────────────
bao_df, bao_data, bao_invcov = load_bao()

# ── Fit Rose+19 age–HR relation once (used for JBP/LOG dynamic correction) ───
_rose_df   = load_rose19()
_lin_slope, _lin_intercept = fit_linear(_rose_df)
_poly_coeffs               = fit_poly(_rose_df, deg=2)
_mean_age                  = _rose_df['Age'].mean()  # reference age [Gyr]


# ── Log-probability ───────────────────────────────────────────────────────────

def log_prob_w0wa(theta, z_sn, mu_sn, invcov_sn, model, corr_type):
    """
    Log-posterior for a w0wa model.

    θ = (Om, w0, wa, H0, ombh2)

    corr_type : 'Uncorr' | 'Lin' | 'Poly'
                For CPL + 'Lin'/'Poly': pre-corrected μ_SN is passed in,
                no further correction applied (Son+2025 static methodology).
                For JBP/LOG + 'Lin'/'Poly': uncorrected μ_SN is passed in
                and correction is applied dynamically here from Rose+19 fit.

    Priors match those in DESI DR2 (Abdul-Karim et al. 2025, Table 2 of [38]):
        w0  ∈ U[-3, 1],  wa ∈ U[-3, 2],  w0+wa < 0  (early matter domination)
    """
    Om, w0, wa, H0, ombh2 = theta

    # ── Hard priors (DESI DR2 standard) ──────────────────────────────────────
    if not (0.10  < Om    < 0.70 ): return -np.inf
    if not (-3.0  < w0    < 1.0  ): return -np.inf   # DESI: U[-3, 1]
    if not (-3.0  < wa    < 2.0  ): return -np.inf   # DESI: U[-3, 2]
    if w0 + wa >= 0:                return -np.inf   # enforce early matter domination
    if not (50.0  < H0    < 100. ): return -np.inf
    if not (0.017 < ombh2 < 0.028): return -np.inf

    # ── Gaussian BBN prior on ω_b ─────────────────────────────────────────
    ln_prior = -0.5 * ((ombh2 - BBN_OMBH2_MEAN) / BBN_OMBH2_SIG) ** 2

    # ── BAO likelihood ────────────────────────────────────────────────────
    bao_th = bao_theory(bao_df, Om, w0, wa, H0, ombh2, model=model)
    if bao_th is None:
        return -np.inf

    d_bao    = bao_data - bao_th
    chi2_bao = d_bao @ bao_invcov @ d_bao

    # ── SN theory ─────────────────────────────────────────────────────────
    mu_th = distance_modulus(z_sn, Om, w0, wa, H0, model=model)
    if mu_th is None:
        return -np.inf

    # ── Age-bias correction (JBP / LOG only) ─────────────────────────────
    # CPL uses statically pre-corrected data files — no further correction here.
    # JBP / LOG have no pre-corrected files, so the correction is applied
    # dynamically at each MCMC step using the current trial cosmology.
    mu_sn_eff = mu_sn
    if model in ('JBP', 'LOG') and corr_type != 'Uncorr':
        if corr_type == 'Lin':
            bias = dynamic_linear_bias(z_sn, Om, w0, wa, H0, _lin_slope, model)
        else:  # 'Poly'
            bias = dynamic_poly_bias(z_sn, Om, w0, wa, H0, _poly_coeffs,
                                     _mean_age, model)
        mu_sn_eff = mu_sn - bias   # subtract bias to obtain corrected μ

    # ── SN likelihood (marginalised over M) ──────────────────────────────
    d_sn = mu_sn_eff - mu_th
    Cd   = invcov_sn @ d_sn
    chi2_sn_marg = d_sn @ Cd - np.sum(Cd) ** 2 / np.sum(invcov_sn)

    return ln_prior - 0.5 * (chi2_bao + chi2_sn_marg)


# ── Run one chain ─────────────────────────────────────────────────────────────

def run_chain(name, z, mu, invcov, model, corr_type,
              nwalkers=NWALKERS_W0WA, nsteps=NSTEPS_W0WA):
    backend_path = os.path.join(OUT_DIR, f'{name}.h5')

    if os.path.exists(backend_path):
        reader = emcee.backends.HDFBackend(backend_path, read_only=True)
        if reader.iteration >= nsteps:
            print(f'  [{name}] Already complete ({reader.iteration} steps). Skipping.')
            return

    ndim = 5
    p0_centre = np.array([0.30, -1.00, 0.00, 70.0, BBN_OMBH2_MEAN])
    p0_sigma  = np.array([0.02,  0.10, 0.10,  1.0,  BBN_OMBH2_SIG])
    pos = p0_centre + p0_sigma * np.random.randn(nwalkers, ndim)

    backend = emcee.backends.HDFBackend(backend_path)
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob_w0wa,
        args=(z, mu, invcov, model, corr_type),
        backend=backend
    )
    print(f'  [{name}] Running {nsteps} steps …')
    sampler.run_mcmc(pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        burnin = int(BURNIN_FRAC * nsteps)
        eff    = (nsteps - burnin) * nwalkers / np.max(tau)
        print(f'  [{name}] tau_max = {np.max(tau):.1f}  |  eff. samples ~ {eff:.0f}')
        if np.max(tau) * 50 > nsteps:
            print(f'  [{name}] WARNING: chain may need more steps '
                  f'(tau_max*50 = {int(np.max(tau)*50)} > {nsteps}).')
    except emcee.autocorr.AutocorrError:
        print(f'  [{name}] Autocorrelation estimate unavailable.')


# ── Worker for multiprocessing ────────────────────────────────────────────────

def worker(args):
    chain_name, dataset, corr, model = args
    try:
        if model == 'CPL':
            # Static pre-correction: load corrected μ_SN directly from files.
            # Matches Son+2025 methodology (Δm applied once before MCMC).
            load_corr = corr
        else:
            # JBP / LOG: no pre-corrected files exist for these models.
            # Always load uncorrected data; dynamic correction is applied
            # inside log_prob_w0wa at each MCMC step.
            load_corr = 'Uncorr'
        z, mu, invcov = load_sn(dataset, load_corr)
    except FileNotFoundError as e:
        print(f'  Skipping {chain_name}: {e}')
        return
    run_chain(chain_name, z, mu, invcov, model, corr)


# ── Task definitions ──────────────────────────────────────────────────────────

MODELS   = ['CPL', 'JBP', 'LOG']
DATASETS = {'Panth': 'Pantheon', 'DES': 'DES'}
CORRS    = ['Uncorr', 'Lin', 'Poly']

if __name__ == '__main__':
    mp.freeze_support()
    np.random.seed(42)

    tasks = []
    for model in MODELS:
        for ds_short, ds_full in DATASETS.items():
            for corr in CORRS:
                cname = f'{model}_{ds_short}_{corr}'
                tasks.append((cname, ds_full, corr, model))

    # Run sequentially to avoid multiprocessing issues on Windows
    # (emcee itself parallelises internally per step via moves)
    for task in tasks:
        print(f'\n=== {task[0]} ===')
        worker(task)

    print('\nDone. Chains saved to', OUT_DIR)
