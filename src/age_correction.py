"""
Age-bias correction routines.

The Rose+19 dataset shows a positive correlation between local stellar age
(in Gyr) and the SN Ia Hubble residual (HR).  We fit this relation and use
it to derive a bias correction for the SN distance moduli.

Two correction strategies:
  1. Static  – fit the HR–Age trend once on Rose+19, then apply a fixed
               correction term to every SN distance modulus.
  2. Dynamic – inside the MCMC, estimate the typical progenitor age at each
               SN redshift using the lookback time for the *current* trial
               cosmology.  This makes the correction self-consistent with the
               cosmological model under test.

Units: Age / delta_age are always in **Gyr** throughout this module.
"""
import numpy as np
from scipy.stats import linregress
from cosmology import lookback_time
from config import ALPHA_SFH


# ── Fit HR–Age relation on Rose+19 ───────────────────────────────────────────

def fit_linear(df):
    """
    Weighted linear regression of HR on Age [Gyr].

    Returns
    -------
    slope, intercept : floats (mag / Gyr, mag)
    """
    mask    = np.isfinite(df['Age']) & np.isfinite(df['HR']) & np.isfinite(df['e_HR'])
    ages    = df.loc[mask, 'Age'].values
    hrs     = df.loc[mask, 'HR'].values
    weights = 1.0 / df.loc[mask, 'e_HR'].values ** 2

    # Numpy-weighted polyfit (degree 1)
    p = np.polyfit(ages, hrs, 1, w=weights)
    return p[0], p[1]          # slope, intercept


def fit_poly(df, deg=2):
    """
    Weighted polynomial regression of HR on Age [Gyr].

    Returns
    -------
    coeffs : 1-D array of polynomial coefficients (highest power first)
    """
    mask    = np.isfinite(df['Age']) & np.isfinite(df['HR']) & np.isfinite(df['e_HR'])
    ages    = df.loc[mask, 'Age'].values
    hrs     = df.loc[mask, 'HR'].values
    weights = 1.0 / df.loc[mask, 'e_HR'].values ** 2
    return np.polyfit(ages, hrs, deg, w=weights)


# ── Static correction (applied once, before MCMC) ────────────────────────────

def static_linear_correction(mu_arr, age_arr, slope, mean_age):
    """
    Apply a static linear age-bias correction to distance moduli.

    mu_corrected = mu_obs − slope * (age − mean_age)

    Parameters
    ----------
    mu_arr   : observed distance moduli
    age_arr  : local stellar ages [Gyr]
    slope    : HR–Age slope [mag / Gyr]
    mean_age : reference age [Gyr]  (typically mean of the calibration sample)
    """
    return mu_arr - slope * (age_arr - mean_age)


def static_poly_correction(mu_arr, age_arr, coeffs, mean_age):
    """
    Apply a static polynomial age-bias correction to distance moduli.

    mu_corrected = mu_obs − [p(age) − p(mean_age)]
    """
    p = np.poly1d(coeffs)
    return mu_arr - (p(age_arr) - p(mean_age))


# ── Dynamic correction (used inside MCMC) ────────────────────────────────────

def delta_age_at_z(z_arr, Om, w0, wa, H0, model='CPL'):
    """
    Estimate the shift in mean progenitor age at each redshift z relative to
    the local universe.

    Model:  δ(age)(z) = −α × t_L(z)

    where α = ALPHA_SFH encodes how the mean stellar population age tracks
    the lookback time (from the star-formation history; see Appendix A of
    the paper).

    Returns delta_age [Gyr] — negative at high z (progenitors are younger).
    """
    tL = lookback_time(z_arr, Om, w0, wa, H0, model)
    return -ALPHA_SFH * tL


def dynamic_linear_bias(z_arr, Om, w0, wa, H0, slope, model='CPL'):
    """
    Systematic bias in μ at each z due to the age–HR trend (linear model).

    bias = slope × δ(age)(z)   [mag]

    This is the amount by which a SN observed at z appears brighter/fainter
    than the purely cosmological prediction owing to progenitor age evolution.
    Positive slope + negative δ(age) → negative bias (SNe appear brighter at
    high z before correction).
    """
    da = delta_age_at_z(z_arr, Om, w0, wa, H0, model)
    return slope * da            # [mag]


def dynamic_poly_bias(z_arr, Om, w0, wa, H0, coeffs, mean_age, model='CPL'):
    """
    Systematic bias in μ at each z due to the age–HR trend (polynomial model).

    bias = p(mean_age + δ(age)) − p(mean_age)   [mag]

    All ages are in **Gyr** — the polynomial was fitted on Gyr data.
    """
    da         = delta_age_at_z(z_arr, Om, w0, wa, H0, model)
    age_at_z   = np.clip(mean_age + da, 0.5, None)   # physical lower bound
    p          = np.poly1d(coeffs)
    return p(age_at_z) - p(mean_age)                 # [mag]
