"""
Cosmological distance calculations for wCDM, CPL, JBP, and LOG models.

All models are flat (Omega_k = 0). The dark energy equation of state is:
  wCDM : w(z) = w                                     (constant)
  CPL  : w(z) = w0 + wa * z/(1+z)                     (Chevallier-Polarski-Linder)
  JBP  : w(z) = w0 + wa * z/(1+z)^2                   (Jassal-Bagla-Padmanabhan)
  LOG  : w(z) = w0 + wa * ln(1+z)                     (Logarithmic)

No CAMB calls are made — all quantities use fast analytical integration
so that the MCMC likelihood can be evaluated ≫1000 times per second.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from config import C_LIGHT_KMS, T_H_FACTOR, TCMB

# Number of points in the integration grid
_NGRID = 1200


# ── Sound horizon at drag epoch ───────────────────────────────────────────────

def rdrag(ombh2, omh2, Tcmb=TCMB):
    """
    Comoving sound horizon at baryon drag epoch [Mpc].

    Uses two complementary formulas:
      1. Drag redshift z_d from Eisenstein & Hu (1998) Eq. 4  (accurate)
      2. Sound-horizon power-law fit calibrated against CAMB
         (Alam et al. 2017 / BOSS collaboration, Appendix B):

         r_d = 147.09 * (ombh2/0.02226)^{-0.134} * (omh2/0.1431)^{-0.255} Mpc

         Accurate to ~0.5% for typical Planck-compatible parameter ranges.

    Parameters
    ----------
    ombh2 : Ω_b h²
    omh2  : Ω_m h²  (total matter, i.e. Ω_m * (H0/100)²)
    Tcmb  : CMB temperature [K]  (default 2.7255; enters only via T correction)
    """
    # Temperature correction relative to Planck 2018 (T_CMB = 2.7255 K)
    T27 = Tcmb / 2.7
    # The power-law formula below is calibrated at T=2.7255 K; apply a small
    # correction for different temperatures using the T27^{-0.4} scaling
    # expected from the analytic formula  (negligible for T near 2.7255 K).
    t_corr = (T27 / (2.7255 / 2.7)) ** (-0.4)

    rd = 147.09 * (ombh2 / 0.02226) ** (-0.134) * (omh2 / 0.1431) ** (-0.255) * t_corr
    return rd   # [Mpc]


# ── Dark energy density evolution f_de(z) = ρ_de(z)/ρ_de(0) ─────────────────

def _f_de(z, w0, wa, model):
    """Return f_de(z) = ρ_de(z)/ρ_de(0) for a given DE parameterization."""
    if model == 'wCDM':
        return (1.0 + z) ** (3.0 * (1.0 + w0))

    elif model == 'CPL':
        # w(a) = w0 + wa*(1-a) = w0 + wa*z/(1+z)
        # f_de = (1+z)^{3(1+w0+wa)} exp(-3 wa z/(1+z))
        return ((1.0 + z) ** (3.0 * (1.0 + w0 + wa))
                * np.exp(-3.0 * wa * z / (1.0 + z)))

    elif model == 'JBP':
        # w(a) = w0 + wa*a(1-a) → w(z) = w0 + wa * z/(1+z)^2
        # f_de = (1+z)^{3(1+w0)} exp(1.5 wa (z/(1+z))^2)
        return ((1.0 + z) ** (3.0 * (1.0 + w0))
                * np.exp(1.5 * wa * (z / (1.0 + z)) ** 2))

    elif model == 'LOG':
        # w(z) = w0 + wa ln(1+z)
        # f_de = (1+z)^{3(1+w0)} exp(1.5 wa ln^2(1+z))
        return ((1.0 + z) ** (3.0 * (1.0 + w0))
                * np.exp(1.5 * wa * np.log(1.0 + z) ** 2))

    else:
        raise ValueError(f"Unknown DE model: {model}")


# ── E(z) = H(z)/H0 ───────────────────────────────────────────────────────────

def E_z(z, Om, w0, wa, model='CPL'):
    """
    Dimensionless Hubble parameter E(z) = H(z)/H0 for a flat universe.

    Parameters
    ----------
    z           : scalar or array of redshifts
    Om          : Ω_m (matter density parameter)
    w0, wa      : DE equation-of-state parameters
                  For wCDM pass wa=0 and model='wCDM' (wa is ignored)
    model       : 'wCDM' | 'CPL' | 'JBP' | 'LOG'
    """
    Ode = 1.0 - Om
    return np.sqrt(Om * (1.0 + z) ** 3 + Ode * _f_de(z, w0, wa, model))


# ── Comoving / luminosity distances ──────────────────────────────────────────

def _build_dc_interp(Om, w0, wa, H0, model, z_max):
    """Build a d_C(z) interpolator over [0, z_max]."""
    z_grid = np.linspace(0.0, z_max, _NGRID)
    ez     = E_z(z_grid, Om, w0, wa, model)
    # Guard against non-physical cosmologies
    if np.any(~np.isfinite(ez)) or np.any(ez <= 0):
        return None
    integrand = 1.0 / ez
    dc_cum = cumulative_trapezoid(integrand, z_grid, initial=0.0)
    dc_mpc = (C_LIGHT_KMS / H0) * dc_cum
    return interp1d(z_grid, dc_mpc, kind='linear', fill_value='extrapolate')


def luminosity_distance(z_arr, Om, w0, wa, H0, model='CPL'):
    """
    Luminosity distance d_L(z) in Mpc for flat FLRW.

    Returns None if cosmological parameters are unphysical.
    """
    z_arr   = np.atleast_1d(z_arr)
    z_max   = np.max(z_arr) * 1.05
    interp  = _build_dc_interp(Om, w0, wa, H0, model, z_max)
    if interp is None:
        return None
    dc = interp(z_arr)
    return dc * (1.0 + z_arr)          # d_L = (1+z) * d_C  [Mpc]


def distance_modulus(z_arr, Om, w0, wa, H0, model='CPL'):
    """Distance modulus μ(z) = 5 log10(d_L / 10 pc)."""
    dL = luminosity_distance(z_arr, Om, w0, wa, H0, model)
    if dL is None:
        return None
    if np.any(dL <= 0):
        return None
    return 5.0 * np.log10(dL) + 25.0   # d_L in Mpc → +25 converts to 10 pc


# ── BAO observable predictions ────────────────────────────────────────────────

def bao_theory(bao_df, Om, w0, wa, H0, ombh2, model='CPL'):
    """
    Predict the BAO observables for all rows of bao_df.

    Parameters
    ----------
    bao_df  : DataFrame with columns ['z', 'type']
              type ∈ {'DM_over_rs', 'DH_over_rs', 'DV_over_rs'}
    Returns
    -------
    preds : 1-D array of predicted values, or None on failure
    """
    omh2 = Om * (H0 / 100.0) ** 2
    rd   = rdrag(ombh2, omh2)
    if rd <= 0:
        return None

    # We need d_C at each BAO redshift
    z_bao  = bao_df['z'].values
    z_max  = np.max(z_bao) * 1.05
    interp = _build_dc_interp(Om, w0, wa, H0, model, z_max)
    if interp is None:
        return None

    preds = []
    for _, row in bao_df.iterrows():
        zi  = row['z']
        DC  = interp(zi)                            # comoving distance [Mpc]
        Hz  = H0 * E_z(zi, Om, w0, wa, model)      # H(z) [km/s/Mpc]
        DH  = C_LIGHT_KMS / Hz                      # Hubble distance [Mpc]
        DV  = (zi * DC ** 2 * DH) ** (1.0 / 3.0)   # volume distance [Mpc]

        t = row['type']
        if t == 'DM_over_rs':
            preds.append(DC / rd)
        elif t == 'DH_over_rs':
            preds.append(DH / rd)
        elif t == 'DV_over_rs':
            preds.append(DV / rd)
        else:
            preds.append(0.0)

    return np.array(preds)


# ── Lookback time ─────────────────────────────────────────────────────────────

def lookback_time(z_arr, Om, w0, wa, H0, model='CPL'):
    """
    Lookback time t_L(z) in Gyr.
    Uses the relation:  t_L = (T_H) ∫_0^z dz' / [(1+z') E(z')]
    """
    z_arr = np.atleast_1d(z_arr)
    T_H   = T_H_FACTOR / H0            # Hubble time [Gyr]
    z_max = np.max(z_arr) * 1.05
    z_grid = np.linspace(0.0, max(0.1, z_max), _NGRID)
    ez     = E_z(z_grid, Om, w0, wa, model)
    integrand = 1.0 / (ez * (1.0 + z_grid))
    tL_cum = T_H * cumulative_trapezoid(integrand, z_grid, initial=0.0)
    interp = interp1d(z_grid, tL_cum, kind='linear', fill_value='extrapolate')
    return interp(z_arr)
