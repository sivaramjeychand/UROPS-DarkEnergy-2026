"""
Data loading utilities for SN Ia, BAO, and Rose+19 datasets.
All functions return numpy arrays ready for likelihood computation.
"""
import io
import numpy as np
import pandas as pd
from config import (ROSE19_T1, ROSE19_T7,
                    PANTH_DAT, PANTH_COV, PANTH_LIN, PANTH_POLY,
                    DES_DAT, DES_COV, DES_LIN, DES_POLY,
                    BAO_MEAN, BAO_COV)


# ── Rose+19 ──────────────────────────────────────────────────────────────────

def load_rose19():
    """
    Load Rose et al. 2019 (ApJ 874, 32) Tables 1 & 7, merge on SNID.

    Returns
    -------
    df : DataFrame with columns [SNID, z, HR, e_HR, Age, ...]
         'Age' is the local stellar age in **Gyr** (column logAl of Table 7,
         which despite the prefix contains linear Gyr values in this dataset).
    """
    t1 = pd.read_csv(ROSE19_T1)
    t7 = pd.read_csv(ROSE19_T7)
    t1['SNID'] = t1['SNID'].astype(int)
    t7['SNID'] = t7['SNID'].astype(int)
    df = pd.merge(t1, t7, on='SNID', how='inner')
    df = df.dropna(subset=['logAl', 'HR', 'e_HR']).copy()
    # logAl values range ~1.7–9.7 → treated as linear age in Gyr
    df['Age'] = df['logAl']
    return df


# ── BAO ───────────────────────────────────────────────────────────────────────

def load_bao():
    """
    Load DESI 2024 DR1 BAO measurements.

    Returns
    -------
    bao_df    : DataFrame with columns ['z', 'value', 'type']
                type ∈ {'DM_over_rs', 'DH_over_rs', 'DV_over_rs'}
    bao_data  : 1-D array of observed values
    bao_invcov: square precision matrix
    """
    bao_df = pd.read_csv(BAO_MEAN, sep=r'\s+', comment='#',
                         names=['z', 'value', 'type'])
    bao_data = bao_df['value'].values
    cov = np.loadtxt(BAO_COV)
    bao_invcov = np.linalg.inv(cov)
    return bao_df, bao_data, bao_invcov


# ── Supernova helpers ─────────────────────────────────────────────────────────

def _load_pantheon_dat(path, panth_cov_path=None):
    """Load a Pantheon+-format whitespace-delimited file."""
    df = pd.read_csv(path, sep=r'\s+')
    if 'zHD' not in df.columns:
        df = pd.read_csv(path, sep=r'\s+', skiprows=1)

    # Low-z cut: Pantheon+ includes calibrators that should not bias the
    # cosmological fit (their distances are anchored, not inferred).
    mask = df['zHD'] > 0.01
    df = df[mask].copy()
    z   = df['zHD'].values
    mu  = df['MU_SH0ES'].values
    idx = df.index.values          # original row indices for covariance slicing

    if panth_cov_path is None:
        return z, mu, None, idx

    # Load and reshape covariance
    with open(panth_cov_path) as f:
        header = f.readline()
    skip = 1 if len(header.split()) == 1 else 0
    raw = np.loadtxt(panth_cov_path, skiprows=skip)
    N   = int(round(np.sqrt(raw.size)))
    cov = raw.reshape(N, N)
    # Slice to the z>0.01 subset
    if cov.shape[0] > len(z):
        cov = cov[np.ix_(idx, idx)]

    invcov = np.linalg.pinv(cov)
    return z, mu, invcov, idx


def _load_des_csv(path, des_cov_path=None):
    """Load a DES-SN5YR SNANA-style file (VARNAMES: header or CSV)."""
    with open(path) as f:
        lines = f.readlines()

    names, start = None, 0
    for i, line in enumerate(lines):
        if line.strip().startswith('VARNAMES:'):
            names = line.replace('VARNAMES:', '').split()
            start = i + 1
            break

    if names:
        df = pd.read_csv(io.StringIO(''.join(lines[start:])),
                         sep=r'\s+', names=names, comment='#')
    else:
        df = pd.read_csv(path, sep=r'\s+', comment='#')

    df['zHD']  = pd.to_numeric(df['zHD'],  errors='coerce')
    col_mu     = 'MU' if 'MU' in df.columns else 'MU_SH0ES'
    df[col_mu] = pd.to_numeric(df[col_mu], errors='coerce')
    df         = df.dropna(subset=['zHD', col_mu])
    z  = df['zHD'].values
    mu = df[col_mu].values

    if des_cov_path is None:
        return z, mu, None

    d    = np.load(des_cov_path)
    n    = int(d['nsn'][0])
    flat = d['cov']
    full = np.zeros((n, n))
    full[np.triu_indices(n)] = flat
    prec = full + full.T - np.diag(np.diag(full))   # precision matrix stored
    cov  = np.linalg.inv(prec)

    if len(z) != n:
        z = z[:n]; mu = mu[:n]

    invcov = np.linalg.pinv(cov)
    return z, mu, invcov


# ── Public interface ──────────────────────────────────────────────────────────

def load_sn(dataset, correction='Uncorr'):
    """
    Load a SN Ia dataset with its covariance matrix.

    Parameters
    ----------
    dataset    : 'Pantheon' or 'DES'
    correction : 'Uncorr' | 'Lin' | 'Poly'
                 'Uncorr' returns the raw distances.
                 'Lin'/'Poly' return statically pre-corrected distances.
                 For *dynamic* correction (applied inside MCMC), use 'Uncorr'
                 and pass correction_type to the log-prob function.

    Returns
    -------
    z, mu, invcov : ndarrays
    """
    if dataset == 'Pantheon':
        path_map = {'Uncorr': PANTH_DAT, 'Lin': PANTH_LIN, 'Poly': PANTH_POLY}
        path = path_map[correction]
        if not __import__('os').path.exists(path):
            raise FileNotFoundError(path)
        z, mu, invcov, _ = _load_pantheon_dat(path, PANTH_COV)
    elif dataset == 'DES':
        path_map = {'Uncorr': DES_DAT, 'Lin': DES_LIN, 'Poly': DES_POLY}
        path = path_map[correction]
        if not __import__('os').path.exists(path):
            raise FileNotFoundError(path)
        z, mu, invcov = _load_des_csv(path, DES_COV)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if invcov is None:
        raise RuntimeError("Covariance matrix could not be loaded.")
    return z, mu, invcov
