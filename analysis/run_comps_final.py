import os
import numpy as np
import pandas as pd
import emcee
import camb
from scipy.interpolate import interp1d
from scipy import integrate
import multiprocessing
import sys
from astropy.cosmology import FlatLambdaCDM

# --- CONFIGURATION ---
OUT_DIR = "chains_comps"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Baseline Data for Progenitor Age
Z_POINTS = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
AGE_SOLID_PROGENITOR = np.array([0, 1.0, 1.8, 2.6, 3.3, 3.9, 4.4, 4.8, 5.1, 5.3, 5.5, 5.6, 5.7, 5.8, 5.85, 5.9, 5.95, 6.0])
F_AGE_PROGENITOR_PAPER = interp1d(Z_POINTS, AGE_SOLID_PROGENITOR, kind='cubic', fill_value="extrapolate")

OM_BASE, W0_BASE, WA_BASE, H0_BASE = 0.353, -0.42, -1.75, 70.0 

# Paths
PANTHEON_DIR = r'../data/external/PantheonPlus/Pantheon+_Data/4_DISTANCES_AND_COVAR'
PANTHEON_UNCORR = os.path.join(PANTHEON_DIR, 'Pantheon+SH0ES.dat')
PANTHEON_COV = os.path.join(PANTHEON_DIR, 'Pantheon+SH0ES_STAT+SYS.cov')
DES_DIR = r'../data/external/DES-SN5YR/4_DISTANCES_COVMAT'
DES_UNCORR = os.path.join(DES_DIR, 'DES-Dovekie_HD.csv')
DES_COV = os.path.join(DES_DIR, 'STAT+SYS.npz')
BAO_MEAN_FILE = r'../data/external/DESI_BAO/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt'
BAO_COV_FILE = r'../data/external/DESI_BAO/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt'

# --- SHARED GLOBAL DICTIONARY (For Workers) ---
data_cache = {}

def load_bao():
    if not os.path.exists(BAO_MEAN_FILE): return None, None, None
    # Fix: sep='\s+' replaces delim_whitespace
    df = pd.read_csv(BAO_MEAN_FILE, sep='\s+', comment='#', names=['z', 'value', 'type'])
    data = df['value'].values
    cov = np.loadtxt(BAO_COV_FILE)
    inv_cov = np.linalg.inv(cov)
    return df, data, inv_cov

def load_sn(data_path, cov_path):
    if not os.path.exists(data_path): return None, None, None
    if 'Pantheon' in data_path:
        df = pd.read_csv(data_path, sep=r'\s+')
        if 'zHD' not in df.columns: df = pd.read_csv(data_path, sep=r'\s+', skiprows=1)
        df = df[df['zHD'] > 0.01]
        z = df['zHD'].values; mu = df['MU_SH0ES'].values; indices = df.index.values
        
        cov_flat = np.loadtxt(cov_path)
        if len(cov_flat) == (1701**2 + 1):
            N = int(cov_flat[0]); cov = cov_flat[1:].reshape((N, N))
        else:
            N = int(np.sqrt(len(cov_flat))); cov = cov_flat.reshape((N, N))
            
        if cov.shape[0] > len(z): cov = cov[np.ix_(indices, indices)]
        return z, mu, np.linalg.inv(cov) # Standard inverse for Pantheon+
    
    else:
        # DES-SN5YR specific loading
        df = pd.read_csv(data_path, sep=r'\s+', comment='#')
        z = df['zHD'].values
        col_mu = 'MU' if 'MU' in df.columns else 'MU_SH0ES'
        mu = df[col_mu].values
        
        d = np.load(cov_path)
        n = d['nsn'][0]
        full = np.zeros((n, n))
        full[np.triu_indices(n)] = d['cov']
        cov = full + full.T - np.diag(np.diag(full))
        
        if len(z) != n: z, mu = z[:n], mu[:n]
        
        # DES often requires a more stable pseudo-inverse due to small eigenvalues
        return z, mu, np.linalg.pinv(cov, rcond=1e-15)

def get_ez2(pars, z, model):
    Om, w0, wa, H0, ombh2 = pars
    a = 1.0 / (1+z)
    ez2 = Om * (1+z)**3
    if model == 'CPL': term = a**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a))
    elif model == 'JBP': term = a**(-3*(1+w0)) * np.exp(1.5 * wa * (a-1)**2)
    elif model == 'LOG': term = a**(-3*(1+w0)) * np.exp(1.5 * wa * np.log(a)**2)
    else: term = a**(-3*(1+w0))
    return ez2 + (1-Om) * term

def get_tL_grid(pars, model, z_max=2.5): # Increased from 2.0 to 2.5
    """Calculates look-back time grid in Gyr for scaling."""
    z_grid = np.linspace(0, z_max, 300)
    inv_E_z = 1.0 / ((1.0 + z_grid) * np.sqrt(get_ez2(pars, z_grid, model)))
    tL_cum = integrate.cumulative_trapezoid(inv_E_z, z_grid, initial=0)
    tL_vals = (977.792 / pars[3]) * tL_cum 
    
    # Added fill_value="extrapolate" to prevent crashes on boundary cases
    return interp1d(z_grid, tL_vals, kind='cubic', fill_value="extrapolate")

# Initialize baseline for workers inside the main or worker function
def get_baseline_f_tl():
    baseline_pars = [OM_BASE, W0_BASE, WA_BASE, H0_BASE, 0.0224]
    return get_tL_grid(baseline_pars, 'CPL')

def log_prob(theta, z_sn, mu_sn, inv_sn_cov, model, corr_type, bao_df, bao_data, bao_inv_cov, f_tl_base):
    Om, w0, wa, H0, ombh2 = theta
    if not (0.1 < Om < 0.6) or not (-3.0 < w0 < 1.0) or not (-5.0 < wa < 5.0) or not (55 < H0 < 85):
        return -np.inf
    
    # SN Theory & BAO Theory
    h = H0 / 100.0; omch2 = (Om * h**2) - ombh2
    if omch2 < 0: return -np.inf

    # CAMB rdrag
    pars_c = camb.CAMBparams()
    pars_c.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0)
    pars_c.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')
    try:
        results = camb.get_background(pars_c)
        rdrag = results.get_derived_params()['rdrag']
    except: return -np.inf

    z_max = max(np.max(z_sn), np.max(bao_df['z']))
    z_grid = np.linspace(0, z_max * 1.1, 300)
    inv_E_grid = 1.0 / np.sqrt(get_ez2(theta, z_grid, model))
    DM_cum = integrate.cumulative_trapezoid(inv_E_grid, z_grid, initial=0)
    DM_interp = interp1d(z_grid, (299792.458 / H0) * DM_cum, kind='cubic')
    
    mu_theory = 5 * np.log10(DM_interp(z_sn) * (1 + z_sn)) + 25

    # Bias Correction
    if corr_type in ['Lin', 'Poly']:
        f_tl_curr = get_tL_grid(theta, model)
        tL_curr, tL_base = f_tl_curr(z_sn), f_tl_base(z_sn)
        delta_age = -1.0 * F_AGE_PROGENITOR_PAPER(z_sn) * (tL_curr / np.where(tL_base==0, 1e-9, tL_base))
        
        if corr_type == 'Lin':
            mu_theory += 0.030 * (-1.0 * delta_age)
        elif corr_type == 'Poly' and data_cache.get('poly_func'):
            age_z = 13.8 + delta_age
            log_age_z = np.log10(np.clip(age_z * 1e9, 1e6, None))
            mu_theory += data_cache['poly_func'](log_age_z) - data_cache['poly_func'](data_cache['log_age_today'])

    # Chi2 calculations
    bao_preds = []
    for _, row in bao_df.iterrows():
        zi = row['z']; dm = DM_interp(zi); hz = H0 * np.sqrt(get_ez2(theta, zi, model))
        if row['type'] == 'DM_over_rs': pred = dm/rdrag
        elif row['type'] == 'DH_over_rs': pred = (299792.458/hz)/rdrag
        elif row['type'] == 'DV_over_rs': pred = (zi * dm**2 * (299792.458/hz))**(1/3)/rdrag
        else: pred = 0
        bao_preds.append(pred)

    chi2_bao = np.dot(bao_data - bao_preds, np.dot(bao_inv_cov, bao_data - bao_preds))
    delta_sn = mu_sn - mu_theory
    chi2_sn = np.dot(delta_sn, np.dot(inv_sn_cov, delta_sn)) - (np.sum(np.dot(inv_sn_cov, delta_sn))**2 / np.sum(inv_sn_cov))
    
    return -0.5 * (chi2_bao + chi2_sn)

def worker(args):
    model, dname, dpath, cpath, corr_type, steps = args
    print(f"Starting {model} on {dname}...")
    
    # Reload local process cache
    b_df, b_data, b_inv = load_bao()
    f_tl_base = get_baseline_f_tl()
    z_sn, mu_sn, inv_sn = load_sn(dpath, cpath)
    
    if z_sn is None: return

    backend_path = os.path.join(OUT_DIR, f"{model}_{dname}.h5")
    backend = emcee.backends.HDFBackend(backend_path)
    backend.reset(32, 5)
    
    pos = [0.32, -0.6, -1.0, 70.0, 0.0224] + 1e-3 * np.random.randn(32, 5)
    sampler = emcee.EnsembleSampler(32, 5, log_prob, 
                                    args=(z_sn, mu_sn, inv_sn, model, corr_type, b_df, b_data, b_inv, f_tl_base), 
                                    backend=backend)
    sampler.run_mcmc(pos, steps, progress=False)
    print(f"Finished {model}_{dname}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # Setup Polynomial if file exists
    if os.path.exists('Rose19_corrected.csv'):
        df_rose = pd.read_csv('Rose19_corrected.csv')
        mask = ~np.isnan(df_rose['HR']) & ~np.isnan(df_rose['logAl'])
        coeffs = np.polyfit(df_rose.loc[mask, 'logAl'], df_rose.loc[mask, 'HR'], 2, w=1.0/(df_rose.loc[mask, 'e_HR']**2))
        data_cache['poly_func'] = np.poly1d(coeffs)
        data_cache['log_age_today'] = np.log10(FlatLambdaCDM(H0=70, Om0=0.3).age(0).value * 0.71 * 1e9)

    tasks = []
    datasets = {
        'Panth_Uncorr': (PANTHEON_UNCORR, PANTHEON_COV, 'Uncorr'),
        'Panth_Lin':    (PANTHEON_UNCORR, PANTHEON_COV, 'Lin'),
        'Panth_Poly':   (PANTHEON_UNCORR, PANTHEON_COV, 'Poly'),
        'DES_Uncorr':   (DES_UNCORR, DES_COV, 'Uncorr'),
        'DES_Lin':      (DES_UNCORR, DES_COV, 'Lin'),
        'DES_Poly':     (DES_UNCORR, DES_COV, 'Poly')
    }
    # Lower step count for DES to check convergence first
    for m in ['CPL', 'JBP', 'LOG']:
        for dname, (dp, cp, ct) in datasets.items():
            tasks.append((m, dname, dp, cp, ct, 400))
    
    # Using 4 cores to prevent Windows threading overhead
    with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 4)) as pool:
        pool.map(worker, tasks) 