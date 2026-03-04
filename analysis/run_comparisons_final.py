
import os
import numpy as np
import pandas as pd
import emcee
import camb
from scipy.interpolate import interp1d
from scipy import integrate
import multiprocessing
import sys

# --- CONFIGURATION ---
OUT_DIR = "chains_comparisons"
os.makedirs(OUT_DIR, exist_ok=True)

# Dataset Paths
PANTHEON_DIR = r'../data/external/PantheonPlus/Pantheon+_Data/4_DISTANCES_AND_COVAR'
PANTHEON_UNCORR = os.path.join(PANTHEON_DIR, 'Pantheon+SH0ES.dat')
PANTHEON_COV = os.path.join(PANTHEON_DIR, 'Pantheon+SH0ES_STAT+SYS.cov')

PANTHEON_CORR_DIR = r'../data/external/PantheonPlus_Corrected/Pantheon+_Data/4_DISTANCES_AND_COVAR'
PANTHEON_LIN = os.path.join(PANTHEON_CORR_DIR, 'Pantheon+SH0ES_Linear.dat')
PANTHEON_POLY = os.path.join(PANTHEON_CORR_DIR, 'Pantheon+SH0ES_Poly.dat')

DES_DIR = r'../data/external/DES-SN5YR/4_DISTANCES_COVMAT'
DES_UNCORR = os.path.join(DES_DIR, 'DES-Dovekie_HD.csv')
DES_COV = os.path.join(DES_DIR, 'STAT+SYS.npz')

DES_CORR_DIR = r'../data/external/DES-SN5YR_Corrected/4_DISTANCES_COVMAT'
DES_LIN = os.path.join(DES_CORR_DIR, 'DES-Dovekie_HD_Linear.csv')
DES_POLY = os.path.join(DES_CORR_DIR, 'DES-Dovekie_HD_Poly.csv')

BAO_MEAN_FILE = r'../data/external/DESI_BAO/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt'
BAO_COV_FILE = r'../data/external/DESI_BAO/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt'

# --- LOADING FUNCTIONS ---

def load_bao():
    if not os.path.exists(BAO_MEAN_FILE): return None, None, None
    df = pd.read_csv(BAO_MEAN_FILE, delim_whitespace=True, comment='#', names=['z', 'value', 'type'])
    data = df['value'].values
    cov = np.loadtxt(BAO_COV_FILE)
    inv_cov = np.linalg.inv(cov)
    return df, data, inv_cov

def load_sn(data_path, cov_path):
    if not os.path.exists(data_path): return None, None, None
    z, mu, cov = None, None, None
    if 'Pantheon' in data_path:
        df = pd.read_csv(data_path, delim_whitespace=True)
        if 'zHD' not in df.columns: df = pd.read_csv(data_path, delim_whitespace=True, skiprows=1)
        df = df[df['zHD'] > 0.01]
        z = df['zHD'].values; mu = df['MU_SH0ES'].values; indices = df.index.values
        with open(cov_path, 'r') as f: header = f.readline()
        if len(header.split()) == 1: cov = np.loadtxt(cov_path, skiprows=1)
        else: cov = np.loadtxt(cov_path)
        N = int(np.sqrt(len(cov.flatten()))); cov = cov.reshape((N, N))
        if cov.shape[0] > len(z): cov = cov[np.ix_(indices, indices)]
    else:
        # DES
        with open(data_path, 'r') as f: lines = f.readlines()
        start_row, names = 0, None
        for i, line in enumerate(lines):
            if line.strip().startswith('VARNAMES:'): names = line.replace('VARNAMES:', '').split(); start_row = i + 1; break
        if names:
            import io
            df = pd.read_csv(io.StringIO("".join(lines[start_row:])), sep=r'\s+', names=names, comment='#')
        else: df = pd.read_csv(data_path, sep=r'\s+', comment='#')
        df['zHD'] = pd.to_numeric(df['zHD'], errors='coerce')
        col_mu = 'MU' if 'MU' in df.columns else 'MU_SH0ES'
        df[col_mu] = pd.to_numeric(df[col_mu], errors='coerce')
        df = df.dropna(subset=['zHD', col_mu])
        z = df['zHD'].values; mu = df[col_mu].values
        d = np.load(cov_path); n = d['nsn'][0]
        full = np.zeros((n, n)); full[np.triu_indices(n)] = d['cov']
        mat = full + full.T - np.diag(np.diag(full))
        cov = np.linalg.inv(mat)
        if len(z) != n: z, mu = z[:n], mu[:n]
    inv_cov = np.linalg.pinv(cov)
    return z, mu, inv_cov

# --- ENGINE ---
bao_df, bao_data, bao_inv_cov = load_bao()

def get_ez2(pars, z, model):
    Om, w0, wa, H0, ombh2 = pars
    a = 1.0 / (1+z)
    ez2 = Om * (1+z)**3
    if model == 'CPL': term = a**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a))
    elif model == 'JBP': term = a**(-3*(1+w0)) * np.exp(1.5 * wa * (a-1)**2)
    elif model == 'LOG': term = a**(-3*(1+w0)) * np.exp(1.5 * wa * np.log(a)**2)
    else: term = a**(-3*(1+w0))
    ez2 += (1-Om) * term
    return ez2

def get_theory_custom(pars, z_sn, model='CPL'):
    Om, w0, wa, H0, ombh2 = pars
    h = H0 / 100.0; omch2 = (Om * h**2) - ombh2
    if omch2 < 0: return None, None
    
    # 1. rdrag from CAMB
    pars_c = camb.CAMBparams()
    pars_c.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0)
    pars_c.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')
    try:
        results = camb.get_background(pars_c)
        rdrag = results.get_derived_params()['rdrag']
    except: return None, None

    # 2. Integration
    z_max = max(np.max(z_sn), np.max(bao_df['z']))
    z_grid = np.linspace(0, z_max * 1.05, 500)
    inv_E_grid = 1.0 / np.sqrt(get_ez2(pars, z_grid, model))
    DM_cum = integrate.cumulative_trapezoid(inv_E_grid, z_grid, initial=0)
    DM_interp = interp1d(z_grid, (299792.458 / H0) * DM_cum, kind='cubic')
    
    mu_theory = 5 * np.log10(DM_interp(z_sn) * (1 + z_sn)) + 25
    bao_preds = []
    for _, row in bao_df.iterrows():
        zi = row['z']; dm = DM_interp(zi); hz = H0 * np.sqrt(get_ez2(pars, zi, model))
        if row['type'] == 'DM_over_rs': pred = dm/rdrag
        elif row['type'] == 'DH_over_rs': pred = (299792.458/hz)/rdrag
        elif row['type'] == 'DV_over_rs': pred = (zi * dm**2 * (299792.458/hz))**(1/3)/rdrag
        else: pred = 0
        bao_preds.append(pred)
    return np.array(bao_preds), mu_theory

def log_prob(theta, z_sn, mu_sn, inv_sn_cov, model):
    Om, w0, wa, H0, ombh2 = theta
    if not (0.1 < Om < 0.6) or not (-3.0 < w0 < 1.0) or not (-5.0 < wa < 5.0) or not (50 < H0 < 90) or not (0.015 < ombh2 < 0.030) or (w0 + wa > 0): return -np.inf
    bao_th, sn_th = get_theory_custom(theta, z_sn, model=model)
    if bao_th is None: return -np.inf
    chi2 = ((ombh2 - 0.0224) / 0.0004)**2 + np.dot(bao_data - bao_th, np.dot(bao_inv_cov, bao_data - bao_th))
    delta_sn = mu_sn - sn_th
    chi2_sn = np.dot(delta_sn, np.dot(inv_sn_cov, delta_sn)) - (np.sum(np.dot(inv_sn_cov, delta_sn))**2 / np.sum(inv_sn_cov))
    return -0.5 * (chi2 + chi2_sn)

def worker(args):
    model, dname, dpath, cpath, steps = args
    name = f"{model}_{dname}"
    backend_path = os.path.join(OUT_DIR, f"{name}.h5")
    if os.path.exists(backend_path) and os.path.getsize(backend_path) > 1300000:
        print(f"[{name}] Done skip.")
        return
    print(f"[{name}] Starting...")
    z, mu, inv = load_sn(dpath, cpath)
    if z is None: return
    pos = [0.3, -1.0, 0.0, 70.0, 0.0224] + 1e-3 * np.random.randn(32, 5)
    backend = emcee.backends.HDFBackend(backend_path); backend.reset(32, 5)
    sampler = emcee.EnsembleSampler(32, 5, log_prob, args=(z, mu, inv, model), backend=backend)
    sampler.run_mcmc(pos, steps, progress=False)
    print(f"[{name}] Finished.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    tasks = []
    models = ['CPL', 'JBP', 'LOG']
    datasets = {
        'Panth_Uncorr': (PANTHEON_UNCORR, PANTHEON_COV),
        'Panth_Lin': (PANTHEON_LIN, PANTHEON_COV),
        'Panth_Poly': (PANTHEON_POLY, PANTHEON_COV),
        'DES_Uncorr': (DES_UNCORR, DES_COV),
        'DES_Lin': (DES_LIN, DES_COV),
        'DES_Poly': (DES_POLY, DES_COV)
    }
    for m in models:
        for dname, (dp, cp) in datasets.items():
            steps = 200 if m != 'CPL' else 250
            tasks.append((m, dname, dp, cp, steps))
    
    num_procs = min(multiprocessing.cpu_count(), 6)
    print(f"Spawning {num_procs} workers...")
    with multiprocessing.Pool(num_procs) as pool:
        pool.map(worker, tasks)
