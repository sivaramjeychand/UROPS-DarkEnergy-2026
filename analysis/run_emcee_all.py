
import os
import numpy as np
import pandas as pd
import emcee
import camb
import sys
import glob

# --- CONFIGURATION ---
OUT_DIR = "chains_emcee_all"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# 1. Dataset Paths
# Pantheon+
PANTHEON_DIR = r'../data/external/PantheonPlus/Pantheon+_Data/4_DISTANCES_AND_COVAR'
PANTHEON_UNCORR = os.path.join(PANTHEON_DIR, 'Pantheon+SH0ES.dat')
PANTHEON_COV = os.path.join(PANTHEON_DIR, 'Pantheon+SH0ES_STAT+SYS.cov')

PANTHEON_CORR_DIR = r'../data/external/PantheonPlus_Corrected/Pantheon+_Data/4_DISTANCES_AND_COVAR'
PANTHEON_LIN = os.path.join(PANTHEON_CORR_DIR, 'Pantheon+SH0ES_Linear.dat')
PANTHEON_POLY = os.path.join(PANTHEON_CORR_DIR, 'Pantheon+SH0ES_Poly.dat')

# DES
DES_DIR = r'../data/external/DES-SN5YR/4_DISTANCES_COVMAT'
DES_UNCORR = os.path.join(DES_DIR, 'DES-Dovekie_HD.csv')
DES_COV = os.path.join(DES_DIR, 'STAT+SYS.npz')

DES_CORR_DIR = r'../data/external/DES-SN5YR_Corrected/4_DISTANCES_COVMAT'
DES_LIN = os.path.join(DES_CORR_DIR, 'DES-Dovekie_HD_Linear.csv')
DES_POLY = os.path.join(DES_CORR_DIR, 'DES-Dovekie_HD_Poly.csv')

# BAO
BAO_MEAN_FILE = r'../data/external/DESI_BAO/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt'
BAO_COV_FILE = r'../data/external/DESI_BAO/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt'

# --- LOADING FUNCTIONS ---

def load_bao():
    print("Loading BAO Data...")
    if not os.path.exists(BAO_MEAN_FILE):
        print("BAO File not found."); return None, None, None
    
    df = pd.read_csv(BAO_MEAN_FILE, delim_whitespace=True, comment='#', names=['z', 'value', 'type'])
    data = df['value'].values
    cov = np.loadtxt(BAO_COV_FILE)
    try:
        inv_cov = np.linalg.inv(cov)
    except:
        print("BAO Inv Cov failed"); return None, None, None
    return df, data, inv_cov

def load_sn(data_path, cov_path):
    print(f"Loading SN: {os.path.basename(data_path)}")
    if not os.path.exists(data_path):
        print(f"SN Data not found: {data_path}"); return None, None, None
    
    z, mu = None, None
    cov = None
    
    # Load Data
    if 'Pantheon' in data_path:
        # Pantheon Format
        try:
             df = pd.read_csv(data_path, delim_whitespace=True)
             if 'zHD' not in df.columns:
                 df = pd.read_csv(data_path, delim_whitespace=True, skiprows=1)
             z = df['zHD'].values
             mu = df['MU_SH0ES'].values
        except:
            return None, None, None
    else:
        # DES Format
        try:
            with open(data_path, 'r') as f:
                lines = f.readlines()
            
            start_row = 0
            names = None
            for i, line in enumerate(lines):
                if line.strip().startswith('VARNAMES:'):
                    names = line.replace('VARNAMES:', '').split()
                    start_row = i + 1
                    break
            
            if names:
                import io
                content = "".join(lines[start_row:])
                df = pd.read_csv(io.StringIO(content), sep=r'\s+', names=names, comment='#')
            else:
                df = pd.read_csv(data_path, sep='\s+', comment='#')

            df['zHD'] = pd.to_numeric(df['zHD'], errors='coerce')
            col_mu = 'MU' if 'MU' in df.columns else 'MU_SH0ES'
            df[col_mu] = pd.to_numeric(df[col_mu], errors='coerce')
            
            df = df.dropna(subset=['zHD', col_mu])
            z = df['zHD'].values
            mu = df[col_mu].values

        except Exception as e:
             print(f"DES Load Error: {e}")
             return None, None, None
    
    # Load Covariance
    if cov_path.endswith('.cov') or cov_path.endswith('.txt'):
        header = open(cov_path).readline()
        if len(header.split()) == 1:
            cov = np.loadtxt(cov_path, skiprows=1)
        else:
            cov = np.loadtxt(cov_path)
        N = int(np.sqrt(len(cov.flatten())))
        cov = cov.reshape((N, N))
        
    elif cov_path.endswith('.npz'):
        try:
            d = np.load(cov_path)
            n = d['nsn'][0]
            flat = d['cov']
            full = np.zeros((n, n))
            full[np.triu_indices(n)] = flat
            mat = full + full.T - np.diag(np.diag(full))
            inv_cov = mat
            
            if len(z) != n:
                try:
                    cov_full = np.linalg.inv(inv_cov)
                except:
                    return None, None, None
                
                if abs(len(z) - n) < 5:
                    z = z[:n]
                    mu = mu[:n]
                    cov = cov_full
                else:
                    return None, None, None
            else:
                cov = np.linalg.inv(inv_cov)
                
        except Exception as e:
            return None, None, None

    if len(z) != cov.shape[0]:
        return None, None, None
        
    try:
        inv_cov = np.linalg.inv(cov)
    except:
        return None, None, None
        
    return z, mu, inv_cov

# --- MCMC ENGINE ---
bao_df, bao_data, bao_inv_cov = load_bao()

def get_theory(pars, z_sn):
    Om, w, H0 = pars
    h = H0 / 100.0
    ombh2 = 0.0224
    omch2 = (Om * h**2) - ombh2
    if omch2 < 0: return None, None
    
    pars_c = camb.CAMBparams()
    pars_c.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0)
    pars_c.set_dark_energy(w=w, wa=0, dark_energy_model='fluid')
    pars_c.InitPower.set_params(As=2e-9, ns=0.96)
    pars_c.set_matter_power(redshifts=[0.0], kmax=2.0)
    
    try:
        results = camb.get_background(pars_c)
    except:
        return None, None

    if bao_df is not None:
        rd = results.get_derived_params()['rdrag']
        bao_preds = []
        for _, row in bao_df.iterrows():
            z = row['z']
            DA = results.angular_diameter_distance(z)
            H = results.hubble_parameter(z)
            DM = (1+z)*DA
            DH = 299792.458/H
            DV = (z * DM**2 * DH)**(1.0/3.0)
            
            if row['type'] == 'DM_over_rs': pred = DM/rd
            elif row['type'] == 'DH_over_rs': pred = DH/rd
            elif row['type'] == 'DV_over_rs': pred = DV/rd
            else: pred = 0
            bao_preds.append(pred)
        bao_theory = np.array(bao_preds)
    else:
        bao_theory = None

    dl = results.luminosity_distance(z_sn)
    mu_theory = 5 * np.log10(dl) + 25
    
    return bao_theory, mu_theory

def log_prob(theta, z_sn, mu_sn, inv_sn_cov):
    Om, w, H0 = theta
    if not (0.1 < Om < 0.6): return -np.inf
    if not (-2.5 < w < -0.3): return -np.inf
    if not (50 < H0 < 90): return -np.inf
    
    bao_th, sn_th = get_theory(theta, z_sn)
    if bao_th is None: return -np.inf
    
    delta_bao = bao_data - bao_th
    chi2_bao = np.dot(delta_bao, np.dot(bao_inv_cov, delta_bao))
    
    delta_sn = mu_sn - sn_th
    chi2_stat = np.dot(delta_sn, np.dot(inv_sn_cov, delta_sn))
    sum_w = np.sum(inv_sn_cov)
    sum_w_delta = np.sum(np.dot(inv_sn_cov, delta_sn))
    chi2_sn_marg = chi2_stat - (sum_w_delta**2 / sum_w)
    
    chi2_prior = ((Om - 0.315) / 0.007)**2
    
    return -0.5 * (chi2_bao + chi2_sn_marg + chi2_prior)

def run_chain(name, z, mu, inv_cov, steps=100):
    print(f"Running Chain: {name} (Steps: {steps})")
    pos = [0.3, -1.0, 73.0] + 1e-2 * np.random.randn(16, 3)
    nwalkers, ndim = pos.shape
    
    backend_path = os.path.join(OUT_DIR, f"{name}.h5")
    if os.path.exists(backend_path):
        os.remove(backend_path)
        
    backend = emcee.backends.HDFBackend(backend_path)
    backend.reset(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(z, mu, inv_cov), backend=backend)
    sampler.run_mcmc(pos, steps, progress=True)
    
    return [0.3, -1.0, 73.0]

tasks = {
    'Pantheon_Uncorr': (PANTHEON_UNCORR, PANTHEON_COV),
    'Pantheon_Lin': (PANTHEON_LIN, PANTHEON_COV),
    'Pantheon_Poly': (PANTHEON_POLY, PANTHEON_COV),
    'DES_Uncorr': (DES_UNCORR, DES_COV),
    'DES_Lin': (DES_LIN, DES_COV),
    'DES_Poly': (DES_POLY, DES_COV)
}

if __name__ == "__main__":
    results = {}
    for name, (dpath, cpath) in tasks.items():
        z, mu, inv = load_sn(dpath, cpath)
        if z is not None:
             pars = run_chain(name, z, mu, inv, steps=120) 
             results[name] = pars
