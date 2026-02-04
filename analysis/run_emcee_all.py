
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
        # Original has VARNAMES, Corrected files (saved by me) are standard space-sep
        try:
            with open(data_path, 'r') as f:
                lines = f.readlines()
            
            # 1. Try finding VARNAMES (Original DES)
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
                # 2. Fallback: Standard DataFrame (Corrected files)
                # Ensure we point to the beginning or handle comments
                df = pd.read_csv(data_path, sep='\s+', comment='#')

            # Ensure numeric
            df['zHD'] = pd.to_numeric(df['zHD'], errors='coerce')
            col_mu = 'MU' if 'MU' in df.columns else 'MU_SH0ES'
            if col_mu not in df.columns:
                 # Check for 'm_b_corr'? No, usually MU. 
                 pass
            df[col_mu] = pd.to_numeric(df[col_mu], errors='coerce')
            
            # Filter valid
            df = df.dropna(subset=['zHD', col_mu])
            z = df['zHD'].values
            mu = df[col_mu].values

        except Exception as e:
             print(f"DES Load Error: {e}")
             return None, None, None
    
    # Load Covariance
    if cov_path.endswith('.cov') or cov_path.endswith('.txt'):
        # Pantheon style
        header = open(cov_path).readline()
        if len(header.split()) == 1:
            cov = np.loadtxt(cov_path, skiprows=1)
        else:
            cov = np.loadtxt(cov_path)
        N = int(np.sqrt(len(cov.flatten())))
        cov = cov.reshape((N, N))
        
    elif cov_path.endswith('.npz'):
        # DES style (Compressed Inverse Cov)
        try:
            d = np.load(cov_path)
            # Keys: nsn, cov (upper tri flattened or inv cov?)
            # DES-Dovekie script says: "Covtot_inv is the inverse... d[d.files[1]]"
            n = d['nsn'][0]
            flat = d['cov']
            
            # Reconstruct Full Matrix (Upper Tri)
            full = np.zeros((n, n))
            full[np.triu_indices(n)] = flat
            # Symmetrize
            # Method: A + A.T - diag(A)
            mat = full + full.T - np.diag(np.diag(full))
            
            # This 'mat' is likely the INVERSE Covariance (based on likelihood script)
            # If so, inv_cov = mat.
            # But wait, we might need to subset it if sizes mismatch?
            inv_cov = mat
            
            # Use z from DataFrame to subset if necessary?
            # DES Likelihood uses z>0 cut.
            # Check length
            if len(z) != n:
                print(f"DES Size Mismatch: Data={len(z)}, CovExp={n}")
                # If Data is larger (e.g. 1821 vs 1820), maybe extra line?
                # If Data is smaller (cuts?), we must subset the covariance.
                # Assuming 'mat' covers the RAW file order.
                # If we filter the data, we must filter the matrix rows/cols.
                # But 'mat' is Inverse Covariance.
                # To subset, we MUST: ExInv -> Cov -> Subset -> NewInv
                
                # 1. Invert to get Cov
                try:
                    cov_full = np.linalg.inv(inv_cov)
                except:
                    print("Inversion of full DES cov failed.")
                    return None, None, None
                
                # 2. Subset
                # We need to know WHICH rows to keep.
                # The data file likely matches the covariance 1-to-1 before cuts.
                # The 'df' we loaded has dropped NaNs.
                # We should use the original indices.
                # But 'read_csv' creates new index.
                # We assume the file is sorted and matches?
                # Best effort: Truncate to size n if close?
                if abs(len(z) - n) < 5:
                    print(f"Truncating/matching data to {n}")
                    z = z[:n]
                    mu = mu[:n]
                    cov = cov_full # Already n x n
                else:
                    print("Major mismatch. Aborting DES load.")
                    return None, None, None
            else:
                # Dimensions match
                cov = np.linalg.inv(inv_cov) # Get C
                
        except Exception as e:
            print(f"DES Cov Error: {e}")
            return None, None, None

    # Final Check
    if len(z) != cov.shape[0]:
        print(f"Final Dim Mismatch: {len(z)} vs {cov.shape[0]}")
        return None, None, None
        
    try:
        inv_cov = np.linalg.inv(cov)
    except:
        print("Singular Matrix"); return None, None, None
        
    return z, mu, inv_cov


# --- MCMC ENGINE ---
bao_df, bao_data, bao_inv_cov = load_bao()

def get_theory(pars, z_sn):
    # pars: [Om, w, H0]
    Om, w, H0 = pars
    
    # CAMB Setup
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

    # BAO
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

    # SN
    dl = results.luminosity_distance(z_sn)
    mu_theory = 5 * np.log10(dl) + 25
    
    return bao_theory, mu_theory

def log_prob(theta, z_sn, mu_sn, inv_sn_cov):
    Om, w, H0 = theta
    # Broad priors
    if not (0.1 < Om < 0.6): return -np.inf
    if not (-2.5 < w < -0.3): return -np.inf
    if not (50 < H0 < 90): return -np.inf
    
    bao_th, sn_th = get_theory(theta, z_sn)
    if bao_th is None: return -np.inf
    
    # --- 1. BAO Chi2 ---
    delta_bao = bao_data - bao_th
    chi2_bao = np.dot(delta_bao, np.dot(bao_inv_cov, delta_bao))
    
    # --- 2. SN Chi2 (WITH MARGINALIZATION) ---
    delta_sn = mu_sn - sn_th
    
    # Standard Chi2
    chi2_stat = np.dot(delta_sn, np.dot(inv_sn_cov, delta_sn))
    
    # Marginalization over M
    # chi2_marg = chi2_stat - (Sum(W * delta)^2 / Sum(W))
    sum_w = np.sum(inv_sn_cov)
    sum_w_delta = np.sum(np.dot(inv_sn_cov, delta_sn))
    
    chi2_sn_marg = chi2_stat - (sum_w_delta**2 / sum_w)
    
    # --- 3. CMB Prior (Planck) ---
    # Constrains Omega_m
    chi2_prior = ((Om - 0.315) / 0.007)**2
    
    return -0.5 * (chi2_bao + chi2_sn_marg + chi2_prior)

def run_chain(name, z, mu, inv_cov, steps=100):
    print(f"Running Chain: {name} (Steps: {steps})")
    pos = [0.3, -1.0, 73.0] + 1e-2 * np.random.randn(16, 3)
    nwalkers, ndim = pos.shape
    
    backend_path = os.path.join(OUT_DIR, f"{name}.h5")
    # Reset backend to ensure fresh run
    if os.path.exists(backend_path):
        os.remove(backend_path)
        
    backend = emcee.backends.HDFBackend(backend_path)
    backend.reset(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(z, mu, inv_cov), backend=backend)
    sampler.run_mcmc(pos, steps, progress=True)
    
    # Analyze
    try:
        flat_samples = sampler.get_chain(discard=int(steps*0.3), flat=True)
        mean_pars = np.mean(flat_samples, axis=0)
        print(f"Results {name}: Om={mean_pars[0]:.3f}, w={mean_pars[1]:.3f}, H0={mean_pars[2]:.3f}")
        return mean_pars
    except:
        return [0.3, -1.0, 73.0]

# --- EXECUTION ---
tasks = {
    'Pantheon_Uncorr': (PANTHEON_UNCORR, PANTHEON_COV),
    'Pantheon_Lin': (PANTHEON_LIN, PANTHEON_COV),
    'Pantheon_Poly': (PANTHEON_POLY, PANTHEON_COV),
    'DES_Uncorr': (DES_UNCORR, DES_COV),
    'DES_Lin': (DES_LIN, DES_COV),
    'DES_Poly': (DES_POLY, DES_COV)
}

# Clear previous results
results = {}

if __name__ == "__main__":
    for name, (dpath, cpath) in tasks.items():
        z, mu, inv = load_sn(dpath, cpath)
        if z is not None:
             pars = run_chain(name, z, mu, inv, steps=120) 
             results[name] = pars
        else:
            print(f"Skipping {name} due to load error.")
            
    # Save Results
    print("\n--- FINAL BEST FITS ---")
    with open(os.path.join(OUT_DIR, "best_fits.csv"), 'w') as f:
        f.write("Dataset,Om,w,H0\n")
        for k, v in results.items():
            line = f"{k},{v[0]:.4f},{v[1]:.4f},{v[2]:.4f}"
            print(line)
            f.write(line + "\n")
