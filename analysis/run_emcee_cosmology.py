
import os
import numpy as np
import pandas as pd
import emcee
import camb
from camb import model, initialpower
import time
from multiprocessing import Pool
import sys

# --- CONFIGURATION ---
# --- CONFIGURATION ---
OUT_DIR = "chains_emcee"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Filenames
BAO_MEAN_FILE = r'../data/external/DESI_BAO/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt'
BAO_COV_FILE = r'../data/external/DESI_BAO/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt'

SN_FILE_UNCORR = r'../data/external/PantheonPlus/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat'
SN_COV_UNCORR = r'../data/external/PantheonPlus/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov'

SN_FILE_CORR = r'../data/external/PantheonPlus_Corrected/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat'
SN_COV_CORR = r'../data/external/PantheonPlus_Corrected/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov'

# --- LOAD BAO DATA ---
print("Loading BAO Data...")
# Inspect headers/format from previous steps
# Mean file: # [z] [value] [type]
bao_mean_df = pd.read_csv(BAO_MEAN_FILE, delim_whitespace=True, comment='#', names=['z', 'value', 'type'])
bao_data_vector = bao_mean_df['value'].values
bao_cov_matrix = np.loadtxt(BAO_COV_FILE)
# Invert BAO cov
try:
    bao_inv_cov = np.linalg.inv(bao_cov_matrix)
except np.linalg.LinAlgError:
    print("Warning: BAO Covariance singular?")
    sys.exit(1)

# --- LOAD SN DATA ---
def load_sn_data(dat_file, cov_file):
    print(f"Loading SN data from {dat_file}...")
    try:
        df = pd.read_csv(dat_file, delim_whitespace=True) # or sep='\s+'
        if 'zHD' not in df.columns: # Skip header issues if persistent
             df = pd.read_csv(dat_file, delim_whitespace=True, skiprows=1)
    except:
        # Fallback for weird headers
         df = pd.read_csv(dat_file, sep='\s+')

    z_sn = df['zHD'].values
    mu_sn = df['MU_SH0ES'].values
    
    # Load covariance
    # The covariance file usually starts with the number of points
    with open(cov_file, 'r') as f:
        n = int(f.readline())
    
    cov = np.loadtxt(cov_file, skiprows=1)
    cov = cov.reshape((n, n))
    
    # Invert
    inv_cov = np.linalg.inv(cov)
    return z_sn, mu_sn, inv_cov

print("Loading SN Uncorrected...")
sn_z_uncorr, sn_mu_uncorr, sn_inv_uncorr = load_sn_data(SN_FILE_UNCORR, SN_COV_UNCORR)

print("Loading SN Corrected...")
sn_z_corr, sn_mu_corr, sn_inv_corr = load_sn_data(SN_FILE_CORR, SN_COV_CORR)


# --- THEORY (CAMB) ---
def get_theory_vector(pars, bao_df, sn_z):
    # pars: [Om, w, H0] (Flat wCDM)
    # We fix Obh2 = 0.0224, As, ns etc to Planck/Standard
    Om, w, H0 = pars
    
    # Setup CAMB
    h = H0 / 100.0
    ombh2 = 0.0224
    omch2 = (Om * h**2) - ombh2
    
    if omch2 < 0: return None, None # Physicality check
    
    # Set parameters
    # Note: set_dark_energy(w=w, wa=0, dark_energy_model='fluid')
    pars_camb = camb.CAMBparams()
    pars_camb.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0)
    pars_camb.set_dark_energy(w=w, wa=0, dark_energy_model='fluid')
    
    # We need background evolution
    pars_camb.InitPower.set_params(As=2e-9, ns=0.96)
    pars_camb.set_matter_power(redshifts=[0.0], kmax=2.0)
    
    # Calculate
    try:
        results = camb.get_background(pars_camb)
    except:
        return None, None
    
    # --- BAO Theory ---
    # We need rs (sound horizon)
    # Note: CAMB results.get_derived_params()['rdrag'] gives rd
    rd = results.get_derived_params()['rdrag']
    
    bao_theory = []
    for idx, row in bao_df.iterrows():
        z = row['z']
        quant = row['type']
        
        # Derived values
        # DA = angular diameter distance
        # H = Hubble parameter
        # DV = (z * (1+z)**2 * DA**2 / H)**(1/3) ? No, DV = [cz (1+z)^2 DA^2 / H]^(1/3)
        # Check specific definitions for 'DM_over_rs', 'DH_over_rs', 'DV_over_rs'
        
        DA = results.angular_diameter_distance(z)
        H = results.hubble_parameter(z)
        
        # Comoving Transverse Distance DM = (1+z) * DA
        DM = (1+z) * DA
        
        # c/H
        c = 299792.458 # km/s
        DH = c / H
        
        # DV
        DV = (z * DM**2 * DH)**(1/3)
        
        if quant == 'DM_over_rs':
            pred = DM / rd
        elif quant == 'DH_over_rs':
            pred = DH / rd
        elif quant == 'DV_over_rs':
            pred = DV / rd
        else:
            pred = 0
            
        bao_theory.append(pred)
        
    # --- SN Theory ---
    # Distance modulus mu = 5 log10(dL) + 25
    dl = results.luminosity_distance(sn_z) # returns Mpc
    mu_theory = 5 * np.log10(dl) + 25
    
    return np.array(bao_theory), mu_theory

# --- LIKELIHOOD ---
def log_likelihood(theta, sn_set_name):
    # theta = [Om, w, H0]
    Om, w, H0 = theta
    
    # Priors
    if not (0.1 < Om < 0.5): return -np.inf
    if not (-2.0 < w < 0.0): return -np.inf # Conservative prior
    if not (50 < H0 < 90): return -np.inf
    
    # Theory
    if sn_set_name == 'uncorr':
        z_sn_use = sn_z_uncorr
    elif sn_set_name == 'corr':
        z_sn_use = sn_z_corr
    else: # BAO only (use uncorr z just for dummy call)
        z_sn_use = sn_z_uncorr # Not used for likelihood
        
    bao_pred, sn_pred = get_theory_vector(theta, bao_mean_df, z_sn_use)
    
    if bao_pred is None: return -np.inf
    
    # --- BAO Chi2 ---
    delta_bao = bao_data_vector - bao_pred
    chi2_bao = np.dot(delta_bao, np.dot(bao_inv_cov, delta_bao))
    
    # --- SN Chi2 ---
    if sn_set_name == 'bao_only':
        chi2_sn = 0
    else:
        if sn_set_name == 'uncorr':
            delta_sn = sn_mu_uncorr - sn_pred
            inv_cov = sn_inv_uncorr
        else:
            delta_sn = sn_mu_corr - sn_pred
            inv_cov = sn_inv_corr
            
        # The matrix is large (1700x1700). Dot product might be slow.
        # Optimize: L = -0.5 * chi2
        chi2_sn = np.dot(delta_sn, np.dot(inv_cov, delta_sn))
        
    return -0.5 * (chi2_bao + chi2_sn)

def log_probability(theta, sn_set_name):
    return log_likelihood(theta, sn_set_name)

# --- RUNNER ---
def run_mcmc(sn_set_name, nsteps=150):
    print(f"Starting MCMC for: {sn_set_name}")
    
    # Init pos
    # [Om, w, H0]
    pos = [0.3, -1.0, 73.0] + 1e-2 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape
    
    filename = os.path.join(OUT_DIR, f"chain_{sn_set_name}.h5")
    # Overwrite
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    
    # Run Serial (No Pool) to avoid Windows multiprocessing overhead with large matrices
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(sn_set_name,), backend=backend)
    sampler.run_mcmc(pos, nsteps, progress=True)
        
    print(f"Finished {sn_set_name}")

if __name__ == "__main__":
    # We run 3 chains: bao, uncorr, corr
    N_STEPS = 150
    
    run_mcmc('bao_only', N_STEPS)
    run_mcmc('uncorr', N_STEPS)
    run_mcmc('corr', N_STEPS)
