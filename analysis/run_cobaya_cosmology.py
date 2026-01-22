
import os
from cobaya.run import run
from cobaya.yaml import yaml_load_file

# Define the base output directory
# We are in analysis/, so chains is adjacent
OUT_DIR = "chains"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Common params for wCDM
params = {
    "omm": {"prior": {"min": 0.1, "max": 0.5}, "ref": 0.3, "proposal": 0.01},
    "w": {"prior": {"min": -3, "max": 1}, "ref": -1, "proposal": 0.05},
    "H0": {"prior": {"min": 50, "max": 90}, "ref": 70, "proposal": 1},
    "ombh2": 0.0224, # Fixed to simplify/speed up if focusing on DE w
    "omch2": 0.12,   # Fixed or effectively derived from omm if H0 varies? 
                     # Better: vary H0 and omm, fix ombh2, let omch2 be derived.
                     # Cobaya can handle omm -> omch2 if we define conversion.
                     # Or simpler: Vary H0, omm.
}

# Actually, for wCDM we usually vary:
# logA, ns, theta_MC, ombh2, omch2, w.
# But for SN/BAO focused analysis, often we just vary omm, w, H0 and maybe nuisance.
# Let's try a minimal set: omm, w, H0. 
# We need to ensure we have a theory code.
theory = {"camb": {"extra_args": {"halofit_version": "takahashi"}}}

# 1. BAO Only (as proxy for BAO+CMB to save time/likelihoods first?)
# The user wants BAO+CMB. Planck likelihoods are huge.
# I will try to use a Gaussian Prior approximation for CMB shift parameters if I can't install full Planck.
# Or just use BAO+SN as the main comparison first.
# Figure 8 has BAO+CMB.
# Let's try to include a simplified CMB likelihood (e.g. gaussian on theta_MC, ombh2, omch2).
# Or just use BAO alone vs SN alone vs Combined.
# Let's start with BAO + SN Uncorrected vs BAO + SN Corrected.
# This should show the shift.

# We will define 3 runs:
# A: BAO
# B: BAO + SN (Uncorrected)
# C: BAO + SN (Corrected)

# Likelihoods
cov_path = r"../data/external/PantheonPlus/Pantheon+_Data/4_DISTANCES_AND_COVAR"
corr_cov_path = r"../data/external/PantheonPlus_Corrected/Pantheon+_Data/4_DISTANCES_AND_COVAR"

# NOTE: Cobaya's sn.pantheonplus uses a specific .dataset file or path. 
# It usually expects the standard Pantheon+ file structure.
# We will point it to the .dat file.

info_bao = {
    "params": params,
    "likelihood": {"bao.desi_2024.bao_all_dr1": None},
    "theory": theory,
    "sampler": {"mcmc": {"Rminus1_stop": 0.1, "max_tries": 1000}},
    "output": os.path.join(OUT_DIR, "bao")
}

info_sn_uncorr = {
    "params": params,
    "likelihood": {
        "bao.desi_2024.bao_all_dr1": None,
        "sn.pantheonplus": {"dataset_file": os.path.join(cov_path, "Pantheon+SH0ES.dat")}
    },
    "theory": theory,
    "sampler": {"mcmc": {"Rminus1_stop": 0.1, "max_tries": 1000}},
    "output": os.path.join(OUT_DIR, "bao_sn_uncorr")
}

info_sn_corr = {
    "params": params,
    "likelihood": {
        "bao.desi_2024.bao_all_dr1": None,
        "sn.pantheonplus": {"dataset_file": os.path.join(corr_cov_path, "Pantheon+SH0ES.dat")}
    },
    "theory": theory,
    "sampler": {"mcmc": {"Rminus1_stop": 0.1, "max_tries": 1000}},
    "output": os.path.join(OUT_DIR, "bao_sn_corr")
}

# Function to run
def run_chain(info, name):
    print(f"--- Running Chain: {name} ---")
    updated_info, sampler = run(info, resume=True)
    print(f"--- Finished {name} ---")

if __name__ == "__main__":
    # run_chain(info_bao, "BAO Only") # Optional, focusing on SN shift
    run_chain(info_sn_uncorr, "BAO + SN Uncorrected")
    run_chain(info_sn_corr, "BAO + SN Corrected")
