
import os
import numpy as np
import pandas as pd
import emcee

# Formula for q0 in Flat wCDM
# q0 = 1/2 * Om + 1/2 * (1 - Om) * (1 + 3w)
def calculate_q0_samples(flat_samples):
    # flat_samples columns: Om, w, H0
    Om = flat_samples[:, 0]
    w = flat_samples[:, 1]
    # H0 is not needed for q0
    
    q0 = 0.5 * Om + 0.5 * (1 - Om) * (1 + 3 * w)
    return q0

CHAIN_DIR = "chains_emcee_all"
OUTPUT_FILE = "output/q0_wCDM_results.csv"

datasets = [
    "Pantheon_Uncorr",
    "Pantheon_Lin",
    "Pantheon_Poly",
    "DES_Uncorr",
    "DES_Lin",
    "DES_Poly"
]

results = []

print(f"{'Dataset':<20} | {'q0':<15} | {'State'}")
print("-" * 50)

for name in datasets:
    h5_path = os.path.join(CHAIN_DIR, f"{name}.h5")
    if not os.path.exists(h5_path):
        print(f"{name:<20} | {'Missing':<15} | -")
        continue

    try:
        reader = emcee.backends.HDFBackend(h5_path)
        # Assuming run_emcee_all.py used roughly 120 steps, discarding 40 is safe (approx 30%)
        # To be safe, we discard based on total steps found
        full_chain = reader.get_chain()
        nsteps = full_chain.shape[0]
        discard = int(nsteps * 0.3)
        
        flat_samples = reader.get_chain(discard=discard, flat=True)
        
        q0_chain = calculate_q0_samples(flat_samples)
        
        q0_mean = np.mean(q0_chain)
        q0_std = np.std(q0_chain)
        
        state = "Accelerating" if q0_mean < 0 else "Decelerating"
        
        print(f"{name:<20} | {q0_mean:>.3f} +/- {q0_std:.3f} | {state}")
        
        results.append({
        "Dataset": name,
            "q0_mean": q0_mean,
            "q0_std": q0_std,
            "State": state
        })
        
    except Exception as e:
        print(f"{name:<20} | Error: {e}")

# Save to CSV
if not os.path.exists("output"):
    os.makedirs("output")
    
df_res = pd.DataFrame(results)
df_res.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved results to {OUTPUT_FILE}")
