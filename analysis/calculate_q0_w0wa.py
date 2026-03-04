
import os
import numpy as np
import pandas as pd
import emcee

# Formula for q0 in Flat w0waCDM at z=0
# q0 = 1/2 * Om + 1/2 * (1 - Om) * (1 + 3*w(0))
# w(0) = w0
def calculate_q0_w0wa(flat_samples):
    # flat_samples columns: Om, w0, wa, H0
    Om = flat_samples[:, 0]
    w0 = flat_samples[:, 1]
    # wa is not needed for q0 (which is at z=0)
    # H0 is not needed
    
    q0 = 0.5 * Om + 0.5 * (1 - Om) * (1 + 3 * w0)
    return q0

CHAIN_DIR = "chains_emcee_w0wa"
OUTPUT_FILE = "output/q0_results_w0wa.csv"

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
        # Assuming we ran enough steps, discard burn-in
        full_chain = reader.get_chain()
        nsteps = full_chain.shape[0]
        discard = int(nsteps * 0.3)
        
        flat_samples = reader.get_chain(discard=discard, flat=True)
        
        q0_chain = calculate_q0_w0wa(flat_samples)
        
        # Calculate statistics
        vals = np.percentile(q0_chain, [16, 50, 84])
        q0_med = vals[1]
        q0_up = vals[2] - vals[1]
        q0_lo = vals[1] - vals[0]
        
        state = "Accelerating" if q0_med < 0 else "Decelerating"
        
        err_str = f"{q0_med:.3f} +{q0_up:.3f}/-{q0_lo:.3f}"
        print(f"{name:<20} | {err_str:<15} | {state}")
        
        results.append({
            "Dataset": name,
            "q0_median": q0_med,
            "q0_upper": q0_up,
            "q0_lower": q0_lo,
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
