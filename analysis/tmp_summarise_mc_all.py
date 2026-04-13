import os
import numpy as np
import pandas as pd
import emcee

CHAIN_DIR = "chains_emcee_w0wa"
OUTPUT_MD = "output/chains_mc_all_report.md"

datasets = ['Pantheon', 'DES']
corrections = ['Uncorr', 'Lin', 'Poly']

results = []
for ds in datasets:
    for corr in corrections:
        name = f"{ds}_{corr}"
        h5_path = os.path.join(CHAIN_DIR, f"{name}.h5")
        if os.path.exists(h5_path):
            try:
                reader = emcee.backends.HDFBackend(h5_path)
                n = reader.iteration
                if n < 20: continue # Basic sanity
                
                chain = reader.get_chain(discard=int(n*0.4), flat=True)
                lnprob = reader.get_log_prob(discard=int(n*0.4), flat=True)
                
                Om_samp = chain[:, 0]; w_samp = chain[:, 1]; H0_samp = chain[:, 2]
                
                def gs(d):
                    p = np.percentile(d, [16, 50, 84])
                    return p[1], p[2]-p[1], p[1]-p[0]
                    
                om_m, om_u, om_l = gs(Om_samp)
                w_m, w_u, w_l = gs(w_samp)
                h0_m, h0_u, h0_l = gs(H0_samp)
                
                # Calculate deceleration parameter q0 for flat wCDM
                q0_samp = 0.5 * Om_samp + 0.5 * (1 - Om_samp) * (1 + 3 * w_samp)
                q0_m, q0_u, q0_l = gs(q0_samp)
                
                chi2_min = -2 * np.max(lnprob)
                
                results.append({
                    'Dataset': ds, 'Correction': corr,
                    'Om_val': om_m,
                    'Om': f"{om_m:.3f} +{om_u:.3f}/-{om_l:.3f}",
                    'w_val': w_m,
                    'w': f"{w_m:.3f} +{w_u:.3f}/-{w_l:.3f}",
                    'H0_val': h0_m,
                    'H0': f"{h0_m:.3f} +{h0_u:.3f}/-{h0_l:.3f}",
                    'q0_val': q0_m,
                    'q0': f"{q0_m:.3f} +{q0_u:.3f}/-{q0_l:.3f}",
                    'chi2': chi2_min
                })
            except Exception as e:
                print(f"Failed parsing {name}: {e}")

df = pd.DataFrame(results)

if not df.empty:
    with open(OUTPUT_MD, 'w') as f:
        f.write("# Chains MC All: wCDM Model Comparison Report\n\n")
        f.write("This report summarizes the MCMC chains from `chains_emcee_all/`. The model parameters are `(Om, w, H0)`, representing a wCDM cosmology. It compares the uncorrected data with linear and polynomial age-bias corrections.\n\n")
        
        for ds in datasets:
            f.write(f"## Dataset: {ds}\n\n")
            sub = df[df['Dataset'] == ds].copy()
            if sub.empty:
                f.write("No results yet.\n\n")
                continue
                
            uncorr_om = sub[sub['Correction'] == 'Uncorr']['Om_val'].values
            if len(uncorr_om) > 0:
                sub['Shift(Om)'] = sub['Om_val'] - uncorr_om[0]
                
            uncorr_w = sub[sub['Correction'] == 'Uncorr']['w_val'].values
            if len(uncorr_w) > 0:
                sub['Shift(w)'] = sub['w_val'] - uncorr_w[0]
                
            uncorr_q0 = sub[sub['Correction'] == 'Uncorr']['q0_val'].values
            if len(uncorr_q0) > 0:
                sub['Shift(q0)'] = sub['q0_val'] - uncorr_q0[0]
                
            display_cols = ['Correction', 'Om', 'w', 'H0', 'q0']
            if 'Shift(Om)' in sub.columns:
                display_cols.extend(['Shift(Om)', 'Shift(w)', 'Shift(q0)'])
            display_cols.append('chi2')
            
            # Format shifts to 3 decimal places
            if 'Shift(Om)' in sub.columns:
                sub['Shift(Om)'] = sub['Shift(Om)'].apply(lambda x: f"{x:+.3f}")
                sub['Shift(w)'] = sub['Shift(w)'].apply(lambda x: f"{x:+.3f}")
                sub['Shift(q0)'] = sub['Shift(q0)'].apply(lambda x: f"{x:+.3f}")
                
            f.write(sub[display_cols].to_markdown(index=False))
            f.write("\n\n")
            
        f.write("## Interpretation & Executive Summary\n")
        f.write("- **Shift in $\Omega_m$**: When applying age-bias corrections (Linear or Polynomial), the estimated matter density parameter $\Omega_m$ typically shifts compared to the uncorrected dataset. This indicates that age-bias impacts the inferred cosmology.\n")
        f.write("- **Impact on $w$**: Similarly, the dark energy equation of state $w$ exhibits shifts across the different corrections, which can push the result slightly further from or closer to the standard $w = -1$ (Cosmological Constant) depending on the direction of the correction.\n")
    print(f"Report generated: {OUTPUT_MD}")
else:
    print("No data parsed.")
