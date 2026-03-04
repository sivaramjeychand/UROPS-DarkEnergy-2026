
import os
import numpy as np
import pandas as pd
import emcee

# --- CONFIG ---
CHAIN_DIR = "chains_emcee_w0wa"
OUTPUT_TXT = "output/comparison_table.txt"
OUTPUT_MD = "output/comparison_table.md"

# --- PAPER VALUES (from User Image) ---
# Format: [Om_val, Om_err, w0_val, w0_err, wa_val, wa_err, q0_val, q0_err]
paper_data = {
    "Pantheon_Uncorr": ["0.311", "0.006", "-0.838", "0.054", "-0.612", "0.201", "-0.366", "0.061"],
    "Pantheon_Lin":    ["0.353", "0.006", "-0.447", "0.059", "-1.585", "0.227", "0.065",  "0.060"],
    "DES_Uncorr":      ["0.319", "0.006", "-0.754", "0.056", "-0.848", "0.217", "-0.271", "0.061"],
    "DES_Lin":         ["0.363", "0.006", "-0.337", "0.062", "-1.902", "0.246", "0.178",  "0.061"]
}

# My Datasets mapping
task_map = {
    "Pantheon_Uncorr": "BAO + CMB + Panth.",
    "Pantheon_Lin":    "BAO + CMB + Panth. (linear)",
    "Pantheon_Poly":   "BAO + CMB + Panth. (poly)",
    "DES_Uncorr":      "BAO + CMB + DES5Y",
    "DES_Lin":         "BAO + CMB + DES5Y (linear)",
    "DES_Poly":        "BAO + CMB + DES5Y (poly)"
}

def get_stats(chain):
    # chain shape: (samples, dims) -> Om, w0, wa, H0, ombh2
    # Calculate q0
    Om = chain[:, 0]
    w0 = chain[:, 1]
    q0 = 0.5 * Om + 0.5 * (1 - Om) * (1 + 3 * w0)
    
    def get_percentiles(data):
        v = np.percentile(data, [16, 50, 84])
        return (v[1], v[2] - v[1], v[1] - v[0]) # (median, upper_err, lower_err)

    return [
        get_percentiles(Om),
        get_percentiles(w0),
        get_percentiles(chain[:, 2]), # wa
        get_percentiles(q0)
    ]

def fmt(med, up, lo):
    # If errors are very close, show as +/-
    if abs(up - lo) < 0.005:
        return f"{med:.3f} ± {up:.3f}"
    return f"{med:.3f} +{up:.3f}/-{lo:.3f}"

rows = []

print("Generating comparison table...")

for key, label in task_map.items():
    # 1. Load Replication Data
    h5_path = os.path.join(CHAIN_DIR, f"{key}.h5")
    if os.path.exists(h5_path):
        reader = emcee.backends.HDFBackend(h5_path)
        chain = reader.get_chain(discard=int(reader.iteration * 0.3), flat=True)
        stats = get_stats(chain) # [(Om, err), (w0, err), (wa, err), (q0, err)]
        
        # Format Replication Row
        row_rep = {
            "Dataset": label,
            "Source": "This Work",
            "Omega_m": fmt(*stats[0]),
            "w0": fmt(*stats[1]),
            "wa": fmt(*stats[2]),
            "q0": fmt(*stats[3])
        }
        rows.append(row_rep)
        
        # 2. Load Paper Data (if exists for this key)
        if key in paper_data:
            p = paper_data[key]
            # p is list of strings
            row_pap = {
                "Dataset": label.replace("(linear)", "(corrected)").replace("(poly)", "(corrected)"), 
                # Note: Paper only has one 'corrected' entry usually implies Linear, but let's match label style
                "Source": "Paper (Son et al.)",
                "Omega_m": f"{p[0]} ± {p[1]}",
                "w0": f"{p[2]} ± {p[3]}",
                "wa": f"{p[4]} ± {p[5]}",
                "q0": f"{p[6]} ± {p[7]}"
            }
            # Insert Paper row immediately after standard/linear rep row for comparison
            # Modify Dataset name to match strictly if needed, but side-by-side is nice
            rows.append(row_pap)
            
            # Add a separator or difference row? Maybe just rows key is fine.

df = pd.DataFrame(rows)
cols = ["Dataset", "Source", "Omega_m", "w0", "wa", "q0"]
df = df[cols]

# Markdown
md_table = df.to_markdown(index=False)
print("\n" + md_table)

with open(OUTPUT_MD, "w") as f:
    f.write(md_table)

# Create a more visual grouped table
# Group by Dataset Base
groups = [
    ("Pantheon", ["Pantheon_Uncorr", "Pantheon_Lin", "Pantheon_Poly"]),
    ("DES5Y", ["DES_Uncorr", "DES_Lin", "DES_Poly"])
]

html = "<h3>Comparison of Results</h3><table><thead><tr><th>Dataset</th><th>Source</th><th>&Omega;<sub>m</sub></th><th>w<sub>0</sub></th><th>w<sub>a</sub></th><th>q<sub>0</sub></th></tr></thead><tbody>"

def get_row_html(d, s, om, w0, wa, q0, bold=False):
    style = "font-weight:bold;" if bold else ""
    return f"<tr style='{style}'><td>{d}</td><td>{s}</td><td>{om}</td><td>{w0}</td><td>{wa}</td><td>{q0}</td></tr>"

for gname, keys in groups:
    html += f"<tr><td colspan='6'><b>--- {gname} ---</b></td></tr>"
    for k in keys:
        # Rep Data
        h5_path = os.path.join(CHAIN_DIR, f"{k}.h5")
        if os.path.exists(h5_path):
            reader = emcee.backends.HDFBackend(h5_path)
            chain = reader.get_chain(discard=int(reader.iteration * 0.3), flat=True)
            s = get_stats(chain)
            
            lbl = task_map[k]
            html += get_row_html(lbl, "This Work", fmt(*s[0]), fmt(*s[1]), fmt(*s[2]), fmt(*s[3]), bold=("Poly" in lbl))
        
        # Paper Data
        if k in paper_data:
            p = paper_data[k]
            lbl_pap = lbl.replace("(linear)", "(corrected)")
            html += get_row_html(lbl_pap, "Paper", f"{p[0]} ± {p[1]}", f"{p[2]} ± {p[3]}", f"{p[4]} ± {p[5]}", f"{p[6]} ± {p[7]}", bold=False)

html += "</tbody></table>"
with open("output/comparison_table.html", "w") as f:
    f.write(html)
