"""
Step 9 — Generate the final parameter comparison table.

Reads all w0wa chains (step 6) and prints a Markdown table comparing
This Work vs. the paper values for each dataset × correction combination.

Output:
  - src/output/tables/comparison_table.md
  - src/output/tables/comparison_table.csv
"""
import os, sys
import numpy as np
import pandas as pd
import emcee

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_TABLES, OUT_CHAINS, BURNIN_FRAC, PAPER_VALUES

CHAIN_DIR = os.path.join(OUT_CHAINS, 'w0wa')


def get_stats(chain):
    """
    Return (median, +err, -err) for Om, w0, wa, q0 from a flat chain.
    q0 = Ω_m/2 + (1 − Ω_m)/2 · (1 + 3w0)  at z=0 (CPL leading order).
    """
    Om = chain[:, 0]
    w0 = chain[:, 1]
    q0 = Om / 2.0 + (1.0 - Om) / 2.0 * (1.0 + 3.0 * w0)

    def pct(x):
        p = np.percentile(x, [16, 50, 84])
        return p[1], p[2] - p[1], p[1] - p[0]

    return {
        'Om':  pct(Om),
        'w0':  pct(w0),
        'wa':  pct(chain[:, 2]),
        'q0':  pct(q0),
        'H0':  pct(chain[:, 3]),
    }


def fmt(med, up, lo):
    return f'{med:.3f} +{up:.3f}/−{lo:.3f}'


rows = []

for model in ['CPL', 'JBP', 'LOG']:
    for ds in ['Panth', 'DES']:
        for corr in ['Uncorr', 'Lin', 'Poly']:
            name = f'{model}_{ds}_{corr}'
            path = os.path.join(CHAIN_DIR, f'{name}.h5')
            if not os.path.exists(path):
                continue

            reader = emcee.backends.HDFBackend(path, read_only=True)
            n      = reader.iteration
            if n < 50:
                continue
            burnin = int(BURNIN_FRAC * n)
            chain  = reader.get_chain(discard=burnin, flat=True)
            st     = get_stats(chain)

            row = {
                'Model': model, 'Dataset': ds, 'Correction': corr,
                'Source': 'This work',
                'Ωm':   fmt(*st['Om']),
                'w0':   fmt(*st['w0']),
                'wa':   fmt(*st['wa']),
                'q0':   fmt(*st['q0']),
                'H0':   fmt(*st['H0']),
            }
            rows.append(row)

            # Append paper row for CPL only (paper uses CPL)
            pkey = f'{ds}_{corr}'
            if model == 'CPL' and pkey in PAPER_VALUES:
                p = PAPER_VALUES[pkey]
                rows.append({
                    'Model': 'CPL', 'Dataset': ds, 'Correction': corr,
                    'Source': 'Paper (Son+2024)',
                    'Ωm': f"{p['Om']:.3f} ± {p['Om_err']:.3f}",
                    'w0': f"{p['w0']:.3f} ± {p['w0_err']:.3f}",
                    'wa': f"{p['wa']:.3f} ± {p['wa_err']:.3f}",
                    'q0': f"{p['q0']:.3f} ± {p['q0_err']:.3f}",
                    'H0': '—',
                })

if not rows:
    print('No chains found. Run 06_run_mcmc_w0wa.py first.')
    sys.exit(0)

df = pd.DataFrame(rows)
cols = ['Model', 'Dataset', 'Correction', 'Source', 'Ωm', 'w0', 'wa', 'q0', 'H0']
df   = df[cols]

md  = df.to_markdown(index=False)
print('\n' + md)

os.makedirs(OUT_TABLES, exist_ok=True)
df.to_csv(os.path.join(OUT_TABLES, 'comparison_table.csv'), index=False)
with open(os.path.join(OUT_TABLES, 'comparison_table.md'), 'w') as f:
    f.write('# Dark Energy Model Comparison\n\n')
    f.write('Comparing CPL, JBP, and LOG parametrizations with and without '
            'age-bias correction.\n\n')
    f.write(md)
    f.write('\n\n> **Note:** `q0 = Ωm/2 + (1−Ωm)/2·(1+3w0)` at z=0.\n'
            '> Positive q0 → decelerating universe.\n')

print(f'\nSaved to {OUT_TABLES}/')
