# Progenitor Age-Bias in Supernova Cosmology (UROPS)

## UROPS-DarkEnergy-2026: SN Ia Age-Bias Replication

This repository contains the replication of the analysis presented in **Son et al. (2025)**, "Strong Progenitor Age-bias in Supernova Cosmology". The project quantifies how the evolution of host galaxy ages introduces a systematic bias in cosmological parameter estimation, masking the true nature of Dark Energy.

## 🚀 Key Results
- **Resolved Tension**: By aligning the BAO standard ruler ($r_d \approx 128.6$ Mpc) with the local SH0ES $H_0$ anchor, we achieved the "New Concordance" where SN and BAO scales match perfectly.
- **Confirmed Age-Bias**: Replicated the local correlation between host galaxy age and Hubble residuals ($r \approx -0.5$) using Rose et al. (2019).
- **Physical "Bridge" Model**: Implemented a physical age-redshift evolution model based on the Delay Time Distribution (DTD), aligning with Figure 2 of the paper.
- **SN-BAO Concordance**: Replicated Figure 4 (Residual Hubble Diagram), showing that corrected SNe shift into strong agreement with BAO data at high redshift.
- **Cosmological Impact**: Bayesian MCMC analysis shows a shift in the Dark Energy equation of state **$w$ from $-1.07 \pm 0.03$ to $-0.98 \pm 0.03$**, resolving the perceived tension with the cosmological constant ($w = -1$).

## 📂 Project Structure
- `analysis/`: Jupyter notebooks for each phase of the replication.
  - `01_...`: Data merging and Fig 1 replication.
  - `06_...`: Physical Age-Redshift Model and precision correction.
  - `07_...`: Resolved SN-BAO agreement replication.
- `data/`: Processed and external datasets (Pantheon+, DESI BAO, Rose19).
- `run_emcee_cosmology.py`: Python script for MCMC Bayesian parameter estimation.
- `plot_contours.py`: Script for generating triangle plots and parameter tables using `GetDist`.

## 🛠️ Requirements
- `camb`, `emcee`, `getdist`, `astropy`

---
*Replication performed for UROPS Dark Energy Project 2026.*

## Roadmap
- [ ] Phase 2: Implement Age-Bias Correction (Eq 1 in Son et al.)
- [ ] Phase 3: Run MCMC with Cobaya (CPL Model)
- [ ] Phase 4: Extension (JBP Model)

## Dependencies
- Cobaya
- CAMB
- GetDist
- Astropy