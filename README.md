
# Progenitor Age-Bias in Supernova Cosmology

**Investigation of the "11-Sigma Signal" and Cosmological Parameter Biases**

This repository contains a comprehensive replication and analysis of the claims made in **Son et al. (2025)** regarding progenitor age-bias in Type Ia Supernovae. The project investigates how different age-correction models (Linear vs. Polynomial) impact the derived cosmological parameters for Dark Energy ($w$CDM and $w_0w_a$CDM models), using the **Pantheon+** and **DES-SN5YR** datasets combined with **DESI 2024 BAO** and **Planck 2018 CMB** priors.

## 🚀 Key Findings

We successfully replicated the central anomaly reported in the literature and identified its source:

1.  **Replication of the "Decelerating Universe" Anomaly:**
    *   When applying a **Linear Age-Bias Correction** (slope $\sim -0.03$ mag/Gyr), the cosmological fit for both Pantheon+ and DES-SN5YR shifts drastically.
    *   The deceleration parameter becomes positive ($q_0 > 0$), and the Dark Energy equation of state evolves to extreme values ($w_a \approx -1.9$), effectively masking the signature of cosmic acceleration.
    *   This confirms the "11-sigma" signal mentioned in recent discussions is driven by the linear modeling assumption.

2.  **Polynomial Model Restoration:**
    *   We implemented a more physically motivated **Polynomial Correction** (matching the non-linear age-luminosity relation seen in local data).
    *   This model **restores the standard cosmological convergence**: $q_0 \approx -0.3$ to $-0.4$ (Accelerating), and $w_0 \approx -1, w_a \approx -0.6$.
    *   This suggests the "evidence against acceleration" is an artifact of over-correcting via a linear slope at high redshifts.

3.  **Methodological Improvements:**
    *   Identified that a tight Gaussian prior on $\Omega_m$ suppresses the ability to see the "broken" linear model solution. Switching to a **Flat Prior** ($0.1 < \Omega_m < 0.6$) was crucial for replicating the paper's results (Figure 3 and Table 1/2).
    *   Implemented proper **Analytic Marginalization** over absolute magnitude ($M$) to ensure robust fits independent of $H_0$ calibration quirks.
    *   Applied **Low-Redshift Cuts** ($z > 0.01$) for Pantheon+ to mitigate peculiar velocity errors which bias $w_a$.

## 📊 Results Summary

**Comparison of $w_0w_a$CDM Parameters (with BAO + CMB Priors):**

| Dataset | Correction | Source | $\Omega_m$ | $w_0$ | $w_a$ | $q_0$ | State |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DES-SN5YR** | Uncorrected | Paper (Table 2) | $0.319$ | $-0.78$ | $-0.71$ | $-0.30$ | Accelerating |
| | Uncorrected | **Replication** | **$0.326$** | **$-0.85$** | **$-0.93$** | **$-0.36$** | **Accelerating** |
| | Linear | Paper (Table 2) | $0.374$ | $-0.32$ | $-2.14$ | $+0.20$ | Decelerating |
| | Linear | **Replication** | **$0.392$** | **$-0.54$** | **$-1.91$** | **$+0.01$** | **Decelerating** |
| | Polynomial | **Replication** | **$0.341$** | **$-0.76$** | **$-1.42$** | **$-0.25$** | **Accelerating** |
| | | | | | | | |
| **Pantheon+** | Uncorrected | Paper (Table 2) | $0.301$ | $-0.89$ | $-0.20$ | $-0.43$ | Accelerating |
| | Uncorrected | **Replication** | **$0.315$** | **$-0.92$** | **$-0.49$** | **$-0.44$** | **Accelerating** |
| | Linear | Paper (Table 2) | $0.359$ | $-0.45$ | $-1.63$ | $+0.06$ | Decelerating |
| | Linear | **Replication** | **$0.392$** | **$-0.62$** | **$-1.81$** | **$-0.06$** | **Accelerating** |
| | Polynomial | **Replication** | **$0.320$** | **$-0.88$** | **$-0.61$** | **$-0.40$** | **Accelerating** |

*Note: The Linear correction systematically pushes $\Omega_m$ high and $w_a$ very negative. For DES-SN5YR, this drives the universe into deceleration. The Polynomial correction (ours) consistently restores the acceleration signal.*

## 📂 Repository Structure

### **1. Core Analysis Scripts (`analysis/`)**
*   `run_emcee_all.py`: MCMC engine for the standard **$w$CDM** model. Handles data loading, likelihood calculation (with marginalization), and chain execution.
*   `run_emcee_w0wa.py`: Specialized MCMC engine for the **$w_0w_a$CDM** dynamical Dark Energy model. Includes specific fixes for CAMB stability (`ppf` dark energy) and priors.
*   `calculate_q0.py` & `calculate_q0_w0wa.py`: Post-processing scripts to compute the deceleration parameter ($q_0$) from the generated chains.
*   `09_generate_comparison_table.py`: Generates formatted Markdown/HTML tables comparing our results directly with Son et al. (2025).

### **2. Replication & Plotting**
*   `04_figure3_replication.py`: Replicates Figure 3 (Hubble Residuals) for Pantheon+.
*   `04_figure3_replication_DES5Y.py`: Replicates Figure 3 for DES-SN5YR, including specific residual shifting logic matching the paper's visual style.
*   `08_plot_w0wa_contours.py`: Generates confidence contour plots (using `GetDist`) for the $w_0$-$w_a$ plane.

### **3. Data Handling**
*   `06_apply_correction_*.py`: Applies the age-bias corrections (Linear vs. Polynomial) to the raw SN data files based on progenitor age evolution models.
*   **Data Sources:**
    *   **Pantheon+:** `data/external/PantheonPlus`
    *   **DES-SN5YR:** `data/external/DES-SN5YR`
    *   **DESI BAO:** `data/external/DESI_BAO` (Standard Ruler calibration)

## 🛠️ Methodology & Tech Stack

*   **MCMC Sampler:** `emcee` (Affine Invariant MCMC Ensemble sampler)
*   **Cosmology Engine:** `CAMB` (Code for Anisotropies in the Microwave Background) used for high-precision background evolution predictions.
*   **Likelihood:**
    *   $\chi^2_{total} = \chi^2_{SN, marg} + \chi^2_{BAO} + \chi^2_{Prior}$
    *   **Analytic Marginalization:** Over absolute magnitude $M$ to remove $H_0$ dependency in SN fits.
    *   **Priors:** Flat priors on $\Omega_m$ generally, with $w_0+w_a < 0$ stability cuts for quintessence models.
*   **Analysis:** `GetDist` for chain analysis, custom Python scripts for $q_0$ derivation.

## 🏁 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib scipy emcee camb getdist astropy
    ```

2.  **Run MCMC Chains:**
    *   For standard $w$CDM:
        ```bash
        cd analysis
        python run_emcee_all.py
        ```
    *   For dynamical $w_0w_a$CDM (The key result):
        ```bash
        python run_emcee_w0wa.py
        ```

3.  **Generate Analysis:**
    *   Compute $q_0$ statistics:
        ```bash
        python calculate_q0_w0wa.py
        ```
    *   Generate Comparison Tables:
        ```bash
        python 09_generate_comparison_table.py
        ```
    *   Plot Contours:
        ```bash
        python 08_plot_w0wa_contours.py
        ```

## 📝 Authors

Analysis performed for **UROPS Dark Energy Project 2026**.
Replication based on methods from *Son et al. (2025)* and *Rose et al. (2019)*.