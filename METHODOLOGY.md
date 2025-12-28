# Rigorous Methodology for Energy Time-Series Forecasting

This document outlines the strict methodological framework implemented to ensure reproducible, leakage-free, and physically defensible energy forecasting.

## 1. Causal Feature Engineering (Leakage Prevention)
To eliminate target leakage, we strictly enforce causality in feature generation:
*   **Lag Features:** $X_{t-k}$ where $k \ge 1$.
*   **Rolling Statistics:** Computed on the **shifted** target series $y_{t-1}$.
    *   *Incorrect (Leaky):* `rolling(window).mean()` (includes $y_t$ by default in some libraries or if not carefully indexed).
    *   *Correct (Implemented):* `y.shift(1).rolling(window).mean()`.
*   **Automated Checks:** The pipeline includes an assertion step (`check_leakage`) that verifies feature vectors at time $t$ do not contain information from $y_t$.

## 2. Chronological Evaluation Protocol
Random splitting is invalid for time-series data due to autocorrelation. We implement:
*   **Nested Time-Series Cross-Validation:**
    *   **Outer Loop (Evaluation):** 5-fold rolling-origin forward validation.
    *   **Inner Loop (Tuning):** Time-series split within the outer training set.
*   **Final Evaluation:** A deterministic chronological split (First 75% Train, Last 25% Test).
*   **No Shuffling:** All splitters are explicitly configured with `shuffle=False`.

## 3. Modeling & Scaling
*   **Scaling:** `StandardScaler` is fitted **strictly** on the training fold and applied to validation/test folds.
*   **Models:**
    *   **XGBoost / LightGBM:** Gradient boosting trees optimized for tabular time-series.
    *   **Random Forest:** Baseline ensemble method.
*   **Handling "Lights":** The pipeline runs two scenarios (Include Lights vs. Exclude Lights) to address potential sensor availability in real-world deployments.

## 4. Statistical Rigor
*   **Metrics:** RMSE, MAE, $R^2$ reported for both Train (in-sample) and Test (out-of-sample).
*   **Significance:** Diebold-Mariano test (available in core pipeline) for comparing predictive accuracy.
*   **Feature Importance:** Permutation importance computed on the **Test Set** (not Train) to avoid overfitting bias.

## 5. Optimization (Illustrative Proxy)
*   **Context:** The dataset lacks HVAC-specific sub-metering.
*   **Approach:** The multi-objective optimization (NSGA-II) is explicitly labeled as **"Illustrative/Toy"**.
*   **Objective Functions:** Uses proxy relationships (e.g., "5% energy savings per °C setback") rather than claiming physical HVAC simulation.
*   **Goal:** To demonstrate the *decision-making framework* (Pareto front analysis) without making false physical claims about the specific building's HVAC system.
