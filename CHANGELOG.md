# Changelog

## [1.0.0] - Rigorous Methodology Update

### 🛑 Critical Fixes (Leakage & Methodology)
*   **Target Leakage Elimination:** Refactored feature engineering to use `shift(1)` before computing rolling statistics. This ensures no information from time $t$ is used to predict $t$.
*   **Chronological Splitting:** Replaced all random `train_test_split` with `get_chronological_split` and `TimeSeriesSplit`. Shuffling is strictly disabled.
*   **Scaling Safety:** Moved scalers inside the cross-validation loops to ensure they are fitted ONLY on training data.

### 🏗️ Architectural Changes
*   **Pipeline Consolidation:** Created `pipeline_core.py` as the single source of truth for data processing, splitting, and evaluation.
*   **Refactored Main Script:** `energy_optimization_research.py` now uses the core pipeline and orchestrates the full experiment (With/Without Lights).
*   **Deprecated Files:** Removed `enhanced_optimization.py` and `generalization_testing.py` to focus on the core, defensible scope.

### 📉 Optimization Downgrade
*   **Conceptual Correction:** Downgraded the optimization section to a "Toy/Illustrative" demonstration.
*   **Proxy Objectives:** Replaced implicit physical claims with explicit proxy functions for energy/comfort trade-offs.
*   **Disclaimers:** Added explicit warnings that the optimization does not represent a validated HVAC simulation.

### 🧪 Analysis Improvements
*   **Sensitivity Analysis:** Rewrote `sensitivity_analysis.py` to use the rigorous pipeline. Added noise robustness and permutation importance on the Test set.
*   **Nested CV:** Implemented a robust 5-fold Nested Time-Series Cross-Validation for reliable error estimation.

### 📦 Reproducibility
*   **Requirements:** Added `requirements.txt`.
*   **Seeding:** Global random seeds set for Numpy, Torch, and Sklearn.
