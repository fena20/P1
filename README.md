# Advanced Energy Optimization Research

A rigorous, leakage-free time-series forecasting and optimization framework for building energy consumption.

## 🚀 Key Features
*   **Leakage-Free Pipeline:** Strict causal feature engineering (Shift-then-Roll).
*   **Rigorous Evaluation:** Nested Time-Series Cross-Validation.
*   **Reproducibility:** Seeded random states, standardized environment.
*   **Illustrative Optimization:** Multi-objective evolutionary algorithm (NSGA-II) for energy/comfort trade-offs (Proxy-based).

## 🛠️ Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

Run the main research protocol. This script executes the end-to-end pipeline, including:
1.  Data Loading & Cleaning
2.  Feature Engineering (with leakage checks)
3.  Nested Cross-Validation (5 folds)
4.  Final Model Training (Chronological Split)
5.  Toy Optimization Demo

```bash
python3 energy_optimization_research.py
```

### Outputs
All artifacts are saved to the `outputs/` directory:
*   `table1_model_comparison.csv`: Metrics for models (With/Without Lights).
*   `nested_cv_results_*.csv`: Detailed fold-by-fold results.
*   `feature_importance/*.png`: Feature importance plots per model.
*   `figure3_toy_pareto.png`: Optimization Pareto front.

## 📊 Methodology
See [METHODOLOGY.md](METHODOLOGY.md) for a detailed explanation of the leakage prevention and evaluation protocols.

## 📂 File Structure
*   `pipeline_core.py`: **Core Logic.** Contains data loading, splitting, feature engineering, and training wrappers.
*   `energy_optimization_research.py`: **Main Entry Point.** Orchestrates the research experiments.
*   `energy_eda_analysis.py`: Exploratory Data Analysis.
*   `sensitivity_analysis.py`: Robustness checks (noise, permutation importance).
