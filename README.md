# Multi-Objective Optimization Framework for Building Energy Management

## Overview

This repository contains a comprehensive Python framework for multi-objective optimization of building energy management systems, targeting publication in **Applied Energy (Q1)**. The framework implements advanced surrogate modeling, rigorous validation, model interpretability, and multi-objective optimization using NSGA-II.

## Features

### 1. Advanced Preprocessing & Feature Engineering
- **Lag Features**: Implements temporal lag features (t-1, t-2) for temperature and humidity variables to capture thermal inertia
- **Scientific Justification**: Thermal inertia is fundamental to building energy systems; lag features capture delayed responses essential for accurate prediction

### 2. Surrogate Modeling
- **Baseline Models**: Random Forest and SVR for comparison
- **Proposed Model**: Stacking Ensemble combining XGBoost, LightGBM, and Extra Trees with meta-learner
- **Hyperparameter Tuning**: Optuna-based optimization using Tree-structured Parzen Estimator (TPE)

### 3. Rigorous Validation (Q1 Journal Standard)
- **5-Fold Cross-Validation**: Robust performance assessment across multiple train/test splits
- **Statistical Significance Testing**: Wilcoxon Signed-Rank Test to prove improvement is statistically significant (p < 0.05)

### 4. Model Interpretability (XAI)
- **SHAP Analysis**: SHapley Additive exPlanations for feature importance and model interpretation
- Identifies key drivers of energy consumption (e.g., T_out, RH_1)

### 5. Multi-Objective Optimization (NSGA-II)
- **Objective 1**: Minimize Energy Consumption (using trained surrogate model)
- **Objective 2**: Minimize Thermal Discomfort Index
  - Formula: `Discomfort = w1 * |T - 21°C| + w2 * |RH - 50%|`
- **Constraints**: Realistic bounds on thermostat setpoints (18°C to 26°C)

### 6. Visualization & Reporting
- **Pareto Front**: Visualization of energy-comfort trade-offs
- **Key Solutions**: Highlights Eco-Centric, Comfort-Centric, and Balanced (Knee Point) solutions
- **Comparison Tables**: Comprehensive performance metrics (RMSE, R², MAE)

## Dataset

The framework uses the energy consumption dataset from:
- **Repository**: https://github.com/Fateme9977/P2
- **File**: `energydata_complete.csv`
- **Target Variable**: `Appliances` (Energy consumption in Wh)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python multi_objective_energy_optimization.py
```

The script will:
1. Load and preprocess the data
2. Train baseline and proposed models
3. Perform cross-validation and statistical testing
4. Generate SHAP interpretability plots
5. Run NSGA-II optimization
6. Create visualizations and comparison tables

## Output Files

- `shap_summary_plot.png`: SHAP feature importance visualization
- `pareto_front.png`: Pareto front showing energy-comfort trade-offs
- `model_comparison.csv`: Performance comparison table for all models

## Scientific Methodology

### Why 5-Fold Cross-Validation?
Cross-validation provides robust performance estimates that are not sensitive to specific train/test splits, demonstrating model stability and generalizability (essential for Q1 journal standards).

### Why Wilcoxon Signed-Rank Test?
A non-parametric statistical test that determines if improvements are statistically significant (p < 0.05), providing rigorous evidence beyond simple metric comparisons.

### Why Lag Features?
Thermal inertia causes delayed responses in building energy systems. Lag features (t-1, t-2) capture these temporal dependencies, which are critical for accurate energy prediction.

### Why Stacking Ensemble?
Combines complementary strengths of different algorithms:
- XGBoost: Complex interaction handling
- LightGBM: Fast, accurate gradient boosting
- Extra Trees: Robust variance reduction
- Meta-learner: Optimal combination learning

### Why NSGA-II?
A well-established evolutionary algorithm for multi-objective optimization, widely used in energy system optimization (Applied Energy, Energy & Buildings). Efficiently explores Pareto-optimal solutions.

## Model Performance

The framework reports:
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)
- **MAE** (Mean Absolute Error)
- **Cross-Validation Statistics** (Mean ± Std)
- **Statistical Significance** (p-value)

## Key Solutions on Pareto Front

1. **Eco-Centric**: Minimum energy consumption
2. **Comfort-Centric**: Minimum thermal discomfort
3. **Balanced (Knee Point)**: Optimal trade-off between objectives

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{your_article,
  title={Multi-Objective Optimization Framework for Building Energy Management},
  author={Your Name},
  journal={Applied Energy},
  year={2024}
}
```

## License

[Specify your license]

## Contact

[Your contact information]

## Acknowledgments

- Dataset: Fateme9977/P2 repository
- Libraries: scikit-learn, XGBoost, LightGBM, Optuna, SHAP, pymoo
