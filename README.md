# Physics-Informed Machine Learning for Building Energy Prediction and Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxx-blue)](https://doi.org/)

## 📋 Overview

This repository contains the complete implementation for the research paper:

**"Physics-Informed Stacking Ensemble with Multi-Objective Optimization for Residential Building Energy Prediction and Thermal Comfort Management"**

*Submitted to: Applied Energy*

### Abstract

This study presents a comprehensive framework for building energy prediction and optimization that integrates physics-informed feature engineering, state-of-the-art machine learning (Quantile TabNet and Stacking Ensemble), multi-objective optimization (NSGA-II), and robust generalization testing. The framework achieves **R² = 0.875** on the UCI Appliances Energy Prediction dataset and demonstrates **14-25% energy savings potential** while maintaining thermal comfort (|PMV| < 0.3).

---

## 🏗️ Repository Structure

```
workspace/
├── 📊 Data & Models
│   ├── energydata_complete.csv          # UCI Dataset (19,735 observations)
│   ├── kitakyushu_data.xlsx             # External validation dataset
│   ├── stacking_model.joblib            # Trained Stacking model
│   ├── scaler_X.joblib                  # Feature scaler
│   └── scaler_y.joblib                  # Target scaler
│
├── 📈 Analysis Scripts
│   ├── energy_eda_analysis.py           # Exploratory Data Analysis
│   ├── energy_optimization_research.py  # Main ML pipeline
│   ├── enhanced_optimization.py         # NSGA-II optimization
│   ├── generalization_testing.py        # External validation
│   └── sensitivity_analysis.py          # Sensitivity analysis
│
├── 📊 Generated Figures
│   ├── figure1_model_comparison.png     # SOTA model comparison
│   ├── figure2_attention_masks.png      # TabNet interpretability
│   ├── figure3_pareto_front_enhanced.png# Pareto optimization
│   ├── figure4_outlier_analysis.png     # Data quality
│   ├── figure5_generalization.png       # External validation
│   ├── figure6_domain_adaptation.png    # Transfer learning
│   └── figure7_sensitivity_analysis.png # Sensitivity analysis
│
├── 📋 Tables & Reports
│   ├── table1_performance_metrics.csv
│   ├── table2_policy_implications.txt
│   ├── generalization_results.csv
│   ├── sensitivity_analysis_report.txt
│   └── macro_validation_summary.txt
│
└── 📖 Documentation
    ├── README.md                        # This file
    └── METHODOLOGY.md                   # Detailed methodology
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install numpy pandas matplotlib seaborn scikit-learn
pip install torch pytorch-tabnet xgboost lightgbm
pip install pymoo scipy joblib openpyxl
```

### Run Complete Analysis

```bash
# Step 1: Exploratory Data Analysis
python energy_eda_analysis.py

# Step 2: Model Training & Optimization
python energy_optimization_research.py

# Step 3: Enhanced Optimization (10-20% savings)
python enhanced_optimization.py

# Step 4: Generalization Testing
python generalization_testing.py

# Step 5: Sensitivity Analysis
python sensitivity_analysis.py
```

---

## 📊 Key Results

### Model Performance Comparison

| Model | RMSE (Wh) | MAE (Wh) | R² | PICP (%) |
|-------|-----------|----------|-----|----------|
| **Quantile TabNet** | **36.91** | **14.17** | **0.8754** | **95.0** |
| Stacking Ensemble | 42.46 | 19.77 | 0.8351 | - |
| XGBoost | 43.76 | 19.31 | 0.8248 | - |
| LightGBM | 42.36 | 19.74 | 0.8358 | - |
| Random Forest | 44.88 | 20.45 | 0.8157 | - |
| MLP | 45.57 | 22.82 | 0.8100 | - |

### Optimization Results (NSGA-II + TOPSIS)

| Scenario | Energy Savings | CO₂ Reduction | Thermal Comfort |
|----------|---------------|---------------|-----------------|
| TOPSIS Optimal | 14.4% | 172 kg/year | |PMV| = 0.00 |
| Maximum Savings | 25.0% | 299 kg/year | |PMV| = 0.03 |

### Generalization Performance

| Dataset | Zero-Shot R² | Few-Shot R² | Improvement |
|---------|--------------|-------------|-------------|
| Original (Belgium) | 0.763 | - | Baseline |
| European (Synthetic) | 0.456 | 0.878 | +92.5% |
| Kitakyushu (Japan) | -5.86 | 0.054 | +63.3% RMSE |

---

## 🔬 Methodology Overview

### Phase 1: Physics-Informed Feature Engineering

- **Cyclical Encoding**: sin/cos transformation for temporal variables
- **Thermodynamic Features**: Dew point (Tₐₚ), thermal gradient (ΔT), heat index
- **Lag Features**: t-1, t-2, t-3, t-6 for thermal inertia
- **Rolling Statistics**: 1-hour, 2-hour moving averages

### Phase 2: SOTA Model - Quantile TabNet

- **Architecture**: Transformer-based with sparse attention
- **Training**: Pinball loss for quantile regression [0.025, 0.5, 0.975]
- **Output**: Point prediction + 95% prediction intervals

### Phase 3: Multi-Objective Optimization

- **Algorithm**: NSGA-II (200 generations, 100 population)
- **Objectives**: Minimize energy, minimize |PMV|
- **Constraints**: Temperature setpoints ∈ [18°C, 26°C]
- **Selection**: TOPSIS for balanced solution

### Phase 4: Generalization & Sensitivity

- **External Validation**: Kitakyushu campus, synthetic European building
- **Transfer Learning**: Few-shot adaptation with 10% calibration
- **Sensitivity**: Feature ablation, hyperparameter, noise robustness

---

## 📁 Dataset Description

### UCI Appliances Energy Prediction Dataset

- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)
- **Period**: January 11 - May 27, 2016 (137 days)
- **Resolution**: 10-minute intervals
- **Observations**: 19,735
- **Location**: Low-energy house, Stambruges, Belgium

### Sensor Mapping

| Sensor | Location | Description |
|--------|----------|-------------|
| T1, RH_1 | Kitchen | Cooking area sensors |
| T2, RH_2 | Living Room | Main living space |
| T3, RH_3 | Laundry Room | Washing/drying area |
| T4, RH_4 | Office | Home office space |
| T5, RH_5 | Bathroom | Wet room sensors |
| T6, RH_6 | Building North | Exterior sensor (WSN) |
| T7, RH_7 | Ironing Room | Utility space |
| T8, RH_8 | Teenager Room | Bedroom 2 |
| T9, RH_9 | Parents Room | Master bedroom |
| T_out | Weather Station | Chièvres Airport (24km) |

---

## 📈 Figures for Publication

All figures are generated in both PNG (300 DPI) and PDF formats:

1. **Figure 1**: SOTA Model Comparison (Boxplot)
2. **Figure 2**: TabNet Attention Masks (Interpretability)
3. **Figure 3**: Pareto Front with Energy Savings Annotation
4. **Figure 4**: Data Quality and Outlier Analysis
5. **Figure 5**: Generalization Across Buildings (Bar Chart)
6. **Figure 6**: Domain Adaptation Time-Series
7. **Figure 7**: Comprehensive Sensitivity Analysis

---

## 🔧 Configuration & Reproducibility

### Random Seeds
```python
np.random.seed(42)
torch.manual_seed(42)
```

### Model Hyperparameters
```python
# TabNet
n_steps = 3
n_a = n_d = 64
gamma = 1.5
epochs = 100
batch_size = 256

# Stacking Ensemble
XGBoost: n_estimators=100, max_depth=6, lr=0.1
LightGBM: n_estimators=100, max_depth=6, lr=0.1
Meta-learner: Ridge(alpha=1.0)

# NSGA-II
pop_size = 200
n_gen = 150
crossover = SBX(prob=0.9, eta=15)
mutation = PM(eta=20)
```

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{author2024physics,
  title={Physics-Informed Stacking Ensemble with Multi-Objective 
         Optimization for Residential Building Energy Prediction 
         and Thermal Comfort Management},
  author={Author, A. and Author, B.},
  journal={Applied Energy},
  year={2024},
  publisher={Elsevier}
}
```

---

## 📖 References

1. Candanedo, L.M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81-97.

2. Arik, S.Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. *AAAI Conference on Artificial Intelligence*.

3. Deb, K., et al. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*.

4. EIA (2020). Residential Energy Consumption Survey (RECS). U.S. Energy Information Administration.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **[Author Name]** - *Primary Researcher* - [Institution]
- **[Co-Author]** - *Supervision* - [Institution]

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Belgian passive house research community
- PyTorch and scikit-learn development teams
