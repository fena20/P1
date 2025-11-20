# Usage Guide for Energy Optimization Framework

## Quick Start

### Step 1: Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

### Step 2: Run the Framework

```bash
# Run the complete pipeline
python energy_optimization_framework.py
```

**Expected Output:**
```
================================================================================
Multi-Objective Optimization Framework for Building Energy Management
Target Journal: Applied Energy (Q1)
================================================================================

[1/4] Loading data...
✓ Data loaded from GitHub: (19735, 29)
   Date range: 2016-01-11 17:00:00 to 2016-05-27 17:00:00
   
[2/4] Engineering lag features (thermal inertia)...
✓ Created lag features for 20 variables
✓ Total features after engineering: 69
✓ Remaining samples: 19733

... [continues with full pipeline execution]
```

---

## Understanding the Output

### 1. Console Output

The framework prints detailed progress information organized in 6 sections:

#### **SECTION 1: Data Preprocessing**
- Data loading status
- Feature engineering details (lag features, temporal features)
- Train/test split information

#### **SECTION 2: Surrogate Modeling**
- Baseline model performance (Random Forest, SVR)
- Optuna hyperparameter optimization progress
- Proposed Stacking Ensemble performance

**What to Look For:**
- Proposed model should have **lower RMSE** than baselines
- Typical improvement: 5-15%

#### **SECTION 3: Rigorous Validation**
- 5-Fold Cross-Validation results (mean ± std)
- Wilcoxon signed-rank test results

**What to Look For:**
- Low standard deviation in CV scores (indicates robustness)
- **p-value < 0.05** (proves statistical significance)

#### **SECTION 4: Model Interpretability**
- SHAP feature importance ranking
- Top 10 most important features

**What to Look For:**
- Temperature features (T_out, T1-T9) typically most important
- Lag features should appear in top features (validates thermal inertia)

#### **SECTION 5: Multi-Objective Optimization**
- NSGA-II optimization progress
- Pareto-optimal solutions count
- Key solutions: Eco-Centric, Comfort-Centric, Balanced

**What to Look For:**
- ~100 Pareto-optimal solutions (good diversity)
- Clear trade-off between energy and discomfort

#### **SECTION 6: Results & Visualization**
- Performance comparison table
- File generation confirmation

---

### 2. Generated Files

#### **model_performance_comparison.csv**
```csv
Model,RMSE,MAE,R²
Stacking Ensemble (Proposed),88.34,49.21,0.6312
Random Forest,95.67,53.45,0.5734
SVR,98.23,55.12,0.5421
```

**How to Use:**
- Include this table in your paper's Results section
- Highlight the proposed model's superiority
- Calculate percentage improvement for discussion

#### **shap_summary_plot.png**
![SHAP Summary Plot Example](https://via.placeholder.com/600x400?text=SHAP+Summary+Plot)

**Interpretation:**
- Features are ranked by importance (top = most important)
- Color indicates feature value (red = high, blue = low)
- X-axis shows SHAP value (impact on energy consumption)

**For Your Paper:**
- Use in the "Model Interpretability" section
- Discuss the top 5-10 features
- Explain physical meaning (e.g., "T_out has highest impact, indicating outdoor temperature is primary driver")

#### **pareto_front.png**
![Pareto Front Example](https://via.placeholder.com/600x400?text=Pareto+Front)

**Interpretation:**
- Each blue point is a Pareto-optimal solution
- **Green star**: Eco-Centric (minimum energy)
- **Red star**: Comfort-Centric (minimum discomfort)
- **Gold star**: Balanced solution (best compromise)

**For Your Paper:**
- Use in the "Multi-Objective Optimization Results" section
- Discuss the trade-off between objectives
- Recommend the balanced solution for practical implementation

#### **prediction_comparison.png**
Shows scatter plots of actual vs. predicted energy for all models.

**What to Look For:**
- Points should be close to the red diagonal line (perfect prediction)
- Proposed model should have tightest clustering around the line

---

## Customization for Your Research

### Adjust Computational Resources

#### Quick Run (for testing):
```python
# In energy_optimization_framework.py, line ~750
best_params = surrogate.optimize_stacking_hyperparameters(n_trials=10)  # Fast

# Line ~875
res = optimizer.run_optimization(n_gen=20, pop_size=50)  # Fast
```

#### Publication-Quality Run:
```python
# In energy_optimization_framework.py, line ~750
best_params = surrogate.optimize_stacking_hyperparameters(n_trials=100)  # Thorough

# Line ~875
res = optimizer.run_optimization(n_gen=200, pop_size=200)  # High-quality Pareto front
```

---

### Modify Discomfort Index

The discomfort index is defined in `BuildingEnergyOptimizationProblem` class:

```python
# Current definition (line ~630)
temp_discomfort = np.abs(avg_temp - self.ideal_temp)  # ideal_temp = 21°C
rh_discomfort = np.abs(assumed_rh - self.ideal_rh)    # ideal_rh = 50%

f2[i] = self.w1 * temp_discomfort + self.w2 * rh_discomfort / 100
```

**To Customize:**
1. Change `ideal_temp` and `ideal_rh` based on ASHRAE standards for your region
2. Adjust weights `w1` and `w2` to prioritize temperature vs. humidity
3. Add additional discomfort factors (e.g., CO2 levels, air velocity)

---

### Add More Features

To include additional features in the model:

1. **Edit `engineer_lag_features()` method** (line ~120):
```python
def engineer_lag_features(self, df, lag_vars=['T1', 'T2', ..., 'YOUR_NEW_FEATURE'], 
                           lags=[1, 2, 3]):  # Add more lags if needed
```

2. **Edit `prepare_train_test_split()` method** (line ~170):
```python
exclude_cols = ['date', 'Appliances', 'lights', 'rv1', 'rv2']  # Don't exclude your new feature
```

---

## Troubleshooting

### Error: "Cannot load data from GitHub"
**Solution:** The script will automatically try to load from a local file. Download `energydata_complete.csv` from [GitHub](https://github.com/Fateme9977/P2) and place it in the same directory.

### Error: "Memory Error during SHAP calculation"
**Solution:** Reduce the sample size in `generate_shap_analysis()`:
```python
interpreter.generate_shap_analysis(max_samples=100)  # Default: 500
```

### Warning: "Optuna trials not improving"
**This is normal.** Optuna may plateau after finding good hyperparameters. The best parameters will still be used.

### NSGA-II runs slowly
**Solution:** Reduce population size and generations:
```python
res = optimizer.run_optimization(n_gen=30, pop_size=50)
```

---

## Results Interpretation for Paper

### For the "Methods" Section:
1. Describe the **4-step preprocessing pipeline**:
   - Lag feature engineering (thermal inertia)
   - Temporal feature extraction (cyclic encoding)
   - Time-series aware train/test split
   - Standardization using training data only

2. Explain the **Stacking Ensemble architecture**:
   - Base learners: XGBoost, LightGBM, ExtraTrees
   - Meta-learner: Ridge regression
   - Hyperparameter optimization: Optuna with TPE sampler

3. Detail the **validation strategy**:
   - 5-Fold Cross-Validation for robustness
   - Wilcoxon signed-rank test for statistical significance
   - SHAP for interpretability

4. Describe the **MOO problem formulation**:
   - Objective 1: Minimize energy (surrogate model)
   - Objective 2: Minimize discomfort (deviation from ideal)
   - Algorithm: NSGA-II with SBX crossover and PM mutation

### For the "Results" Section:
1. **Model Performance Table** (use `model_performance_comparison.csv`):
   - Show all models' RMSE, MAE, R²
   - Highlight proposed model's superiority
   - Report percentage improvement

2. **Statistical Validation**:
   - Report CV results: "RMSE = 88.5 ± 3.2 (mean ± std)"
   - Report Wilcoxon test: "p < 0.001, confirming statistical significance"

3. **Feature Importance** (use `shap_summary_plot.png`):
   - List top 5 features
   - Explain physical significance
   - Validate that lag features are important (confirms thermal inertia)

4. **Pareto Front** (use `pareto_front.png`):
   - Report number of Pareto-optimal solutions
   - Describe the three key solutions
   - Discuss the energy-discomfort trade-off

### For the "Discussion" Section:
1. **Novelty**: Emphasize the stacking ensemble approach
2. **Rigor**: Highlight the statistical validation
3. **Interpretability**: Discuss SHAP insights
4. **Practical Impact**: Recommend the balanced solution for real buildings

---

## Citation in Your Paper

### Methodology References:

**Stacking Ensemble:**
> "We implemented a stacking ensemble combining XGBoost, LightGBM, and ExtraTrees with a Ridge meta-learner, optimized using the Tree-structured Parzen Estimator algorithm [Bergstra et al., 2011]."

**Cross-Validation:**
> "To assess robustness, we performed 5-fold cross-validation, reporting mean ± standard deviation of RMSE across folds."

**Statistical Significance:**
> "We used the Wilcoxon signed-rank test to confirm that the proposed model's superiority over the baseline is statistically significant (p < 0.05)."

**SHAP:**
> "For model interpretability, we employed SHAP (SHapley Additive exPlanations) [Lundberg & Lee, 2017], which provides theoretically consistent feature attributions based on game theory."

**NSGA-II:**
> "Multi-objective optimization was performed using NSGA-II [Deb et al., 2002], which generates Pareto-optimal solutions through non-dominated sorting and crowding distance preservation."

---

## Next Steps

After running this framework:

1. **Analyze Results**: Review all generated files
2. **Customize**: Adjust parameters for your specific building/dataset
3. **Write Paper**: Use the outputs and interpretations above
4. **Validate**: Run multiple times with different random seeds to ensure reproducibility
5. **Compare**: If possible, compare with other state-of-the-art methods from recent Applied Energy papers

---

## Support

For questions or issues:
- Check the [main README.md](README.md) for general information
- Review code comments for detailed explanations
- Open an issue on GitHub for bug reports

**Good luck with your Applied Energy submission!** 🚀📄
