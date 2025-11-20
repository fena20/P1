# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Running the Framework

```bash
python multi_objective_energy_optimization.py
```

## Expected Runtime

- **Data Loading & Preprocessing**: ~30 seconds
- **Baseline Model Training**: ~1-2 minutes
- **Stacking Model Optimization**: ~5-10 minutes (30 Optuna trials)
- **Cross-Validation**: ~5-10 minutes (5-fold CV)
- **SHAP Analysis**: ~2-3 minutes
- **NSGA-II Optimization**: ~3-5 minutes (30 generations, 50 population)

**Total Runtime**: Approximately 20-30 minutes

## Output Files

After execution, you will find:

1. **shap_summary_plot.png**: Feature importance visualization
2. **pareto_front.png**: Pareto optimal solutions plot
3. **model_comparison.csv**: Performance metrics comparison table

## Customization

### Adjust Optimization Parameters

In `main()` function, modify:
- `n_trials=30`: Number of Optuna optimization trials
- `n_gen=30`: NSGA-II generations
- `pop_size=50`: NSGA-II population size

### Modify Discomfort Formula

In `EnergyOptimizationProblem.__init__()`:
- `ideal_temp=21.0`: Ideal temperature (°C)
- `ideal_rh=50.0`: Ideal relative humidity (%)
- `w1=1.0, w2=1.0`: Weights for temperature and humidity discomfort

### Change Lag Periods

In `main()` function:
```python
X_lagged = create_lag_features(X, lag_periods=[1, 2, 3])  # Add t-3
```

## Troubleshooting

### Memory Issues
- Reduce `pop_size` in NSGA-II optimization
- Reduce number of Optuna trials
- Use smaller sample for SHAP analysis (modify `X_sample` size)

### Long Runtime
- Reduce `n_trials` for Optuna
- Reduce `n_gen` and `pop_size` for NSGA-II
- Use fewer lag periods

### Missing Features
- The code auto-detects temperature/humidity features
- If detection fails, manually specify in `create_lag_features()`

## Citation

When using this framework, please cite:

```bibtex
@article{your_article,
  title={Multi-Objective Optimization Framework for Building Energy Management},
  author={Your Name},
  journal={Applied Energy},
  year={2024}
}
```
