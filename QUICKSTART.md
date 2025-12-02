# Quick Start Guide

This guide will help you get started with the Surrogate-Assisted Building Energy Optimization project.

## Prerequisites

- Python 3.8 or higher
- Git (for cloning BDG2 dataset)
- ~2GB disk space (for BDG2 data)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Verify BDG2 Data

The BDG2 dataset should be cloned in the `bdg2_data/` directory. If not already present:

```bash
git clone https://github.com/buds-lab/building-data-genome-project-2.git bdg2_data
```

**Note**: The BDG2 dataset uses Git LFS (Large File Storage). Ensure you have Git LFS installed:
```bash
git lfs install
cd bdg2_data
git lfs pull
```

## Step 3: Run the Pipeline

### Option A: Run Complete Pipeline

Execute all phases sequentially:
```bash
python run_pipeline.py --all
```

### Option B: Run Individual Phases

**Phase 1 - Data Curation**:
```bash
python run_pipeline.py --phase 1
```

This will:
- Select residential/lodging buildings from BDG2 metadata
- Integrate meter readings with weather data
- Clean and normalize the data
- Split into train/validation/test sets

**Phase 2 - Model Training**:
```bash
python run_pipeline.py --phase 2
```

This will:
- Train LSTM surrogate model
- Train XGBoost surrogate model
- Save trained models to `data/processed/models/`

**Phase 3 - Optimization**:
```bash
python run_pipeline.py --phase 3
```

This will:
- Run GA optimization with surrogate models
- Generate Pareto-optimal solutions
- Save results to `data/processed/optimization/`

**Phase 4 - Results**:
```bash
python run_pipeline.py --phase 4
```

This will:
- Compare baseline vs. optimized controllers
- Generate publication-quality figures
- Save results to `data/processed/results/` and `figures/`

## Step 4: View Results

### Generated Figures

Check the `figures/` directory for:
- `figure1_framework_schematic.png`: Framework overview
- `figure2_daily_optimization_profile.png`: Daily control profiles
- `figure3_pareto_front.png`: Cost-comfort trade-off
- `figure4_cross_building_performance.png`: Cross-building results

### Results Files

- `data/processed/results/comparative_results.csv`: Performance comparison
- `data/processed/optimization/pareto_solutions.csv`: Pareto-optimal solutions

## Troubleshooting

### Issue: BDG2 data files not found

**Solution**: Ensure Git LFS is installed and data files are pulled:
```bash
cd bdg2_data
git lfs pull
```

### Issue: Memory errors during training

**Solution**: Reduce batch size or sequence length in model training scripts, or use a subset of buildings.

### Issue: Model files not found in Phase 3

**Solution**: Ensure Phase 2 completed successfully. Check `data/processed/models/` for model files.

### Issue: Import errors

**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Customization

### Modify Building Selection Criteria

Edit `src/phase1_data_curation/building_selection.py`:
- Change `residential_keywords` list
- Adjust `min_coverage` threshold

### Adjust Model Architecture

Edit model training scripts:
- `src/phase2_surrogate/train_lstm.py`: Modify LSTM layers, units, dropout
- `src/phase2_surrogate/train_xgboost.py`: Adjust XGBoost parameters

### Change Optimization Parameters

Edit `src/phase3_optimization/ga_optimizer.py`:
- Modify `n_population` and `n_generations`
- Adjust setpoint bounds (`setpoint_min`, `setpoint_max`)
- Change comfort weight (`comfort_weight`)

### Customize Price Schedule

Edit `src/phase3_optimization/ga_optimizer.py`:
- Modify `create_price_schedule()` function
- Adjust TOU pricing structure

## Next Steps

1. **Explore the Data**: Use Jupyter notebooks in `notebooks/` for data exploration
2. **Hyperparameter Tuning**: Optimize model parameters for your specific use case
3. **Sensitivity Analysis**: Vary comfort weight to explore Pareto front
4. **Cross-Validation**: Evaluate across multiple buildings and climate zones

## Getting Help

- Check `README.md` for detailed documentation
- Review `PROJECT_SUMMARY.md` for implementation details
- Examine source code comments for algorithm explanations

## Citation

If you use this code in your research, please cite:

Miller, C., Kathirgamanathan, A., Picchetti, B. et al. The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. *Scientific Data* 7, 368 (2020).
