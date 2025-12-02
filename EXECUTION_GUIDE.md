# Execution Guide

## Quick Start

### Option 1: Complete Pipeline (Recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run complete pipeline
python main.py
```

This will execute all components in sequence:
- Data preparation
- Deep learning training
- RL training (optional)
- Federated learning
- Optimization comparison
- Sensitivity analysis
- Paper generation

### Option 2: Individual Components

```bash
# Data preparation
python data_preparation.py

# Deep learning training
python deep_learning_model.py

# RL training (time-consuming)
python rl_training.py

# Federated learning
python federated_learning.py

# Optimization comparison
python optimization_simulation.py

# Sensitivity analysis
python sensitivity_analysis.py

# Generate paper
python paper_drafting.py
```

### Option 3: Paper Only (No Dependencies)

```bash
# Generate paper with default values
python generate_paper_standalone.py
```

## Dataset Setup

### Option A: Kaggle API (Recommended)

1. Install Kaggle API:
```bash
pip install kaggle
```

2. Set up credentials:
   - Download `kaggle.json` from Kaggle account settings
   - Place in `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. Download dataset:
```bash
python data_preparation.py
```

### Option B: Manual Download

1. Download from: https://www.kaggle.com/competitions/ashrae-energy-prediction-iii/data
2. Extract to `data/` directory:
   - `building_metadata.csv`
   - `weather_train.csv`
   - `train.csv`

### Option C: Synthetic Data (Fallback)

If download fails, the code will automatically generate synthetic data for demonstration.

## Expected Outputs

After running `main.py`, you should have:

```
output/
├── figures/
│   ├── fig0.png  # System architecture
│   ├── fig1.png  # Prediction scatter plot
│   ├── fig2.png  # Comparison bar charts
│   ├── fig3.png  # Time-series plot
│   ├── fig4.png  # Sensitivity heatmap
│   └── fig5.png  # Pareto front
├── tables/
│   ├── table1.tex  # Summary statistics
│   ├── table1.csv
│   ├── table2.tex  # Method comparison
│   └── table2.csv
├── models/
│   ├── lstm_best.pth
│   ├── ppo_*.pth
│   └── federated_model.pth
├── paper/
│   ├── paper.md  # Markdown version
│   └── paper.tex # LaTeX version
└── [other results files]
```

## Troubleshooting

### Missing Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### CUDA/GPU Issues

The code automatically falls back to CPU if GPU is not available. To force CPU:
```python
# In config.py or before imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### Memory Issues

Reduce data size in `config.py`:
- Limit number of buildings
- Reduce sequence length
- Use smaller batch sizes

### Kaggle API Issues

If Kaggle download fails:
1. Check `~/.kaggle/kaggle.json` exists
2. Verify API token is valid
3. Code will use synthetic data as fallback

## Configuration

Edit `config.py` to adjust:

- **Model parameters**: Hidden sizes, layers, learning rates
- **Training parameters**: Epochs, batch sizes, seeds
- **RL parameters**: PPO hyperparameters
- **Paths**: Data and output directories
- **Constants**: Energy costs, CO₂ factors, comfort thresholds

## Reproducibility

All random processes use `RANDOM_SEED = 42` for reproducibility. Results should be consistent across runs with the same seed.

## Performance Notes

- **DL Training**: ~10-30 minutes (CPU), ~2-5 minutes (GPU)
- **RL Training**: ~30-60 minutes per building (CPU), ~5-10 minutes (GPU)
- **Federated Learning**: ~15-30 minutes
- **Full Pipeline**: ~1-2 hours (CPU), ~15-30 minutes (GPU)

## Next Steps

1. Review generated paper: `output/paper/paper.md`
2. Check figures: `output/figures/`
3. Verify tables: `output/tables/`
4. Customize for your needs
5. Submit to Applied Energy journal

## Support

For issues or questions:
- Check `README.md` for detailed documentation
- Review `PROJECT_SUMMARY.md` for overview
- Examine code comments for implementation details
