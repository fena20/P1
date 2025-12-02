# Surrogate-Assisted Building Energy Optimization using BDG2

This project implements a data-driven surrogate model for building energy optimization using the Building Data Genome Project 2 (BDG2) dataset. The framework combines neural network and gradient-boosted surrogate models with Genetic Algorithm (GA) optimization to enable real-time HVAC setpoint scheduling that balances energy cost and thermal comfort.

## Project Overview

The project is organized into four main phases:

1. **Phase 1 - Data Curation and Pre-Processing**: Building selection, data integration, and cleaning
2. **Phase 2 - Surrogate Model Development**: LSTM and XGBoost-based digital twin models
3. **Phase 3 - Optimization Framework**: GA-based multi-objective optimization with surrogate models
4. **Phase 4 - Results and Analysis**: Comparative analysis and visualization

## Repository Structure

```
.
├── src/
│   ├── phase1_data_curation/    # Data preprocessing and building selection
│   │   ├── building_selection.py
│   │   ├── data_integration.py
│   │   └── data_cleaning.py
│   ├── phase2_surrogate/        # Surrogate model implementations
│   │   ├── train_lstm.py
│   │   └── train_xgboost.py
│   ├── phase3_optimization/     # GA optimization framework
│   │   └── ga_optimizer.py
│   ├── phase4_results/          # Results analysis and visualization
│   │   ├── comparative_analysis.py
│   │   └── generate_figures.py
│   └── utils.py                # Utility functions
├── notebooks/                   # Jupyter notebooks for exploration
├── figures/                     # Generated figures for publication
├── data/
│   └── processed/               # Processed datasets and models
├── bdg2_data/                   # BDG2 dataset (cloned repository)
├── requirements.txt             # Python dependencies
├── run_pipeline.py              # Main pipeline script
└── README.md                    # This file
```

## Installation

1. Clone the repository and BDG2 dataset:
```bash
git clone https://github.com/buds-lab/building-data-genome-project-2.git bdg2_data
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start - Run Complete Pipeline

Run all phases sequentially:
```bash
python run_pipeline.py --all
```

Or run individual phases:
```bash
python run_pipeline.py --phase 1  # Data curation
python run_pipeline.py --phase 2  # Model training
python run_pipeline.py --phase 3  # Optimization
python run_pipeline.py --phase 4  # Results and figures
```

### Individual Phase Execution

#### Phase 1: Data Curation
```bash
python src/phase1_data_curation/building_selection.py
python src/phase1_data_curation/data_integration.py
python src/phase1_data_curation/data_cleaning.py
```

#### Phase 2: Surrogate Model Training
```bash
python src/phase2_surrogate/train_lstm.py
python src/phase2_surrogate/train_xgboost.py
```

#### Phase 3: Optimization
```bash
python src/phase3_optimization/ga_optimizer.py
```

#### Phase 4: Results and Visualization
```bash
python src/phase4_results/comparative_analysis.py
python src/phase4_results/generate_figures.py
```

## Key Features

- **Real-data-driven**: Uses BDG2 dataset with 3,053 meters from 1,636 buildings
- **Multi-building generalization**: Cross-building and cross-climate model training
- **Multi-objective optimization**: Pareto-optimal solutions for cost-comfort trade-offs
- **Tariff-aware control**: Time-of-use pricing integration
- **Computational efficiency**: Orders-of-magnitude speed-up vs. physics-based simulation

## Methodology

### Phase 1: Data Curation
- Filters residential and lodging-type buildings from BDG2 metadata
- Integrates hourly meter readings with weather data
- Handles missing data and outliers
- Splits data into train/validation/test sets (70/15/15)

### Phase 2: Surrogate Models
- **LSTM Model**: Sequence-to-one LSTM with 24-hour input window
- **XGBoost Model**: Gradient-boosted trees with engineered lag features
- Both models predict next-hour energy consumption and indoor temperature

### Phase 3: Optimization
- Multi-objective Genetic Algorithm (NSGA-II)
- Objectives: Minimize energy cost and comfort penalty
- Constraints: Setpoint bounds (19-26°C), comfort band (PMV: -0.5 to +0.5)
- Output: Pareto-optimal setpoint schedules

### Phase 4: Analysis
- Comparative analysis: Baseline (fixed setpoint) vs. Optimized controller
- Performance metrics: Energy savings, cost reduction, comfort violations
- Visualization: Framework schematic, daily profiles, Pareto fronts, cross-building performance

## Output Files

### Processed Data
- `data/processed/selected_buildings.csv`: Selected building metadata
- `data/processed/integrated_data.csv`: Merged meter and weather data
- `data/processed/train_data.csv`, `val_data.csv`, `test_data.csv`: Split datasets

### Trained Models
- `data/processed/models/lstm_model.h5`: Trained LSTM model
- `data/processed/models/xgboost_model.pkl`: Trained XGBoost model

### Optimization Results
- `data/processed/optimization/pareto_solutions.csv`: Pareto-optimal solutions
- `data/processed/results/comparative_results.csv`: Baseline vs. optimized comparison

### Figures
- `figures/figure1_framework_schematic.png`: Framework overview
- `figures/figure2_daily_optimization_profile.png`: Daily control profiles
- `figures/figure3_pareto_front.png`: Cost-comfort trade-off
- `figures/figure4_cross_building_performance.png`: Cross-building generalization

## Citation

If you use this code, please cite:

- BDG2 Dataset: Miller et al. (2020). The Building Data Genome Project 2. *Scientific Data*, 7, 368.
  ```
  @article{miller2020building,
    title={The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition},
    author={Miller, Clayton and Kathirgamanathan, Anjukan and Picchetti, Bianca and others},
    journal={Scientific Data},
    volume={7},
    pages={368},
    year={2020}
  }
  ```

## License

See LICENSE file for details.
