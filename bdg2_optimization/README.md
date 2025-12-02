# BDG2 Surrogate-Assisted Building Energy Optimization Framework

A comprehensive framework for building energy optimization using the **Building Data Genome Project 2 (BDG2)** dataset. This project implements a data-driven surrogate model combined with Genetic Algorithm (GA) optimization for real-time HVAC setpoint scheduling while balancing energy cost and thermal comfort.

## 📚 Research Context

This framework supports research on:
- **Real-data-driven surrogate optimization** using the BDG2 dataset (3,053 meters from 1,636 buildings)
- **Unified digital-twin plus GA framework** for day-ahead HVAC control
- **Cross-building and cross-climate generalization** analysis
- **Multi-objective cost-comfort optimization** with Pareto analysis

## 🏗️ Project Structure

```
bdg2_optimization/
├── src/
│   ├── config.py              # Configuration parameters
│   ├── data_preprocessing.py  # BDG2 data loading and preprocessing
│   ├── surrogate_models.py    # LSTM and XGBoost surrogate models
│   ├── optimization.py        # GA-based optimization framework
│   ├── visualization.py       # Publication-quality figure generation
│   └── main.py                # Main execution script
├── figures/                   # Generated publication figures
├── results/                   # Output tables and metrics
├── models/                    # Trained model files
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Installation

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure the BDG2 dataset is available:

```bash
git clone https://github.com/buds-lab/building-data-genome-project-2.git bdg2_data
```

## 🚀 Usage

Run the complete optimization pipeline:

```bash
cd src
python main.py
```

This executes:
1. **Phase 1**: Data curation and preprocessing from BDG2
2. **Phase 2**: Surrogate model training (XGBoost)
3. **Phase 3**: GA-based setpoint optimization
4. **Phase 4**: Results analysis and visualization

## 📊 Generated Outputs

### Tables

| Table | Description | File |
|-------|-------------|------|
| Table 1 | Selected Case Study Buildings | `results/table1_buildings.csv` |
| Table 2 | Input Variables for Prediction Model | `results/table2_variables.csv` |
| Table 3 | Optimization Constraints | `results/table3_optimization.csv` |
| Table 4 | Comparative Results Summary | `results/table4_results.csv` |

### Figures (Publication-Ready, 300 DPI)

| Figure | Description | File |
|--------|-------------|------|
| Figure 1 | Framework Schematic | `figures/figure1_framework.png` |
| Figure 2 | Daily Optimization Profile | `figures/figure2_daily_optimization.png` |
| Figure 3 | Pareto Front Analysis | `figures/figure3_pareto_front.png` |
| Figure 4 | Cross-Building Performance | `figures/figure4_cross_building.png` |

## 🧠 Methodology

### Surrogate Model

The framework uses an **XGBoost-based surrogate** with engineered features:
- **Lag features**: Energy consumption at t-1, t-2, t-3, t-6, t-12, t-24
- **Rolling statistics**: 6-hour and 24-hour means, max, min, std
- **Temporal features**: Cyclical encoding of hour, day-of-week, month
- **Weather features**: Temperature, humidity, heating/cooling degree hours

### Optimization

**Genetic Algorithm Configuration:**
- Population size: 50
- Generations: 100
- Crossover probability: 0.8
- Mutation probability: 0.2

**Objective Function:**
```
J = C_energy + w · D_comfort
```

Where:
- `C_energy`: Total energy cost based on time-of-use tariff
- `D_comfort`: PMV-based comfort penalty
- `w`: Comfort weight (default: 100)

**Constraints:**
- Setpoint range: 19°C ≤ T_set ≤ 26°C
- Comfort band: -0.5 ≤ PMV ≤ +0.5

### Time-of-Use Tariff

| Period | Hours | Rate ($/kWh) |
|--------|-------|--------------|
| Super Off-Peak | 00:00-06:00 | $0.05 |
| Off-Peak | 06:00-14:00, 19:00-24:00 | $0.10 |
| Peak | 14:00-19:00 | $0.25 |

## 📈 Sample Results

From the demonstration run:

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Comfort Violations (hours) | 17 | 2 | 88.2% ↓ |
| Computation Time | ~1 hour (physics sim) | ~34 seconds | 99.1% ↓ |

**Note**: The tradeoff between energy and comfort depends on climate conditions. In cold climates, improving comfort may require additional heating energy. The Pareto front analysis reveals all non-dominated solutions.

## 📦 Dependencies

- **Core**: pandas, numpy, scikit-learn
- **Machine Learning**: xgboost, tensorflow
- **Optimization**: deap (Distributed Evolutionary Algorithms in Python)
- **Thermal Comfort**: pythermalcomfort
- **Visualization**: matplotlib, seaborn

## 📖 Citation

If you use this framework, please cite:

```bibtex
@dataset{Miller2020,
  author = {Miller, Clayton and Meggers, Forrest},
  title = {Building Data Genome Project 2},
  year = {2020},
  publisher = {Scientific Data, Nature Research},
  doi = {10.1038/s41597-020-00712-x}
}
```

## 📝 License

This project is provided for research purposes. The BDG2 dataset is subject to its own license terms.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

*Developed for research on surrogate-assisted building energy optimization using the BDG2 dataset.*
