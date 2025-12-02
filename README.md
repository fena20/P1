# Edge AI with Hybrid RL and Deep Learning for Occupant-Centric Optimization of Energy Consumption in Residential Buildings

This repository contains a research-grade implementation of a hybrid deep learning and reinforcement learning framework for occupant-centric energy optimization in residential buildings, using the Building Data Genome Project 2 (BDG2) / ASHRAE Great Energy Predictor III dataset.

## Project structure
- `data/`: raw and processed data (BDG2 / ASHRAE)
- `src/`: Python modules for data preparation, modeling, RL environment, simulation, and analysis
- `notebooks/`: Jupyter notebooks for end-to-end experiments and paper reproduction
- `figures/`: publication-ready figures (PNG/EPS, 300 DPI)
- `tables/`: LaTeX tables (e.g., `table1.tex`, `table2.tex`)
- `paper/`: paper draft in Markdown/LaTeX for Applied Energy submission
- `models/`: saved deep learning and RL models (PyTorch / Stable-Baselines3)

## Getting started
1. Obtain the Building Data Genome Project 2 / ASHRAE Great Energy Predictor III dataset from Kaggle.
2. Place the downloaded CSV files (e.g., `building_metadata.csv`, `weather_train.csv`, `train.csv`) into the `data/` directory.
3. Run the main pipeline scripts and notebooks in `src/` and `notebooks/` to reproduce the experiments and figures.

All code targets Python 3 and relies on standard scientific Python libraries (NumPy, pandas, matplotlib, seaborn), PyTorch for deep learning, and Stable-Baselines3 + Gym for reinforcement learning.
