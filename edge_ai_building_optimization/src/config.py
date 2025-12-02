"""
Configuration file for Edge AI Building Energy Optimization Project
All hyperparameters and settings are documented here for reproducibility.
"""

import os

# Random seeds for reproducibility
RANDOM_SEED = 42

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
TABLES_DIR = os.path.join(BASE_DIR, 'tables')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Data parameters
TRAIN_YEAR = 2016
TEST_YEAR = 2017
METER_TYPE = 0  # Electricity

# Deep Learning parameters
DL_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'sequence_length': 24,  # 24 hours lookback
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'comfort_loss_weight': 0.1,
}

# RL parameters
RL_CONFIG = {
    'total_timesteps': 100000,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'learning_rate': 0.0003,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
}

# Environment parameters
ENV_CONFIG = {
    'temp_setpoint_range': (18, 26),  # Celsius
    'temp_action_delta': 1.0,  # +/- 1 degree
    'lighting_levels': 5,  # 0-4 discrete levels
    'energy_price': 0.10,  # $/kWh
    'co2_factor': 0.5,  # kgCO2/kWh
    'comfort_weight': 0.3,
    'energy_weight': 0.7,
}

# Comfort parameters (PMV/PPD)
COMFORT_CONFIG = {
    'metabolic_rate': 1.2,  # met (seated office work)
    'clothing_insulation': 0.7,  # clo (typical indoor clothing)
    'air_velocity': 0.1,  # m/s
    'ppd_threshold': 10,  # % maximum acceptable PPD
}

# Federated Learning parameters
FL_CONFIG = {
    'num_clients': 5,
    'local_epochs': 5,
    'rounds': 10,
}

# Visualization settings
VIZ_CONFIG = {
    'figsize': (10, 6),
    'dpi': 300,
    'style': 'seaborn-v0_8-muted',
    'palette': 'colorblind',
    'font_size': 12,
}

# Create directories if they don't exist
for dir_path in [DATA_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)
