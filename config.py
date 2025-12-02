"""
Configuration file for the Edge AI Energy Optimization Project
"""
import numpy as np

# Data paths
DATA_DIR = "data"
BUILDING_METADATA_PATH = f"{DATA_DIR}/building_metadata.csv"
WEATHER_TRAIN_PATH = f"{DATA_DIR}/weather_train.csv"
TRAIN_PATH = f"{DATA_DIR}/train.csv"

# Data filtering
PRIMARY_USE_FILTER = "Lodging/residential"
TRAIN_YEAR = 2016
TEST_YEAR = 2017
METER_TYPE = 0  # Electricity

# Model parameters
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
TRANSFORMER_D_MODEL = 128
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 4
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
TRAIN_VAL_SPLIT = 0.8

# RL parameters
RL_LEARNING_RATE = 3e-4
RL_BATCH_SIZE = 64
RL_N_STEPS = 2048
RL_N_EPOCHS = 10
RL_GAMMA = 0.99
RL_GAE_LAMBDA = 0.95
RL_CLIP_RANGE = 0.2
RL_ENT_COEF = 0.01
RL_VF_COEF = 0.5

# Environment parameters
MAX_SETPOINT_ADJUSTMENT = 2.0  # degrees Celsius
NUM_ACTIONS = 5  # discrete action space
COMFORT_PENALTY_THRESHOLD = 10.0  # PPD threshold
ENERGY_COST_PER_KWH = 0.1  # USD
CO2_EMISSION_FACTOR = 0.5  # kg CO2 per kWh

# Multi-agent parameters
NUM_AGENTS = 2  # HVAC and Lighting

# Federated learning parameters
NUM_FEDERATED_ROUNDS = 10
NUM_CLIENTS = 5
FEDERATED_FRACTION = 0.3

# Edge AI simulation
EDGE_INFERENCE_BATCH_SIZE = 1
USE_TORCHSCRIPT = True

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Figure settings
DPI = 300
FIG_SIZE = (10, 6)
FIG_STYLE = 'seaborn-v0_8-paper'

# Output paths
OUTPUT_DIR = "output"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"
TABLES_DIR = f"{OUTPUT_DIR}/tables"
MODELS_DIR = f"{OUTPUT_DIR}/models"
PAPER_DIR = f"{OUTPUT_DIR}/paper"
