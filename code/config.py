"""
Configuration file for Building Energy Optimization Project
Applied Energy Journal Submission
"""

import numpy as np
import random
import torch

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Data Configuration
DATA_DIR = '../data'
FIGURES_DIR = '../figures'
TABLES_DIR = '../tables'
RESULTS_DIR = '../results'

# Dataset Parameters
TRAIN_YEAR = 2016
TEST_YEAR = 2017
NUM_BUILDINGS = 30  # Residential buildings from BDG2
BUILDING_TYPE = 'Lodging/residential'

# Building Parameters
BUILDING_AREA_RANGE = (5000, 50000)  # square feet
OCCUPANCY_RANGE = (20, 200)  # number of occupants

# Weather Parameters (realistic ranges)
TEMP_MEAN = 20.0  # Celsius
TEMP_STD = 8.0
HUMIDITY_MEAN = 60.0
HUMIDITY_STD = 15.0

# Energy Parameters
BASE_ENERGY_INTENSITY = 100  # kWh per 1000 sq ft per day
ENERGY_COST = 0.1  # $ per kWh
CO2_INTENSITY = 0.5  # kg CO2 per kWh

# Deep Learning Parameters
DL_INPUT_FEATURES = ['hour', 'day_of_week', 'air_temperature', 'humidity', 
                     'square_feet', 'occupancy', 'energy_lag1', 'energy_lag24']
DL_EPOCHS = 10  # Reduced for faster execution
DL_BATCH_SIZE = 64
DL_LEARNING_RATE = 0.001
DL_HIDDEN_SIZE = 128
DL_NUM_LAYERS = 2
DL_DROPOUT = 0.2

# Reinforcement Learning Parameters
RL_TIMESTEPS = 10000  # Reduced for faster execution
RL_LEARNING_RATE = 0.0003
RL_GAMMA = 0.99
RL_N_STEPS = 2048
RL_BATCH_SIZE = 64

# Control Parameters
HVAC_TEMP_RANGE = (18, 26)  # Celsius
HVAC_SETPOINT_DELTA = 0.5
LIGHTING_LEVELS = [0.3, 0.5, 0.7, 1.0]  # Fraction of max

# Comfort Parameters (ISO 7730)
PMV_TARGET = 0.0  # Predicted Mean Vote (-3 to +3, 0 is neutral)
PPD_THRESHOLD = 10.0  # Predicted Percentage Dissatisfied (%)
COMFORT_WEIGHT = 0.3
ENERGY_WEIGHT = 0.7

# Multi-Agent Parameters
NUM_AGENTS = 2  # HVAC and Lighting
AGENT_NAMES = ['hvac', 'lighting']

# Federated Learning Parameters
FL_NUM_CLIENTS = 5
FL_ROUNDS = 10
FL_LOCAL_EPOCHS = 5

# Visualization Parameters
DPI = 300
FIGURE_SIZE = (10, 6)
STYLE = 'seaborn-v0_8-paper'

# Statistical Parameters
CONFIDENCE_LEVEL = 0.95
NUM_BOOTSTRAP_SAMPLES = 1000
