"""
Configuration parameters for the BDG2 Surrogate-Assisted Optimization Framework.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "bdg2_data" / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
for dir_path in [FIGURES_DIR, RESULTS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Configuration for data preprocessing."""
    # Data paths
    metadata_path: Path = DATA_ROOT / "metadata" / "metadata.csv"
    electricity_path: Path = DATA_ROOT / "meters" / "cleaned" / "electricity_cleaned.csv"
    weather_path: Path = DATA_ROOT / "weather" / "weather.csv"
    
    # Building selection criteria
    primary_use_types: List[str] = field(default_factory=lambda: ["Lodging/residential"])
    min_data_coverage: float = 0.95  # Minimum 95% data availability
    
    # Data splitting
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Time parameters
    sequence_length: int = 24  # 24 hours lookback
    prediction_horizon: int = 1  # Next hour prediction
    
    # Missing data handling
    max_gap_hours: int = 6  # Maximum gap for linear interpolation
    
    # Normalization
    normalization_method: str = "minmax"  # 'minmax' or 'standard'


@dataclass
class ModelConfig:
    """Configuration for surrogate models."""
    # LSTM parameters
    lstm_units: List[int] = field(default_factory=lambda: [64, 32])
    lstm_dropout: float = 0.2
    lstm_recurrent_dropout: float = 0.1
    lstm_batch_size: int = 32
    lstm_epochs: int = 50
    lstm_patience: int = 10
    lstm_learning_rate: float = 0.001
    
    # XGBoost parameters
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_random_state: int = 42
    
    # Lag features for XGBoost
    lag_hours: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24])


@dataclass
class OptimizationConfig:
    """Configuration for GA-based optimization."""
    # Optimization horizon
    horizon_hours: int = 24  # Day-ahead optimization
    
    # HVAC setpoint constraints
    min_setpoint: float = 19.0  # Minimum cooling setpoint (°C)
    max_setpoint: float = 26.0  # Maximum cooling setpoint (°C)
    setpoint_resolution: float = 0.5  # Setpoint granularity
    
    # Baseline controller
    baseline_setpoint: float = 23.0  # Fixed setpoint for baseline
    
    # Genetic Algorithm parameters
    population_size: int = 50
    generations: int = 100
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2
    tournament_size: int = 3
    
    # Objective weights
    comfort_weight: float = 100.0  # Weight for comfort penalty
    
    # Comfort parameters (PMV-based)
    pmv_comfort_low: float = -0.5
    pmv_comfort_high: float = 0.5
    
    # Default metabolic rate and clothing insulation
    default_met: float = 1.2  # Seated, light activity
    default_clo: float = 0.5  # Light summer clothing
    default_air_velocity: float = 0.1  # m/s
    
    # Multi-objective Pareto analysis
    n_pareto_points: int = 50


@dataclass
class ElectricityTariff:
    """Time-of-use electricity tariff structure."""
    # Peak hours (weekdays 14:00-19:00)
    peak_hours: List[int] = field(default_factory=lambda: list(range(14, 20)))
    peak_rate: float = 0.25  # $/kWh
    
    # Off-peak hours
    offpeak_rate: float = 0.10  # $/kWh
    
    # Super off-peak (night: 00:00-06:00)
    super_offpeak_hours: List[int] = field(default_factory=lambda: list(range(0, 7)))
    super_offpeak_rate: float = 0.05  # $/kWh
    
    def get_rate(self, hour: int, is_weekday: bool = True) -> float:
        """Get electricity rate for a given hour."""
        if not is_weekday:
            return self.offpeak_rate
        if hour in self.super_offpeak_hours:
            return self.super_offpeak_rate
        if hour in self.peak_hours:
            return self.peak_rate
        return self.offpeak_rate


@dataclass  
class VisualizationConfig:
    """Configuration for publication-quality visualizations."""
    figure_dpi: int = 300
    figure_format: str = "png"  # 'png', 'pdf', 'svg'
    
    # Color scheme (professional scientific palette)
    colors: Dict[str, str] = field(default_factory=lambda: {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'accent': '#F18F01',
        'success': '#C73E1D',
        'dark': '#1B1B1B',
        'light': '#E8E8E8',
        'baseline': '#7B7B7B',
        'optimized': '#2E86AB',
        'pareto': '#A23B72'
    })
    
    # Font settings
    font_family: str = "sans-serif"
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    legend_size: int = 10
    
    # Figure sizes (width, height in inches)
    single_column_width: float = 3.5
    double_column_width: float = 7.0
    standard_height: float = 3.0


# Climate zone mapping based on latitude
CLIMATE_ZONES = {
    "Hot-Humid": (25.0, 32.0),      # Latitude range
    "Hot-Dry": (30.0, 38.0),
    "Mixed-Humid": (32.0, 40.0),
    "Mixed-Dry": (35.0, 42.0),
    "Cold": (40.0, 50.0),
    "Very-Cold": (45.0, 60.0)
}


def get_climate_zone(latitude: float) -> str:
    """Determine climate zone based on latitude."""
    abs_lat = abs(latitude)
    if abs_lat < 25:
        return "Hot-Humid"
    elif abs_lat < 32:
        return "Hot-Humid" if latitude > 0 else "Hot-Dry"
    elif abs_lat < 38:
        return "Mixed-Humid"
    elif abs_lat < 45:
        return "Cold"
    else:
        return "Very-Cold"
