"""
Utility functions for the project.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def calculate_pmv(air_temp, mean_radiant_temp, air_speed, rel_humidity, 
                  metabolic_rate=1.0, clothing_insulation=0.5):
    """
    Calculate Predicted Mean Vote (PMV) for thermal comfort.
    
    Parameters:
    -----------
    air_temp : float or np.ndarray
        Air temperature (°C)
    mean_radiant_temp : float or np.ndarray
        Mean radiant temperature (°C)
    air_speed : float or np.ndarray
        Air speed (m/s)
    rel_humidity : float or np.ndarray
        Relative humidity (%)
    metabolic_rate : float
        Metabolic rate (met), default 1.0
    clothing_insulation : float
        Clothing insulation (clo), default 0.5
    
    Returns:
    --------
    float or np.ndarray
        PMV value
    """
    # Simplified PMV calculation (ASHRAE 55)
    # Full implementation would use Fanger's equations
    
    # Convert to arrays for vectorized operations
    air_temp = np.asarray(air_temp)
    mean_radiant_temp = np.asarray(mean_radiant_temp)
    
    # Operative temperature
    t_op = 0.5 * (air_temp + mean_radiant_temp)
    
    # Simplified PMV (linear approximation)
    # PMV = 0 indicates thermal neutrality
    # PMV > 0 indicates warm, PMV < 0 indicates cool
    pmv = 0.303 * np.exp(-0.036 * metabolic_rate) + 0.028
    
    # Temperature effect
    pmv += 0.1 * (t_op - 22.0)
    
    # Humidity effect (simplified)
    if isinstance(rel_humidity, (int, float)):
        if rel_humidity > 60:
            pmv += 0.1
        elif rel_humidity < 30:
            pmv -= 0.1
    
    return pmv

def create_time_of_use_tariff(peak_hours_start=9, peak_hours_end=21, 
                              super_peak_start=18, super_peak_end=21,
                              base_price=0.10, peak_price=0.15, super_peak_price=0.20):
    """
    Create time-of-use electricity price schedule.
    
    Parameters:
    -----------
    peak_hours_start : int
        Start of peak hours (default: 9)
    peak_hours_end : int
        End of peak hours (default: 21)
    super_peak_start : int
        Start of super peak hours (default: 18)
    super_peak_end : int
        End of super peak hours (default: 21)
    base_price : float
        Base electricity price ($/kWh)
    peak_price : float
        Peak electricity price ($/kWh)
    super_peak_price : float
        Super peak electricity price ($/kWh)
    
    Returns:
    --------
    np.ndarray
        24-hour price schedule
    """
    prices = np.full(24, base_price)
    prices[peak_hours_start:peak_hours_end] = peak_price
    prices[super_peak_start:super_peak_end] = super_peak_price
    return prices

def load_bdg2_metadata(metadata_path=None):
    """
    Load BDG2 building metadata.
    
    Parameters:
    -----------
    metadata_path : Path, optional
        Path to metadata file
    
    Returns:
    --------
    pd.DataFrame
        Building metadata
    """
    if metadata_path is None:
        metadata_path = Path(__file__).parent.parent / "bdg2_data" / "data" / "metadata" / "metadata.csv"
    
    return pd.read_csv(metadata_path)

def validate_setpoint_schedule(setpoint_schedule, min_temp=19.0, max_temp=26.0):
    """
    Validate and clip setpoint schedule to acceptable bounds.
    
    Parameters:
    -----------
    setpoint_schedule : np.ndarray
        Setpoint schedule
    min_temp : float
        Minimum allowed temperature (°C)
    max_temp : float
        Maximum allowed temperature (°C)
    
    Returns:
    --------
    np.ndarray
        Validated setpoint schedule
    """
    return np.clip(setpoint_schedule, min_temp, max_temp)
