"""
Data Preprocessing Module for Building Energy Optimization
Handles BDG2 dataset loading, preprocessing, and synthetic data generation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, Optional
import os
import warnings
warnings.filterwarnings('ignore')

from config import (
    RANDOM_SEED, DATA_DIR, TRAIN_YEAR, TEST_YEAR, 
    METER_TYPE, TABLES_DIR
)

np.random.seed(RANDOM_SEED)


def generate_synthetic_bdg2_data(
    n_buildings: int = 50,
    start_date: str = '2016-01-01',
    end_date: str = '2017-12-31'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic data mimicking BDG2 dataset structure.
    This is used when actual dataset is not available.
    
    Returns:
        building_metadata: Building characteristics
        weather_data: Weather variables
        meter_readings: Energy consumption data
    """
    print("Generating synthetic BDG2-like data...")
    
    # Generate timestamps
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Building metadata
    building_ids = [f'building_{i}' for i in range(n_buildings)]
    building_metadata = pd.DataFrame({
        'building_id': building_ids,
        'primary_use': ['Lodging/residential'] * n_buildings,
        'square_feet': np.random.uniform(500, 5000, n_buildings).astype(int),
        'year_built': np.random.randint(1950, 2015, n_buildings),
        'floor_count': np.random.randint(1, 5, n_buildings),
    })
    
    # Weather data with realistic patterns
    weather_records = []
    for ts in timestamps:
        hour = ts.hour
        day_of_year = ts.dayofyear
        
        # Seasonal temperature variation (Northern Hemisphere)
        seasonal_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        # Diurnal variation
        diurnal_temp = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
        # Random noise
        noise = np.random.normal(0, 2)
        
        air_temp = seasonal_temp + diurnal_temp + noise
        
        # Dew point (typically lower than air temp)
        dew_temp = air_temp - np.random.uniform(3, 10)
        
        # Wind speed
        wind_speed = np.abs(np.random.normal(3, 2))
        
        # Cloud cover (0-9 scale)
        cloud_cover = np.random.randint(0, 10)
        
        # Sea level pressure
        sea_level_pressure = np.random.normal(1013, 10)
        
        weather_records.append({
            'timestamp': ts,
            'air_temperature': air_temp,
            'dew_temperature': dew_temp,
            'wind_speed': wind_speed,
            'cloud_coverage': cloud_cover,
            'sea_level_pressure': sea_level_pressure,
        })
    
    weather_data = pd.DataFrame(weather_records)
    
    # Meter readings with occupancy-based patterns
    meter_records = []
    for building_id in building_ids:
        building_info = building_metadata[building_metadata['building_id'] == building_id].iloc[0]
        base_load = building_info['square_feet'] * 0.01  # Base load proportional to size
        
        for idx, ts in enumerate(timestamps):
            hour = ts.hour
            day_of_week = ts.dayofweek
            month = ts.month
            
            # Occupancy pattern (residential)
            if day_of_week < 5:  # Weekday
                if 8 <= hour < 18:
                    occupancy_factor = 0.3  # Low during work hours
                elif 18 <= hour < 23:
                    occupancy_factor = 0.9  # High in evening
                else:
                    occupancy_factor = 0.5  # Moderate at night
            else:  # Weekend
                if 9 <= hour < 22:
                    occupancy_factor = 0.8
                else:
                    occupancy_factor = 0.5
            
            # Seasonal HVAC load
            temp = weather_data.iloc[idx]['air_temperature']
            if temp < 15:  # Heating
                hvac_factor = (15 - temp) / 20
            elif temp > 24:  # Cooling
                hvac_factor = (temp - 24) / 15
            else:
                hvac_factor = 0.1
            
            # Total energy with noise
            meter_reading = base_load * (1 + occupancy_factor + hvac_factor)
            meter_reading *= np.random.uniform(0.9, 1.1)  # Random variation
            meter_reading = max(0, meter_reading)
            
            meter_records.append({
                'building_id': building_id,
                'meter': METER_TYPE,
                'timestamp': ts,
                'meter_reading': meter_reading,
            })
    
    meter_readings = pd.DataFrame(meter_records)
    
    print(f"Generated data for {n_buildings} buildings over {len(timestamps)} timestamps")
    
    return building_metadata, weather_data, meter_readings


def load_or_generate_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Try to load BDG2 data from disk, or generate synthetic data.
    """
    # Check for actual data files
    metadata_path = os.path.join(DATA_DIR, 'building_metadata.csv')
    weather_path = os.path.join(DATA_DIR, 'weather_train.csv')
    train_path = os.path.join(DATA_DIR, 'train.csv')
    
    if all(os.path.exists(p) for p in [metadata_path, weather_path, train_path]):
        print("Loading BDG2 data from disk...")
        building_metadata = pd.read_csv(metadata_path)
        weather_data = pd.read_csv(weather_path)
        meter_readings = pd.read_csv(train_path)
        
        # Filter for residential buildings
        residential_ids = building_metadata[
            building_metadata['primary_use'] == 'Lodging/residential'
        ]['building_id'].tolist()
        
        meter_readings = meter_readings[
            (meter_readings['building_id'].isin(residential_ids)) &
            (meter_readings['meter'] == METER_TYPE)
        ]
        
        return building_metadata, weather_data, meter_readings
    else:
        return generate_synthetic_bdg2_data()


def extract_occupant_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract occupant behavior proxy features from meter readings.
    """
    df = df.copy()
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Diurnal pattern features (per building)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Weekly pattern
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Seasonal pattern
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Rolling statistics per building (occupancy proxy)
    df = df.sort_values(['building_id', 'timestamp'])
    
    # Calculate rolling statistics
    for window in [6, 24]:  # 6-hour and 24-hour windows
        df[f'rolling_mean_{window}h'] = df.groupby('building_id')['meter_reading'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}h'] = df.groupby('building_id')['meter_reading'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
    
    # Fill NaN values from rolling calculations
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    # Peak detection (binary feature for high usage periods)
    threshold = df.groupby('building_id')['meter_reading'].transform(
        lambda x: x.quantile(0.75)
    )
    df['is_peak'] = (df['meter_reading'] > threshold).astype(int)
    
    return df


def preprocess_data(
    building_metadata: pd.DataFrame,
    weather_data: pd.DataFrame,
    meter_readings: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, StandardScaler]]:
    """
    Preprocess and merge all data sources.
    """
    print("Preprocessing data...")
    
    # Ensure timestamp is datetime
    if 'timestamp' in meter_readings.columns:
        meter_readings['timestamp'] = pd.to_datetime(meter_readings['timestamp'])
    if 'timestamp' in weather_data.columns:
        weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
    
    # Filter residential buildings
    residential_ids = building_metadata[
        building_metadata['primary_use'] == 'Lodging/residential'
    ]['building_id'].tolist()
    
    meter_readings = meter_readings[
        meter_readings['building_id'].isin(residential_ids)
    ].copy()
    
    # Extract occupant proxies
    meter_readings = extract_occupant_proxies(meter_readings)
    
    # Merge with weather data
    merged_df = meter_readings.merge(weather_data, on='timestamp', how='left')
    
    # Merge with building metadata
    merged_df = merged_df.merge(building_metadata, on='building_id', how='left')
    
    # Handle missing values
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if merged_df[col].isna().sum() > 0:
            merged_df[col] = merged_df[col].interpolate(method='linear')
            merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
    
    # Normalize features
    scalers = {}
    features_to_scale = [
        'air_temperature', 'dew_temperature', 'wind_speed',
        'square_feet', 'meter_reading',
        'rolling_mean_6h', 'rolling_std_6h',
        'rolling_mean_24h', 'rolling_std_24h'
    ]
    
    for feature in features_to_scale:
        if feature in merged_df.columns:
            scaler = StandardScaler()
            merged_df[f'{feature}_scaled'] = scaler.fit_transform(
                merged_df[[feature]]
            )
            scalers[feature] = scaler
    
    print(f"Preprocessed {len(merged_df)} records from {len(residential_ids)} buildings")
    
    return merged_df, scalers


def create_train_test_split(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training (2016) and test (2017) sets.
    """
    df['year'] = df['timestamp'].dt.year
    
    train_df = df[df['year'] == TRAIN_YEAR].copy()
    test_df = df[df['year'] == TEST_YEAR].copy()
    
    print(f"Training set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")
    
    return train_df, test_df


def generate_summary_statistics(
    train_df: pd.DataFrame,
    save_path: str
) -> pd.DataFrame:
    """
    Generate Table 1: Summary statistics for residential buildings.
    """
    stats_dict = {}
    
    # Energy use statistics
    energy_stats = train_df['meter_reading'].describe()
    stats_dict['Energy Use (kWh)'] = {
        'Mean': energy_stats['mean'],
        'Std': energy_stats['std'],
        'Min': energy_stats['min'],
        'Max': energy_stats['max'],
    }
    
    # Weather variables
    for var in ['air_temperature', 'dew_temperature', 'wind_speed']:
        if var in train_df.columns:
            var_stats = train_df[var].describe()
            var_name = var.replace('_', ' ').title()
            stats_dict[var_name] = {
                'Mean': var_stats['mean'],
                'Std': var_stats['std'],
                'Min': var_stats['min'],
                'Max': var_stats['max'],
            }
    
    # Occupant proxies
    stats_dict['Rolling Mean 24h (kWh)'] = {
        'Mean': train_df['rolling_mean_24h'].mean(),
        'Std': train_df['rolling_mean_24h'].std(),
        'Min': train_df['rolling_mean_24h'].min(),
        'Max': train_df['rolling_mean_24h'].max(),
    }
    
    stats_dict['Rolling Std 24h (kWh)'] = {
        'Mean': train_df['rolling_std_24h'].mean(),
        'Std': train_df['rolling_std_24h'].std(),
        'Min': train_df['rolling_std_24h'].min(),
        'Max': train_df['rolling_std_24h'].max(),
    }
    
    stats_dict['Peak Usage Ratio'] = {
        'Mean': train_df['is_peak'].mean(),
        'Std': train_df['is_peak'].std(),
        'Min': 0,
        'Max': 1,
    }
    
    stats_dict['Building Size (sq ft)'] = {
        'Mean': train_df['square_feet'].mean(),
        'Std': train_df['square_feet'].std(),
        'Min': train_df['square_feet'].min(),
        'Max': train_df['square_feet'].max(),
    }
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_dict).T
    stats_df = stats_df.round(3)
    stats_df.index.name = 'Variable'
    stats_df = stats_df.reset_index()
    
    # Save to CSV
    stats_df.to_csv(save_path, index=False)
    print(f"Table 1 saved to {save_path}")
    
    return stats_df


if __name__ == "__main__":
    # Test the data preprocessing
    building_metadata, weather_data, meter_readings = load_or_generate_data()
    merged_df, scalers = preprocess_data(building_metadata, weather_data, meter_readings)
    train_df, test_df = create_train_test_split(merged_df)
    
    table1_path = os.path.join(TABLES_DIR, 'table1.csv')
    stats_df = generate_summary_statistics(train_df, table1_path)
    print("\nTable 1 - Summary Statistics:")
    print(stats_df)
