"""
Data Preparation Module for BDG2 Dataset
Handles downloading, filtering, preprocessing, and feature engineering
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import *

def ensure_directories():
    """Create necessary directories"""
    for dir_path in [DATA_DIR, OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, PAPER_DIR]:
        os.makedirs(dir_path, exist_ok=True)

def download_bdg2_data():
    """
    Download BDG2 dataset from Kaggle
    Note: Requires Kaggle API credentials (kaggle.json)
    """
    print("Checking for BDG2 dataset...")
    
    # Check if data already exists
    if os.path.exists(BUILDING_METADATA_PATH) and os.path.exists(WEATHER_TRAIN_PATH) and os.path.exists(TRAIN_PATH):
        print("Data files already exist. Skipping download.")
        return
    
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print("Downloading BDG2 dataset from Kaggle...")
        api.competition_download_files('ashrae-energy-prediction-iii', path=DATA_DIR, unzip=True)
        print("Download complete!")
        
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Please ensure kaggle.json is in ~/.kaggle/")
        print("Or manually download from: https://www.kaggle.com/competitions/ashrae-energy-prediction-iii/data")
        print("Creating synthetic data for demonstration...")
        create_synthetic_data()

def create_synthetic_data():
    """Create synthetic BDG2-like data for demonstration if download fails"""
    print("Generating synthetic residential building data...")
    
    # Generate building metadata
    n_buildings = 50
    building_ids = range(1, n_buildings + 1)
    
    metadata = pd.DataFrame({
        'building_id': building_ids,
        'site_id': np.random.randint(1, 5, n_buildings),
        'primary_use': ['Lodging/residential'] * n_buildings,
        'square_feet': np.random.normal(50000, 20000, n_buildings).astype(int),
        'year_built': np.random.randint(1950, 2020, n_buildings),
        'floor_count': np.random.randint(1, 5, n_buildings)
    })
    metadata['square_feet'] = np.abs(metadata['square_feet'])
    
    # Generate weather data (2016-2017)
    dates = pd.date_range('2016-01-01', '2017-12-31 23:00:00', freq='H')
    n_sites = metadata['site_id'].nunique()
    
    weather_data = []
    for site_id in range(1, n_sites + 1):
        for date in dates:
            weather_data.append({
                'timestamp': date,
                'site_id': site_id,
                'air_temperature': 20 + 10 * np.sin(2 * np.pi * date.dayofyear / 365) + np.random.normal(0, 3),
                'dew_temperature': 15 + 8 * np.sin(2 * np.pi * date.dayofyear / 365) + np.random.normal(0, 2),
                'sea_level_pressure': 1013 + np.random.normal(0, 5),
                'wind_direction': np.random.uniform(0, 360),
                'wind_speed': np.random.uniform(0, 15),
                'cloud_coverage': np.random.uniform(0, 10)
            })
    
    weather_df = pd.DataFrame(weather_data)
    
    # Generate meter readings
    train_data = []
    for building_id in building_ids:
        building_meta = metadata[metadata['building_id'] == building_id].iloc[0]
        site_id = building_meta['site_id']
        
        for date in dates:
            # Get corresponding weather
            weather = weather_df[(weather_df['site_id'] == site_id) & 
                                (weather_df['timestamp'] == date)].iloc[0]
            
            # Simulate energy consumption based on weather and building characteristics
            base_load = building_meta['square_feet'] / 1000
            temp_effect = abs(weather['air_temperature'] - 22) * 0.5  # Heating/cooling load
            diurnal_effect = 0.3 * np.sin(2 * np.pi * date.hour / 24) + 0.7
            seasonal_effect = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
            
            meter_reading = (base_load + temp_effect) * diurnal_effect * seasonal_effect
            meter_reading += np.random.normal(0, 0.1 * meter_reading)
            meter_reading = max(0, meter_reading)
            
            train_data.append({
                'building_id': building_id,
                'meter': METER_TYPE,
                'timestamp': date,
                'meter_reading': meter_reading
            })
    
    train_df = pd.DataFrame(train_data)
    
    # Save synthetic data
    metadata.to_csv(BUILDING_METADATA_PATH, index=False)
    weather_df.to_csv(WEATHER_TRAIN_PATH, index=False)
    train_df.to_csv(TRAIN_PATH, index=False)
    
    print(f"Synthetic data created: {n_buildings} buildings, {len(dates)} timestamps")

def load_and_filter_data():
    """Load and filter residential building data"""
    print("Loading BDG2 data...")
    
    # Load datasets
    building_meta = pd.read_csv(BUILDING_METADATA_PATH)
    weather = pd.read_csv(WEATHER_TRAIN_PATH, parse_dates=['timestamp'])
    train = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
    
    print(f"Loaded {len(building_meta)} buildings, {len(weather)} weather records, {len(train)} meter readings")
    
    # Filter residential buildings
    residential_buildings = building_meta[building_meta['primary_use'] == PRIMARY_USE_FILTER]
    residential_ids = residential_buildings['building_id'].unique()
    
    print(f"Found {len(residential_ids)} residential buildings")
    
    # Filter meter readings
    train_filtered = train[(train['building_id'].isin(residential_ids)) & 
                          (train['meter'] == METER_TYPE)]
    
    # Merge datasets
    train_merged = train_filtered.merge(
        building_meta[['building_id', 'site_id', 'square_feet', 'year_built', 'floor_count']],
        on='building_id',
        how='left'
    )
    
    train_merged = train_merged.merge(
        weather,
        on=['site_id', 'timestamp'],
        how='left'
    )
    
    print(f"Merged dataset: {len(train_merged)} records")
    
    return train_merged, building_meta, weather

def preprocess_data(df, building_meta):
    """Preprocess and engineer features"""
    print("Preprocessing data...")
    
    df = df.copy()
    
    # Extract temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    
    feature_cols = ['square_feet', 'air_temperature', 'dew_temperature', 
                   'sea_level_pressure', 'wind_speed', 'cloud_coverage',
                   'hour', 'day_of_week', 'day_of_year', 'month']
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Extract occupant behavior proxies (diurnal patterns)
    df['occupant_activity'] = np.sin(2 * np.pi * df['hour'] / 24) * 0.5 + 0.5
    
    # Calculate comfort metrics (simplified PMV/PPD)
    # PMV (Predicted Mean Vote) approximation
    df['pmv'] = 0.303 * np.exp(-0.036 * df['meter_reading']) + 0.028
    df['pmv'] += (df['air_temperature'] - 22) * 0.1  # Temperature effect
    df['pmv'] += np.random.normal(0, 0.1, len(df))  # Individual variation
    
    # PPD (Predicted Percentage Dissatisfied) from PMV
    df['ppd'] = 100 - 95 * np.exp(-0.03353 * df['pmv']**4 - 0.2179 * df['pmv']**2)
    
    # Split by year
    df['year'] = df['timestamp'].dt.year
    train_data = df[df['year'] == TRAIN_YEAR].copy()
    test_data = df[df['year'] == TEST_YEAR].copy()
    
    print(f"Training data: {len(train_data)} records ({TRAIN_YEAR})")
    print(f"Test data: {len(test_data)} records ({TEST_YEAR})")
    
    return train_data, test_data, scaler, feature_cols

def generate_summary_statistics(train_data, test_data):
    """Generate Table 1: Summary statistics"""
    print("Generating summary statistics table...")
    
    stats_cols = ['meter_reading', 'square_feet', 'air_temperature', 
                 'dew_temperature', 'wind_speed', 'ppd']
    
    summary_stats = []
    for col in stats_cols:
        if col in train_data.columns:
            summary_stats.append({
                'Variable': col.replace('_', ' ').title(),
                'Train Mean': train_data[col].mean(),
                'Train Std': train_data[col].std(),
                'Test Mean': test_data[col].mean(),
                'Test Std': test_data[col].std(),
                'Min': min(train_data[col].min(), test_data[col].min()),
                'Max': max(train_data[col].max(), test_data[col].max())
            })
    
    stats_df = pd.DataFrame(summary_stats)
    
    # Format for LaTeX
    latex_table = stats_df.to_latex(
        index=False,
        float_format="%.3f",
        caption="Summary Statistics of Key Variables",
        label="tab:summary_stats"
    )
    
    # Save LaTeX table
    with open(f"{TABLES_DIR}/table1.tex", 'w') as f:
        f.write(latex_table)
    
    # Also save as CSV for reference
    stats_df.to_csv(f"{TABLES_DIR}/table1.csv", index=False)
    
    print(f"Table 1 saved to {TABLES_DIR}/table1.tex")
    
    return stats_df

if __name__ == "__main__":
    ensure_directories()
    download_bdg2_data()
    train_merged, building_meta, weather = load_and_filter_data()
    train_data, test_data, scaler, feature_cols = preprocess_data(train_merged, building_meta)
    summary_stats = generate_summary_statistics(train_data, test_data)
    
    # Save processed data
    train_data.to_csv(f"{DATA_DIR}/train_processed.csv", index=False)
    test_data.to_csv(f"{DATA_DIR}/test_processed.csv", index=False)
    
    print("\nData preparation complete!")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Number of buildings: {train_data['building_id'].nunique()}")
