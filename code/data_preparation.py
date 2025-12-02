"""
Data Preparation Module for Building Energy Optimization
Simulates BDG2 (Building Data Genome 2) dataset structure
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import config

def generate_building_metadata(num_buildings=config.NUM_BUILDINGS):
    """Generate building metadata similar to BDG2"""
    np.random.seed(config.RANDOM_SEED)
    
    buildings = []
    for i in range(num_buildings):
        building = {
            'building_id': i,
            'site_id': i,
            'primary_use': config.BUILDING_TYPE,
            'square_feet': np.random.randint(*config.BUILDING_AREA_RANGE),
            'year_built': np.random.randint(1980, 2010),
            'floor_count': np.random.randint(2, 8),
            'occupancy': np.random.randint(*config.OCCUPANCY_RANGE)
        }
        buildings.append(building)
    
    df = pd.DataFrame(buildings)
    return df

def generate_weather_data(start_date, end_date, num_sites=config.NUM_BUILDINGS):
    """Generate realistic weather data"""
    np.random.seed(config.RANDOM_SEED)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    weather_data = []
    
    for site_id in range(num_sites):
        for timestamp in dates:
            # Add seasonality
            day_of_year = timestamp.timetuple().tm_yday
            seasonal_temp = config.TEMP_MEAN + 10 * np.sin(2 * np.pi * day_of_year / 365)
            daily_variation = 5 * np.sin(2 * np.pi * timestamp.hour / 24)
            
            temp = seasonal_temp + daily_variation + np.random.normal(0, 2)
            
            # Humidity inversely correlated with temperature
            humidity = config.HUMIDITY_MEAN - 0.5 * (temp - config.TEMP_MEAN) + np.random.normal(0, 5)
            humidity = np.clip(humidity, 20, 100)
            
            weather = {
                'site_id': site_id,
                'timestamp': timestamp,
                'air_temperature': temp,
                'dew_temperature': temp - 5,
                'cloud_coverage': np.random.randint(0, 9),
                'wind_speed': np.random.exponential(3),
                'precip_depth_1_hr': np.random.exponential(0.1) if np.random.rand() < 0.1 else 0,
                'sea_level_pressure': 1013 + np.random.normal(0, 10),
                'humidity': humidity
            }
            weather_data.append(weather)
    
    df = pd.DataFrame(weather_data)
    return df

def generate_energy_data(building_metadata, weather_data, start_date, end_date):
    """Generate energy consumption data with realistic patterns"""
    np.random.seed(config.RANDOM_SEED)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    energy_data = []
    
    for _, building in building_metadata.iterrows():
        building_id = building['building_id']
        base_load = building['square_feet'] / 1000 * config.BASE_ENERGY_INTENSITY / 24  # Per hour
        
        for timestamp in dates:
            # Get weather for this timestamp
            weather = weather_data[
                (weather_data['site_id'] == building_id) & 
                (weather_data['timestamp'] == timestamp)
            ]
            
            if len(weather) == 0:
                continue
                
            temp = weather['air_temperature'].values[0]
            
            # Energy components
            # 1. Base load
            energy = base_load
            
            # 2. Occupancy pattern (higher during day, weekdays)
            hour = timestamp.hour
            is_weekend = timestamp.weekday() >= 5
            occupancy_factor = 1.0
            if 8 <= hour <= 22:
                occupancy_factor = 1.5 if not is_weekend else 1.2
            else:
                occupancy_factor = 0.6
            
            energy *= occupancy_factor
            
            # 3. Temperature-dependent HVAC load
            comfort_temp = 21.0
            if abs(temp - comfort_temp) > 2:
                hvac_load = base_load * 0.5 * abs(temp - comfort_temp) / 10
                energy += hvac_load
            
            # 4. Add noise
            energy *= (1 + np.random.normal(0, 0.1))
            energy = max(0, energy)
            
            meter_reading = {
                'building_id': building_id,
                'timestamp': timestamp,
                'meter': 0,  # Electricity
                'meter_reading': energy
            }
            energy_data.append(meter_reading)
    
    df = pd.DataFrame(energy_data)
    return df

def prepare_dataset():
    """Main function to prepare complete dataset"""
    print("Generating building metadata...")
    building_metadata = generate_building_metadata()
    building_metadata.to_csv(os.path.join(config.DATA_DIR, 'building_metadata.csv'), index=False)
    print(f"Generated {len(building_metadata)} buildings")
    
    print("\nGenerating training data (2016)...")
    weather_train = generate_weather_data('2016-01-01', '2016-12-31 23:00:00')
    weather_train.to_csv(os.path.join(config.DATA_DIR, 'weather_train.csv'), index=False)
    print(f"Generated {len(weather_train)} weather records")
    
    energy_train = generate_energy_data(building_metadata, weather_train, '2016-01-01', '2016-12-31 23:00:00')
    energy_train.to_csv(os.path.join(config.DATA_DIR, 'train.csv'), index=False)
    print(f"Generated {len(energy_train)} energy records")
    
    print("\nGenerating testing data (2017)...")
    weather_test = generate_weather_data('2017-01-01', '2017-12-31 23:00:00')
    weather_test.to_csv(os.path.join(config.DATA_DIR, 'weather_test.csv'), index=False)
    
    energy_test = generate_energy_data(building_metadata, weather_test, '2017-01-01', '2017-12-31 23:00:00')
    energy_test.to_csv(os.path.join(config.DATA_DIR, 'test.csv'), index=False)
    print(f"Generated {len(energy_test)} test energy records")
    
    return building_metadata, weather_train, energy_train, weather_test, energy_test

def load_and_merge_data(train=True):
    """Load and merge datasets"""
    print("\nLoading and merging data...")
    
    building_metadata = pd.read_csv(os.path.join(config.DATA_DIR, 'building_metadata.csv'))
    
    if train:
        weather = pd.read_csv(os.path.join(config.DATA_DIR, 'weather_train.csv'))
        energy = pd.read_csv(os.path.join(config.DATA_DIR, 'train.csv'))
    else:
        weather = pd.read_csv(os.path.join(config.DATA_DIR, 'weather_test.csv'))
        energy = pd.read_csv(os.path.join(config.DATA_DIR, 'test.csv'))
    
    # Convert timestamp to datetime
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])
    energy['timestamp'] = pd.to_datetime(energy['timestamp'])
    
    # Merge datasets
    # First merge energy with building metadata
    df = energy.merge(building_metadata, on='building_id', how='left')
    
    # Then merge with weather (on site_id and timestamp)
    df = df.merge(weather, on=['site_id', 'timestamp'], how='left')
    
    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # Create lag features
    df = df.sort_values(['building_id', 'timestamp'])
    df['energy_lag1'] = df.groupby('building_id')['meter_reading'].shift(1)
    df['energy_lag24'] = df.groupby('building_id')['meter_reading'].shift(24)
    
    # Fill missing values
    df['energy_lag1'].fillna(df['meter_reading'].mean(), inplace=True)
    df['energy_lag24'].fillna(df['meter_reading'].mean(), inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    print(f"Merged dataset shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def create_summary_statistics_table(df):
    """Generate Table 1: Summary statistics for paper"""
    print("\nGenerating summary statistics table...")
    
    stats_dict = {
        'Variable': [],
        'Mean': [],
        'Std': [],
        'Min': [],
        'Max': [],
        'Unit': []
    }
    
    variables = [
        ('meter_reading', 'Energy Consumption', 'kWh'),
        ('air_temperature', 'Air Temperature', '°C'),
        ('humidity', 'Relative Humidity', '%'),
        ('square_feet', 'Building Area', 'sq ft'),
        ('occupancy', 'Occupancy', 'persons'),
        ('wind_speed', 'Wind Speed', 'm/s')
    ]
    
    for col, name, unit in variables:
        if col in df.columns:
            stats_dict['Variable'].append(name)
            stats_dict['Mean'].append(f"{df[col].mean():.2f}")
            stats_dict['Std'].append(f"{df[col].std():.2f}")
            stats_dict['Min'].append(f"{df[col].min():.2f}")
            stats_dict['Max'].append(f"{df[col].max():.2f}")
            stats_dict['Unit'].append(unit)
    
    stats_df = pd.DataFrame(stats_dict)
    
    # Save as LaTeX
    latex_str = stats_df.to_latex(index=False, escape=False, column_format='lrrrrr')
    
    with open(os.path.join(config.TABLES_DIR, 'table1_summary_statistics.tex'), 'w') as f:
        f.write("% Table 1: Summary Statistics of Building Energy Dataset\n")
        f.write(latex_str)
    
    print("Table 1 saved to tables/table1_summary_statistics.tex")
    
    return stats_df

if __name__ == "__main__":
    # Prepare dataset
    prepare_dataset()
    
    # Load and process
    train_df = load_and_merge_data(train=True)
    test_df = load_and_merge_data(train=False)
    
    # Generate summary table
    summary_stats = create_summary_statistics_table(train_df)
    print("\nSummary Statistics:")
    print(summary_stats)
    
    print("\nData preparation complete!")
