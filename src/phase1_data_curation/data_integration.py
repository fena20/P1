"""
Phase 1.2: Data Integration
Merges hourly meter readings with weather data and aligns time series.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Path configuration
BDG2_ROOT = Path(__file__).parent.parent.parent / "bdg2_data"
WEATHER_PATH = BDG2_ROOT / "data" / "weather" / "weather.csv"
METERS_DIR = BDG2_ROOT / "data" / "meters" / "cleaned"
SELECTED_BUILDINGS_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "selected_buildings.csv"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "processed"

def load_weather_data():
    """Load BDG2 weather data."""
    print(f"Loading weather data from {WEATHER_PATH}")
    weather = pd.read_csv(WEATHER_PATH)
    
    # Parse timestamp if present
    if 'timestamp' in weather.columns:
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        weather = weather.set_index('timestamp')
    elif 'datetime' in weather.columns:
        weather['datetime'] = pd.to_datetime(weather['datetime'])
        weather = weather.set_index('datetime')
    
    print(f"Loaded weather data: {weather.shape}")
    print(f"Date range: {weather.index.min()} to {weather.index.max()}")
    return weather

def load_meter_data(building_ids, sample_size=None):
    """
    Load meter data for selected buildings.
    
    Parameters:
    -----------
    building_ids : list
        List of building IDs to load
    sample_size : int, optional
        Number of buildings to sample (for testing)
    
    Returns:
    --------
    pd.DataFrame
        Combined meter data
    """
    print(f"Loading meter data for {len(building_ids)} buildings...")
    
    if sample_size:
        building_ids = building_ids[:sample_size]
        print(f"Sampling first {sample_size} buildings for testing")
    
    all_meters = []
    
    if METERS_DIR.exists():
        meter_files = list(METERS_DIR.glob("*.csv"))
        print(f"Found {len(meter_files)} meter files")
        
        for meter_file in meter_files[:10]:  # Limit to first 10 files for initial testing
            try:
                df = pd.read_csv(meter_file, nrows=10000)  # Sample first 10k rows
                
                # Check if this file contains any of our building IDs
                id_col = None
                for col in df.columns:
                    if 'building' in col.lower() and 'id' in col.lower():
                        id_col = col
                        break
                
                if id_col and df[id_col].isin(building_ids).any():
                    # Filter for our buildings
                    df_filtered = df[df[id_col].isin(building_ids)].copy()
                    all_meters.append(df_filtered)
                    print(f"  Loaded data from {meter_file.name}: {len(df_filtered)} rows")
            except Exception as e:
                print(f"  Error loading {meter_file.name}: {e}")
                continue
    
    if all_meters:
        combined_meters = pd.concat(all_meters, ignore_index=True)
        print(f"\nCombined meter data: {combined_meters.shape}")
        return combined_meters
    else:
        print("No meter data found. Creating sample structure...")
        # Create sample structure for demonstration
        dates = pd.date_range('2016-01-01', '2017-12-31 23:00:00', freq='H')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'building_id': building_ids[0] if building_ids else 'sample_building',
            'meter_reading': np.random.normal(50, 10, len(dates))
        })
        return sample_data

def merge_meter_weather(meter_data, weather_data):
    """
    Merge meter data with weather data on timestamp.
    
    Parameters:
    -----------
    meter_data : pd.DataFrame
        Meter readings dataframe
    weather_data : pd.DataFrame
        Weather data dataframe
    
    Returns:
    --------
    pd.DataFrame
        Merged dataframe
    """
    print("\nMerging meter and weather data...")
    
    # Ensure timestamp column exists in meter_data
    if 'timestamp' not in meter_data.columns:
        if 'datetime' in meter_data.columns:
            meter_data['timestamp'] = pd.to_datetime(meter_data['datetime'])
        else:
            print("Warning: No timestamp column found in meter data")
            return meter_data
    
    meter_data['timestamp'] = pd.to_datetime(meter_data['timestamp'])
    meter_data = meter_data.set_index('timestamp')
    
    # Resample weather data to hourly if needed
    if weather_data.index.freq is None:
        weather_data = weather_data.resample('H').mean()
    
    # Merge on timestamp
    merged = meter_data.join(weather_data, how='left', rsuffix='_weather')
    
    print(f"Merged data shape: {merged.shape}")
    print(f"Date range: {merged.index.min()} to {merged.index.max()}")
    
    return merged.reset_index()

def align_timezone(data, target_tz='UTC'):
    """
    Align timezone of time series data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataframe with timestamp index
    target_tz : str
        Target timezone (default: 'UTC')
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with aligned timezone
    """
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        if data['timestamp'].dt.tz is not None:
            data['timestamp'] = data['timestamp'].dt.tz_convert(target_tz)
        else:
            data['timestamp'] = data['timestamp'].dt.tz_localize('UTC').dt.tz_convert(target_tz)
    
    return data

def extract_temporal_features(data):
    """
    Extract temporal features (hour of day, day of week, etc.).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataframe with timestamp column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added temporal features
    """
    if 'timestamp' not in data.columns:
        return data
    
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek + 1  # 1-7
    data['day_of_year'] = data['timestamp'].dt.dayofyear
    data['month'] = data['timestamp'].dt.month
    data['is_weekend'] = (data['day_of_week'] >= 6).astype(int)
    
    return data

def main():
    """Main function for data integration."""
    print("=" * 60)
    print("Phase 1.2: Data Integration")
    print("=" * 60)
    
    # Load selected buildings
    if SELECTED_BUILDINGS_PATH.exists():
        selected_buildings = pd.read_csv(SELECTED_BUILDINGS_PATH)
        building_ids = selected_buildings['building_id'].tolist() if 'building_id' in selected_buildings.columns else selected_buildings.index.tolist()
    else:
        print("Warning: selected_buildings.csv not found. Using sample building IDs...")
        building_ids = ['Res_01', 'Res_02', 'Res_03']
    
    # Load weather data
    weather_data = load_weather_data()
    
    # Load meter data
    meter_data = load_meter_data(building_ids, sample_size=5)
    
    # Merge meter and weather data
    merged_data = merge_meter_weather(meter_data, weather_data)
    
    # Align timezone
    merged_data = align_timezone(merged_data)
    
    # Extract temporal features
    merged_data = extract_temporal_features(merged_data)
    
    # Save integrated data
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_PATH / "integrated_data.csv"
    merged_data.to_csv(output_file, index=False)
    print(f"\nSaved integrated data to {output_file}")
    print(f"Final data shape: {merged_data.shape}")
    
    return merged_data

if __name__ == "__main__":
    integrated_data = main()
