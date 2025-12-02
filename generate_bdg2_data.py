"""
Generate realistic synthetic BDG2-style data for EDA analysis
This creates data matching the BDG2 structure and patterns for demonstration
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

print("Generating synthetic BDG2-style dataset...")

# Create data directory
os.makedirs('data', exist_ok=True)

# Parameters
start_date = datetime(2016, 1, 1)
end_date = datetime(2017, 12, 31)
n_buildings = 50
n_sites = 3

# Site information
sites = {
    'site_0': {'name': 'Temperate Climate Site', 'timezone': 'America/Los_Angeles', 
               'lat': 37.7749, 'lng': -122.4194},
    'site_1': {'name': 'Cold Climate Site', 'timezone': 'America/New_York',
               'lat': 40.7128, 'lng': -74.0060},
    'site_2': {'name': 'Warm Climate Site', 'timezone': 'America/Phoenix',
               'lat': 33.4484, 'lng': -112.0740}
}

# Building types
building_types = ['Office', 'Education', 'Public Assembly', 'Retail', 'Healthcare']
sub_types = {
    'Office': ['Corporate', 'Government', 'Mixed Use'],
    'Education': ['University', 'K-12', 'Library'],
    'Public Assembly': ['Convention Center', 'Museum', 'Theater'],
    'Retail': ['Shopping Mall', 'Supermarket', 'Store'],
    'Healthcare': ['Hospital', 'Clinic', 'Nursing Home']
}

# Generate building metadata
buildings = []
for i in range(n_buildings):
    site_id = f'site_{i % n_sites}'
    btype = np.random.choice(building_types)
    sub_btype = np.random.choice(sub_types[btype])
    sqm = np.random.lognormal(mean=9.5, sigma=0.8)  # Realistic building sizes
    sqm = max(500, min(50000, sqm))  # Clamp between 500-50000 m²
    
    buildings.append({
        'building_id': f'building_{i:03d}',
        'site_id': site_id,
        'primaryspaceusage': btype,
        'sub_primaryspaceusage': sub_btype,
        'sqm': sqm,
        'lat': sites[site_id]['lat'] + np.random.normal(0, 0.01),
        'lng': sites[site_id]['lng'] + np.random.normal(0, 0.01),
        'timezone': sites[site_id]['timezone']
    })

metadata_df = pd.DataFrame(buildings)
metadata_df.to_csv('data/metadata.csv', index=False)
print(f"✓ Generated metadata.csv with {len(metadata_df)} buildings")

# Generate hourly timestamps
timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
n_hours = len(timestamps)

# Generate weather data for each site
weather_data = []
for site_id, site_info in sites.items():
    # Base temperature varies by site and season
    for ts in timestamps:
        day_of_year = ts.timetuple().tm_yday
        # Seasonal temperature variation
        base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Site-specific adjustments
        if 'Cold' in site_info['name']:
            base_temp -= 8
        elif 'Warm' in site_info['name']:
            base_temp += 12
        
        # Add daily and hourly variation
        daily_var = 5 * np.sin(2 * np.pi * day_of_year / 365)
        hourly_var = 3 * np.sin(2 * np.pi * (ts.hour - 6) / 24)
        noise = np.random.normal(0, 2)
        
        air_temp = base_temp + daily_var + hourly_var + noise
        
        weather_data.append({
            'timestamp': ts,
            'site_id': site_id,
            'airTemperature': air_temp,
            'dewTemperature': air_temp - np.random.uniform(2, 8),
            'windSpeed': np.random.gamma(2, 2),  # Skewed distribution
            'seaLvlPressure': 1013 + np.random.normal(0, 5),
            'cloudCoverage': np.random.uniform(0, 1),
            'precipDepth1HR': np.random.exponential(0.5) if np.random.random() < 0.1 else 0
        })

weather_df = pd.DataFrame(weather_data)
weather_df.to_csv('data/weather.csv', index=False)
print(f"✓ Generated weather.csv with {len(weather_df)} hourly records")

# Generate meter readings (electricity only, meter=0)
meter_data = []
for building in buildings:
    building_id = building['building_id']
    site_id = building['site_id']
    sqm = building['sqm']
    btype = building['primaryspaceusage']
    
    # Base load intensity (kWh/m²·h) varies by building type
    base_intensity = {
        'Office': 0.08,
        'Education': 0.06,
        'Public Assembly': 0.12,
        'Retail': 0.15,
        'Healthcare': 0.20
    }.get(btype, 0.10)
    
    for ts in timestamps:
        # Get corresponding weather
        weather_row = weather_df[(weather_df['site_id'] == site_id) & 
                                 (weather_df['timestamp'] == ts)].iloc[0]
        air_temp = weather_row['airTemperature']
        
        # Base load
        base_load = base_intensity * sqm
        
        # Temperature-dependent load (cooling/heating)
        # Cooling load increases above 24°C, heating load increases below 18°C
        cooling_load = max(0, (air_temp - 24) * 0.02 * sqm) if air_temp > 24 else 0
        heating_load = max(0, (18 - air_temp) * 0.015 * sqm) if air_temp < 18 else 0
        
        # Time-of-day pattern (stronger for offices)
        hour = ts.hour
        if btype == 'Office':
            # Office hours: 7-18 peak, lower at night
            time_factor = 0.3 + 0.7 * (1 - abs(hour - 12.5) / 12.5)
        elif btype == 'Retail':
            # Retail: 9-21 peak
            time_factor = 0.4 + 0.6 * (1 - abs(hour - 15) / 12)
        else:
            # More constant for other types
            time_factor = 0.6 + 0.4 * (1 - abs(hour - 12) / 12)
        
        # Day-of-week pattern
        day_of_week = ts.weekday()
        if day_of_week < 5:  # Weekday
            dow_factor = 1.0
        else:  # Weekend
            dow_factor = 0.6 if btype == 'Office' else 0.8
        
        # Seasonal variation
        day_of_year = ts.timetuple().tm_yday
        seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * (day_of_year - 200) / 365)
        
        # Random variation
        noise_factor = np.random.lognormal(mean=0, sigma=0.1)
        
        # Calculate total load
        total_load = (base_load + cooling_load + heating_load) * time_factor * dow_factor * seasonal_factor * noise_factor
        
        meter_data.append({
            'timestamp': ts,
            'building_id': building_id,
            'meter': 0,  # 0 = electricity
            'meter_reading': max(0, total_load)
        })

meter_df = pd.DataFrame(meter_data)
# Save in chunks to manage memory
meter_df.to_csv('data/meter_data.csv', index=False)
print(f"✓ Generated meter_data.csv with {len(meter_df)} hourly records")

print("\n" + "="*60)
print("Synthetic BDG2 dataset generation complete!")
print("="*60)
print(f"Buildings: {len(metadata_df)}")
print(f"Weather records: {len(weather_df)}")
print(f"Meter records: {len(meter_df)}")
print(f"Time period: {start_date.date()} to {end_date.date()}")
