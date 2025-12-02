"""
Data Preprocessing Module for BDG2 Surrogate-Assisted Optimization.

This module handles:
- Building selection based on metadata criteria
- Data integration (merging electricity and weather data)
- Cleaning, normalization, and train/val/test splitting
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config import DataConfig, get_climate_zone


@dataclass
class BuildingData:
    """Container for processed building data."""
    building_id: str
    site_id: str
    climate_zone: str
    floor_area_sqm: float
    year_built: int
    
    # Processed time series
    data: pd.DataFrame
    
    # Scalers for inverse transformation
    energy_scaler: Optional[object] = None
    feature_scalers: Optional[Dict] = None


class BDG2DataProcessor:
    """
    Preprocessor for BDG2 dataset.
    
    Handles building selection, data integration, cleaning and normalization.
    """
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.metadata = None
        self.electricity_data = None
        self.weather_data = None
        self.selected_buildings = []
        
    def load_raw_data(self) -> None:
        """Load raw data from BDG2 dataset."""
        print("Loading BDG2 data...")
        
        # Load metadata
        self.metadata = pd.read_csv(self.config.metadata_path)
        print(f"  Loaded metadata: {len(self.metadata)} buildings")
        
        # Load electricity data
        print("  Loading electricity data (this may take a moment)...")
        self.electricity_data = pd.read_csv(
            self.config.electricity_path,
            parse_dates=['timestamp'],
            index_col='timestamp'
        )
        print(f"  Loaded electricity data: {self.electricity_data.shape}")
        
        # Load weather data
        self.weather_data = pd.read_csv(
            self.config.weather_path,
            parse_dates=['timestamp']
        )
        print(f"  Loaded weather data: {len(self.weather_data)} records")
        
    def select_buildings(self) -> pd.DataFrame:
        """
        Select buildings based on criteria specified in config.
        
        Returns:
            DataFrame with selected building metadata.
        """
        if self.metadata is None:
            self.load_raw_data()
            
        # Filter by primary use type
        selected = self.metadata[
            self.metadata['primaryspaceusage'].isin(self.config.primary_use_types)
        ].copy()
        
        # Filter by electricity meter availability
        selected = selected[selected['electricity'] == 'Yes']
        
        print(f"\nBuilding Selection:")
        print(f"  Total lodging/residential buildings: {len(selected)}")
        
        # Check data coverage for each building
        buildings_with_coverage = []
        
        for _, row in selected.iterrows():
            building_id = row['building_id']
            if building_id in self.electricity_data.columns:
                series = self.electricity_data[building_id].dropna()
                coverage = len(series) / len(self.electricity_data)
                if coverage >= self.config.min_data_coverage:
                    buildings_with_coverage.append(building_id)
        
        selected = selected[selected['building_id'].isin(buildings_with_coverage)]
        print(f"  Buildings with ≥{self.config.min_data_coverage*100:.0f}% data coverage: {len(selected)}")
        
        # Add climate zone
        selected['climate_zone'] = selected['lat'].apply(get_climate_zone)
        
        self.selected_buildings = selected
        return selected
    
    def get_building_list(self, n_buildings: int = 10) -> List[str]:
        """
        Get a diverse list of buildings for analysis.
        
        Args:
            n_buildings: Number of buildings to select
            
        Returns:
            List of building IDs
        """
        if len(self.selected_buildings) == 0:
            self.select_buildings()
            
        # Try to get diversity across sites and building sizes
        selected = self.selected_buildings.copy()
        
        # Sort by floor area to get variety
        selected = selected.sort_values('sqm')
        
        # Sample from different size quartiles
        building_ids = []
        n_per_quartile = max(1, n_buildings // 4)
        
        for i in range(4):
            quartile = selected.iloc[i * len(selected) // 4:(i + 1) * len(selected) // 4]
            sample_n = min(n_per_quartile, len(quartile))
            sampled = quartile.sample(n=sample_n, random_state=42)
            building_ids.extend(sampled['building_id'].tolist())
            
        return building_ids[:n_buildings]
    
    def process_building(self, building_id: str) -> BuildingData:
        """
        Process data for a single building.
        
        Args:
            building_id: Building identifier
            
        Returns:
            BuildingData object with processed data
        """
        if self.metadata is None:
            self.load_raw_data()
            
        # Get building metadata
        meta = self.metadata[self.metadata['building_id'] == building_id].iloc[0]
        site_id = meta['site_id']
        
        # Get electricity data
        energy = self.electricity_data[[building_id]].copy()
        energy.columns = ['energy_kwh']
        
        # Get weather data for this site
        site_weather = self.weather_data[self.weather_data['site_id'] == site_id].copy()
        site_weather = site_weather.set_index('timestamp')
        
        # Merge energy and weather
        merged = energy.join(site_weather, how='left')
        
        # Clean column names
        merged = merged.rename(columns={
            'airTemperature': 'outdoor_temp',
            'dewTemperature': 'dew_temp',
            'cloudCoverage': 'cloud_coverage',
            'windSpeed': 'wind_speed'
        })
        
        # Calculate relative humidity from dew point
        if 'dew_temp' in merged.columns and 'outdoor_temp' in merged.columns:
            merged['relative_humidity'] = self._calculate_rh(
                merged['outdoor_temp'], 
                merged['dew_temp']
            )
        
        # Add temporal features
        merged['hour'] = merged.index.hour
        merged['day_of_week'] = merged.index.dayofweek
        merged['month'] = merged.index.month
        merged['is_weekend'] = (merged['day_of_week'] >= 5).astype(int)
        
        # Handle missing values
        merged = self._handle_missing_values(merged)
        
        # Drop rows with any remaining NaN
        merged = merged.dropna()
        
        # Create BuildingData object
        building_data = BuildingData(
            building_id=building_id,
            site_id=site_id,
            climate_zone=get_climate_zone(meta['lat']),
            floor_area_sqm=meta['sqm'],
            year_built=int(meta['yearbuilt']) if pd.notna(meta['yearbuilt']) else 2000,
            data=merged
        )
        
        return building_data
    
    def _calculate_rh(self, temp: pd.Series, dew_temp: pd.Series) -> pd.Series:
        """Calculate relative humidity from temperature and dew point."""
        # Magnus formula approximation
        a = 17.27
        b = 237.7
        
        gamma_t = (a * temp) / (b + temp) + np.log(6.112)
        gamma_td = (a * dew_temp) / (b + dew_temp) + np.log(6.112)
        
        rh = 100 * np.exp(gamma_td - gamma_t + np.log(6.112) - np.log(6.112))
        rh = rh.clip(0, 100)
        
        return rh
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values through interpolation."""
        # Identify gaps
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32, float]:
                # Create mask of missing values
                missing_mask = df[col].isna()
                
                # Find consecutive missing segments
                if missing_mask.sum() > 0:
                    # Count consecutive missing values
                    missing_groups = (missing_mask != missing_mask.shift()).cumsum()
                    missing_lengths = df.groupby(missing_groups)[col].transform('count')
                    
                    # Only interpolate short gaps
                    short_gaps = missing_lengths <= self.config.max_gap_hours
                    
                    # Linear interpolation for short gaps
                    df.loc[missing_mask & short_gaps, col] = df[col].interpolate(
                        method='linear', limit=self.config.max_gap_hours
                    ).loc[missing_mask & short_gaps]
        
        return df
    
    def create_sequences(
        self, 
        data: pd.DataFrame,
        target_cols: List[str] = ['energy_kwh'],
        feature_cols: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: DataFrame with time series data
            target_cols: Columns to predict
            feature_cols: Feature columns (if None, auto-select)
            
        Returns:
            X: Input sequences (n_samples, seq_length, n_features)
            y: Target values (n_samples, n_targets)
        """
        if feature_cols is None:
            feature_cols = [
                'outdoor_temp', 'relative_humidity', 'hour', 
                'day_of_week', 'is_weekend', 'energy_kwh'
            ]
            feature_cols = [c for c in feature_cols if c in data.columns]
        
        seq_len = self.config.sequence_length
        horizon = self.config.prediction_horizon
        
        features = data[feature_cols].values
        targets = data[target_cols].values
        
        X, y = [], []
        for i in range(len(data) - seq_len - horizon + 1):
            X.append(features[i:i + seq_len])
            y.append(targets[i + seq_len + horizon - 1])
            
        return np.array(X), np.array(y)
    
    def create_lag_features(
        self, 
        data: pd.DataFrame,
        lag_hours: List[int] = None
    ) -> pd.DataFrame:
        """
        Create lag features for XGBoost.
        
        Args:
            data: DataFrame with time series data
            lag_hours: List of lag periods
            
        Returns:
            DataFrame with lag features added
        """
        if lag_hours is None:
            lag_hours = [1, 2, 3, 6, 12, 24]
            
        df = data.copy()
        
        # Create lag features for energy
        for lag in lag_hours:
            df[f'energy_lag_{lag}'] = df['energy_kwh'].shift(lag)
            
        # Create lag features for temperature
        if 'outdoor_temp' in df.columns:
            for lag in [1, 3, 6]:
                df[f'temp_lag_{lag}'] = df['outdoor_temp'].shift(lag)
        
        # Rolling statistics
        df['energy_rolling_mean_24h'] = df['energy_kwh'].rolling(24).mean()
        df['energy_rolling_std_24h'] = df['energy_kwh'].rolling(24).std()
        
        if 'outdoor_temp' in df.columns:
            df['temp_rolling_mean_24h'] = df['outdoor_temp'].rolling(24).mean()
        
        return df.dropna()
    
    def normalize_data(
        self, 
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        cols_to_normalize: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Normalize data using training set statistics.
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            cols_to_normalize: Columns to normalize
            
        Returns:
            Normalized train, val, test DataFrames and scalers
        """
        if cols_to_normalize is None:
            cols_to_normalize = [
                'energy_kwh', 'outdoor_temp', 'relative_humidity',
                'wind_speed', 'cloud_coverage'
            ]
            cols_to_normalize = [c for c in cols_to_normalize if c in train_df.columns]
        
        scalers = {}
        train_norm = train_df.copy()
        val_norm = val_df.copy() if val_df is not None else None
        test_norm = test_df.copy() if test_df is not None else None
        
        for col in cols_to_normalize:
            if self.config.normalization_method == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
                
            train_norm[col] = scaler.fit_transform(train_df[[col]])
            scalers[col] = scaler
            
            if val_norm is not None and col in val_df.columns:
                val_norm[col] = scaler.transform(val_df[[col]])
            if test_norm is not None and col in test_df.columns:
                test_norm[col] = scaler.transform(test_df[[col]])
        
        return train_norm, val_norm, test_norm, scalers
    
    def split_data(
        self, 
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets preserving temporal order.
        
        Args:
            data: DataFrame with time series
            
        Returns:
            train_df, val_df, test_df
        """
        n = len(data)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        train_df = data.iloc[:train_end].copy()
        val_df = data.iloc[train_end:val_end].copy()
        test_df = data.iloc[val_end:].copy()
        
        return train_df, val_df, test_df


def create_case_study_table(buildings: List[BuildingData]) -> pd.DataFrame:
    """
    Create Table 1: Characteristics of Selected Case Study Buildings.
    
    Args:
        buildings: List of BuildingData objects
        
    Returns:
        DataFrame formatted for publication
    """
    rows = []
    for i, b in enumerate(buildings):
        rows.append({
            'Building ID': f'Res_{i+1:02d}',
            'Primary Use': 'Residence Hall',
            'Floor Area (m²)': f'{b.floor_area_sqm:,.0f}',
            'Climate Zone': b.climate_zone,
            'Year Built': b.year_built,
            'Data Resolution': '1-Hour'
        })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Test the data processor
    processor = BDG2DataProcessor()
    processor.load_raw_data()
    
    # Select buildings
    selected = processor.select_buildings()
    print("\nSelected Buildings Summary:")
    print(selected[['building_id', 'site_id', 'sqm', 'climate_zone']].head(10))
    
    # Get building list
    building_ids = processor.get_building_list(n_buildings=5)
    print(f"\nSelected {len(building_ids)} buildings for analysis:")
    for bid in building_ids:
        print(f"  - {bid}")
    
    # Process one building
    print("\nProcessing first building...")
    building_data = processor.process_building(building_ids[0])
    print(f"Building: {building_data.building_id}")
    print(f"Climate Zone: {building_data.climate_zone}")
    print(f"Data shape: {building_data.data.shape}")
    print(f"Columns: {list(building_data.data.columns)}")
