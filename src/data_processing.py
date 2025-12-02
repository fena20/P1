"""
Phase 1: Data Curation and Pre-Processing

This module handles:
1. Building selection (residential/lodging from BDG2 metadata)
2. Data integration (meter readings + weather data)
3. Cleaning and normalization
4. Train/validation/test splitting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class BDG2DataProcessor:
    """Process BDG2 data for surrogate model training and optimization."""
    
    def __init__(self, bdg2_path: str = "/workspace/bdg2_data"):
        self.bdg2_path = Path(bdg2_path)
        self.data_path = self.bdg2_path / "data"
        self.output_path = Path("/workspace/data")
        
        # Load metadata
        self.metadata = pd.read_csv(self.data_path / "metadata" / "metadata.csv")
        print(f"Loaded metadata for {len(self.metadata)} buildings")
        
    def select_residential_buildings(self, min_sqm: float = 500, 
                                    max_missing_ratio: float = 0.1) -> pd.DataFrame:
        """
        Select residential/lodging buildings with electricity meters and sufficient data coverage.
        
        Args:
            min_sqm: Minimum floor area in square meters
            max_missing_ratio: Maximum acceptable ratio of missing data
            
        Returns:
            DataFrame with selected buildings metadata
        """
        print("\n=== Building Selection ===")
        
        # Filter for residential/lodging buildings
        residential_mask = (
            self.metadata['primaryspaceusage'].str.contains('Lodging/residential', na=False) |
            self.metadata['primaryspaceusage'].str.contains('Residential', na=False)
        )
        
        # Filter for electricity availability
        electricity_mask = self.metadata['electricity'] == 'Yes'
        
        # Filter for minimum size
        size_mask = self.metadata['sqm'] >= min_sqm
        
        # Combine filters
        selected = self.metadata[residential_mask & electricity_mask & size_mask].copy()
        
        print(f"Found {len(selected)} residential buildings with electricity meters")
        print(f"  - {residential_mask.sum()} total residential buildings")
        print(f"  - {electricity_mask.sum()} total buildings with electricity")
        print(f"  - {size_mask.sum()} buildings >= {min_sqm} sqm")
        
        # Add climate zone classification based on lat/lng
        selected['climate_zone'] = selected.apply(self._classify_climate, axis=1)
        
        # Sort by data quality and diversity
        selected = selected.sort_values(['climate_zone', 'sqm'], ascending=[True, False])
        
        # Save selected buildings
        output_file = self.output_path / "selected_buildings" / "residential_buildings.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(output_file, index=False)
        print(f"\nSaved {len(selected)} selected buildings to {output_file}")
        
        # Print summary statistics
        print("\n=== Selected Buildings Summary ===")
        print(f"Climate zones: {selected['climate_zone'].value_counts().to_dict()}")
        print(f"Floor area range: {selected['sqm'].min():.0f} - {selected['sqm'].max():.0f} sqm")
        print(f"Year built range: {selected['yearbuilt'].min():.0f} - {selected['yearbuilt'].max():.0f}")
        
        return selected
    
    def _classify_climate(self, row) -> str:
        """Classify climate zone based on latitude."""
        lat = row['lat']
        if pd.isna(lat):
            return 'Unknown'
        elif lat > 40:
            return 'Cold'
        elif lat > 30:
            return 'Mixed'
        else:
            return 'Hot-Humid'
    
    def load_and_integrate_data(self, building_ids: List[str], 
                               year: str = '2016') -> Dict[str, pd.DataFrame]:
        """
        Load and integrate meter readings with weather data for selected buildings.
        
        Args:
            building_ids: List of building IDs to process
            year: Year to process ('2016' or '2017')
            
        Returns:
            Dictionary mapping building_id to integrated DataFrame
        """
        print(f"\n=== Loading and Integrating Data for {year} ===")
        
        # Load electricity meter data (wide format: buildings as columns)
        elec_file = self.data_path / "meters" / "cleaned" / "electricity_cleaned.csv"
        print(f"Loading electricity data from {elec_file}...")
        electricity = pd.read_csv(elec_file, low_memory=False)
        electricity['timestamp'] = pd.to_datetime(electricity['timestamp'])
        
        # Load weather data
        weather_file = self.data_path / "weather" / "weather.csv"
        print(f"Loading weather data from {weather_file}...")
        weather = pd.read_csv(weather_file)
        weather['timestamp'] = pd.to_datetime(weather['timestamp'])
        
        # Filter for specified year
        year_mask = electricity['timestamp'].dt.year == int(year)
        electricity = electricity[year_mask]
        
        year_mask_weather = weather['timestamp'].dt.year == int(year)
        weather = weather[year_mask_weather]
        
        print(f"Electricity records (timestamps): {len(electricity)}")
        print(f"Weather records: {len(weather)}")
        print(f"Available building columns: {len([c for c in electricity.columns if c != 'timestamp'])}")
        
        # Process each building
        integrated_data = {}
        
        for building_id in building_ids:
            print(f"\nProcessing {building_id}...")
            
            # Check if building has electricity data
            if building_id not in electricity.columns:
                print(f"  No electricity column found for {building_id}, skipping...")
                continue
            
            # Get building metadata
            building_meta = self.metadata[self.metadata['building_id'] == building_id].iloc[0]
            site_id = building_meta['site_id']
            
            # Add climate zone if not present
            if 'climate_zone' not in building_meta:
                building_meta = building_meta.copy()
                building_meta['climate_zone'] = self._classify_climate(building_meta)
            
            # Extract building electricity data
            building_elec = electricity[['timestamp', building_id]].copy()
            building_elec.columns = ['timestamp', 'meter_reading']
            
            # Remove rows with missing meter readings
            building_elec = building_elec.dropna(subset=['meter_reading'])
            
            if len(building_elec) == 0:
                print(f"  No valid electricity data found for {building_id}, skipping...")
                continue
            
            # Filter weather data for this site
            site_weather = weather[weather['site_id'] == site_id].copy()
            
            if len(site_weather) == 0:
                print(f"  No weather data found for site {site_id}, skipping...")
                continue
            
            # Merge electricity and weather data on timestamp
            integrated = pd.merge(
                building_elec,
                site_weather,
                on='timestamp',
                how='inner'
            )
            
            # Add temporal features
            integrated['hour'] = integrated['timestamp'].dt.hour
            integrated['day_of_week'] = integrated['timestamp'].dt.dayofweek + 1
            integrated['month'] = integrated['timestamp'].dt.month
            integrated['day_of_year'] = integrated['timestamp'].dt.dayofyear
            
            # Add building metadata
            integrated['building_id'] = building_id
            integrated['sqm'] = building_meta['sqm']
            integrated['yearbuilt'] = building_meta['yearbuilt']
            integrated['climate_zone'] = building_meta['climate_zone']
            
            print(f"  Integrated {len(integrated)} hourly records")
            print(f"  Date range: {integrated['timestamp'].min()} to {integrated['timestamp'].max()}")
            print(f"  Missing data: {integrated.isnull().sum().sum()} values")
            print(f"  Mean energy: {integrated['meter_reading'].mean():.2f} kWh")
            
            integrated_data[building_id] = integrated
        
        return integrated_data
    
    def clean_and_normalize(self, data: pd.DataFrame, 
                           building_id: str) -> pd.DataFrame:
        """
        Clean and normalize integrated building data.
        
        Args:
            data: Integrated DataFrame
            building_id: Building identifier
            
        Returns:
            Cleaned and normalized DataFrame
        """
        print(f"\n=== Cleaning and Normalizing {building_id} ===")
        
        df = data.copy()
        initial_rows = len(df)
        
        # Select relevant columns
        feature_cols = [
            'timestamp', 'building_id', 'meter_reading',
            'airTemperature', 'cloudCoverage', 'dewTemperature', 
            'seaLvlPressure', 'windSpeed',
            'hour', 'day_of_week', 'month', 'day_of_year',
            'sqm', 'yearbuilt', 'climate_zone'
        ]
        
        # Keep only available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        df = df[available_cols].copy()
        
        # Handle missing values in weather data
        # Short gaps (up to 3 hours): linear interpolation
        # Longer gaps: forward fill then backward fill
        weather_cols = ['airTemperature', 'cloudCoverage', 'dewTemperature', 
                       'seaLvlPressure', 'windSpeed']
        
        for col in weather_cols:
            if col in df.columns:
                # Interpolate short gaps
                df[col] = df[col].interpolate(method='linear', limit=3)
                # Fill remaining with forward/backward fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Remove rows with missing meter readings (these are the target)
        df = df.dropna(subset=['meter_reading'])
        
        # Remove outliers in meter readings (> 5 std from mean)
        mean_reading = df['meter_reading'].mean()
        std_reading = df['meter_reading'].std()
        outlier_threshold = mean_reading + 5 * std_reading
        df = df[df['meter_reading'] <= outlier_threshold]
        
        # Ensure no negative readings
        df = df[df['meter_reading'] >= 0]
        
        print(f"  Initial rows: {initial_rows}")
        print(f"  After cleaning: {len(df)}")
        print(f"  Rows removed: {initial_rows - len(df)} ({100*(initial_rows - len(df))/initial_rows:.1f}%)")
        
        # Check data coverage
        expected_hours = 365 * 24  # for one year
        coverage = len(df) / expected_hours
        print(f"  Data coverage: {coverage*100:.1f}%")
        
        # Save cleaned data
        output_file = self.output_path / "processed" / f"{building_id}_cleaned.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"  Saved cleaned data to {output_file}")
        
        return df
    
    def create_train_val_test_splits(self, data: pd.DataFrame, 
                                     building_id: str,
                                     train_ratio: float = 0.7,
                                     val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets preserving temporal order.
        
        Args:
            data: Cleaned DataFrame
            building_id: Building identifier
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"\n=== Creating Train/Val/Test Splits for {building_id} ===")
        
        # Sort by timestamp
        df = data.sort_values('timestamp').reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"  Total samples: {n}")
        print(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
        print(f"    Date range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        print(f"  Validation: {len(val_df)} ({len(val_df)/n*100:.1f}%)")
        print(f"    Date range: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
        print(f"  Test: {len(test_df)} ({len(test_df)/n*100:.1f}%)")
        print(f"    Date range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
        
        # Save splits
        splits_dir = self.output_path / "splits" / building_id
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(splits_dir / "train.csv", index=False)
        val_df.to_csv(splits_dir / "val.csv", index=False)
        test_df.to_csv(splits_dir / "test.csv", index=False)
        
        print(f"  Saved splits to {splits_dir}")
        
        return train_df, val_df, test_df
    
    def compute_normalization_params(self, train_df: pd.DataFrame) -> Dict:
        """
        Compute normalization parameters from training data.
        
        Args:
            train_df: Training DataFrame
            
        Returns:
            Dictionary with means and stds for each numeric column
        """
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        
        params = {}
        for col in numeric_cols:
            if col not in ['timestamp', 'hour', 'day_of_week', 'month', 'day_of_year']:
                params[col] = {
                    'mean': train_df[col].mean(),
                    'std': train_df[col].std(),
                    'min': train_df[col].min(),
                    'max': train_df[col].max()
                }
        
        return params


def main():
    """Run complete data processing pipeline."""
    
    print("="*80)
    print("PHASE 1: DATA CURATION AND PRE-PROCESSING")
    print("="*80)
    
    # Initialize processor
    processor = BDG2DataProcessor()
    
    # Step 1: Select residential buildings
    selected_buildings = processor.select_residential_buildings(
        min_sqm=500,
        max_missing_ratio=0.2
    )
    
    # Select top buildings from different climate zones for case studies
    # Pick 2-3 buildings per climate zone
    case_study_buildings = []
    for climate in ['Hot-Humid', 'Mixed', 'Cold']:
        climate_buildings = selected_buildings[
            selected_buildings['climate_zone'] == climate
        ]['building_id'].head(3).tolist()
        case_study_buildings.extend(climate_buildings)
    
    print(f"\n=== Selected {len(case_study_buildings)} case study buildings ===")
    for bid in case_study_buildings:
        building = selected_buildings[selected_buildings['building_id'] == bid].iloc[0]
        print(f"  {bid}: {building['sqm']:.0f} sqm, "
              f"{building['climate_zone']}, "
              f"built {building['yearbuilt']:.0f}")
    
    # Step 2: Load and integrate data for 2016 (primary analysis year)
    integrated_data = processor.load_and_integrate_data(
        building_ids=case_study_buildings,
        year='2016'
    )
    
    # Step 3: Clean, normalize, and split each building's data
    all_splits = {}
    
    for building_id, data in integrated_data.items():
        # Clean and normalize
        cleaned_data = processor.clean_and_normalize(data, building_id)
        
        # Create splits
        train_df, val_df, test_df = processor.create_train_val_test_splits(
            cleaned_data, building_id
        )
        
        # Compute normalization parameters
        norm_params = processor.compute_normalization_params(train_df)
        
        all_splits[building_id] = {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'norm_params': norm_params
        }
    
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE")
    print("="*80)
    print(f"Processed {len(all_splits)} buildings")
    print(f"Data saved to /workspace/data/")
    
    return all_splits


if __name__ == "__main__":
    main()
