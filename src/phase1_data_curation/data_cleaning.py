"""
Phase 1.3: Data Cleaning and Normalization
Handles missing data, outliers, and normalizes features for model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Path configuration
INTEGRATED_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "integrated_data.csv"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "processed"

def load_integrated_data():
    """Load integrated data from Phase 1.2."""
    print(f"Loading integrated data from {INTEGRATED_DATA_PATH}")
    if not INTEGRATED_DATA_PATH.exists():
        raise FileNotFoundError(f"Integrated data not found at {INTEGRATED_DATA_PATH}. Run data_integration.py first.")
    
    data = pd.read_csv(INTEGRATED_DATA_PATH)
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')
    
    print(f"Loaded data: {data.shape}")
    return data

def handle_missing_data(data, max_gap_hours=6):
    """
    Handle missing data by imputing short gaps and excluding longer gaps.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    max_gap_hours : int
        Maximum gap length to impute (default: 6 hours)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing data handled
    """
    print("\nHandling missing data...")
    print(f"Initial missing values: {data.isnull().sum().sum()}")
    
    data_cleaned = data.copy()
    
    # Identify numeric columns for imputation
    numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove columns that are identifiers or temporal features
    exclude_cols = ['building_id', 'meter_id', 'site_id']
    numeric_cols = [col for col in numeric_cols if not any(exc in col.lower() for exc in exclude_cols)]
    
    for col in numeric_cols:
        if data_cleaned[col].isnull().sum() > 0:
            # Forward fill for short gaps
            data_cleaned[col] = data_cleaned[col].fillna(method='ffill', limit=max_gap_hours)
            # Backward fill for remaining short gaps
            data_cleaned[col] = data_cleaned[col].fillna(method='bfill', limit=max_gap_hours)
            # Linear interpolation for remaining gaps
            data_cleaned[col] = data_cleaned[col].interpolate(method='linear', limit=max_gap_hours)
    
    print(f"Remaining missing values: {data_cleaned.isnull().sum().sum()}")
    
    # Drop rows with excessive missing data (>50% missing)
    threshold = len(numeric_cols) * 0.5
    data_cleaned = data_cleaned.dropna(thresh=threshold)
    
    print(f"Final data shape after cleaning: {data_cleaned.shape}")
    
    return data_cleaned

def remove_outliers(data, method='iqr', factor=1.5):
    """
    Remove outliers using IQR method or Z-score.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    method : str
        Method to use ('iqr' or 'zscore')
    factor : float
        Factor for outlier detection (default: 1.5 for IQR)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers removed
    """
    print(f"\nRemoving outliers using {method} method...")
    
    data_cleaned = data.copy()
    numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['building_id', 'meter_id', 'site_id', 'hour', 'day_of_week', 'day_of_year', 'month']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols and col not in data_cleaned.index.names]
    
    initial_rows = len(data_cleaned)
    
    if method == 'iqr':
        for col in numeric_cols:
            Q1 = data_cleaned[col].quantile(0.25)
            Q3 = data_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Cap outliers instead of removing (less data loss)
            data_cleaned[col] = data_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
    
    elif method == 'zscore':
        for col in numeric_cols:
            z_scores = np.abs((data_cleaned[col] - data_cleaned[col].mean()) / data_cleaned[col].std())
            data_cleaned = data_cleaned[z_scores < factor]
    
    removed_rows = initial_rows - len(data_cleaned)
    print(f"Removed/capped outliers affecting {removed_rows} rows ({removed_rows/initial_rows*100:.2f}%)")
    
    return data_cleaned

def normalize_features(data, method='standard', feature_cols=None):
    """
    Normalize features using min-max or standard scaling.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    method : str
        Normalization method ('standard' or 'minmax')
    feature_cols : list, optional
        List of columns to normalize. If None, normalizes all numeric columns.
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with normalized features
    tuple
        (scaler, feature_columns) for inverse transformation
    """
    print(f"\nNormalizing features using {method} scaling...")
    
    data_normalized = data.copy()
    
    if feature_cols is None:
        numeric_cols = data_normalized.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['building_id', 'meter_id', 'site_id', 'hour', 'day_of_week', 'day_of_year', 'month']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols and col not in data_normalized.index.names]
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Store original values for inverse transform
    scaler.fit(data_normalized[feature_cols])
    data_normalized[feature_cols] = scaler.transform(data_normalized[feature_cols])
    
    print(f"Normalized {len(feature_cols)} features")
    
    return data_normalized, (scaler, feature_cols)

def split_temporal_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train/validation/test sets while preserving temporal ordering.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe (must have timestamp index)
    train_ratio : float
        Training set ratio (default: 0.7)
    val_ratio : float
        Validation set ratio (default: 0.15)
    test_ratio : float
        Test set ratio (default: 0.15)
    
    Returns:
    --------
    tuple
        (train_data, val_data, test_data)
    """
    print(f"\nSplitting data: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'timestamp' in data.columns:
            data = data.set_index('timestamp')
        else:
            raise ValueError("Data must have timestamp index or 'timestamp' column")
    
    # Sort by timestamp
    data = data.sort_index()
    
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = data.iloc[:n_train]
    val_data = data.iloc[n_train:n_train + n_val]
    test_data = data.iloc[n_train + n_val:]
    
    print(f"Train: {len(train_data)} samples ({train_data.index.min()} to {train_data.index.max()})")
    print(f"Validation: {len(val_data)} samples ({val_data.index.min()} to {val_data.index.max()})")
    print(f"Test: {len(test_data)} samples ({test_data.index.min()} to {test_data.index.max()})")
    
    return train_data, val_data, test_data

def main():
    """Main function for data cleaning."""
    print("=" * 60)
    print("Phase 1.3: Data Cleaning and Normalization")
    print("=" * 60)
    
    # Load integrated data
    data = load_integrated_data()
    
    # Handle missing data
    data_cleaned = handle_missing_data(data, max_gap_hours=6)
    
    # Remove outliers
    data_cleaned = remove_outliers(data_cleaned, method='iqr', factor=1.5)
    
    # Normalize features
    data_normalized, scaler_info = normalize_features(data_cleaned, method='standard')
    
    # Split into train/val/test
    train_data, val_data, test_data = split_temporal_data(data_normalized)
    
    # Save cleaned and split data
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    train_data.to_csv(OUTPUT_PATH / "train_data.csv")
    val_data.to_csv(OUTPUT_PATH / "val_data.csv")
    test_data.to_csv(OUTPUT_PATH / "test_data.csv")
    
    # Save scaler info (would use pickle in production)
    import json
    scaler_dict = {
        'method': 'standard',
        'feature_columns': scaler_info[1],
        'mean': scaler_info[0].mean_.tolist() if hasattr(scaler_info[0], 'mean_') else None,
        'scale': scaler_info[0].scale_.tolist() if hasattr(scaler_info[0], 'scale_') else None
    }
    with open(OUTPUT_PATH / "scaler_info.json", 'w') as f:
        json.dump(scaler_dict, f, indent=2)
    
    print(f"\nSaved cleaned data:")
    print(f"  Train: {OUTPUT_PATH / 'train_data.csv'}")
    print(f"  Validation: {OUTPUT_PATH / 'val_data.csv'}")
    print(f"  Test: {OUTPUT_PATH / 'test_data.csv'}")
    
    return train_data, val_data, test_data

if __name__ == "__main__":
    train_data, val_data, test_data = main()
