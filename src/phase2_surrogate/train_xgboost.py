"""
Phase 2.2: XGBoost-based Surrogate Model
Trains a gradient-boosted decision tree model with engineered lag features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

# Path configuration
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "models"

class XGBoostSurrogate:
    """XGBoost-based surrogate model for building energy prediction."""
    
    def __init__(self, max_lag=24):
        """
        Initialize XGBoost surrogate model.
        
        Parameters:
        -----------
        max_lag : int
            Maximum lag for feature engineering (default: 24 hours)
        """
        self.max_lag = max_lag
        self.model_energy = None
        self.model_temp = None
        self.feature_cols = None
    
    def create_lag_features(self, data, target_col, feature_cols):
        """
        Create lag features for time series prediction.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
        target_col : str
            Target column name
        feature_cols : list
            List of feature column names
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with lag features
        """
        df = data.copy()
        
        # Create lag features for target variable
        for lag in [1, 3, 6, 12, 24]:
            if lag <= self.max_lag:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Create lag features for key features
        key_features = [col for col in feature_cols if any(x in col.lower() for x in ['temp', 'energy', 'meter'])]
        for feature in key_features[:3]:  # Limit to first 3 key features
            for lag in [1, 6, 24]:
                if lag <= self.max_lag:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        # Rolling statistics
        for window in [6, 24]:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        
        return df
    
    def prepare_features(self, data, feature_cols, target_cols):
        """
        Prepare features for XGBoost training.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
        feature_cols : list
            List of base feature column names
        target_cols : list
            List of target column names
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with engineered features
        """
        df = data.copy()
        
        # Create lag features for primary target
        primary_target = target_cols[0]
        df = self.create_lag_features(df, primary_target, feature_cols)
        
        # Select all feature columns (original + lag features)
        all_feature_cols = feature_cols.copy()
        
        # Add lag feature columns
        lag_cols = [col for col in df.columns if '_lag_' in col or '_rolling_' in col]
        all_feature_cols.extend(lag_cols)
        
        # Remove any columns that don't exist
        all_feature_cols = [col for col in all_feature_cols if col in df.columns]
        
        self.feature_cols = all_feature_cols
        
        return df
    
    def train(self, train_data, val_data, feature_cols, target_cols, params=None):
        """
        Train XGBoost models for each target.
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        val_data : pd.DataFrame
            Validation data
        feature_cols : list
            Base feature columns
        target_cols : list
            Target columns
        params : dict, optional
            XGBoost parameters
        
        Returns:
        --------
        dict
            Training history
        """
        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        # Prepare features
        train_prep = self.prepare_features(train_data, feature_cols, target_cols)
        val_prep = self.prepare_features(val_data, feature_cols, target_cols)
        
        # Remove rows with NaN (from lag features)
        train_prep = train_prep.dropna()
        val_prep = val_prep.dropna()
        
        X_train = train_prep[self.feature_cols]
        X_val = val_prep[self.feature_cols]
        
        models = {}
        histories = {}
        
        for target_col in target_cols:
            print(f"\nTraining model for {target_col}...")
            
            y_train = train_prep[target_col]
            y_val = val_prep[target_col]
            
            # Train model
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            models[target_col] = model
            
            # Get feature importance
            importance = model.feature_importances_
            feature_importance = dict(zip(self.feature_cols, importance))
            histories[target_col] = {
                'feature_importance': feature_importance,
                'best_iteration': model.best_iteration
            }
        
        # Store primary model (for energy prediction)
        self.model_energy = models[target_cols[0]]
        if len(target_cols) > 1:
            self.model_temp = models[target_cols[1]]
        
        return histories
    
    def predict(self, data, feature_cols, target_cols):
        """
        Make predictions.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        feature_cols : list
            Base feature columns
        target_cols : list
            Target columns
        
        Returns:
        --------
        dict
            Predictions for each target
        """
        data_prep = self.prepare_features(data, feature_cols, target_cols)
        data_prep = data_prep.dropna()
        
        if len(data_prep) == 0:
            return {col: np.array([]) for col in target_cols}
        
        X = data_prep[self.feature_cols]
        
        predictions = {}
        for target_col in target_cols:
            if target_col == target_cols[0]:
                model = self.model_energy
            elif len(target_cols) > 1 and target_col == target_cols[1]:
                model = self.model_temp
            else:
                continue
            
            pred = model.predict(X)
            predictions[target_col] = pred
        
        return predictions
    
    def evaluate(self, test_data, feature_cols, target_cols):
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            Test data
        feature_cols : list
            Base feature columns
        target_cols : list
            Target columns
        
        Returns:
        --------
        dict
            Evaluation metrics for each target
        """
        predictions = self.predict(test_data, feature_cols, target_cols)
        
        metrics = {}
        for target_col in target_cols:
            if target_col not in predictions:
                continue
            
            test_prep = self.prepare_features(test_data, feature_cols, target_cols)
            test_prep = test_prep.dropna()
            
            if len(test_prep) == 0:
                continue
            
            y_true = test_prep[target_col].values
            y_pred = predictions[target_col]
            
            # Align lengths
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            metrics[target_col] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
        
        return metrics

def load_prepared_data():
    """Load and prepare data for training."""
    print("Loading data...")
    
    train_data = pd.read_csv(DATA_PATH / "train_data.csv", index_col=0, parse_dates=True)
    val_data = pd.read_csv(DATA_PATH / "val_data.csv", index_col=0, parse_dates=True)
    test_data = pd.read_csv(DATA_PATH / "test_data.csv", index_col=0, parse_dates=True)
    
    # Define feature columns
    feature_cols = [
        'airTemperature', 'dewTemperature', 'cloudCoverage',
        'precipDepth1HR', 'seaLevelPressure', 'windSpeed',
        'hour', 'day_of_week', 'month'
    ]
    
    # Use available columns
    available_features = [col for col in feature_cols if col in train_data.columns]
    if len(available_features) < 5:
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['building_id', 'meter_id', 'site_id']
        available_features = [col for col in numeric_cols if col not in exclude][:10]
    
    # Define target columns
    target_cols = ['meter_reading']
    if 'meter_reading' not in train_data.columns:
        energy_cols = [col for col in train_data.columns if 'energy' in col.lower() or 'power' in col.lower() or 'meter' in col.lower()]
        if energy_cols:
            target_cols = [energy_cols[0]]
        else:
            target_cols = [available_features[0]]
    
    print(f"Features: {available_features}")
    print(f"Targets: {target_cols}")
    
    return train_data, val_data, test_data, available_features, target_cols

def main():
    """Main training function."""
    print("=" * 60)
    print("Phase 2.2: XGBoost Surrogate Model Training")
    print("=" * 60)
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_data, val_data, test_data, feature_cols, target_cols = load_prepared_data()
    
    # Initialize model
    model = XGBoostSurrogate(max_lag=24)
    
    # Train model
    print("\nTraining XGBoost models...")
    histories = model.train(
        train_data, val_data,
        feature_cols, target_cols
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = model.evaluate(test_data, feature_cols, target_cols)
    
    print("\n" + "=" * 60)
    print("Test Set Performance:")
    print("=" * 60)
    for target, target_metrics in metrics.items():
        print(f"\n{target}:")
        for metric, value in target_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save models
    import pickle
    with open(OUTPUT_PATH / "xgboost_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # Save model info
    model_info = {
        'feature_cols': model.feature_cols,
        'target_cols': target_cols,
        'max_lag': model.max_lag,
        'histories': histories
    }
    with open(OUTPUT_PATH / "xgboost_model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    
    print(f"\nModel saved to {OUTPUT_PATH}")
    
    return model, metrics

if __name__ == "__main__":
    model, metrics = main()
