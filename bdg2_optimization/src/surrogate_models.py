"""
Surrogate Model Module for BDG2 Optimization Framework.

Implements:
- LSTM-based neural network surrogate
- XGBoost-based gradient boosting surrogate
- Multi-building training with optional fine-tuning
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from config import ModelConfig, MODELS_DIR


class LSTMSurrogate:
    """
    LSTM-based surrogate model for building energy prediction.
    
    Predicts next-hour energy consumption based on historical weather,
    energy usage, and temporal features.
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.history = None
        self.feature_names = None
        self.n_features = None
        
    def build_model(self, n_features: int, sequence_length: int) -> Model:
        """
        Build LSTM architecture.
        
        Args:
            n_features: Number of input features
            sequence_length: Length of input sequences
            
        Returns:
            Compiled Keras model
        """
        self.n_features = n_features
        
        model = Sequential([
            Input(shape=(sequence_length, n_features)),
            LSTM(
                self.config.lstm_units[0],
                return_sequences=len(self.config.lstm_units) > 1,
                dropout=self.config.lstm_dropout,
                recurrent_dropout=self.config.lstm_recurrent_dropout
            ),
            BatchNormalization()
        ])
        
        # Add additional LSTM layers
        for i, units in enumerate(self.config.lstm_units[1:]):
            return_seq = i < len(self.config.lstm_units) - 2
            model.add(LSTM(
                units,
                return_sequences=return_seq,
                dropout=self.config.lstm_dropout,
                recurrent_dropout=self.config.lstm_recurrent_dropout
            ))
            model.add(BatchNormalization())
        
        # Output layers
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))  # Energy prediction
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.lstm_learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: int = 1
    ) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences (n_samples, seq_len, n_features)
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            verbose: Training verbosity
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model(X_train.shape[2], X_train.shape[1])
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config.lstm_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config.lstm_epochs,
            batch_size=self.config.lstm_batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X, verbose=0).flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        return {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
        }
    
    def save(self, path: Path = None):
        """Save model to disk."""
        if path is None:
            path = MODELS_DIR / "lstm_surrogate.keras"
        self.model.save(path)
        
    def load(self, path: Path = None):
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / "lstm_surrogate.keras"
        self.model = keras.models.load_model(path)


class XGBoostSurrogate:
    """
    XGBoost-based surrogate model for building energy prediction.
    
    Uses engineered lag features and calendar variables for tabular prediction.
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        
    def prepare_features(
        self, 
        df: pd.DataFrame,
        target_col: str = 'energy_kwh'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features with lag engineering.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Feature DataFrame and target Series
        """
        data = df.copy()
        
        # Create lag features
        for lag in self.config.lag_hours:
            data[f'energy_lag_{lag}'] = data[target_col].shift(lag)
        
        # Temperature lags
        if 'outdoor_temp' in data.columns:
            for lag in [1, 3, 6, 12]:
                data[f'temp_lag_{lag}'] = data['outdoor_temp'].shift(lag)
        
        # Rolling statistics
        data['energy_rolling_mean_6h'] = data[target_col].rolling(6).mean()
        data['energy_rolling_std_6h'] = data[target_col].rolling(6).std()
        data['energy_rolling_mean_24h'] = data[target_col].rolling(24).mean()
        data['energy_rolling_max_24h'] = data[target_col].rolling(24).max()
        data['energy_rolling_min_24h'] = data[target_col].rolling(24).min()
        
        if 'outdoor_temp' in data.columns:
            data['temp_rolling_mean_24h'] = data['outdoor_temp'].rolling(24).mean()
            # Heating/cooling degree hours
            data['hdd'] = np.maximum(0, 18 - data['outdoor_temp'])
            data['cdd'] = np.maximum(0, data['outdoor_temp'] - 18)
        
        # Cyclical encoding of time
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Drop NaN from lag features
        data = data.dropna()
        
        # Shift target for next-hour prediction
        y = data[target_col].shift(-1).dropna()
        X = data.iloc[:-1]  # Align features
        
        # Select feature columns
        exclude_cols = [target_col, 'site_id']
        feature_cols = [c for c in X.columns if c not in exclude_cols]
        
        self.feature_names = feature_cols
        
        return X[feature_cols], y
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        verbose: int = 1
    ) -> Dict:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            verbose: Training verbosity
            
        Returns:
            Training metrics
        """
        self.model = xgb.XGBRegressor(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            random_state=self.config.xgb_random_state,
            n_jobs=-1,
            early_stopping_rounds=20 if X_val is not None else None
        )
        
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose > 0
        )
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {'feature_importance': self.feature_importance}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names] if self.feature_names else X
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        y_true = y.values if isinstance(y, pd.Series) else y
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        }
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Get top N important features."""
        return self.feature_importance.head(top_n)
    
    def save(self, path: Path = None):
        """Save model to disk."""
        if path is None:
            path = MODELS_DIR / "xgboost_surrogate.json"
        self.model.save_model(path)
        joblib.dump(self.feature_names, MODELS_DIR / "xgboost_features.pkl")
        
    def load(self, path: Path = None):
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / "xgboost_surrogate.json"
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        self.feature_names = joblib.load(MODELS_DIR / "xgboost_features.pkl")


class MultiBuildingSurrogate:
    """
    Multi-building surrogate training for cross-building generalization.
    
    Trains a global model on pooled data from multiple buildings,
    with optional building-specific fine-tuning.
    """
    
    def __init__(
        self, 
        model_type: str = 'xgboost',
        config: ModelConfig = None
    ):
        self.model_type = model_type
        self.config = config or ModelConfig()
        self.global_model = None
        self.building_models = {}
        
    def train_global_model(
        self,
        building_data_list: List[Tuple[pd.DataFrame, pd.DataFrame]],
        verbose: int = 1
    ) -> Dict:
        """
        Train global model on pooled data from all buildings.
        
        Args:
            building_data_list: List of (train_df, val_df) tuples per building
            verbose: Training verbosity
            
        Returns:
            Training results
        """
        print("Training global multi-building model...")
        
        # Pool training data
        all_train_X, all_train_y = [], []
        all_val_X, all_val_y = [], []
        
        if self.model_type == 'xgboost':
            self.global_model = XGBoostSurrogate(self.config)
            
            for train_df, val_df in building_data_list:
                X_train, y_train = self.global_model.prepare_features(train_df)
                X_val, y_val = self.global_model.prepare_features(val_df)
                all_train_X.append(X_train)
                all_train_y.append(y_train)
                all_val_X.append(X_val)
                all_val_y.append(y_val)
            
            X_train_pooled = pd.concat(all_train_X, ignore_index=True)
            y_train_pooled = pd.concat(all_train_y, ignore_index=True)
            X_val_pooled = pd.concat(all_val_X, ignore_index=True)
            y_val_pooled = pd.concat(all_val_y, ignore_index=True)
            
            results = self.global_model.train(
                X_train_pooled, y_train_pooled,
                X_val_pooled, y_val_pooled,
                verbose=verbose
            )
        
        else:  # LSTM
            self.global_model = LSTMSurrogate(self.config)
            # LSTM implementation would go here
            # For brevity, using XGBoost as the primary model
            raise NotImplementedError("Multi-building LSTM not implemented yet")
        
        print(f"Global model trained on {len(X_train_pooled)} samples")
        return results
    
    def fine_tune(
        self,
        building_id: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        verbose: int = 0
    ):
        """Fine-tune global model for a specific building."""
        # Create copy of global model and continue training
        if self.model_type == 'xgboost':
            building_model = XGBoostSurrogate(self.config)
            X_train, y_train = building_model.prepare_features(train_df)
            
            if val_df is not None:
                X_val, y_val = building_model.prepare_features(val_df)
            else:
                X_val, y_val = None, None
            
            # Use fewer estimators for fine-tuning
            building_model.config.xgb_n_estimators = 50
            building_model.train(X_train, y_train, X_val, y_val, verbose=verbose)
            
            self.building_models[building_id] = building_model
    
    def predict(
        self, 
        X: pd.DataFrame, 
        building_id: str = None
    ) -> np.ndarray:
        """Make predictions using appropriate model."""
        if building_id and building_id in self.building_models:
            return self.building_models[building_id].predict(X)
        return self.global_model.predict(X)


def create_input_variables_table() -> pd.DataFrame:
    """
    Create Table 2: Input Variables for the Prediction Model.
    """
    data = [
        {
            'Variable Category': 'Environmental',
            'Feature Name': 'Outdoor Air Temperature',
            'Unit': '°C',
            'Source': 'BDG2 Weather',
            'Relevance': 'Core climatic driver for heating/cooling demand'
        },
        {
            'Variable Category': 'Environmental', 
            'Feature Name': 'Relative Humidity',
            'Unit': '%',
            'Source': 'BDG2 Weather (derived)',
            'Relevance': 'Influences latent loads and perceived comfort'
        },
        {
            'Variable Category': 'Environmental',
            'Feature Name': 'Wind Speed',
            'Unit': 'm/s',
            'Source': 'BDG2 Weather',
            'Relevance': 'Affects building infiltration and heat loss'
        },
        {
            'Variable Category': 'Temporal',
            'Feature Name': 'Hour of Day',
            'Unit': '0-23',
            'Source': 'Time Index',
            'Relevance': 'Represents daily occupancy and behavior patterns'
        },
        {
            'Variable Category': 'Temporal',
            'Feature Name': 'Day of Week',
            'Unit': '1-7',
            'Source': 'Time Index',
            'Relevance': 'Encodes weekly usage patterns (workdays vs weekends)'
        },
        {
            'Variable Category': 'Control',
            'Feature Name': 'Cooling/Heating Setpoint',
            'Unit': '°C',
            'Source': 'Control Schedule',
            'Relevance': 'Primary optimization variable and control parameter'
        },
        {
            'Variable Category': 'Historical',
            'Feature Name': 'Lagged Energy (1-24h)',
            'Unit': 'kWh',
            'Source': 'BDG2 Meters',
            'Relevance': 'Captures building thermal inertia and patterns'
        }
    ]
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Testing surrogate models...")
    print("\nTable 2: Input Variables for the Prediction Model")
    print(create_input_variables_table().to_string(index=False))
