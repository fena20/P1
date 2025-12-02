"""
Phase 2: Surrogate Model Development

This module implements:
1. LSTM-based temporal prediction model
2. XGBoost gradient-boosted model  
3. Multi-building training and fine-tuning
4. Model evaluation and comparison
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers, models, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available, LSTM models will be skipped")


class SurrogateModelBuilder:
    """Build and train surrogate models for building energy prediction."""
    
    def __init__(self, data_path: str = "/workspace/data"):
        self.data_path = Path(data_path)
        self.models_path = Path("/workspace/models")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self.feature_cols = [
            'airTemperature', 'windSpeed', 'hour', 'day_of_week', 'month'
        ]
        self.target_col = 'meter_reading'
        
    def load_building_data(self, building_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train/val/test splits for a building."""
        splits_dir = self.data_path / "splits" / building_id
        
        train_df = pd.read_csv(splits_dir / "train.csv")
        val_df = pd.read_csv(splits_dir / "val.csv")
        test_df = pd.read_csv(splits_dir / "test.csv")
        
        # Convert timestamp
        for df in [train_df, val_df, test_df]:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return train_df, val_df, test_df
    
    def prepare_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                        test_df: pd.DataFrame) -> Tuple:
        """Prepare and normalize features."""
        
        # Select available features
        available_features = [col for col in self.feature_cols if col in train_df.columns]
        
        # Extract features and targets
        X_train = train_df[available_features].fillna(method='ffill').fillna(method='bfill')
        y_train = train_df[self.target_col].values
        
        X_val = val_df[available_features].fillna(method='ffill').fillna(method='bfill')
        y_val = val_df[self.target_col].values
        
        X_test = test_df[available_features].fillna(method='ffill').fillna(method='bfill')
        y_test = test_df[self.target_col].values
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Normalize target (for stable training)
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        return (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 
                X_test_scaled, y_test_scaled, scaler, target_scaler, available_features)
    
    def create_sequence_data(self, X: np.ndarray, y: np.ndarray, 
                            lookback: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X_seq, y_seq = [], []
        
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_lstm_model(self, input_shape: Tuple, name: str = "lstm_energy") -> models.Model:
        """Build LSTM model architecture."""
        
        model = models.Sequential(name=name)
        
        # LSTM layers
        model.add(layers.LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(32, return_sequences=False))
        model.add(layers.Dropout(0.2))
        
        # Dense layers
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_lstm(self, building_id: str, lookback: int = 24, 
                   epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train LSTM model for a building."""
        
        if not TF_AVAILABLE:
            print("TensorFlow not available, skipping LSTM training")
            return {}
        
        print(f"\n=== Training LSTM Model for {building_id} ===")
        
        # Load data
        train_df, val_df, test_df = self.load_building_data(building_id)
        
        # Prepare features
        (X_train, y_train, X_val, y_val, X_test, y_test, 
         scaler, target_scaler, features) = self.prepare_features(train_df, val_df, test_df)
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequence_data(X_train, y_train, lookback)
        X_val_seq, y_val_seq = self.create_sequence_data(X_val, y_val, lookback)
        X_test_seq, y_test_seq = self.create_sequence_data(X_test, y_test, lookback)
        
        print(f"  Train sequences: {X_train_seq.shape}")
        print(f"  Val sequences: {X_val_seq.shape}")
        print(f"  Test sequences: {X_test_seq.shape}")
        
        # Build model
        input_shape = (lookback, X_train.shape[1])
        model = self.build_lstm_model(input_shape, f"lstm_{building_id}")
        
        print(f"  Model parameters: {model.count_params():,}")
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Train
        print("  Training...")
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        y_pred_val_scaled = model.predict(X_val_seq, verbose=0).flatten()
        y_pred_test_scaled = model.predict(X_test_seq, verbose=0).flatten()
        
        # Inverse transform
        y_pred_val = target_scaler.inverse_transform(y_pred_val_scaled.reshape(-1, 1)).flatten()
        y_pred_test = target_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
        
        y_val_orig = target_scaler.inverse_transform(y_val_seq.reshape(-1, 1)).flatten()
        y_test_orig = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
        
        # Metrics
        val_metrics = self.compute_metrics(y_val_orig, y_pred_val, "Validation")
        test_metrics = self.compute_metrics(y_test_orig, y_pred_test, "Test")
        
        # Save model
        model_dir = self.models_path / "lstm" / building_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model.save(str(model_dir / "model.keras"))
        joblib.dump(scaler, model_dir / "feature_scaler.pkl")
        joblib.dump(target_scaler, model_dir / "target_scaler.pkl")
        
        # Save metadata
        metadata = {
            'building_id': building_id,
            'model_type': 'LSTM',
            'lookback': lookback,
            'features': features,
            'input_shape': list(input_shape),
            'epochs_trained': len(history.history['loss']),
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Model saved to {model_dir}")
        
        return metadata
    
    def train_xgboost(self, building_id: str, lookback: int = 24) -> Dict:
        """Train XGBoost model for a building."""
        
        print(f"\n=== Training XGBoost Model for {building_id} ===")
        
        # Load data
        train_df, val_df, test_df = self.load_building_data(building_id)
        
        # Prepare features
        (X_train, y_train, X_val, y_val, X_test, y_test, 
         scaler, target_scaler, features) = self.prepare_features(train_df, val_df, test_df)
        
        # Create lag features (alternative to sequences for XGBoost)
        X_train_aug = self.create_lag_features(X_train, lookback)
        X_val_aug = self.create_lag_features(X_val, lookback)
        X_test_aug = self.create_lag_features(X_test, lookback)
        
        # Remove rows with NaN from lagging
        X_train_aug = X_train_aug[lookback:]
        y_train_aug = y_train[lookback:]
        X_val_aug = X_val_aug[lookback:]
        y_val_aug = y_val[lookback:]
        X_test_aug = X_test_aug[lookback:]
        y_test_aug = y_test[lookback:]
        
        print(f"  Train samples: {X_train_aug.shape}")
        print(f"  Val samples: {X_val_aug.shape}")
        print(f"  Test samples: {X_test_aug.shape}")
        
        # Train XGBoost
        print("  Training...")
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=20
        )
        
        model.fit(
            X_train_aug, y_train_aug,
            eval_set=[(X_val_aug, y_val_aug)],
            verbose=False
        )
        
        # Predict
        y_pred_val_scaled = model.predict(X_val_aug)
        y_pred_test_scaled = model.predict(X_test_aug)
        
        # Inverse transform
        y_pred_val = target_scaler.inverse_transform(y_pred_val_scaled.reshape(-1, 1)).flatten()
        y_pred_test = target_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
        
        y_val_orig = target_scaler.inverse_transform(y_val_aug.reshape(-1, 1)).flatten()
        y_test_orig = target_scaler.inverse_transform(y_test_aug.reshape(-1, 1)).flatten()
        
        # Metrics
        val_metrics = self.compute_metrics(y_val_orig, y_pred_val, "Validation")
        test_metrics = self.compute_metrics(y_test_orig, y_pred_test, "Test")
        
        # Feature importance (convert to float for JSON)
        feature_importance = dict(zip(
            [f"feature_{i}" for i in range(X_train_aug.shape[1])],
            [float(x) for x in model.feature_importances_]
        ))
        
        # Save model
        model_dir = self.models_path / "xgboost" / building_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, model_dir / "model.pkl")
        joblib.dump(scaler, model_dir / "feature_scaler.pkl")
        joblib.dump(target_scaler, model_dir / "target_scaler.pkl")
        
        # Save metadata
        metadata = {
            'building_id': building_id,
            'model_type': 'XGBoost',
            'lookback': lookback,
            'features': features,
            'n_features': X_train_aug.shape[1],
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Model saved to {model_dir}")
        
        return metadata
    
    def create_lag_features(self, X: np.ndarray, lookback: int) -> np.ndarray:
        """Create lagged features for XGBoost."""
        n_samples, n_features = X.shape
        X_lagged = np.zeros((n_samples, n_features * (lookback + 1)))
        
        for i in range(n_samples):
            for lag in range(lookback + 1):
                if i - lag >= 0:
                    X_lagged[i, lag*n_features:(lag+1)*n_features] = X[i-lag]
        
        return X_lagged
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       dataset_name: str = "") -> Dict:
        """Compute regression metrics."""
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # CVRMSE (Coefficient of Variation of RMSE)
        cvrmse = (rmse / np.mean(y_true)) * 100
        
        # NMBE (Normalized Mean Bias Error)
        nmbe = (np.mean(y_pred - y_true) / np.mean(y_true)) * 100
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'cvrmse': float(cvrmse),
            'nmbe': float(nmbe)
        }
        
        if dataset_name:
            print(f"\n  {dataset_name} Metrics:")
            print(f"    RMSE: {rmse:.2f} kWh")
            print(f"    MAE: {mae:.2f} kWh")
            print(f"    R²: {r2:.4f}")
            print(f"    CV(RMSE): {cvrmse:.2f}%")
            print(f"    NMBE: {nmbe:.2f}%")
        
        return metrics


def main():
    """Train surrogate models for all buildings."""
    
    print("="*80)
    print("PHASE 2: SURROGATE MODEL DEVELOPMENT")
    print("="*80)
    
    # Get list of processed buildings
    splits_dir = Path("/workspace/data/splits")
    building_ids = [d.name for d in splits_dir.iterdir() if d.is_dir()]
    
    print(f"\nFound {len(building_ids)} buildings to process:")
    for bid in building_ids:
        print(f"  - {bid}")
    
    # Initialize builder
    builder = SurrogateModelBuilder()
    
    # Train models for each building
    all_results = {
        'lstm': {},
        'xgboost': {}
    }
    
    for building_id in building_ids:
        print(f"\n{'='*80}")
        print(f"Processing: {building_id}")
        print(f"{'='*80}")
        
        # Train LSTM
        try:
            lstm_results = builder.train_lstm(building_id, lookback=24, epochs=50)
            all_results['lstm'][building_id] = lstm_results
        except Exception as e:
            print(f"  LSTM training failed: {e}")
            all_results['lstm'][building_id] = {'error': str(e)}
        
        # Train XGBoost
        try:
            xgb_results = builder.train_xgboost(building_id, lookback=24)
            all_results['xgboost'][building_id] = xgb_results
        except Exception as e:
            print(f"  XGBoost training failed: {e}")
            all_results['xgboost'][building_id] = {'error': str(e)}
    
    # Save summary
    results_file = Path("/workspace/results/model_training_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("PHASE 2 COMPLETE")
    print("="*80)
    print(f"Results saved to {results_file}")
    
    # Print summary
    print("\n=== Model Performance Summary ===")
    for model_type in ['lstm', 'xgboost']:
        print(f"\n{model_type.upper()} Models:")
        for building_id, results in all_results[model_type].items():
            if 'test_metrics' in results:
                print(f"  {building_id}:")
                print(f"    Test R²: {results['test_metrics']['r2']:.4f}")
                print(f"    Test CVRMSE: {results['test_metrics']['cvrmse']:.2f}%")


if __name__ == "__main__":
    main()
