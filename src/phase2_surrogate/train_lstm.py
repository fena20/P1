"""
Phase 2.1: LSTM-based Surrogate Model
Trains an LSTM neural network to predict next-hour energy consumption and indoor temperature.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

# Path configuration
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "models"

class LSTMSurrogate:
    """LSTM-based surrogate model for building energy prediction."""
    
    def __init__(self, sequence_length=24, n_features=None, n_outputs=2):
        """
        Initialize LSTM surrogate model.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequence (default: 24 hours)
        n_features : int
            Number of input features
        n_outputs : int
            Number of output predictions (default: 2 for energy and indoor temp)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.history = None
    
    def create_sequences(self, X, y):
        """
        Create sequences for LSTM input.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features (n_samples, n_features)
        y : np.ndarray
            Target values (n_samples, n_outputs)
        
        Returns:
        --------
        tuple
            (X_seq, y_seq) - sequences for LSTM
        """
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, lstm_units=[64, 32], dropout_rate=0.2):
        """
        Build LSTM model architecture.
        
        Parameters:
        -----------
        lstm_units : list
            Number of units in each LSTM layer
        dropout_rate : float
            Dropout rate for regularization
        """
        model = keras.Sequential()
        
        # First LSTM layer
        model.add(layers.LSTM(
            lstm_units[0],
            return_sequences=len(lstm_units) > 1,
            input_shape=(self.sequence_length, self.n_features)
        ))
        model.add(layers.Dropout(dropout_rate))
        
        # Additional LSTM layers
        for units in lstm_units[1:]:
            model.add(layers.LSTM(units, return_sequences=False))
            model.add(layers.Dropout(dropout_rate))
        
        # Dense layers
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(self.n_outputs))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, data, feature_cols, target_cols):
        """
        Prepare data for training.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
        feature_cols : list
            List of feature column names
        target_cols : list
            List of target column names
        
        Returns:
        --------
        tuple
            (X_seq, y_seq) - prepared sequences
        """
        # Extract features and targets
        X = data[feature_cols].values
        y = data[target_cols].values
        
        # Handle missing values
        X = pd.DataFrame(X).fillna(method='ffill').fillna(method='bfill').values
        y = pd.DataFrame(y).fillna(method='ffill').fillna(method='bfill').values
        
        # Normalize
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        return X_seq, y_seq
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=1):
        """
        Train the LSTM model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training input sequences
        y_train : np.ndarray
            Training targets
        X_val : np.ndarray
            Validation input sequences
        y_val : np.ndarray
            Validation targets
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        verbose : int
            Verbosity level
        """
        if self.model is None:
            self.build_model()
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Model checkpoint
        checkpoint = keras.callbacks.ModelCheckpoint(
            str(OUTPUT_PATH / "lstm_best_model.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint],
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences
        
        Returns:
        --------
        np.ndarray
            Predictions (denormalized)
        """
        y_pred_scaled = self.model.predict(X)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test input sequences
        y_test : np.ndarray
            Test targets
        
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_test_actual = self.scaler_y.inverse_transform(y_test)
        
        mae = mean_absolute_error(y_test_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        r2 = r2_score(y_test_actual, y_pred)
        
        metrics = {
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
    
    # Define feature columns (adjust based on actual data)
    feature_cols = [
        'airTemperature', 'dewTemperature', 'cloudCoverage',
        'precipDepth1HR', 'seaLevelPressure', 'windSpeed',
        'hour', 'day_of_week', 'month'
    ]
    
    # Use available columns
    available_features = [col for col in feature_cols if col in train_data.columns]
    if len(available_features) < 5:
        # Fallback to numeric columns
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['building_id', 'meter_id', 'site_id']
        available_features = [col for col in numeric_cols if col not in exclude][:10]
    
    # Define target columns
    target_cols = ['meter_reading']
    if 'meter_reading' not in train_data.columns:
        # Find energy-related columns
        energy_cols = [col for col in train_data.columns if 'energy' in col.lower() or 'power' in col.lower() or 'meter' in col.lower()]
        if energy_cols:
            target_cols = [energy_cols[0]]
        else:
            # Use first numeric column as placeholder
            target_cols = [available_features[0]]
    
    print(f"Features: {available_features}")
    print(f"Targets: {target_cols}")
    
    return train_data, val_data, test_data, available_features, target_cols

def main():
    """Main training function."""
    print("=" * 60)
    print("Phase 2.1: LSTM Surrogate Model Training")
    print("=" * 60)
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_data, val_data, test_data, feature_cols, target_cols = load_prepared_data()
    
    # Initialize model
    model = LSTMSurrogate(sequence_length=24, n_outputs=len(target_cols))
    model.n_features = len(feature_cols)
    
    # Prepare data
    print("\nPreparing training data...")
    X_train, y_train = model.prepare_data(train_data, feature_cols, target_cols)
    X_val, y_val = model.prepare_data(val_data, feature_cols, target_cols)
    X_test, y_test = model.prepare_data(test_data, feature_cols, target_cols)
    
    print(f"Training sequences: {X_train.shape}")
    print(f"Validation sequences: {X_val.shape}")
    print(f"Test sequences: {X_test.shape}")
    
    # Build model
    print("\nBuilding LSTM model...")
    model.build_model(lstm_units=[64, 32], dropout_rate=0.2)
    model.model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\n" + "=" * 60)
    print("Test Set Performance:")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model and scalers
    model.model.save(str(OUTPUT_PATH / "lstm_model.h5"))
    
    # Save scaler parameters
    scaler_info = {
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'sequence_length': model.sequence_length
    }
    with open(OUTPUT_PATH / "lstm_scaler_info.json", 'w') as f:
        json.dump(scaler_info, f, indent=2)
    
    print(f"\nModel saved to {OUTPUT_PATH}")
    
    return model, metrics

if __name__ == "__main__":
    model, metrics = main()
