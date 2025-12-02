"""
Deep Learning Module for Energy Prediction and Comfort Modeling
Implements LSTM-based prediction with PMV/PPD thermal comfort metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from typing import Tuple, Dict, List, Optional
import time
import os

from config import (
    RANDOM_SEED, DL_CONFIG, COMFORT_CONFIG, 
    MODELS_DIR, TABLES_DIR
)

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


class EnergyDataset(Dataset):
    """PyTorch Dataset for energy prediction sequences."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 24,
        feature_columns: List[str] = None,
        target_column: str = 'meter_reading_scaled'
    ):
        self.sequence_length = sequence_length
        self.target_column = target_column
        
        if feature_columns is None:
            self.feature_columns = [
                'air_temperature_scaled', 'dew_temperature_scaled',
                'wind_speed_scaled', 'square_feet_scaled',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                'month_sin', 'month_cos', 'is_weekend',
                'rolling_mean_24h_scaled', 'rolling_std_24h_scaled'
            ]
        else:
            self.feature_columns = feature_columns
        
        # Filter to available columns
        self.feature_columns = [c for c in self.feature_columns if c in data.columns]
        
        # Group by building and create sequences
        self.sequences = []
        self.targets = []
        self.temp_values = []  # For comfort calculations
        
        for building_id, group in data.groupby('building_id'):
            group = group.sort_values('timestamp')
            features = group[self.feature_columns].values
            targets = group[self.target_column].values
            temps = group['air_temperature'].values if 'air_temperature' in group.columns else np.zeros(len(group))
            
            for i in range(len(group) - sequence_length):
                self.sequences.append(features[i:i+sequence_length])
                self.targets.append(targets[i+sequence_length])
                self.temp_values.append(temps[i+sequence_length])
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.temp_values = np.array(self.temp_values, dtype=np.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.targets[idx]]),
            torch.FloatTensor([self.temp_values[idx]])
        )


class LSTMEnergyPredictor(nn.Module):
    """LSTM-based model for energy consumption prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 2  # Energy + Comfort
    ):
        super(LSTMEnergyPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_energy = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.fc_comfort = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output comfort score between 0 and 1
        )
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Predict energy
        energy_pred = self.fc_energy(last_hidden)
        
        # Predict comfort index (normalized PMV)
        comfort_pred = self.fc_comfort(last_hidden)
        
        return energy_pred, comfort_pred


def calculate_pmv(
    temperature: float,
    metabolic_rate: float = COMFORT_CONFIG['metabolic_rate'],
    clothing_insulation: float = COMFORT_CONFIG['clothing_insulation'],
    air_velocity: float = COMFORT_CONFIG['air_velocity'],
    relative_humidity: float = 50.0
) -> float:
    """
    Simplified PMV calculation inspired by ISO 7730.
    Full PMV equation is complex; this is a linearized approximation.
    
    Returns:
        PMV value (-3 cold to +3 hot)
    """
    # Neutral temperature based on clothing and metabolic rate
    t_neutral = 21.0 + (clothing_insulation - 0.5) * 2 - (metabolic_rate - 1.0) * 3
    
    # Simplified PMV calculation
    pmv = 0.303 * np.exp(-0.036 * metabolic_rate * 58.15) * (
        (metabolic_rate * 58.15 - 3.05e-3 * (5733 - 6.99 * metabolic_rate * 58.15 - 
         relative_humidity * 10 * np.exp(16.6536 - 4030.183 / (temperature + 235)))) -
        (clothing_insulation * 0.155 * 3.96e-8 * ((temperature + 273)**4 - 
         (temperature + 273 - 3)**4)) -
        (clothing_insulation * 0.155 * 2.38 * abs(temperature - t_neutral)**0.25 * 
         (temperature - t_neutral))
    )
    
    # Clamp to reasonable range
    pmv = np.clip(pmv, -3, 3)
    
    # Simplified linear approximation for faster computation
    pmv_simple = 0.2 * (temperature - t_neutral)
    pmv_simple = np.clip(pmv_simple, -3, 3)
    
    return pmv_simple


def calculate_ppd(pmv: float) -> float:
    """
    Calculate Predicted Percentage of Dissatisfied (PPD) from PMV.
    Based on ISO 7730 equation.
    
    Returns:
        PPD value (0-100%)
    """
    ppd = 100 - 95 * np.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
    return np.clip(ppd, 5, 100)  # PPD is never below 5%


def comfort_loss(
    pred_comfort: torch.Tensor,
    temperature: torch.Tensor,
    ppd_threshold: float = COMFORT_CONFIG['ppd_threshold']
) -> torch.Tensor:
    """
    Custom loss function for thermal comfort.
    Penalizes predictions when PPD exceeds threshold.
    """
    # Calculate PPD from temperature (simplified)
    t_neutral = 21.0
    pmv = 0.2 * (temperature - t_neutral)
    ppd = 100 - 95 * torch.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
    
    # Normalize PPD to 0-1 range
    ppd_normalized = ppd / 100.0
    
    # Loss: MSE between predicted comfort and actual comfort
    # Plus penalty for exceeding threshold
    comfort_score = 1 - ppd_normalized
    mse_loss = nn.functional.mse_loss(pred_comfort, comfort_score.unsqueeze(-1))
    
    # Penalty for exceeding threshold
    threshold_exceeded = torch.relu(ppd - ppd_threshold)
    penalty = threshold_exceeded.mean() * 0.01
    
    return mse_loss + penalty


class EnergyPredictionTrainer:
    """Trainer class for the LSTM energy prediction model."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = DL_CONFIG['learning_rate']
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.energy_criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        comfort_weight: float = DL_CONFIG['comfort_loss_weight']
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (sequences, targets, temps) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            temps = temps.to(self.device)
            
            self.optimizer.zero_grad()
            
            energy_pred, comfort_pred = self.model(sequences)
            
            # Combined loss
            energy_loss = self.energy_criterion(energy_pred, targets)
            c_loss = comfort_loss(comfort_pred, temps)
            loss = energy_loss + comfort_weight * c_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float, Dict]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_temps = []
        all_comfort_preds = []
        
        with torch.no_grad():
            for sequences, targets, temps in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                temps = temps.to(self.device)
                
                energy_pred, comfort_pred = self.model(sequences)
                
                loss = self.energy_criterion(energy_pred, targets)
                total_loss += loss.item()
                
                all_preds.extend(energy_pred.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                all_temps.extend(temps.cpu().numpy().flatten())
                all_comfort_preds.extend(comfort_pred.cpu().numpy().flatten())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_temps = np.array(all_temps)
        
        r2 = r2_score(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        
        # Calculate comfort metrics
        ppd_values = np.array([calculate_ppd(calculate_pmv(t)) for t in all_temps])
        ppd_below_threshold = (ppd_values < COMFORT_CONFIG['ppd_threshold']).mean() * 100
        
        metrics = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'ppd_below_threshold': ppd_below_threshold,
            'avg_ppd': ppd_values.mean()
        }
        
        return total_loss / len(val_loader), r2, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = DL_CONFIG['epochs'],
        patience: int = DL_CONFIG['early_stopping_patience']
    ) -> Dict:
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print("Starting training...")
        start_time = time.time()
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_r2, metrics = self.evaluate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_r2'].append(val_r2)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        training_time = time.time() - start_time
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        return {'training_time': training_time, 'best_val_loss': best_val_loss}


def train_baseline_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Train a simple linear regression baseline."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def generate_model_comparison_table(
    results: Dict,
    save_path: str
) -> pd.DataFrame:
    """Generate Table 2: Model performance comparison."""
    rows = []
    
    for model_name, metrics in results.items():
        rows.append({
            'Model': model_name,
            'R²': round(metrics['r2'], 4),
            'MAE': round(metrics['mae'], 4),
            'RMSE': round(metrics['rmse'], 4),
            'Training Time (s)': round(metrics.get('training_time', 0), 2),
            'Comfort (PPD < 10%)': round(metrics.get('ppd_below_threshold', 0), 2)
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Table 2 saved to {save_path}")
    
    return df


def export_model_torchscript(
    model: nn.Module,
    input_size: int,
    sequence_length: int,
    save_path: str
) -> None:
    """Export model to TorchScript for edge deployment."""
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, sequence_length, input_size)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save the traced model
    traced_model.save(save_path)
    print(f"TorchScript model saved to {save_path}")


if __name__ == "__main__":
    # Test the module
    print("Testing Deep Learning module...")
    
    # Test PMV/PPD calculations
    for temp in [18, 20, 22, 24, 26, 28]:
        pmv = calculate_pmv(temp)
        ppd = calculate_ppd(pmv)
        print(f"Temperature: {temp}°C -> PMV: {pmv:.2f}, PPD: {ppd:.1f}%")
