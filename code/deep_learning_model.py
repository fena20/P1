"""
Deep Learning Module for Energy Prediction
LSTM-based model with comfort prediction (PMV/PPD)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
import config

# Set matplotlib style
plt.style.use(config.STYLE)

class EnergyDataset(Dataset):
    """Dataset for energy prediction"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EnergyLSTM(nn.Module):
    """LSTM model for energy and comfort prediction"""
    def __init__(self, input_size, hidden_size=config.DL_HIDDEN_SIZE, 
                 num_layers=config.DL_NUM_LAYERS, dropout=config.DL_DROPOUT):
        super(EnergyLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layers
        self.fc_energy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.fc_comfort = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)  # PMV and PPD
        )
    
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Predictions
        energy_pred = self.fc_energy(context)
        comfort_pred = self.fc_comfort(context)
        
        return energy_pred, comfort_pred

def calculate_pmv_ppd(temp, humidity, activity=1.2, clothing=0.5):
    """
    Calculate PMV (Predicted Mean Vote) and PPD (Predicted Percentage Dissatisfied)
    Simplified version based on ISO 7730
    
    Args:
        temp: Air temperature (°C)
        humidity: Relative humidity (%)
        activity: Metabolic rate (met)
        clothing: Clothing insulation (clo)
    
    Returns:
        pmv, ppd
    """
    # Simplified PMV calculation
    temp_comfort = 22.0  # Optimal comfort temperature
    pmv = 0.303 * np.exp(-0.036 * activity * 58.15) + 0.028
    pmv *= (activity * 58.15) - 3.05e-3 * (5733 - 6.99 * activity * 58.15 - humidity)
    pmv -= 0.42 * (activity * 58.15 - 58.15)
    pmv -= 1.7e-5 * activity * (5867 - humidity)
    pmv -= 0.0014 * activity * (34 - temp)
    pmv -= 3.96e-8 * clothing * ((temp + 273)**4 - (temp_comfort + 273)**4)
    pmv -= clothing * (temp - temp_comfort)
    
    # Normalize PMV
    pmv = np.clip(pmv / 100, -3, 3)
    
    # Calculate PPD from PMV
    ppd = 100 - 95 * np.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
    ppd = np.clip(ppd, 5, 100)
    
    return pmv, ppd

def prepare_sequences(df, sequence_length=24):
    """Prepare sequences for LSTM"""
    feature_cols = config.DL_INPUT_FEATURES
    
    # Ensure all required columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X_list = []
    y_list = []
    
    # Group by building
    for building_id in df['building_id'].unique():
        building_data = df[df['building_id'] == building_id].copy()
        building_data = building_data.sort_values('timestamp')
        
        # Calculate PMV and PPD
        pmv, ppd = calculate_pmv_ppd(
            building_data['air_temperature'].values,
            building_data['humidity'].values
        )
        building_data['pmv'] = pmv
        building_data['ppd'] = ppd
        
        # Create sequences
        for i in range(len(building_data) - sequence_length):
            X_seq = building_data.iloc[i:i+sequence_length][feature_cols].values
            y_energy = building_data.iloc[i+sequence_length]['meter_reading']
            y_pmv = building_data.iloc[i+sequence_length]['pmv']
            y_ppd = building_data.iloc[i+sequence_length]['ppd']
            
            X_list.append(X_seq)
            y_list.append([y_energy, y_pmv, y_ppd])
    
    return np.array(X_list), np.array(y_list)

def train_model(train_df, val_df, sequence_length=24):
    """Train the LSTM model"""
    print("\nPreparing sequences for LSTM...")
    X_train, y_train = prepare_sequences(train_df, sequence_length)
    X_val, y_val = prepare_sequences(val_df, sequence_length)
    
    print(f"Training sequences: {X_train.shape}")
    print(f"Validation sequences: {X_val.shape}")
    
    # Normalize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    # Create datasets
    train_dataset = EnergyDataset(X_train_scaled, y_train_scaled)
    val_dataset = EnergyDataset(X_val_scaled, y_val_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=config.DL_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.DL_BATCH_SIZE, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_size = X_train.shape[-1]
    model = EnergyLSTM(input_size).to(device)
    
    # Loss and optimizer
    criterion_energy = nn.MSELoss()
    criterion_comfort = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.DL_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nTraining for {config.DL_EPOCHS} epochs...")
    for epoch in range(config.DL_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            energy_pred, comfort_pred = model(X_batch)
            
            loss_energy = criterion_energy(energy_pred.squeeze(), y_batch[:, 0])
            loss_comfort = criterion_comfort(comfort_pred, y_batch[:, 1:])
            
            # Combined loss
            loss = config.ENERGY_WEIGHT * loss_energy + config.COMFORT_WEIGHT * loss_comfort
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                energy_pred, comfort_pred = model(X_batch)
                
                loss_energy = criterion_energy(energy_pred.squeeze(), y_batch[:, 0])
                loss_comfort = criterion_comfort(comfort_pred, y_batch[:, 1:])
                
                loss = config.ENERGY_WEIGHT * loss_energy + config.COMFORT_WEIGHT * loss_comfort
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.DL_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, os.path.join(config.RESULTS_DIR, 'best_model.pth'))
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    return model, scaler_X, scaler_y

def plot_training_curves(train_losses, val_losses):
    """Plot training and validation losses"""
    plt.figure(figsize=config.FIGURE_SIZE)
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Deep Learning Model Training Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig_training_curves.png'), dpi=config.DPI)
    plt.close()
    print("Training curves saved to figures/fig_training_curves.png")

def evaluate_model(model, test_df, scaler_X, scaler_y, sequence_length=24):
    """Evaluate model on test set"""
    print("\nEvaluating model on test set...")
    
    X_test, y_test = prepare_sequences(test_df, sequence_length)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    predictions = []
    with torch.no_grad():
        test_dataset = EnergyDataset(X_test_scaled, np.zeros_like(y_test))
        test_loader = DataLoader(test_dataset, batch_size=config.DL_BATCH_SIZE, shuffle=False)
        
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            energy_pred, comfort_pred = model(X_batch)
            
            # Combine predictions
            pred = torch.cat([energy_pred, comfort_pred], dim=1)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.vstack(predictions)
    predictions = scaler_y.inverse_transform(predictions)
    
    # Calculate metrics for energy prediction
    y_true = y_test[:, 0]
    y_pred = predictions[:, 0]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    print(f"\nEnergy Prediction Metrics:")
    print(f"RMSE: {rmse:.2f} kWh")
    print(f"MAE: {mae:.2f} kWh")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Plot predictions vs actual
    plot_predictions(y_true, y_pred)
    plot_time_series(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'predictions': predictions,
        'actuals': y_test
    }

def plot_predictions(y_true, y_pred):
    """Generate Figure 1: Predicted vs Actual Energy Consumption"""
    plt.figure(figsize=config.FIGURE_SIZE)
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=10, label='Predictions')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), 'g-', linewidth=2, label=f'Linear Fit (y={z[0]:.2f}x+{z[1]:.2f})')
    
    plt.xlabel('Actual Energy Consumption (kWh)', fontsize=12)
    plt.ylabel('Predicted Energy Consumption (kWh)', fontsize=12)
    plt.title('Figure 1: Deep Learning Model Performance - Predicted vs Actual', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig1_prediction_scatter.png'), dpi=config.DPI)
    plt.close()
    print("Figure 1 saved to figures/fig1_prediction_scatter.png")

def plot_time_series(y_true, y_pred, num_samples=500):
    """Plot time series comparison"""
    plt.figure(figsize=(12, 6))
    
    samples = min(num_samples, len(y_true))
    x = np.arange(samples)
    
    plt.plot(x, y_true[:samples], label='Actual', linewidth=1.5, alpha=0.7)
    plt.plot(x, y_pred[:samples], label='Predicted', linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Energy Consumption (kWh)', fontsize=12)
    plt.title('Time Series: Actual vs Predicted Energy Consumption', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig_timeseries_comparison.png'), dpi=config.DPI)
    plt.close()
    print("Time series comparison saved to figures/fig_timeseries_comparison.png")

if __name__ == "__main__":
    print("Deep Learning Model for Energy Prediction")
    print("=" * 50)
