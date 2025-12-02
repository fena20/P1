"""
Deep Learning Component: LSTM/Transformer for Energy and Comfort Prediction
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
from tqdm import tqdm

from config import *

class EnergyDataset(Dataset):
    """Dataset for energy prediction"""
    def __init__(self, data, feature_cols, sequence_length=24):
        self.data = data.sort_values(['building_id', 'timestamp']).reset_index(drop=True)
        self.feature_cols = feature_cols
        self.sequence_length = sequence_length
        self.sequences = self._create_sequences()
    
    def _create_sequences(self):
        """Create sequences for LSTM"""
        sequences = []
        for building_id in self.data['building_id'].unique():
            building_data = self.data[self.data['building_id'] == building_id].copy()
            building_data = building_data.sort_values('timestamp')
            
            for i in range(len(building_data) - self.sequence_length):
                seq = building_data.iloc[i:i+self.sequence_length]
                target = building_data.iloc[i+self.sequence_length]
                
                sequences.append({
                    'features': seq[self.feature_cols].values.astype(np.float32),
                    'target_energy': np.float32(target['meter_reading']),
                    'target_ppd': np.float32(target['ppd']),
                    'building_id': building_id
                })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'features': torch.FloatTensor(seq['features']),
            'target_energy': torch.FloatTensor([seq['target_energy']]),
            'target_ppd': torch.FloatTensor([seq['target_ppd']])
        }

class HybridLSTM(nn.Module):
    """Hybrid LSTM model for energy and comfort prediction"""
    def __init__(self, input_size, hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS):
        super(HybridLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Energy prediction head
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Comfort (PPD) prediction head
        self.comfort_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # PPD is between 0-100
        )
    
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Predictions
        energy_pred = self.energy_head(last_hidden)
        comfort_pred = self.comfort_head(last_hidden) * 100  # Scale to 0-100
        
        return energy_pred, comfort_pred

class TransformerModel(nn.Module):
    """Transformer model as alternative architecture"""
    def __init__(self, input_size, d_model=TRANSFORMER_D_MODEL, nhead=TRANSFORMER_NHEAD, 
                 num_layers=TRANSFORMER_NUM_LAYERS):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.energy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1)
        )
        
        self.comfort_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Project input to d_model
        x = self.input_projection(x)
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Use last timestep
        last_hidden = transformer_out[:, -1, :]
        
        energy_pred = self.energy_head(last_hidden)
        comfort_pred = self.comfort_head(last_hidden) * 100
        
        return energy_pred, comfort_pred

def train_model(model, train_loader, val_loader, device, model_type='lstm'):
    """Train the deep learning model"""
    print(f"Training {model_type.upper()} model...")
    
    criterion_energy = nn.MSELoss()
    criterion_comfort = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            features = batch['features'].to(device)
            target_energy = batch['target_energy'].to(device)
            target_ppd = batch['target_ppd'].to(device)
            
            optimizer.zero_grad()
            energy_pred, comfort_pred = model(features)
            
            loss_energy = criterion_energy(energy_pred, target_energy)
            loss_comfort = criterion_comfort(comfort_pred, target_ppd)
            loss = loss_energy + 0.1 * loss_comfort  # Weighted combination
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                target_energy = batch['target_energy'].to(device)
                target_ppd = batch['target_ppd'].to(device)
                
                energy_pred, comfort_pred = model(features)
                loss_energy = criterion_energy(energy_pred, target_energy)
                loss_comfort = criterion_comfort(comfort_pred, target_ppd)
                loss = loss_energy + 0.1 * loss_comfort
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{MODELS_DIR}/{model_type}_best.pth")
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device):
    """Evaluate model performance"""
    model.eval()
    all_energy_preds = []
    all_energy_targets = []
    all_comfort_preds = []
    all_comfort_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            target_energy = batch['target_energy'].cpu().numpy()
            target_ppd = batch['target_ppd'].cpu().numpy()
            
            energy_pred, comfort_pred = model(features)
            
            all_energy_preds.extend(energy_pred.cpu().numpy().flatten())
            all_energy_targets.extend(target_energy.flatten())
            all_comfort_preds.extend(comfort_pred.cpu().numpy().flatten())
            all_comfort_targets.extend(target_ppd.flatten())
    
    # Calculate metrics
    energy_r2 = r2_score(all_energy_targets, all_energy_preds)
    energy_mae = mean_absolute_error(all_energy_targets, all_energy_preds)
    energy_rmse = np.sqrt(mean_squared_error(all_energy_targets, all_energy_preds))
    
    comfort_r2 = r2_score(all_comfort_targets, all_comfort_preds)
    comfort_mae = mean_absolute_error(all_comfort_targets, all_comfort_preds)
    
    return {
        'energy': {
            'r2': energy_r2,
            'mae': energy_mae,
            'rmse': energy_rmse,
            'predictions': np.array(all_energy_preds),
            'targets': np.array(all_energy_targets)
        },
        'comfort': {
            'r2': comfort_r2,
            'mae': comfort_mae,
            'predictions': np.array(all_comfort_preds),
            'targets': np.array(all_comfort_targets)
        }
    }

def plot_predictions(results, save_path):
    """Generate Figure 1: Predicted vs Actual Energy Consumption"""
    plt.style.use(FIG_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Energy prediction plot
    energy_preds = results['energy']['predictions']
    energy_targets = results['energy']['targets']
    
    axes[0].scatter(energy_targets, energy_preds, alpha=0.5, s=10)
    axes[0].plot([energy_targets.min(), energy_targets.max()], 
                 [energy_targets.min(), energy_targets.max()], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Energy Consumption (kWh)', fontsize=12)
    axes[0].set_ylabel('Predicted Energy Consumption (kWh)', fontsize=12)
    axes[0].set_title(f'Energy Prediction (R² = {results["energy"]["r2"]:.3f})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Comfort prediction plot
    comfort_preds = results['comfort']['predictions']
    comfort_targets = results['comfort']['targets']
    
    axes[1].scatter(comfort_targets, comfort_preds, alpha=0.5, s=10, color='green')
    axes[1].plot([comfort_targets.min(), comfort_targets.max()], 
                 [comfort_targets.min(), comfort_targets.max()], 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual PPD (%)', fontsize=12)
    axes[1].set_ylabel('Predicted PPD (%)', fontsize=12)
    axes[1].set_title(f'Comfort Prediction (R² = {results["comfort"]["r2"]:.3f})', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"Figure 1 saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load processed data
    train_data = pd.read_csv(f"{DATA_DIR}/train_processed.csv", parse_dates=['timestamp'])
    test_data = pd.read_csv(f"{DATA_DIR}/test_processed.csv", parse_dates=['timestamp'])
    
    # Get feature columns
    feature_cols = ['square_feet', 'air_temperature', 'dew_temperature', 
                   'sea_level_pressure', 'wind_speed', 'cloud_coverage',
                   'hour', 'day_of_week', 'day_of_year', 'month', 'occupant_activity']
    
    # Create datasets
    train_dataset = EnergyDataset(train_data, feature_cols, sequence_length=24)
    test_dataset = EnergyDataset(test_data, feature_cols, sequence_length=24)
    
    # Split train into train/val
    train_size = int(TRAIN_VAL_SPLIT * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    input_size = len(feature_cols)
    model = HybridLSTM(input_size).to(device)
    
    # Train
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    
    # Load best model
    model.load_state_dict(torch.load(f"{MODELS_DIR}/lstm_best.pth"))
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    print("\n=== Model Evaluation ===")
    print(f"Energy R²: {results['energy']['r2']:.4f}")
    print(f"Energy MAE: {results['energy']['mae']:.4f}")
    print(f"Energy RMSE: {results['energy']['rmse']:.4f}")
    print(f"Comfort R²: {results['comfort']['r2']:.4f}")
    print(f"Comfort MAE: {results['comfort']['mae']:.4f}")
    
    # Generate Figure 1
    plot_predictions(results, f"{FIGURES_DIR}/fig1.png")
    
    # Save predictions for RL
    np.save(f"{OUTPUT_DIR}/dl_predictions.npy", {
        'energy_preds': results['energy']['predictions'],
        'comfort_preds': results['comfort']['predictions']
    })
