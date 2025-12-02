"""
Federated Learning Simulation for Privacy-Preserving Energy Optimization
Simulates distributed training across multiple buildings without sharing raw data
"""
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from typing import List, Dict
import copy

from config import *
from deep_learning_model import HybridLSTM

class FederatedLearningSimulator:
    """
    Simulates federated learning for energy prediction models
    Each building trains locally, only model parameters are shared
    """
    
    def __init__(self, model_class, model_args, num_clients=NUM_CLIENTS):
        self.model_class = model_class
        self.model_args = model_args
        self.num_clients = num_clients
        self.global_model = None
        self.client_models = []
        self.training_history = []
    
    def initialize_global_model(self):
        """Initialize global model"""
        self.global_model = self.model_class(**self.model_args)
        return self.global_model
    
    def create_client_models(self):
        """Create client models initialized with global weights"""
        self.client_models = []
        for i in range(self.num_clients):
            client_model = self.model_class(**self.model_args)
            client_model.load_state_dict(self.global_model.state_dict())
            self.client_models.append(client_model)
        return self.client_models
    
    def federated_averaging(self, client_weights: List[Dict], sample_sizes: List[int]):
        """
        Federated Averaging (FedAvg) algorithm
        Aggregate client model weights weighted by sample sizes
        """
        total_samples = sum(sample_sizes)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        for key in client_weights[0].keys():
            aggregated_weights[key] = torch.zeros_like(client_weights[0][key])
        
        # Weighted average
        for weights, n_samples in zip(client_weights, sample_sizes):
            weight_factor = n_samples / total_samples
            for key in aggregated_weights.keys():
                aggregated_weights[key] += weight_factor * weights[key]
        
        return aggregated_weights
    
    def train_round(self, client_data: List[pd.DataFrame], client_loaders: List, 
                   device, num_local_epochs=3, learning_rate=LEARNING_RATE):
        """
        One round of federated learning
        1. Distribute global model to clients
        2. Clients train locally
        3. Aggregate model updates
        """
        # Initialize client models with global weights
        global_state = self.global_model.state_dict()
        for client_model in self.client_models:
            client_model.load_state_dict(global_state)
        
        # Local training on each client
        client_weights = []
        sample_sizes = []
        
        criterion_energy = nn.MSELoss()
        criterion_comfort = nn.MSELoss()
        
        for client_idx, (client_model, data_loader) in enumerate(zip(self.client_models, client_loaders)):
            client_model.train()
            optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate)
            
            # Local training
            for epoch in range(num_local_epochs):
                for batch in data_loader:
                    features = batch['features'].to(device)
                    target_energy = batch['target_energy'].to(device)
                    target_ppd = batch['target_ppd'].to(device)
                    
                    optimizer.zero_grad()
                    energy_pred, comfort_pred = client_model(features)
                    
                    loss_energy = criterion_energy(energy_pred, target_energy)
                    loss_comfort = criterion_comfort(comfort_pred, target_ppd)
                    loss = loss_energy + 0.1 * loss_comfort
                    
                    loss.backward()
                    optimizer.step()
            
            # Collect client weights and sample size
            client_weights.append(copy.deepcopy(client_model.state_dict()))
            sample_sizes.append(len(client_data[client_idx]))
        
        # Aggregate using FedAvg
        aggregated_weights = self.federated_averaging(client_weights, sample_sizes)
        self.global_model.load_state_dict(aggregated_weights)
        
        return aggregated_weights
    
    def simulate_federated_training(self, all_client_data: List[pd.DataFrame], 
                                   all_client_loaders: List, device, 
                                   num_rounds=NUM_FEDERATED_ROUNDS):
        """
        Simulate complete federated learning process
        """
        print(f"Starting federated learning simulation ({num_rounds} rounds)...")
        
        self.initialize_global_model()
        self.create_client_models()
        
        for round_num in range(num_rounds):
            print(f"\nFederated Round {round_num + 1}/{num_rounds}")
            
            # Select random subset of clients (fraction)
            num_selected = max(1, int(self.num_clients * FEDERATED_FRACTION))
            selected_indices = np.random.choice(self.num_clients, num_selected, replace=False)
            
            selected_data = [all_client_data[i] for i in selected_indices]
            selected_loaders = [all_client_loaders[i] for i in selected_indices]
            
            # Train round
            aggregated_weights = self.train_round(
                selected_data, selected_loaders, device, num_local_epochs=2
            )
            
            # Evaluate global model (optional - would require validation data)
            self.training_history.append({
                'round': round_num + 1,
                'clients_participated': len(selected_indices),
                'total_samples': sum([len(all_client_data[i]) for i in selected_indices])
            })
        
        print("Federated learning simulation complete!")
        return self.global_model
    
    def get_privacy_metrics(self):
        """
        Calculate privacy-preserving metrics
        In federated learning, raw data never leaves the client
        """
        total_data_transferred = 0  # No raw data transferred
        model_updates_transferred = len(self.training_history) * self.num_clients
        
        # Estimate model size (parameters)
        total_params = sum(p.numel() for p in self.global_model.parameters())
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        return {
            'raw_data_transferred_mb': 0,  # Key privacy benefit
            'model_updates_transferred': model_updates_transferred,
            'model_size_mb': model_size_mb,
            'total_model_data_transferred_mb': model_updates_transferred * model_size_mb,
            'privacy_preserved': True
        }

def simulate_federated_learning(train_data: pd.DataFrame, feature_cols: List[str], device):
    """
    Main function to simulate federated learning
    """
    from deep_learning_model import EnergyDataset
    from torch.utils.data import DataLoader
    
    # Split data by building (each building is a client)
    building_ids = train_data['building_id'].unique()[:NUM_CLIENTS]
    
    client_data = []
    client_loaders = []
    
    for building_id in building_ids:
        building_data = train_data[train_data['building_id'] == building_id]
        if len(building_data) > 500:  # Limit for faster simulation
            building_data = building_data.head(500)
        
        client_data.append(building_data)
        
        # Create dataset and loader
        dataset = EnergyDataset(building_data, feature_cols, sequence_length=24)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        client_loaders.append(loader)
    
    # Initialize federated learning simulator
    input_size = len(feature_cols)
    simulator = FederatedLearningSimulator(
        HybridLSTM,
        {'input_size': input_size},
        num_clients=len(building_ids)
    )
    
    # Run federated training
    global_model = simulator.simulate_federated_training(
        client_data, client_loaders, device, num_rounds=NUM_FEDERATED_ROUNDS
    )
    
    # Get privacy metrics
    privacy_metrics = simulator.get_privacy_metrics()
    
    print("\n=== Federated Learning Privacy Metrics ===")
    print(f"Raw Data Transferred: {privacy_metrics['raw_data_transferred_mb']} MB")
    print(f"Model Updates Transferred: {privacy_metrics['model_updates_transferred']}")
    print(f"Model Size: {privacy_metrics['model_size_mb']:.2f} MB")
    print(f"Total Model Data Transferred: {privacy_metrics['total_model_data_transferred_mb']:.2f} MB")
    print(f"Privacy Preserved: {privacy_metrics['privacy_preserved']}")
    
    # Save federated model
    torch.save(global_model.state_dict(), f"{MODELS_DIR}/federated_model.pth")
    
    # Save privacy metrics
    import json
    with open(f"{OUTPUT_DIR}/federated_privacy_metrics.json", 'w') as f:
        json.dump(privacy_metrics, f, indent=2)
    
    return global_model, privacy_metrics

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_data = pd.read_csv(f"{DATA_DIR}/train_processed.csv", parse_dates=['timestamp'])
    
    feature_cols = ['square_feet', 'air_temperature', 'dew_temperature', 
                   'sea_level_pressure', 'wind_speed', 'cloud_coverage',
                   'hour', 'day_of_week', 'day_of_year', 'month', 'occupant_activity']
    
    # Simulate federated learning
    global_model, privacy_metrics = simulate_federated_learning(train_data, feature_cols, device)
