"""
Reinforcement Learning Training with Stable-Baselines3
Implements PPO with LSTM policy for hybrid RL
"""
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import os

from config import *
from rl_environment import EnergyOptimizationEnv, MultiAgentEnergyEnv

class LSTMFeatureExtractor:
    """Custom feature extractor using LSTM for RL"""
    def __init__(self, observation_space, features_dim=64):
        self.observation_space = observation_space
        self.features_dim = features_dim

def train_rl_agent(data: pd.DataFrame, dl_predictions: Dict, building_id: int = None, 
                   use_multi_agent: bool = False, save_path: str = None):
    """
    Train RL agent using PPO with LSTM policy
    
    Args:
        data: Preprocessed building data
        dl_predictions: Deep learning predictions for hybrid approach
        building_id: Specific building ID (None for all buildings)
        use_multi_agent: Whether to use multi-agent setup
        save_path: Path to save trained model
    """
    print("Training RL agent...")
    
    if save_path is None:
        save_path = f"{MODELS_DIR}/ppo_energy_optimizer"
    
    # Create environment
    if use_multi_agent:
        env = MultiAgentEnergyEnv(data, dl_predictions, building_id)
        # For multi-agent, we'll train separate agents
        print("Multi-agent training not fully implemented in stable-baselines3")
        print("Training single HVAC agent instead...")
        use_multi_agent = False
    
    if not use_multi_agent:
        env = EnergyOptimizationEnv(data, dl_predictions, building_id)
        env = DummyVecEnv([lambda: env])
    
    # Create PPO agent with LSTM policy
    model = PPO(
        "MlpLstmPolicy",  # LSTM policy for sequence modeling
        env,
        learning_rate=RL_LEARNING_RATE,
        n_steps=RL_N_STEPS,
        batch_size=RL_BATCH_SIZE,
        n_epochs=RL_N_EPOCHS,
        gamma=RL_GAMMA,
        gae_lambda=RL_GAE_LAMBDA,
        clip_range=RL_CLIP_RANGE,
        ent_coef=RL_ENT_COEF,
        vf_coef=RL_VF_COEF,
        verbose=1,
        tensorboard_log=f"{OUTPUT_DIR}/tensorboard_logs",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{MODELS_DIR}/ppo_best",
        log_path=f"{OUTPUT_DIR}/eval_logs",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{MODELS_DIR}/checkpoints",
        name_prefix='ppo_model'
    )
    
    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=100000,  # Adjust based on data size
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model

def evaluate_rl_agent(model, data: pd.DataFrame, dl_predictions: Dict, 
                     building_id: int = None, num_episodes: int = 5):
    """Evaluate trained RL agent"""
    print("Evaluating RL agent...")
    
    env = EnergyOptimizationEnv(data, dl_predictions, building_id)
    
    total_rewards = []
    total_energies = []
    total_costs = []
    avg_ppds = []
    comfort_violations = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        total_energies.append(env.total_energy)
        total_costs.append(env.total_cost)
        avg_ppds.append(env.total_ppd / max(1, env.current_step))
        comfort_violations.append(env.comfort_violations)
    
    results = {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_energy': np.mean(total_energies),
        'mean_cost': np.mean(total_costs),
        'mean_ppd': np.mean(avg_ppds),
        'mean_comfort_violations': np.mean(comfort_violations)
    }
    
    print("\n=== RL Agent Evaluation ===")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Energy: {results['mean_energy']:.2f} kWh")
    print(f"Mean Cost: ${results['mean_cost']:.2f}")
    print(f"Mean PPD: {results['mean_ppd']:.2f}%")
    print(f"Comfort Violations: {results['mean_comfort_violations']:.1f}")
    
    return results

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Load data
    train_data = pd.read_csv(f"{DATA_DIR}/train_processed.csv", parse_dates=['timestamp'])
    
    # Load DL predictions
    try:
        dl_preds = np.load(f"{OUTPUT_DIR}/dl_predictions.npy", allow_pickle=True).item()
    except:
        print("DL predictions not found. Using baseline predictions.")
        dl_preds = {
            'energy_preds': train_data['meter_reading'].values,
            'comfort_preds': train_data['ppd'].values
        }
    
    # Train on a subset of buildings for faster training
    building_ids = train_data['building_id'].unique()[:5]  # Use first 5 buildings
    
    for building_id in building_ids:
        print(f"\nTraining for building {building_id}...")
        building_data = train_data[train_data['building_id'] == building_id]
        
        # Limit data size for faster training
        if len(building_data) > 1000:
            building_data = building_data.head(1000)
        
        model = train_rl_agent(
            building_data,
            dl_preds,
            building_id=building_id,
            save_path=f"{MODELS_DIR}/ppo_building_{building_id}"
        )
        
        # Evaluate
        results = evaluate_rl_agent(model, building_data, dl_preds, building_id=building_id)
        
        # Save results
        results_df = pd.DataFrame([results])
        results_df.to_csv(f"{OUTPUT_DIR}/rl_results_building_{building_id}.csv", index=False)
