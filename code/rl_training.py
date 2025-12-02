"""
Reinforcement Learning Training Module
PPO with LSTM policy for building energy optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
try:
    from sb3_contrib import RecurrentPPO
    USE_LSTM = True
except ImportError:
    USE_LSTM = False
import os
import config
from rl_environment import BuildingEnergyEnv, create_building_env

plt.style.use(config.STYLE)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging training metrics
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        return True
    
    def _on_rollout_end(self):
        # Log metrics
        if len(self.model.ep_info_buffer) > 0:
            ep_rew_mean = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            ep_len_mean = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
            self.episode_rewards.append(ep_rew_mean)
            self.episode_lengths.append(ep_len_mean)

def train_rl_agent(train_data, agent_type='hvac', timesteps=config.RL_TIMESTEPS):
    """Train a single RL agent"""
    print(f"\nTraining {agent_type.upper()} agent with PPO...")
    
    # Create environment
    env = BuildingEnergyEnv(train_data, dl_model=None, scaler_X=None, agent_type=agent_type)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Create PPO agent with LSTM policy (if available) or MLP policy
    if USE_LSTM:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=config.RL_LEARNING_RATE,
            n_steps=config.RL_N_STEPS,
            batch_size=config.RL_BATCH_SIZE,
            gamma=config.RL_GAMMA,
            verbose=1,
            tensorboard_log=None
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.RL_LEARNING_RATE,
            n_steps=config.RL_N_STEPS,
            batch_size=config.RL_BATCH_SIZE,
            gamma=config.RL_GAMMA,
            verbose=1,
            tensorboard_log=None
        )
    
    # Callback
    callback = TensorboardCallback()
    
    # Train
    print(f"Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=callback)
    
    # Save model
    model_path = os.path.join(config.RESULTS_DIR, f'ppo_{agent_type}_model')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model, callback.episode_rewards

def train_multi_agent(train_data, timesteps=config.RL_TIMESTEPS):
    """Train multiple agents (HVAC and Lighting)"""
    print("\n" + "="*60)
    print("Training Multi-Agent System")
    print("="*60)
    
    agents = {}
    rewards_history = {}
    
    for agent_name in config.AGENT_NAMES:
        model, rewards = train_rl_agent(train_data, agent_type=agent_name, timesteps=timesteps)
        agents[agent_name] = model
        rewards_history[agent_name] = rewards
    
    # Plot training progress
    plot_training_progress(rewards_history)
    
    return agents

def plot_training_progress(rewards_history):
    """Plot RL training progress"""
    plt.figure(figsize=config.FIGURE_SIZE)
    
    for agent_name, rewards in rewards_history.items():
        if len(rewards) > 0:
            # Smooth rewards
            window = min(10, len(rewards))
            smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
            plt.plot(smoothed, label=f'{agent_name.upper()} Agent', linewidth=2)
    
    plt.xlabel('Training Episode', fontsize=12)
    plt.ylabel('Average Episode Reward', fontsize=12)
    plt.title('Reinforcement Learning Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig_rl_training_progress.png'), dpi=config.DPI)
    plt.close()
    print("RL training progress saved to figures/fig_rl_training_progress.png")

def evaluate_rl_agent(model, test_data, agent_type='hvac', num_episodes=10):
    """Evaluate trained RL agent"""
    print(f"\nEvaluating {agent_type.upper()} agent...")
    
    env = BuildingEnergyEnv(test_data, dl_model=None, scaler_X=None, agent_type=agent_type)
    
    episode_rewards = []
    episode_energies = []
    episode_costs = []
    episode_ppds = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_energy = 0
        episode_cost = 0
        episode_ppd = 0
        steps = 0
        
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_energy += info['energy']
            episode_cost += info['cost']
            episode_ppd += info['ppd']
            steps += 1
        
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy)
        episode_costs.append(episode_cost)
        episode_ppds.append(episode_ppd / steps if steps > 0 else 0)
    
    results = {
        'avg_reward': np.mean(episode_rewards),
        'avg_energy': np.mean(episode_energies),
        'avg_cost': np.mean(episode_costs),
        'avg_ppd': np.mean(episode_ppds),
        'std_reward': np.std(episode_rewards),
        'std_energy': np.std(episode_energies),
        'std_cost': np.std(episode_costs),
        'std_ppd': np.std(episode_ppds)
    }
    
    print(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Average Energy: {results['avg_energy']:.2f} ± {results['std_energy']:.2f} kWh")
    print(f"Average Cost: ${results['avg_cost']:.2f} ± ${results['std_cost']:.2f}")
    print(f"Average PPD: {results['avg_ppd']:.2f}% ± {results['std_ppd']:.2f}%")
    
    return results

def simulate_baseline_control(test_data, method='rule_based'):
    """Simulate baseline control methods for comparison"""
    print(f"\nSimulating {method} baseline...")
    
    total_energy = 0
    total_cost = 0
    ppd_values = []
    
    for idx, row in test_data.iterrows():
        if method == 'rule_based':
            # Simple rule-based: fixed setpoint
            hvac_setpoint = 22.0
            lighting_level = 1.0 if 8 <= row['hour'] <= 22 else 0.3
            
        elif method == 'simple_mpc':
            # Simplified MPC: adjust based on outdoor temp
            outdoor_temp = row['air_temperature']
            if outdoor_temp < 18:
                hvac_setpoint = 23.0
            elif outdoor_temp > 25:
                hvac_setpoint = 20.0
            else:
                hvac_setpoint = 21.0
            
            # Time-based lighting
            if row['hour'] >= 6 and row['hour'] <= 8:
                lighting_level = 0.5
            elif row['hour'] > 8 and row['hour'] <= 20:
                lighting_level = 0.8
            elif row['hour'] > 20 and row['hour'] <= 22:
                lighting_level = 0.5
            else:
                lighting_level = 0.2
        
        else:  # no_control
            hvac_setpoint = row['air_temperature']
            lighting_level = 1.0
        
        # Calculate energy (simplified)
        hvac_energy = abs(hvac_setpoint - row['air_temperature']) * row['square_feet'] / 10000
        lighting_energy = lighting_level * row['square_feet'] / 100
        base_energy = row['meter_reading'] * 0.3
        
        energy = base_energy + hvac_energy + lighting_energy
        cost = energy * config.ENERGY_COST
        
        total_energy += energy
        total_cost += cost
        
        # Comfort
        pmv = 0.5 * (hvac_setpoint - 21.0)
        ppd = 100 - 95 * np.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
        ppd = np.clip(ppd, 5, 100)
        ppd_values.append(ppd)
    
    results = {
        'total_energy': total_energy,
        'total_cost': total_cost,
        'avg_ppd': np.mean(ppd_values),
        'method': method
    }
    
    print(f"Total Energy: {total_energy:.2f} kWh")
    print(f"Total Cost: ${total_cost:.2f}")
    print(f"Average PPD: {results['avg_ppd']:.2f}%")
    
    return results

def load_trained_agent(agent_type='hvac'):
    """Load a trained agent"""
    model_path = os.path.join(config.RESULTS_DIR, f'ppo_{agent_type}_model')
    if os.path.exists(model_path + '.zip'):
        if USE_LSTM:
            model = RecurrentPPO.load(model_path)
        else:
            model = PPO.load(model_path)
        print(f"Loaded {agent_type} agent from {model_path}")
        return model
    else:
        print(f"Model not found at {model_path}")
        return None

if __name__ == "__main__":
    print("Reinforcement Learning Training Module")
    print("=" * 60)
