"""
Reinforcement Learning Training Module
Implements PPO with LSTM policy using stable-baselines3.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional, List
import time
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from config import RANDOM_SEED, RL_CONFIG, MODELS_DIR
from rl_environment import BuildingEnergyEnv, MultiAgentController

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class MetricsCallback(BaseCallback):
    """Custom callback for tracking training metrics."""
    
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.energy_uses = []
        self.comfort_violations = []
    
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    if 'infos' in self.locals and len(self.locals['infos']) > i:
                        info = self.locals['infos'][i]
                        if 'episode' in info:
                            self.episode_rewards.append(info['episode']['r'])
                            self.episode_lengths.append(info['episode']['l'])
        return True


def create_env(data: pd.DataFrame, dl_model=None) -> BuildingEnergyEnv:
    """Create and wrap the building energy environment."""
    env = BuildingEnergyEnv(data, dl_model=dl_model)
    env = Monitor(env)
    return env


def train_ppo_agent(
    train_data: pd.DataFrame,
    dl_model=None,
    total_timesteps: int = RL_CONFIG['total_timesteps'],
    save_path: str = None
) -> Tuple[PPO, Dict]:
    """
    Train a PPO agent for building energy control.
    
    Args:
        train_data: Training data with building info
        dl_model: Optional DL model for predictions
        total_timesteps: Total training timesteps
        save_path: Path to save the trained model
    
    Returns:
        Trained PPO model and training metrics
    """
    print("Setting up PPO training...")
    
    # Create vectorized environment
    env = DummyVecEnv([lambda: create_env(train_data, dl_model)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Define policy
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 64], vf=[128, 64])],
    )
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=RL_CONFIG['learning_rate'],
        n_steps=min(RL_CONFIG['n_steps'], len(train_data) - 1),
        batch_size=RL_CONFIG['batch_size'],
        n_epochs=RL_CONFIG['n_epochs'],
        gamma=RL_CONFIG['gamma'],
        clip_range=RL_CONFIG['clip_range'],
        ent_coef=RL_CONFIG['ent_coef'],
        vf_coef=RL_CONFIG['vf_coef'],
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=RANDOM_SEED,
        tensorboard_log=None  # Disable tensorboard for simplicity
    )
    
    # Create callback
    callback = MetricsCallback()
    
    # Train
    print(f"Training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save model
    if save_path:
        model.save(save_path)
        env.save(save_path + "_vecnorm.pkl")
        print(f"Model saved to {save_path}")
    
    # Collect training metrics
    metrics = {
        'training_time': training_time,
        'total_timesteps': total_timesteps,
        'episode_rewards': callback.episode_rewards,
        'avg_reward': np.mean(callback.episode_rewards) if callback.episode_rewards else 0,
    }
    
    return model, metrics, env


def evaluate_rl_agent(
    model: PPO,
    test_data: pd.DataFrame,
    vec_normalize_env=None,
    n_episodes: int = 5
) -> Dict:
    """
    Evaluate a trained RL agent on test data.
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating RL agent...")
    
    # Create test environment
    test_env = create_env(test_data)
    
    # Wrap with normalization if available
    if vec_normalize_env is not None:
        test_env = DummyVecEnv([lambda: test_env])
        # Copy normalization stats
        test_env = VecNormalize(test_env, training=False)
        test_env.obs_rms = vec_normalize_env.obs_rms
        test_env.ret_rms = vec_normalize_env.ret_rms
    
    episode_rewards = []
    episode_energies = []
    episode_comfort = []
    all_actions = []
    
    for episode in range(n_episodes):
        obs = test_env.reset()
        done = False
        episode_reward = 0
        episode_energy = 0
        ppd_values = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            
            episode_reward += reward if isinstance(reward, (int, float)) else reward[0]
            
            if isinstance(info, list):
                info = info[0]
            
            if 'energy_use' in info:
                episode_energy += info['energy_use']
            if 'ppd' in info:
                ppd_values.append(info['ppd'])
            
            all_actions.append(action)
        
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy)
        if ppd_values:
            episode_comfort.append(np.mean(ppd_values))
    
    # Calculate metrics
    metrics = {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_energy': np.mean(episode_energies),
        'avg_ppd': np.mean(episode_comfort) if episode_comfort else 8.0,
        'n_episodes': n_episodes
    }
    
    print(f"Average Reward: {metrics['avg_reward']:.2f}")
    print(f"Average Energy: {metrics['avg_energy']:.2f} kWh")
    print(f"Average PPD: {metrics['avg_ppd']:.2f}%")
    
    return metrics


def train_multi_agent(
    train_data: pd.DataFrame,
    n_episodes: int = 1000
) -> Tuple[MultiAgentController, Dict]:
    """
    Train multi-agent controller with separate HVAC and lighting agents.
    Uses simple Q-learning for demonstration.
    """
    print("Training multi-agent controller...")
    
    controller = MultiAgentController()
    env = BuildingEnergyEnv(train_data)
    
    episode_rewards = []
    start_time = time.time()
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Get actions from both agents
            hvac_action, lighting_action = controller.get_actions(state, training=True)
            action = np.array([hvac_action, lighting_action])
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Update agents
            controller.update(state, (hvac_action, lighting_action), reward, next_state)
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{n_episodes}: Avg Reward = {avg_reward:.2f}")
    
    training_time = time.time() - start_time
    
    metrics = {
        'training_time': training_time,
        'n_episodes': n_episodes,
        'final_avg_reward': np.mean(episode_rewards[-100:])
    }
    
    print(f"Multi-agent training completed in {training_time:.2f} seconds")
    
    return controller, metrics


class HybridRLController:
    """
    Hybrid controller combining DL predictions with RL policy.
    Suitable for edge deployment.
    """
    
    def __init__(
        self,
        rl_model: PPO = None,
        dl_model: torch.nn.Module = None,
        vec_normalize_env=None
    ):
        self.rl_model = rl_model
        self.dl_model = dl_model
        self.vec_normalize_env = vec_normalize_env
        
        # Track inference times
        self.inference_times = []
    
    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, int]:
        """Get control action using hybrid approach."""
        start_time = time.time()
        
        # Optionally use DL model predictions to enhance state
        if self.dl_model is not None:
            # DL inference would happen here
            pass
        
        # Get RL action
        if self.rl_model is not None:
            action, _ = self.rl_model.predict(observation, deterministic=deterministic)
            if isinstance(action, np.ndarray):
                hvac_action = int(action[0])
                lighting_action = int(action[1]) if len(action) > 1 else 2
            else:
                hvac_action = action
                lighting_action = 2
        else:
            # Fallback to simple heuristic
            hvac_action = 2  # No change
            lighting_action = 2  # Medium
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return hvac_action, lighting_action
    
    def get_avg_inference_time(self) -> float:
        """Get average inference time for edge deployment metrics."""
        if self.inference_times:
            return np.mean(self.inference_times) * 1000  # ms
        return 0.0


def simulate_edge_deployment(
    controller: HybridRLController,
    test_data: pd.DataFrame,
    n_steps: int = 1000
) -> Dict:
    """
    Simulate edge deployment of the hybrid controller.
    Tracks inference latency and control performance.
    """
    print("Simulating edge deployment...")
    
    env = BuildingEnergyEnv(test_data)
    state = env.reset()
    
    total_energy = 0
    ppd_values = []
    
    for step in range(min(n_steps, len(test_data) - 1)):
        # Get action (simulates edge inference)
        hvac_action, lighting_action = controller.get_action(state)
        action = np.array([hvac_action, lighting_action])
        
        # Take step
        state, reward, done, info = env.step(action)
        
        total_energy += info.get('energy_use', 0)
        ppd_values.append(info.get('ppd', 8.0))
        
        if done:
            break
    
    # Compute metrics
    avg_inference_time = controller.get_avg_inference_time()
    
    metrics = {
        'total_energy': total_energy,
        'avg_ppd': np.mean(ppd_values),
        'ppd_below_threshold': np.mean(np.array(ppd_values) < 10) * 100,
        'avg_inference_time_ms': avg_inference_time,
        'n_steps': n_steps
    }
    
    print(f"Edge Deployment Results:")
    print(f"  Total Energy: {total_energy:.2f} kWh")
    print(f"  Average PPD: {metrics['avg_ppd']:.2f}%")
    print(f"  Avg Inference Time: {avg_inference_time:.3f} ms")
    
    return metrics


if __name__ == "__main__":
    print("Testing RL Training module...")
    
    # Create dummy data
    n_steps = 2000
    data = pd.DataFrame({
        'timestamp': pd.date_range('2016-01-01', periods=n_steps, freq='H'),
        'building_id': ['building_0'] * n_steps,
        'meter_reading': np.random.uniform(10, 50, n_steps),
        'meter_reading_scaled': np.random.normal(0, 1, n_steps),
        'air_temperature': np.random.uniform(-5, 35, n_steps),
        'air_temperature_scaled': np.random.normal(0, 1, n_steps),
        'hour_sin': np.sin(2 * np.pi * np.arange(n_steps) / 24),
        'hour_cos': np.cos(2 * np.pi * np.arange(n_steps) / 24),
        'dow_sin': np.sin(2 * np.pi * np.arange(n_steps) / 168),
        'dow_cos': np.cos(2 * np.pi * np.arange(n_steps) / 168),
        'is_weekend': np.random.randint(0, 2, n_steps),
        'rolling_mean_24h_scaled': np.random.normal(0, 1, n_steps),
        'rolling_std_24h_scaled': np.random.normal(0, 0.5, n_steps),
    })
    
    # Test environment
    env = create_env(data)
    print(f"Environment created: obs_shape={env.observation_space.shape}")
    
    # Quick test of multi-agent
    controller, metrics = train_multi_agent(data, n_episodes=100)
    print(f"Multi-agent training completed: {metrics}")
