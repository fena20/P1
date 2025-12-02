"""
Custom Gym Environment for Building Energy Control
Implements multi-agent RL for HVAC and lighting subsystems.
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
import torch

from config import (
    RANDOM_SEED, ENV_CONFIG, COMFORT_CONFIG, RL_CONFIG
)
from deep_learning import calculate_pmv, calculate_ppd, LSTMEnergyPredictor

np.random.seed(RANDOM_SEED)


class BuildingEnergyEnv(gym.Env):
    """
    Custom Gym environment for building energy control.
    
    State:
        - Current energy use (normalized)
        - Current/forecast weather (temperature, humidity)
        - Predicted occupant needs (from DL model)
        - Indoor temperature
        - Current comfort index (PPD)
        - Hour of day, day of week
        
    Actions:
        - HVAC setpoint adjustment: Discrete (-2, -1, 0, +1, +2) °C
        - Lighting level: Discrete (0, 1, 2, 3, 4)
        
    Reward:
        - Negative energy cost
        - Positive comfort score
        - Penalty for PPD > threshold
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        dl_model: Optional[torch.nn.Module] = None,
        energy_scaler: Optional[object] = None,
        config: Dict = None
    ):
        super(BuildingEnergyEnv, self).__init__()
        
        self.config = config or ENV_CONFIG
        self.data = data.reset_index(drop=True)
        self.dl_model = dl_model
        self.energy_scaler = energy_scaler
        
        # Environment parameters
        self.current_step = 0
        self.max_steps = len(data) - 1
        
        # Indoor temperature state
        self.indoor_temp = 21.0  # Starting indoor temperature
        self.setpoint = 21.0  # HVAC setpoint
        self.lighting_level = 2  # Default lighting level
        
        # Action space: Multi-discrete for HVAC and lighting
        # HVAC: 5 actions (-2, -1, 0, +1, +2 degrees)
        # Lighting: 5 levels (0-4)
        self.action_space = spaces.MultiDiscrete([5, 5])
        
        # Observation space
        # [energy_use, outdoor_temp, indoor_temp, ppd, hour_sin, hour_cos, 
        #  dow_sin, dow_cos, is_weekend, predicted_energy, setpoint, lighting]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )
        
        # Tracking metrics
        self.episode_energy = 0
        self.episode_comfort_violations = 0
        self.episode_rewards = []
    
    def _get_state(self) -> np.ndarray:
        """Get current state observation."""
        row = self.data.iloc[self.current_step]
        
        # Calculate current PPD
        pmv = calculate_pmv(self.indoor_temp)
        ppd = calculate_ppd(pmv) / 100.0  # Normalize to 0-1
        
        # Get DL model prediction if available
        if self.dl_model is not None:
            predicted_energy = self._get_dl_prediction()
        else:
            predicted_energy = row.get('meter_reading_scaled', 0)
        
        state = np.array([
            row.get('meter_reading_scaled', 0),
            row.get('air_temperature_scaled', 0),
            (self.indoor_temp - 21) / 10,  # Normalized indoor temp
            ppd,
            row.get('hour_sin', 0),
            row.get('hour_cos', 0),
            row.get('dow_sin', 0),
            row.get('dow_cos', 0),
            row.get('is_weekend', 0),
            predicted_energy,
            (self.setpoint - 21) / 10,  # Normalized setpoint
            self.lighting_level / 4  # Normalized lighting
        ], dtype=np.float32)
        
        return state
    
    def _get_dl_prediction(self) -> float:
        """Get energy prediction from DL model."""
        # This would use the actual model in practice
        # For now, return scaled meter reading with some noise
        row = self.data.iloc[self.current_step]
        base_pred = row.get('meter_reading_scaled', 0)
        return base_pred * (1 + np.random.normal(0, 0.05))
    
    def _simulate_indoor_temp(
        self,
        outdoor_temp: float,
        hvac_action: int
    ) -> float:
        """Simulate indoor temperature dynamics."""
        # Simple thermal model
        # Indoor temp moves towards outdoor temp, modulated by HVAC
        
        thermal_mass = 0.95  # Higher = slower temperature change
        hvac_effectiveness = 0.8
        
        # Setpoint adjustment
        setpoint_delta = (hvac_action - 2) * self.config['temp_action_delta']
        new_setpoint = np.clip(
            self.setpoint + setpoint_delta,
            self.config['temp_setpoint_range'][0],
            self.config['temp_setpoint_range'][1]
        )
        self.setpoint = new_setpoint
        
        # Temperature dynamics
        temp_diff = outdoor_temp - self.indoor_temp
        natural_drift = temp_diff * (1 - thermal_mass)
        
        # HVAC effect (tries to maintain setpoint)
        hvac_effect = (self.setpoint - self.indoor_temp) * hvac_effectiveness * 0.2
        
        new_indoor_temp = self.indoor_temp + natural_drift + hvac_effect
        
        return new_indoor_temp
    
    def _calculate_energy_use(
        self,
        outdoor_temp: float,
        hvac_action: int,
        lighting_action: int
    ) -> float:
        """Calculate energy consumption based on actions."""
        row = self.data.iloc[self.current_step]
        base_energy = row.get('meter_reading', 1.0)
        
        # HVAC energy depends on setpoint-outdoor temp difference
        temp_diff = abs(self.setpoint - outdoor_temp)
        hvac_energy = base_energy * 0.5 * (temp_diff / 20)  # HVAC portion
        
        # Lighting energy
        lighting_energy = base_energy * 0.1 * (lighting_action / 4)
        
        # Other loads (constant)
        other_energy = base_energy * 0.3
        
        total_energy = hvac_energy + lighting_energy + other_energy
        
        return total_energy
    
    def _calculate_reward(
        self,
        energy_use: float,
        ppd: float
    ) -> float:
        """Calculate reward based on energy and comfort."""
        # Energy cost
        energy_cost = energy_use * self.config['energy_price']
        
        # Comfort score (inverse of PPD, normalized)
        comfort_score = (100 - ppd) / 100
        
        # Penalty for exceeding PPD threshold
        if ppd > COMFORT_CONFIG['ppd_threshold']:
            comfort_penalty = (ppd - COMFORT_CONFIG['ppd_threshold']) * 0.1
            self.episode_comfort_violations += 1
        else:
            comfort_penalty = 0
        
        # Combined reward
        reward = (
            -self.config['energy_weight'] * energy_cost +
            self.config['comfort_weight'] * comfort_score -
            comfort_penalty
        )
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        hvac_action = action[0]
        lighting_action = action[1]
        
        # Get current outdoor temperature
        row = self.data.iloc[self.current_step]
        outdoor_temp = row.get('air_temperature', 20)
        
        # Simulate indoor temperature
        self.indoor_temp = self._simulate_indoor_temp(outdoor_temp, hvac_action)
        self.lighting_level = lighting_action
        
        # Calculate energy use
        energy_use = self._calculate_energy_use(
            outdoor_temp, hvac_action, lighting_action
        )
        self.episode_energy += energy_use
        
        # Calculate comfort
        pmv = calculate_pmv(self.indoor_temp)
        ppd = calculate_ppd(pmv)
        
        # Calculate reward
        reward = self._calculate_reward(energy_use, ppd)
        self.episode_rewards.append(reward)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Get new state
        state = self._get_state()
        
        info = {
            'energy_use': energy_use,
            'indoor_temp': self.indoor_temp,
            'outdoor_temp': outdoor_temp,
            'ppd': ppd,
            'setpoint': self.setpoint,
            'lighting': self.lighting_level
        }
        
        return state, reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.current_step = 0
        self.indoor_temp = 21.0
        self.setpoint = 21.0
        self.lighting_level = 2
        self.episode_energy = 0
        self.episode_comfort_violations = 0
        self.episode_rewards = []
        
        return self._get_state()
    
    def render(self, mode='human'):
        """Render the environment state."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Indoor Temp: {self.indoor_temp:.1f}°C")
            print(f"Setpoint: {self.setpoint:.1f}°C")
            print(f"Lighting Level: {self.lighting_level}")
            print(f"Episode Energy: {self.episode_energy:.2f} kWh")


class HVACAgent:
    """Single agent for HVAC control (part of multi-agent setup)."""
    
    def __init__(self, action_space_size: int = 5):
        self.action_space = spaces.Discrete(action_space_size)
        self.q_table = {}  # Simple Q-learning for demonstration
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 0.1
    
    def get_action(self, state: Tuple, training: bool = True) -> int:
        """Get action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return self.action_space.sample()
        
        state_key = tuple(np.round(state, 2))
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        
        return np.argmax(self.q_table[state_key])
    
    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple
    ):
        """Update Q-table."""
        state_key = tuple(np.round(state, 2))
        next_state_key = tuple(np.round(next_state, 2))
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space.n)
        
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        self.q_table[state_key][action] = new_q


class LightingAgent:
    """Single agent for lighting control (part of multi-agent setup)."""
    
    def __init__(self, action_space_size: int = 5):
        self.action_space = spaces.Discrete(action_space_size)
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 0.1
    
    def get_action(self, state: Tuple, training: bool = True) -> int:
        """Get action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return self.action_space.sample()
        
        state_key = tuple(np.round(state, 2))
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        
        return np.argmax(self.q_table[state_key])
    
    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple
    ):
        """Update Q-table."""
        state_key = tuple(np.round(state, 2))
        next_state_key = tuple(np.round(next_state, 2))
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space.n)
        
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        self.q_table[state_key][action] = new_q


class MultiAgentController:
    """
    Multi-agent controller coordinating HVAC and lighting agents.
    Uses decentralized control with centralized training (CTDE).
    """
    
    def __init__(self):
        self.hvac_agent = HVACAgent()
        self.lighting_agent = LightingAgent()
        self.coordination_weight = 0.5  # Weight for coordinating rewards
    
    def get_actions(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> Tuple[int, int]:
        """Get coordinated actions from both agents."""
        # Each agent sees the full state but controls its own action
        hvac_action = self.hvac_agent.get_action(state, training)
        lighting_action = self.lighting_agent.get_action(state, training)
        
        return hvac_action, lighting_action
    
    def update(
        self,
        state: np.ndarray,
        actions: Tuple[int, int],
        reward: float,
        next_state: np.ndarray
    ):
        """Update both agents with shared reward."""
        hvac_action, lighting_action = actions
        
        # Distribute reward between agents
        # In practice, could decompose reward into HVAC and lighting components
        self.hvac_agent.update(state, hvac_action, reward, next_state)
        self.lighting_agent.update(state, lighting_action, reward, next_state)


class BaselineController:
    """Rule-based baseline controller for comparison."""
    
    def __init__(
        self,
        temp_setpoint: float = 21.0,
        schedule: Dict = None
    ):
        self.temp_setpoint = temp_setpoint
        self.schedule = schedule or {
            'night': (22, 6, 18),  # (start_hour, end_hour, setpoint)
            'day': (6, 22, 21)
        }
    
    def get_actions(self, hour: int, outdoor_temp: float) -> Tuple[int, int]:
        """Get rule-based actions."""
        # Simple schedule-based control
        if 6 <= hour < 22:
            target_setpoint = 21
            lighting = 3
        else:
            target_setpoint = 18
            lighting = 1
        
        # Simple bang-bang control for HVAC
        if outdoor_temp < target_setpoint - 2:
            hvac_action = 4  # +2°C
        elif outdoor_temp < target_setpoint:
            hvac_action = 3  # +1°C
        elif outdoor_temp > target_setpoint + 2:
            hvac_action = 0  # -2°C
        elif outdoor_temp > target_setpoint:
            hvac_action = 1  # -1°C
        else:
            hvac_action = 2  # No change
        
        return hvac_action, lighting


if __name__ == "__main__":
    # Test the environment
    print("Testing Building Energy Environment...")
    
    # Create dummy data
    n_steps = 1000
    data = pd.DataFrame({
        'timestamp': pd.date_range('2016-01-01', periods=n_steps, freq='H'),
        'meter_reading': np.random.uniform(10, 50, n_steps),
        'meter_reading_scaled': np.random.normal(0, 1, n_steps),
        'air_temperature': np.random.uniform(-5, 35, n_steps),
        'air_temperature_scaled': np.random.normal(0, 1, n_steps),
        'hour_sin': np.sin(2 * np.pi * np.arange(n_steps) / 24),
        'hour_cos': np.cos(2 * np.pi * np.arange(n_steps) / 24),
        'dow_sin': np.sin(2 * np.pi * np.arange(n_steps) / 168),
        'dow_cos': np.cos(2 * np.pi * np.arange(n_steps) / 168),
        'is_weekend': np.random.randint(0, 2, n_steps),
    })
    
    env = BuildingEnergyEnv(data)
    state = env.reset()
    
    print(f"State shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print(f"Step {i+1}: Reward={reward:.3f}, PPD={info['ppd']:.1f}%")
