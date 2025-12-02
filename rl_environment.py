"""
Reinforcement Learning Environment for Energy Optimization
Implements Gym-compatible environment with multi-agent support
"""
import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from config import *

class EnergyOptimizationEnv(gym.Env):
    """
    Gym environment for energy optimization in residential buildings
    State: [current_energy, weather_features, predicted_energy, predicted_comfort, time_features]
    Action: HVAC setpoint adjustment (discrete: -2, -1, 0, +1, +2 degrees C)
    Reward: -energy_cost + comfort_score - penalty_for_high_PPD
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data: pd.DataFrame, dl_predictions: Dict = None, building_id: int = None):
        super(EnergyOptimizationEnv, self).__init__()
        
        self.data = data.sort_values('timestamp').reset_index(drop=True)
        if building_id is not None:
            self.data = self.data[self.data['building_id'] == building_id].reset_index(drop=True)
        
        self.dl_predictions = dl_predictions
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        
        # State space: [energy, temp, humidity, predicted_energy, predicted_ppd, hour, day_of_week]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )
        
        # Action space: discrete setpoint adjustments
        self.action_space = spaces.Discrete(NUM_ACTIONS)  # 0: -2°C, 1: -1°C, 2: 0°C, 3: +1°C, 4: +2°C
        
        # Current state
        self.current_setpoint = 22.0  # Initial setpoint in Celsius
        self.current_energy = None
        self.current_ppd = None
        
        # Statistics
        self.total_energy = 0.0
        self.total_cost = 0.0
        self.total_ppd = 0.0
        self.comfort_violations = 0
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_setpoint = 22.0
        self.total_energy = 0.0
        self.total_cost = 0.0
        self.total_ppd = 0.0
        self.comfort_violations = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation"""
        if self.current_step >= len(self.data):
            # Return zero observation if out of bounds
            return np.zeros(7, dtype=np.float32)
        
        row = self.data.iloc[self.current_step]
        
        # Get DL predictions if available
        if self.dl_predictions is not None and self.current_step < len(self.dl_predictions['energy_preds']):
            pred_energy = self.dl_predictions['energy_preds'][self.current_step]
            pred_ppd = self.dl_predictions['comfort_preds'][self.current_step]
        else:
            # Fallback to current values
            pred_energy = row['meter_reading']
            pred_ppd = row['ppd']
        
        observation = np.array([
            row['meter_reading'],  # Current energy
            row['air_temperature'],  # Temperature
            row['dew_temperature'],  # Humidity proxy
            pred_energy,  # Predicted energy from DL
            pred_ppd,  # Predicted PPD from DL
            row['hour'] / 24.0,  # Normalized hour
            row['day_of_week'] / 7.0  # Normalized day of week
        ], dtype=np.float32)
        
        return observation
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= len(self.data):
            # Episode finished
            return self._get_observation(), 0.0, True, {}
        
        # Convert action to setpoint adjustment
        action_map = [-2.0, -1.0, 0.0, 1.0, 2.0]
        setpoint_adjustment = action_map[action]
        self.current_setpoint += setpoint_adjustment
        self.current_setpoint = np.clip(self.current_setpoint, 18.0, 26.0)  # Reasonable range
        
        # Get current data
        row = self.data.iloc[self.current_step]
        base_energy = row['meter_reading']
        base_ppd = row['ppd']
        
        # Simulate energy consumption based on setpoint adjustment
        # Energy increases when setpoint is far from outdoor temp
        outdoor_temp = row['air_temperature']
        temp_diff = abs(self.current_setpoint - outdoor_temp)
        energy_multiplier = 1.0 + 0.1 * temp_diff  # 10% increase per degree difference
        
        # Simulate comfort based on setpoint
        # PPD increases when setpoint is far from optimal (22°C)
        optimal_temp = 22.0
        temp_deviation = abs(self.current_setpoint - optimal_temp)
        ppd_adjustment = 2.0 * temp_deviation  # 2% PPD per degree deviation
        adjusted_ppd = base_ppd + ppd_adjustment
        adjusted_ppd = np.clip(adjusted_ppd, 0, 100)
        
        # Calculate actual energy (with some randomness)
        actual_energy = base_energy * energy_multiplier * (1 + np.random.normal(0, 0.05))
        actual_energy = max(0, actual_energy)
        
        self.current_energy = actual_energy
        self.current_ppd = adjusted_ppd
        
        # Calculate reward
        energy_cost = actual_energy * ENERGY_COST_PER_KWH
        comfort_score = 1.0 if adjusted_ppd < COMFORT_PENALTY_THRESHOLD else 0.0
        comfort_penalty = max(0, adjusted_ppd - COMFORT_PENALTY_THRESHOLD) * 0.1
        
        reward = -energy_cost + comfort_score - comfort_penalty
        
        # Update statistics
        self.total_energy += actual_energy
        self.total_cost += energy_cost
        self.total_ppd += adjusted_ppd
        if adjusted_ppd > COMFORT_PENALTY_THRESHOLD:
            self.comfort_violations += 1
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data)
        
        info = {
            'energy': actual_energy,
            'ppd': adjusted_ppd,
            'setpoint': self.current_setpoint,
            'cost': energy_cost
        }
        
        return self._get_observation(), reward, done, info
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Setpoint: {self.current_setpoint:.2f}°C")
            print(f"Energy: {self.current_energy:.2f} kWh")
            print(f"PPD: {self.current_ppd:.2f}%")
            print(f"Total Cost: ${self.total_cost:.2f}")

class MultiAgentEnergyEnv:
    """
    Multi-agent wrapper for energy optimization
    Agents: HVAC and Lighting subsystems
    """
    def __init__(self, data: pd.DataFrame, dl_predictions: Dict = None, building_id: int = None):
        self.hvac_env = EnergyOptimizationEnv(data, dl_predictions, building_id)
        
        # Lighting environment (simplified - uses similar structure)
        self.lighting_env = EnergyOptimizationEnv(data, dl_predictions, building_id)
        
        # Override action space for lighting (dimming levels)
        self.lighting_env.action_space = spaces.Discrete(5)  # 0-4 dimming levels
        
    def reset(self):
        """Reset all agents"""
        hvac_obs = self.hvac_env.reset()
        lighting_obs = self.lighting_env.reset()
        return {'hvac': hvac_obs, 'lighting': lighting_obs}
    
    def step(self, actions: Dict):
        """Step all agents"""
        hvac_obs, hvac_reward, hvac_done, hvac_info = self.hvac_env.step(actions['hvac'])
        
        # Lighting action affects energy consumption
        lighting_action = actions['lighting']
        lighting_obs = self.lighting_env._get_observation()
        
        # Simulate lighting energy (simplified)
        base_lighting_energy = 0.1  # Base lighting load
        dimming_factor = 1.0 - (lighting_action * 0.2)  # 0-80% reduction
        lighting_energy = base_lighting_energy * dimming_factor
        
        lighting_reward = -lighting_energy * ENERGY_COST_PER_KWH
        lighting_done = hvac_done
        
        lighting_info = {'energy': lighting_energy}
        
        done = hvac_done and lighting_done
        
        return (
            {'hvac': hvac_obs, 'lighting': lighting_obs},
            {'hvac': hvac_reward, 'lighting': lighting_reward},
            done,
            {'hvac': hvac_info, 'lighting': lighting_info}
        )
    
    def get_statistics(self):
        """Get combined statistics"""
        return {
            'total_energy': self.hvac_env.total_energy + self.lighting_env.total_energy,
            'total_cost': self.hvac_env.total_cost + self.lighting_env.total_cost,
            'avg_ppd': self.hvac_env.total_ppd / max(1, self.hvac_env.current_step),
            'comfort_violations': self.hvac_env.comfort_violations
        }
