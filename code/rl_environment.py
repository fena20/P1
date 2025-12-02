"""
Reinforcement Learning Environment for Building Energy Optimization
Multi-Agent System for HVAC and Lighting Control
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import config

class BuildingEnergyEnv(gym.Env):
    """
    Custom Gym Environment for Building Energy Optimization
    
    State Space:
        - Current time features (hour, day_of_week)
        - Weather conditions (temperature, humidity)
        - Building characteristics
        - Current energy consumption
        - Predicted future energy (from DL model)
        - Occupancy level
        - Current comfort metrics (PMV, PPD)
    
    Action Space:
        - HVAC temperature setpoint adjustment
        - Lighting level control
    
    Reward:
        Negative cost (energy + comfort penalty)
    """
    
    def __init__(self, building_data, dl_model=None, scaler_X=None, agent_type='hvac'):
        super(BuildingEnergyEnv, self).__init__()
        
        self.building_data = building_data.reset_index(drop=True)
        self.dl_model = dl_model
        self.scaler_X = scaler_X
        self.agent_type = agent_type
        
        # State space: 15 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        
        # Action space
        if agent_type == 'hvac':
            # Discrete actions: decrease temp, maintain, increase temp
            self.action_space = spaces.Discrete(5)
        else:  # lighting
            # Discrete lighting levels
            self.action_space = spaces.Discrete(len(config.LIGHTING_LEVELS))
        
        self.current_step = 0
        self.max_steps = len(self.building_data) - 1
        
        # State variables
        self.current_hvac_setpoint = 21.0  # Initial comfort temperature
        self.current_lighting_level = 0.7  # Initial lighting level
        
        # Episode tracking
        self.episode_energy = 0
        self.episode_cost = 0
        self.episode_discomfort = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = np.random.randint(0, max(1, self.max_steps - 1000))
        self.episode_energy = 0
        self.episode_cost = 0
        self.episode_discomfort = 0
        self.current_hvac_setpoint = 21.0
        self.current_lighting_level = 0.7
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def _get_observation(self):
        """Get current state observation"""
        if self.current_step >= len(self.building_data):
            self.current_step = 0
        
        row = self.building_data.iloc[self.current_step]
        
        # Extract features
        obs = np.array([
            row['hour'] / 24.0,
            row['day_of_week'] / 7.0,
            row['air_temperature'] / 40.0,  # Normalize
            row['humidity'] / 100.0,
            row['square_feet'] / 50000.0,
            row['occupancy'] / 200.0,
            row['meter_reading'] / 1000.0,  # Normalize energy
            self.current_hvac_setpoint / 30.0,
            self.current_lighting_level,
            row.get('wind_speed', 0) / 20.0,
            row.get('cloud_coverage', 0) / 9.0,
            np.sin(2 * np.pi * row['hour'] / 24),  # Cyclical time
            np.cos(2 * np.pi * row['hour'] / 24),
            np.sin(2 * np.pi * row['day_of_year'] / 365),
            np.cos(2 * np.pi * row['day_of_year'] / 365)
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_comfort_metrics(self, indoor_temp, humidity):
        """Calculate PMV and PPD for comfort assessment"""
        # Simplified PMV calculation
        temp_comfort = 21.0
        pmv = 0.5 * (indoor_temp - temp_comfort)
        pmv = np.clip(pmv, -3, 3)
        
        # PPD from PMV
        ppd = 100 - 95 * np.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
        ppd = np.clip(ppd, 5, 100)
        
        return pmv, ppd
    
    def _calculate_energy_consumption(self, row, hvac_setpoint, lighting_level):
        """Calculate energy consumption based on control actions"""
        base_energy = row['meter_reading']
        
        # HVAC energy adjustment
        outdoor_temp = row['air_temperature']
        hvac_energy = abs(hvac_setpoint - outdoor_temp) * row['square_feet'] / 10000
        
        # Lighting energy
        lighting_energy = lighting_level * row['square_feet'] / 100
        
        # Occupancy factor
        occupancy_factor = 1.0
        if 8 <= row['hour'] <= 22:
            occupancy_factor = 1.2
        else:
            occupancy_factor = 0.8
        
        total_energy = (base_energy * 0.3 + hvac_energy + lighting_energy) * occupancy_factor
        
        return total_energy
    
    def step(self, action):
        """Execute one step in the environment"""
        row = self.building_data.iloc[self.current_step]
        
        # Apply action
        if self.agent_type == 'hvac':
            # Map action to setpoint adjustment
            delta = (action - 2) * config.HVAC_SETPOINT_DELTA  # Actions: 0,1,2,3,4 -> -1,-.5,0,.5,1
            self.current_hvac_setpoint += delta
            self.current_hvac_setpoint = np.clip(
                self.current_hvac_setpoint, 
                config.HVAC_TEMP_RANGE[0], 
                config.HVAC_TEMP_RANGE[1]
            )
        else:  # lighting
            self.current_lighting_level = config.LIGHTING_LEVELS[action]
        
        # Calculate energy consumption
        energy_consumed = self._calculate_energy_consumption(
            row, self.current_hvac_setpoint, self.current_lighting_level
        )
        
        # Calculate comfort
        pmv, ppd = self._calculate_comfort_metrics(
            self.current_hvac_setpoint, row['humidity']
        )
        
        # Calculate costs
        energy_cost = energy_consumed * config.ENERGY_COST
        
        # Comfort penalty
        comfort_penalty = 0
        if ppd > config.PPD_THRESHOLD:
            comfort_penalty = (ppd - config.PPD_THRESHOLD) * 0.1
        
        # Reward function: minimize cost and discomfort
        reward = -(energy_cost + comfort_penalty)
        
        # Track episode metrics
        self.episode_energy += energy_consumed
        self.episode_cost += energy_cost
        self.episode_discomfort += ppd
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= min(self.max_steps, len(self.building_data) - 1)
        
        # Get next observation
        obs = self._get_observation()
        
        # Info
        info = {
            'energy': energy_consumed,
            'cost': energy_cost,
            'pmv': pmv,
            'ppd': ppd,
            'hvac_setpoint': self.current_hvac_setpoint,
            'lighting_level': self.current_lighting_level
        }
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Energy: {self.episode_energy:.2f} kWh, "
                  f"Cost: ${self.episode_cost:.2f}, Avg PPD: {self.episode_discomfort / (self.current_step + 1):.2f}%")

class MultiAgentBuildingEnv:
    """
    Multi-agent environment coordinating HVAC and Lighting agents
    """
    
    def __init__(self, building_data, dl_model=None, scaler_X=None):
        self.envs = {
            'hvac': BuildingEnergyEnv(building_data, dl_model, scaler_X, agent_type='hvac'),
            'lighting': BuildingEnergyEnv(building_data, dl_model, scaler_X, agent_type='lighting')
        }
        
        self.observation_spaces = {
            name: env.observation_space for name, env in self.envs.items()
        }
        
        self.action_spaces = {
            name: env.action_space for name, env in self.envs.items()
        }
    
    def reset(self):
        """Reset all agents"""
        return {name: env.reset() for name, env in self.envs.items()}
    
    def step(self, actions):
        """Execute actions for all agents"""
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        for name, env in self.envs.items():
            obs, reward, done, info = env.step(actions[name])
            observations[name] = obs
            rewards[name] = reward
            dones[name] = done
            infos[name] = info
        
        # Check if all done
        done_all = all(dones.values())
        
        return observations, rewards, done_all, infos

def create_building_env(building_data, dl_model=None, scaler_X=None, multi_agent=False):
    """Factory function to create building environment"""
    if multi_agent:
        return MultiAgentBuildingEnv(building_data, dl_model, scaler_X)
    else:
        return BuildingEnergyEnv(building_data, dl_model, scaler_X, agent_type='hvac')

if __name__ == "__main__":
    print("Building Energy Environment for Reinforcement Learning")
    print("=" * 60)
