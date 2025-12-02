"""
Phase 3: Genetic Algorithm Optimization Framework
Multi-objective optimization for HVAC setpoint scheduling using surrogate models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from deap import base, creator, tools, algorithms
import random
import warnings
warnings.filterwarnings('ignore')

# Path configuration
MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "models"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "optimization"

class SurrogateModelWrapper:
    """Wrapper for surrogate models to be used in optimization."""
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize surrogate model wrapper.
        
        Parameters:
        -----------
        model_type : str
            Type of model ('lstm' or 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.feature_cols = None
        self.target_cols = None
        self.scaler_info = None
        self.load_model()
    
    def load_model(self):
        """Load trained surrogate model."""
        if self.model_type == 'xgboost':
            import pickle
            with open(MODEL_PATH / "xgboost_model.pkl", 'rb') as f:
                self.model = pickle.load(f)
            
            with open(MODEL_PATH / "xgboost_model_info.json", 'r') as f:
                model_info = json.load(f)
                self.feature_cols = model_info['feature_cols']
                self.target_cols = model_info['target_cols']
        
        elif self.model_type == 'lstm':
            import tensorflow as tf
            self.model = tf.keras.models.load_model(MODEL_PATH / "lstm_model.h5")
            
            with open(MODEL_PATH / "lstm_scaler_info.json", 'r') as f:
                scaler_info = json.load(f)
                self.feature_cols = scaler_info['feature_cols']
                self.target_cols = scaler_info['target_cols']
    
    def predict(self, weather_data, setpoint_schedule, current_state=None):
        """
        Predict energy consumption for given setpoint schedule.
        
        Parameters:
        -----------
        weather_data : pd.DataFrame
            Weather data for 24 hours
        setpoint_schedule : np.ndarray
            HVAC setpoint schedule (24 values)
        current_state : dict, optional
            Current building state (for LSTM sequence)
        
        Returns:
        --------
        np.ndarray
            Predicted energy consumption (24 values)
        """
        if self.model_type == 'xgboost':
            return self._predict_xgboost(weather_data, setpoint_schedule)
        else:
            return self._predict_lstm(weather_data, setpoint_schedule, current_state)
    
    def _predict_xgboost(self, weather_data, setpoint_schedule):
        """Predict using XGBoost model."""
        # Create feature dataframe
        n_hours = len(setpoint_schedule)
        features = weather_data.copy()
        features['setpoint'] = setpoint_schedule
        
        # Add temporal features
        if 'hour' not in features.columns:
            features['hour'] = np.arange(n_hours) % 24
        if 'day_of_week' not in features.columns:
            features['day_of_week'] = 1  # Default to Monday
        
        # Prepare features (simplified - would need full feature engineering)
        predictions = []
        for i in range(n_hours):
            # Use model's predict method (simplified)
            # In practice, would need to create proper lag features
            pred = self.model.model_energy.predict(features.iloc[[i]][self.feature_cols[:10]])[0]
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _predict_lstm(self, weather_data, setpoint_schedule, current_state):
        """Predict using LSTM model."""
        # Would need to create sequences - simplified for now
        # In practice, would use proper sequence preparation
        return np.random.normal(50, 10, len(setpoint_schedule))  # Placeholder

class OptimizationProblem:
    """Multi-objective optimization problem for HVAC control."""
    
    def __init__(self, surrogate_model, weather_data, price_schedule, comfort_weight=1.0):
        """
        Initialize optimization problem.
        
        Parameters:
        -----------
        surrogate_model : SurrogateModelWrapper
            Surrogate model for energy prediction
        weather_data : pd.DataFrame
            Weather forecast for 24 hours
        price_schedule : np.ndarray
            Electricity price schedule (24 values, $/kWh)
        comfort_weight : float
            Weight for comfort penalty in objective function
        """
        self.surrogate = surrogate_model
        self.weather_data = weather_data
        self.price_schedule = price_schedule
        self.comfort_weight = comfort_weight
        
        # Constraints
        self.setpoint_min = 19.0  # °C
        self.setpoint_max = 26.0  # °C
        self.comfort_band = [-0.5, 0.5]  # PMV comfort band
    
    def calculate_energy_cost(self, setpoint_schedule):
        """
        Calculate total energy cost for setpoint schedule.
        
        Parameters:
        -----------
        setpoint_schedule : np.ndarray
            HVAC setpoint schedule (24 values)
        
        Returns:
        --------
        float
            Total energy cost ($)
        """
        # Predict energy consumption
        energy_consumption = self.surrogate.predict(
            self.weather_data,
            setpoint_schedule
        )
        
        # Calculate cost
        cost = np.sum(energy_consumption * self.price_schedule)
        return cost
    
    def calculate_comfort_penalty(self, setpoint_schedule, indoor_temp_pred=None):
        """
        Calculate comfort penalty based on PMV or setpoint deviations.
        
        Parameters:
        -----------
        setpoint_schedule : np.ndarray
            HVAC setpoint schedule
        indoor_temp_pred : np.ndarray, optional
            Predicted indoor temperatures
        
        Returns:
        --------
        float
            Comfort penalty (discomfort hours or PMV violations)
        """
        # Simplified comfort penalty based on setpoint deviations from comfort zone
        comfort_setpoint = 22.5  # °C (typical comfort setpoint)
        deviations = np.abs(setpoint_schedule - comfort_setpoint)
        
        # Penalty for deviations outside comfort band
        penalty = np.sum(np.maximum(0, deviations - 2.0))  # Penalty for >2°C deviation
        
        return penalty
    
    def evaluate(self, setpoint_schedule):
        """
        Evaluate objective function for given setpoint schedule.
        
        Parameters:
        -----------
        setpoint_schedule : np.ndarray
            HVAC setpoint schedule (24 values)
        
        Returns:
        --------
        tuple
            (energy_cost, comfort_penalty) for multi-objective optimization
        """
        # Ensure setpoints are within bounds
        setpoint_schedule = np.clip(setpoint_schedule, self.setpoint_min, self.setpoint_max)
        
        # Calculate objectives
        energy_cost = self.calculate_energy_cost(setpoint_schedule)
        comfort_penalty = self.calculate_comfort_penalty(setpoint_schedule)
        
        return (energy_cost, comfort_penalty)
    
    def evaluate_weighted(self, setpoint_schedule):
        """
        Evaluate weighted single objective.
        
        Parameters:
        -----------
        setpoint_schedule : np.ndarray
            HVAC setpoint schedule
        
        Returns:
        --------
        float
            Weighted objective value
        """
        energy_cost, comfort_penalty = self.evaluate(setpoint_schedule)
        return energy_cost + self.comfort_weight * comfort_penalty

class GAOptimizer:
    """Genetic Algorithm optimizer for HVAC setpoint scheduling."""
    
    def __init__(self, problem, n_population=50, n_generations=100):
        """
        Initialize GA optimizer.
        
        Parameters:
        -----------
        problem : OptimizationProblem
            Optimization problem instance
        n_population : int
            Population size
        n_generations : int
            Number of generations
        """
        self.problem = problem
        self.n_population = n_population
        self.n_generations = n_generations
        
        # Setup DEAP
        self.setup_deap()
    
    def setup_deap(self):
        """Setup DEAP framework for multi-objective optimization."""
        # Create fitness and individual classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # Minimize both objectives
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        # Attribute generator
        self.toolbox.register("attr_setpoint", random.uniform, 
                             self.problem.setpoint_min, self.problem.setpoint_max)
        
        # Structure initializers
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.attr_setpoint, n=24)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Evaluation function
        self.toolbox.register("evaluate", self.problem.evaluate)
        
        # Genetic operators
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
        self.toolbox.register("select", tools.selNSGA2)
    
    def optimize(self, verbose=True):
        """
        Run genetic algorithm optimization.
        
        Parameters:
        -----------
        verbose : bool
            Print progress
        
        Returns:
        --------
        list
            Pareto-optimal solutions
        """
        # Initialize population
        population = self.toolbox.population(n=self.n_population)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution loop
        for generation in range(self.n_generations):
            if verbose and generation % 10 == 0:
                print(f"Generation {generation}/{self.n_generations}")
            
            # Select next generation
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=0.7, mutpb=0.3)
            
            # Evaluate offspring
            fitnesses = list(map(self.toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit
            
            # Select next generation
            population = self.toolbox.select(offspring + population, self.n_population)
        
        # Get Pareto front
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        
        return pareto_front

def create_sample_weather_data():
    """Create sample weather data for 24 hours."""
    hours = np.arange(24)
    
    # Simulate daily temperature cycle
    base_temp = 25.0
    temp_variation = 5.0 * np.sin(2 * np.pi * (hours - 6) / 24)
    temperatures = base_temp + temp_variation
    
    weather_data = pd.DataFrame({
        'airTemperature': temperatures,
        'dewTemperature': temperatures - 5,
        'cloudCoverage': np.random.uniform(0, 1, 24),
        'precipDepth1HR': np.random.uniform(0, 0.1, 24),
        'seaLevelPressure': np.full(24, 1013.25),
        'windSpeed': np.random.uniform(2, 8, 24),
        'hour': hours,
        'day_of_week': np.full(24, 1),
        'month': np.full(24, 7)  # July
    })
    
    return weather_data

def create_price_schedule(price_type='tou'):
    """
    Create time-of-use price schedule.
    
    Parameters:
    -----------
    price_type : str
        Price schedule type ('tou' or 'flat')
    
    Returns:
    --------
    np.ndarray
        Price schedule (24 values, $/kWh)
    """
    if price_type == 'tou':
        # Time-of-use pricing: higher during peak hours (9-21)
        prices = np.full(24, 0.10)  # Base price $0.10/kWh
        prices[9:21] = 0.15  # Peak hours: $0.15/kWh
        prices[18:21] = 0.20  # Super peak: $0.20/kWh
    else:
        # Flat pricing
        prices = np.full(24, 0.12)
    
    return prices

def main():
    """Main optimization function."""
    print("=" * 60)
    print("Phase 3: GA Optimization Framework")
    print("=" * 60)
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load surrogate model
    print("\nLoading surrogate model...")
    try:
        surrogate = SurrogateModelWrapper(model_type='xgboost')
    except Exception as e:
        print(f"Warning: Could not load model ({e}). Using placeholder.")
        surrogate = None
    
    # Create sample data
    weather_data = create_sample_weather_data()
    price_schedule = create_price_schedule(price_type='tou')
    
    # Create optimization problem
    if surrogate is None:
        # Use placeholder for demonstration
        class PlaceholderSurrogate:
            def predict(self, weather, setpoints):
                # Simple linear model for demonstration
                base_load = 50
                temp_effect = (weather['airTemperature'].values - 22) * 2
                setpoint_effect = (setpoints - 22) * -1.5
                return base_load + temp_effect + setpoint_effect
        
        surrogate = PlaceholderSurrogate()
    
    problem = OptimizationProblem(
        surrogate,
        weather_data,
        price_schedule,
        comfort_weight=10.0
    )
    
    # Run optimization
    print("\nRunning GA optimization...")
    optimizer = GAOptimizer(problem, n_population=50, n_generations=100)
    pareto_solutions = optimizer.optimize(verbose=True)
    
    # Extract results
    results = []
    for sol in pareto_solutions[:10]:  # Top 10 solutions
        energy_cost, comfort_penalty = sol.fitness.values
        results.append({
            'setpoint_schedule': list(sol),
            'energy_cost': energy_cost,
            'comfort_penalty': comfort_penalty,
            'total_cost': energy_cost + problem.comfort_weight * comfort_penalty
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH / "pareto_solutions.csv", index=False)
    
    print(f"\nFound {len(pareto_solutions)} Pareto-optimal solutions")
    print(f"Results saved to {OUTPUT_PATH / 'pareto_solutions.csv'}")
    
    # Print best solution (lowest total cost)
    best_solution = min(results, key=lambda x: x['total_cost'])
    print("\nBest Solution:")
    print(f"  Energy Cost: ${best_solution['energy_cost']:.2f}")
    print(f"  Comfort Penalty: {best_solution['comfort_penalty']:.2f}")
    print(f"  Total Cost: ${best_solution['total_cost']:.2f}")
    print(f"  Setpoint Schedule: {[f'{s:.1f}' for s in best_solution['setpoint_schedule']]}")
    
    return pareto_solutions, results

if __name__ == "__main__":
    pareto_solutions, results = main()
