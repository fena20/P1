"""
Phase 3: GA-Based Optimization Framework

This module implements:
1. Day-ahead HVAC setpoint optimization using Genetic Algorithm
2. Multi-objective optimization (energy cost vs thermal comfort)
3. Time-varying electricity pricing
4. PMV-based comfort constraints
5. Pareto front generation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Callable
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# Optimization libraries
from deap import base, creator, tools, algorithms
import random

# Load models
try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

import xgboost as xgb


class BuildingSurrogateOptimizer:
    """Optimize HVAC setpoints using surrogate models and GA."""
    
    def __init__(self, building_id: str, model_type: str = 'xgboost'):
        """
        Initialize optimizer for a building.
        
        Args:
            building_id: Building identifier
            model_type: 'lstm' or 'xgboost'
        """
        self.building_id = building_id
        self.model_type = model_type
        
        # Paths
        self.models_path = Path("/workspace/models")
        self.data_path = Path("/workspace/data")
        self.results_path = Path("/workspace/results")
        
        # Load surrogate model
        self.load_surrogate_model()
        
        # HVAC setpoint constraints (°C)
        self.T_MIN = 19.0
        self.T_MAX = 26.0
        
        # Comfort bounds (PMV scale: -3 (cold) to +3 (hot))
        self.PMV_MIN = -0.5
        self.PMV_MAX = 0.5
        
        # Time-of-use electricity pricing ($/kWh)
        # Peak: 8am-8pm weekdays
        # Off-peak: all other times
        self.PRICE_PEAK = 0.15
        self.PRICE_OFFPEAK = 0.08
        
    def load_surrogate_model(self):
        """Load trained surrogate model and scalers."""
        
        model_dir = self.models_path / self.model_type / self.building_id
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")
        
        print(f"Loading {self.model_type} model for {self.building_id}...")
        
        if self.model_type == 'xgboost':
            self.model = joblib.load(model_dir / "model.pkl")
        elif self.model_type == 'lstm' and TF_AVAILABLE:
            self.model = keras.models.load_model(model_dir / "model.keras")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Load scalers
        self.feature_scaler = joblib.load(model_dir / "feature_scaler.pkl")
        self.target_scaler = joblib.load(model_dir / "target_scaler.pkl")
        
        # Load metadata
        with open(model_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        print(f"  Model loaded successfully")
        print(f"  Features: {self.metadata['features']}")
    
    def predict_energy(self, weather_features: np.ndarray, 
                      setpoint: float = 22.0) -> float:
        """
        Predict energy consumption given weather conditions and setpoint.
        
        Args:
            weather_features: Array of weather and temporal features
            setpoint: HVAC setpoint temperature (°C)
            
        Returns:
            Predicted energy consumption (kWh)
        """
        # In a real system, setpoint would directly influence prediction
        # For this prototype, we use the trained model which doesn't have setpoint as input
        # We'll approximate the effect by scaling the prediction based on setpoint deviation
        
        # Make prediction with weather features
        if self.model_type == 'xgboost':
            # For XGBoost, need to create lagged features
            # Simplified: use current features only
            features_expanded = np.tile(weather_features, 25).reshape(1, -1)
            energy_scaled = self.model.predict(features_expanded)[0]
        else:  # LSTM
            # For LSTM, need sequence
            # Simplified: replicate current features for sequence
            features_seq = np.tile(weather_features, (24, 1)).reshape(1, 24, -1)
            energy_scaled = self.model.predict(features_seq, verbose=0)[0][0]
        
        # Inverse transform
        energy_pred = self.target_scaler.inverse_transform([[energy_scaled]])[0][0]
        
        # Apply setpoint effect (simplified model)
        # Assumption: 1°C change in setpoint changes energy by ~5% 
        baseline_setpoint = 22.0  # Assumed baseline
        setpoint_effect = 1.0 + 0.05 * (setpoint - baseline_setpoint)
        energy_adjusted = max(0, energy_pred * setpoint_effect)
        
        return energy_adjusted
    
    def compute_pmv(self, T_indoor: float, T_outdoor: float) -> float:
        """
        Simplified PMV (Predicted Mean Vote) calculation.
        
        In a full implementation, this would use the complete Fanger PMV equation
        with metabolic rate, clothing insulation, air velocity, etc.
        
        Args:
            T_indoor: Indoor air temperature (°C)
            T_outdoor: Outdoor air temperature (°C)
            
        Returns:
            PMV value (-3 to +3 scale)
        """
        # Simplified PMV based on indoor temperature deviation from comfort
        T_comfort = 22.0  # Neutral comfort temperature
        
        # PMV approximately scales with temperature deviation
        pmv = (T_indoor - T_comfort) / 3.0
        
        return np.clip(pmv, -3.0, 3.0)
    
    def get_electricity_price(self, hour: int, day_of_week: int) -> float:
        """
        Get time-of-use electricity price.
        
        Args:
            hour: Hour of day (0-23)
            day_of_week: Day of week (1=Mon, 7=Sun)
            
        Returns:
            Electricity price ($/kWh)
        """
        # Peak pricing on weekdays (Mon-Fri) from 8am to 8pm
        is_weekday = day_of_week <= 5
        is_peak_hour = 8 <= hour < 20
        
        if is_weekday and is_peak_hour:
            return self.PRICE_PEAK
        else:
            return self.PRICE_OFFPEAK
    
    def evaluate_schedule(self, setpoint_schedule: List[float],
                         weather_forecast: pd.DataFrame,
                         comfort_weight: float = 1.0) -> Tuple[float, float, Dict]:
        """
        Evaluate an HVAC setpoint schedule over 24 hours.
        
        Args:
            setpoint_schedule: List of 24 hourly setpoint temperatures (°C)
            weather_forecast: DataFrame with 24 hours of weather forecast
            comfort_weight: Weight for comfort penalty in objective
            
        Returns:
            Tuple of (total_cost, comfort_penalty, details_dict)
        """
        total_energy = 0.0
        total_cost = 0.0
        comfort_violations = 0
        total_pmv_deviation = 0.0
        
        hourly_details = []
        
        for hour in range(24):
            # Get weather features for this hour
            weather_row = weather_forecast.iloc[hour]
            
            # Extract features (matching model training)
            features = []
            for feat in self.metadata['features']:
                if feat in weather_row:
                    features.append(weather_row[feat])
                else:
                    features.append(0.0)  # Default if missing
            
            features = np.array(features).reshape(1, -1)
            features_scaled = self.feature_scaler.transform(features)
            
            # Get setpoint for this hour
            setpoint = setpoint_schedule[hour]
            
            # Predict energy consumption
            energy = self.predict_energy(features_scaled[0], setpoint)
            
            # Get electricity price
            hour_of_day = int(weather_row.get('hour', hour))
            day_of_week = int(weather_row.get('day_of_week', 1))
            price = self.get_electricity_price(hour_of_day, day_of_week)
            
            # Calculate cost
            cost = energy * price
            
            # Estimate indoor temperature (simplified)
            # In reality, this would come from a thermal model
            T_outdoor = weather_row.get('airTemperature', 20.0)
            T_indoor = setpoint  # Simplified: assume perfect control
            
            # Compute PMV
            pmv = self.compute_pmv(T_indoor, T_outdoor)
            
            # Check comfort violation
            if pmv < self.PMV_MIN or pmv > self.PMV_MAX:
                comfort_violations += 1
                pmv_deviation = max(0, abs(pmv) - self.PMV_MAX)
            else:
                pmv_deviation = 0.0
            
            total_energy += energy
            total_cost += cost
            total_pmv_deviation += pmv_deviation
            
            hourly_details.append({
                'hour': int(hour),
                'setpoint': float(setpoint),
                'energy': float(energy),
                'price': float(price),
                'cost': float(cost),
                'T_outdoor': float(T_outdoor),
                'T_indoor': float(T_indoor),
                'pmv': float(pmv),
                'comfort_violation': bool(pmv_deviation > 0)
            })
        
        # Comfort penalty (penalize violations and deviations)
        comfort_penalty = comfort_violations + total_pmv_deviation * 10.0
        
        # Combined objective
        objective = total_cost + comfort_weight * comfort_penalty
        
        details = {
            'total_energy': total_energy,
            'total_cost': total_cost,
            'comfort_violations': comfort_violations,
            'comfort_penalty': comfort_penalty,
            'objective': objective,
            'hourly': hourly_details
        }
        
        return total_cost, comfort_penalty, details
    
    def optimize_day_ahead(self, weather_forecast: pd.DataFrame,
                          comfort_weight: float = 1.0,
                          population_size: int = 100,
                          n_generations: int = 100) -> Dict:
        """
        Optimize 24-hour HVAC setpoint schedule using Genetic Algorithm.
        
        Args:
            weather_forecast: 24-hour weather forecast
            comfort_weight: Weight for comfort in objective function
            population_size: GA population size
            n_generations: Number of GA generations
            
        Returns:
            Optimization results dictionary
        """
        print(f"\n=== Optimizing Day-Ahead Schedule ===")
        print(f"  Building: {self.building_id}")
        print(f"  Comfort weight: {comfort_weight}")
        print(f"  GA: {population_size} population, {n_generations} generations")
        
        # Define GA fitness function (minimize)
        def eval_individual(individual):
            setpoints = list(individual)
            cost, comfort, _ = self.evaluate_schedule(
                setpoints, weather_forecast, comfort_weight
            )
            # Return as tuple for DEAP (minimization)
            return (cost + comfort_weight * comfort,)
        
        # Set up DEAP
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        # Gene: setpoint temperature between T_MIN and T_MAX
        toolbox.register("attr_setpoint", random.uniform, self.T_MIN, self.T_MAX)
        
        # Individual: 24 genes (one per hour)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.attr_setpoint, n=24)
        
        # Population
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Genetic operators
        toolbox.register("evaluate", eval_individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Constraint: keep setpoints in bounds
        def checkBounds(individual):
            for i in range(len(individual)):
                if individual[i] < self.T_MIN:
                    individual[i] = self.T_MIN
                elif individual[i] > self.T_MAX:
                    individual[i] = self.T_MAX
            return individual
        
        toolbox.decorate("mate", tools.DeltaPenalty(checkBounds, 1e6))
        toolbox.decorate("mutate", tools.DeltaPenalty(checkBounds, 1e6))
        
        # Initialize population
        pop = toolbox.population(n=population_size)
        
        # Statistics
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        
        # Run GA
        print("  Running genetic algorithm...")
        pop, logbook = algorithms.eaSimple(
            pop, toolbox,
            cxpb=0.7, mutpb=0.3,
            ngen=n_generations,
            stats=stats,
            verbose=False
        )
        
        # Get best individual
        best_ind = tools.selBest(pop, k=1)[0]
        best_setpoints = list(best_ind)
        
        # Evaluate best solution
        cost, comfort, details = self.evaluate_schedule(
            best_setpoints, weather_forecast, comfort_weight
        )
        
        print(f"\n  Optimization complete!")
        print(f"    Total energy: {details['total_energy']:.2f} kWh")
        print(f"    Total cost: ${details['total_cost']:.2f}")
        print(f"    Comfort violations: {details['comfort_violations']} hours")
        print(f"    Comfort penalty: {details['comfort_penalty']:.2f}")
        
        results = {
            'building_id': self.building_id,
            'comfort_weight': comfort_weight,
            'best_setpoints': best_setpoints,
            'total_energy': details['total_energy'],
            'total_cost': details['total_cost'],
            'comfort_violations': details['comfort_violations'],
            'comfort_penalty': details['comfort_penalty'],
            'objective': details['objective'],
            'hourly_details': details['hourly'],
            'ga_stats': {
                'generations': n_generations,
                'final_min': logbook[-1]['min'],
                'final_avg': logbook[-1]['avg']
            }
        }
        
        return results
    
    def baseline_controller(self, weather_forecast: pd.DataFrame,
                           constant_setpoint: float = 22.0) -> Dict:
        """
        Evaluate baseline fixed-setpoint controller.
        
        Args:
            weather_forecast: 24-hour weather forecast
            constant_setpoint: Fixed setpoint temperature
            
        Returns:
            Evaluation results dictionary
        """
        print(f"\n=== Evaluating Baseline Controller ===")
        print(f"  Constant setpoint: {constant_setpoint}°C")
        
        # Create constant schedule
        setpoint_schedule = [constant_setpoint] * 24
        
        # Evaluate
        cost, comfort, details = self.evaluate_schedule(
            setpoint_schedule, weather_forecast, comfort_weight=1.0
        )
        
        print(f"    Total energy: {details['total_energy']:.2f} kWh")
        print(f"    Total cost: ${details['total_cost']:.2f}")
        print(f"    Comfort violations: {details['comfort_violations']} hours")
        
        results = {
            'building_id': self.building_id,
            'controller_type': 'baseline',
            'setpoint': constant_setpoint,
            'total_energy': details['total_energy'],
            'total_cost': details['total_cost'],
            'comfort_violations': details['comfort_violations'],
            'comfort_penalty': details['comfort_penalty'],
            'hourly_details': details['hourly']
        }
        
        return results
    
    def generate_pareto_front(self, weather_forecast: pd.DataFrame,
                             n_points: int = 10) -> List[Dict]:
        """
        Generate Pareto front by varying comfort weight.
        
        Args:
            weather_forecast: 24-hour weather forecast
            n_points: Number of Pareto points to generate
            
        Returns:
            List of optimization results for different comfort weights
        """
        print(f"\n=== Generating Pareto Front ===")
        print(f"  Computing {n_points} Pareto-optimal solutions")
        
        # Comfort weights from 0 (cost-only) to 10 (comfort-priority)
        comfort_weights = np.logspace(-2, 1, n_points)
        
        pareto_results = []
        
        for i, w in enumerate(comfort_weights):
            print(f"\n  Point {i+1}/{n_points}: comfort_weight = {w:.3f}")
            
            results = self.optimize_day_ahead(
                weather_forecast,
                comfort_weight=w,
                population_size=50,
                n_generations=50
            )
            
            pareto_results.append(results)
        
        print(f"\n  Pareto front generation complete!")
        
        return pareto_results


def main():
    """Run optimization for multiple buildings and scenarios."""
    
    print("="*80)
    print("PHASE 3: GA-BASED OPTIMIZATION FRAMEWORK")
    print("="*80)
    
    # Load a test day's weather forecast
    data_path = Path("/workspace/data/splits")
    
    # Get available buildings
    building_ids = [d.name for d in data_path.iterdir() if d.is_dir()][:3]  # Test with 3 buildings
    
    print(f"\nProcessing {len(building_ids)} buildings:")
    for bid in building_ids:
        print(f"  - {bid}")
    
    all_results = {}
    
    for building_id in building_ids:
        print(f"\n{'='*80}")
        print(f"Building: {building_id}")
        print(f"{'='*80}")
        
        try:
            # Load test data for weather forecast
            test_df = pd.read_csv(data_path / building_id / "test.csv")
            test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
            
            # Use first 24 hours as forecast
            weather_forecast = test_df.head(24).copy()
            
            # Initialize optimizer
            optimizer = BuildingSurrogateOptimizer(building_id, model_type='xgboost')
            
            # Run baseline
            baseline_results = optimizer.baseline_controller(weather_forecast)
            
            # Run optimized
            optimized_results = optimizer.optimize_day_ahead(
                weather_forecast,
                comfort_weight=1.0,
                population_size=100,
                n_generations=100
            )
            
            # Compute improvements
            energy_savings = ((baseline_results['total_energy'] - optimized_results['total_energy']) 
                            / baseline_results['total_energy'] * 100)
            cost_savings = ((baseline_results['total_cost'] - optimized_results['total_cost'])
                          / baseline_results['total_cost'] * 100)
            
            print(f"\n=== Results Summary ===")
            print(f"  Energy savings: {energy_savings:.1f}%")
            print(f"  Cost savings: {cost_savings:.1f}%")
            print(f"  Baseline comfort violations: {baseline_results['comfort_violations']}")
            print(f"  Optimized comfort violations: {optimized_results['comfort_violations']}")
            
            all_results[building_id] = {
                'baseline': baseline_results,
                'optimized': optimized_results,
                'savings': {
                    'energy_pct': float(energy_savings),
                    'cost_pct': float(cost_savings)
                }
            }
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[building_id] = {'error': str(e)}
    
    # Save results
    results_file = Path("/workspace/results/optimization_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("PHASE 3 COMPLETE")
    print("="*80)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
