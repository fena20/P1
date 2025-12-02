"""
Phase 4.1: Comparative Analysis
Compares baseline controller vs. optimized controller performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Path configuration
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed"
OPTIMIZATION_PATH = DATA_PATH / "optimization"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "results"

class BaselineController:
    """Baseline fixed setpoint controller."""
    
    def __init__(self, fixed_setpoint=23.0):
        """
        Initialize baseline controller.
        
        Parameters:
        -----------
        fixed_setpoint : float
            Fixed HVAC setpoint temperature (°C)
        """
        self.fixed_setpoint = fixed_setpoint
    
    def get_schedule(self, n_hours=24):
        """
        Get fixed setpoint schedule.
        
        Parameters:
        -----------
        n_hours : int
            Number of hours
        
        Returns:
        --------
        np.ndarray
            Setpoint schedule
        """
        return np.full(n_hours, self.fixed_setpoint)

class PerformanceAnalyzer:
    """Analyzer for comparing controller performance."""
    
    def __init__(self, surrogate_model, weather_data, price_schedule):
        """
        Initialize performance analyzer.
        
        Parameters:
        -----------
        surrogate_model : object
            Surrogate model for energy prediction
        weather_data : pd.DataFrame
            Weather data
        price_schedule : np.ndarray
            Price schedule
        """
        self.surrogate = surrogate_model
        self.weather_data = weather_data
        self.price_schedule = price_schedule
    
    def evaluate_controller(self, setpoint_schedule):
        """
        Evaluate controller performance.
        
        Parameters:
        -----------
        setpoint_schedule : np.ndarray
            Setpoint schedule
        
        Returns:
        --------
        dict
            Performance metrics
        """
        # Predict energy consumption
        if hasattr(self.surrogate, 'predict'):
            energy_consumption = self.surrogate.predict(
                self.weather_data,
                setpoint_schedule
            )
        else:
            # Placeholder
            base_load = 50
            temp_effect = (self.weather_data['airTemperature'].values - 22) * 2
            setpoint_effect = (setpoint_schedule - 22) * -1.5
            energy_consumption = base_load + temp_effect + setpoint_effect
        
        # Calculate metrics
        total_energy = np.sum(energy_consumption)
        total_cost = np.sum(energy_consumption * self.price_schedule)
        
        # Comfort violations (simplified)
        comfort_setpoint = 22.5
        deviations = np.abs(setpoint_schedule - comfort_setpoint)
        comfort_violations = np.sum(deviations > 2.0)
        
        return {
            'total_energy_kwh': total_energy,
            'total_cost_usd': total_cost,
            'comfort_violations_hours': comfort_violations,
            'avg_setpoint': np.mean(setpoint_schedule),
            'setpoint_range': np.max(setpoint_schedule) - np.min(setpoint_schedule)
        }
    
    def compare_controllers(self, baseline_schedule, optimized_schedule):
        """
        Compare baseline vs optimized controller.
        
        Parameters:
        -----------
        baseline_schedule : np.ndarray
            Baseline setpoint schedule
        optimized_schedule : np.ndarray
            Optimized setpoint schedule
        
        Returns:
        --------
        pd.DataFrame
            Comparison results
        """
        baseline_metrics = self.evaluate_controller(baseline_schedule)
        optimized_metrics = self.evaluate_controller(optimized_schedule)
        
        # Calculate improvements
        improvements = {}
        for key in baseline_metrics:
            if isinstance(baseline_metrics[key], (int, float)):
                baseline_val = baseline_metrics[key]
                optimized_val = optimized_metrics[key]
                if baseline_val != 0:
                    improvement_pct = ((baseline_val - optimized_val) / baseline_val) * 100
                else:
                    improvement_pct = 0
                improvements[f'{key}_improvement_pct'] = improvement_pct
        
        # Combine results
        results = {
            'Metric': [],
            'Baseline': [],
            'Optimized': [],
            'Improvement (%)': []
        }
        
        for key in baseline_metrics:
            if isinstance(baseline_metrics[key], (int, float)):
                results['Metric'].append(key.replace('_', ' ').title())
                results['Baseline'].append(baseline_metrics[key])
                results['Optimized'].append(optimized_metrics[key])
                results['Improvement (%)'].append(improvements.get(f'{key}_improvement_pct', 0))
        
        return pd.DataFrame(results)

def create_sample_data():
    """Create sample data for analysis."""
    hours = np.arange(24)
    
    # Weather data
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
        'month': np.full(24, 7)
    })
    
    # Price schedule
    prices = np.full(24, 0.10)
    prices[9:21] = 0.15
    prices[18:21] = 0.20
    
    return weather_data, prices

def main():
    """Main analysis function."""
    print("=" * 60)
    print("Phase 4.1: Comparative Analysis")
    print("=" * 60)
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Create sample data
    weather_data, price_schedule = create_sample_data()
    
    # Placeholder surrogate model
    class PlaceholderSurrogate:
        def predict(self, weather, setpoints):
            base_load = 50
            temp_effect = (weather['airTemperature'].values - 22) * 2
            setpoint_effect = (setpoints - 22) * -1.5
            return base_load + temp_effect + setpoint_effect
    
    surrogate = PlaceholderSurrogate()
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(surrogate, weather_data, price_schedule)
    
    # Baseline controller
    baseline_controller = BaselineController(fixed_setpoint=23.0)
    baseline_schedule = baseline_controller.get_schedule(24)
    
    # Optimized controller (load from optimization results if available)
    if (OPTIMIZATION_PATH / "pareto_solutions.csv").exists():
        opt_results = pd.read_csv(OPTIMIZATION_PATH / "pareto_solutions.csv")
        best_solution = opt_results.loc[opt_results['total_cost'].idxmin()]
        optimized_schedule = np.array(eval(best_solution['setpoint_schedule']))
    else:
        # Use sample optimized schedule
        optimized_schedule = np.array([
            22.0, 22.0, 21.5, 21.0, 20.5, 20.0, 20.0, 20.5,
            21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5,
            25.0, 25.0, 24.5, 24.0, 23.5, 23.0, 22.5, 22.0
        ])
    
    # Compare controllers
    comparison = analyzer.compare_controllers(baseline_schedule, optimized_schedule)
    
    # Save results
    comparison.to_csv(OUTPUT_PATH / "comparative_results.csv", index=False)
    
    print("\n" + "=" * 60)
    print("Comparative Results:")
    print("=" * 60)
    print(comparison.to_string(index=False))
    
    print(f"\nResults saved to {OUTPUT_PATH / 'comparative_results.csv'}")
    
    return comparison

if __name__ == "__main__":
    comparison = main()
