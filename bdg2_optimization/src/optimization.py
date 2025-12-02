"""
Genetic Algorithm Optimization Module for Building Energy Control.

Implements:
- Day-ahead HVAC setpoint optimization
- Multi-objective cost-comfort trade-off
- Pareto front analysis
- PMV-based thermal comfort evaluation
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
import random
import warnings
warnings.filterwarnings('ignore')

from deap import base, creator, tools, algorithms
from pythermalcomfort.models.pmv_ppd_ashrae import pmv_ppd_ashrae

from config import OptimizationConfig, ElectricityTariff


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    optimized_setpoints: np.ndarray
    baseline_setpoints: np.ndarray
    optimized_energy: np.ndarray
    baseline_energy: np.ndarray
    optimized_cost: float
    baseline_cost: float
    optimized_comfort_violations: int
    baseline_comfort_violations: int
    optimized_indoor_temps: np.ndarray
    baseline_indoor_temps: np.ndarray
    pareto_front: Optional[List] = None
    computation_time: float = 0.0


class ThermalComfortCalculator:
    """
    PMV-based thermal comfort calculator.
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
    
    def calculate_pmv(
        self,
        indoor_temp: float,
        outdoor_temp: float,
        relative_humidity: float = 50.0,
        air_velocity: float = None,
        met: float = None,
        clo: float = None
    ) -> float:
        """
        Calculate PMV for given conditions.
        
        Args:
            indoor_temp: Indoor air temperature (°C)
            outdoor_temp: Outdoor temperature (for MRT estimation)
            relative_humidity: Relative humidity (%)
            air_velocity: Air velocity (m/s)
            met: Metabolic rate (met)
            clo: Clothing insulation (clo)
            
        Returns:
            PMV value
        """
        if air_velocity is None:
            air_velocity = self.config.default_air_velocity
        if met is None:
            met = self.config.default_met
        if clo is None:
            clo = self.config.default_clo
            
        # Estimate mean radiant temperature (simplified)
        # Assume MRT is close to indoor air temp with slight outdoor influence
        tr = indoor_temp * 0.9 + outdoor_temp * 0.1
        
        try:
            result = pmv_ppd_ashrae(
                tdb=indoor_temp,
                tr=tr,
                vr=air_velocity,
                rh=relative_humidity,
                met=met,
                clo=clo
            )
            return result.pmv
        except Exception:
            # Fallback to simplified PMV
            return (indoor_temp - 24) / 3  # Simplified linear approximation
    
    def is_comfortable(self, pmv: float) -> bool:
        """Check if PMV is within comfort bounds."""
        return self.config.pmv_comfort_low <= pmv <= self.config.pmv_comfort_high
    
    def comfort_penalty(self, pmv: float) -> float:
        """Calculate comfort penalty for PMV deviation."""
        if self.is_comfortable(pmv):
            return 0.0
        # Quadratic penalty for deviation from comfort band
        if pmv < self.config.pmv_comfort_low:
            deviation = self.config.pmv_comfort_low - pmv
        else:
            deviation = pmv - self.config.pmv_comfort_high
        return deviation ** 2


class BuildingSurrogateWrapper:
    """
    Wrapper to interface surrogate model with optimizer.
    
    Simulates building response to setpoint changes.
    """
    
    def __init__(
        self,
        surrogate_model,
        base_features: pd.DataFrame,
        energy_scaler=None,
        config: OptimizationConfig = None
    ):
        self.model = surrogate_model
        self.base_features = base_features.copy()
        self.energy_scaler = energy_scaler
        self.config = config or OptimizationConfig()
        
        # Simplified thermal model parameters
        self.thermal_mass = 2.5  # hours time constant
        self.hvac_efficiency = 3.0  # COP
        
    def simulate_day(
        self,
        setpoints: np.ndarray,
        weather_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate building energy and indoor temperature for a day.
        
        Uses a hybrid approach combining historical data with thermal physics.
        The model captures pre-cooling/pre-heating dynamics for load shifting.
        
        Args:
            setpoints: 24-hour HVAC setpoint profile (°C)
            weather_data: Weather conditions for the day
            
        Returns:
            energy: Hourly energy consumption (kWh)
            indoor_temps: Hourly indoor temperatures (°C)
        """
        hours = len(setpoints)
        energy = np.zeros(hours)
        indoor_temps = np.zeros(hours)
        
        # Reference energy from historical data
        ref_energy = weather_data['energy_kwh'].values[:hours].copy()
        avg_ref_energy = np.nanmean(ref_energy[ref_energy > 0]) if np.any(ref_energy > 0) else 10.0
        
        # Determine if heating or cooling dominant based on outdoor temps
        outdoor_temps = weather_data['outdoor_temp'].values[:hours]
        avg_outdoor = np.mean(outdoor_temps)
        is_cooling_mode = avg_outdoor > 22.0  # Cooling if outdoor > 22°C
        
        # Initial indoor temperature
        indoor_temp = 22.0  # Start at neutral
        stored_thermal_mass = 0.0  # Stored thermal energy from pre-conditioning
        
        for h in range(hours):
            outdoor_temp = weather_data['outdoor_temp'].iloc[h] if h < len(weather_data) else avg_outdoor
            setpoint = setpoints[h]
            
            # Building thermal dynamics with mass storage
            # Indoor temp naturally drifts toward outdoor temp
            temp_drift = (outdoor_temp - indoor_temp) / (self.thermal_mass * 2)
            
            # HVAC effort to reach setpoint
            temp_error = setpoint - indoor_temp
            hvac_effort = temp_error * 0.5
            
            # Pre-conditioning effect: if we cooled/heated below/above setpoint,
            # we have stored thermal mass that reduces future HVAC needs
            if is_cooling_mode:
                # Cooling mode: lower indoor temp = stored cooling capacity
                stored_thermal_mass = max(0, 23.0 - indoor_temp) * 0.3
            else:
                # Heating mode: higher indoor temp = stored heating capacity
                stored_thermal_mass = max(0, indoor_temp - 20.0) * 0.3
            
            # Update indoor temperature
            indoor_temp = indoor_temp + temp_drift + hvac_effort
            
            # Clamp to reasonable range around setpoint
            indoor_temp = np.clip(indoor_temp, setpoint - 3, setpoint + 3)
            indoor_temps[h] = indoor_temp
            
            # Energy calculation
            # Base load: non-HVAC consumption (approximately 30% of total)
            base_load = avg_ref_energy * 0.30
            
            # HVAC load depends on thermal lift (difference from outdoor)
            thermal_lift = abs(setpoint - outdoor_temp)
            
            # Normalize by typical lift
            typical_lift = abs(23.0 - outdoor_temp)
            
            if is_cooling_mode:
                # Cooling: energy increases with lower setpoints
                # Pre-cooling during off-peak saves during peak
                hvac_load = avg_ref_energy * 0.70 * (thermal_lift / max(typical_lift, 1.0))
                # Benefit from stored cooling (reduced current load)
                hvac_load = hvac_load * (1.0 - stored_thermal_mass * 0.1)
            else:
                # Heating: energy increases with higher setpoints
                hvac_load = avg_ref_energy * 0.70 * (thermal_lift / max(typical_lift, 1.0))
                hvac_load = hvac_load * (1.0 - stored_thermal_mass * 0.1)
            
            # Total energy with reasonable bounds
            energy[h] = np.clip(base_load + hvac_load, avg_ref_energy * 0.3, avg_ref_energy * 2.0)
        
        return energy, indoor_temps
    
    def predict_with_setpoint(
        self,
        features: pd.DataFrame,
        setpoint: float
    ) -> float:
        """Predict energy for given features and setpoint."""
        # Modify features with setpoint effect
        modified_features = features.copy()
        
        # Estimate energy adjustment based on setpoint
        outdoor_temp = features.get('outdoor_temp', 25)
        base_energy = self.model.predict(modified_features)
        
        # Energy scales with setpoint deviation from outdoor
        setpoint_effect = abs(setpoint - outdoor_temp) * 0.05
        adjusted_energy = base_energy * (1 + setpoint_effect)
        
        return adjusted_energy


class GAOptimizer:
    """
    Genetic Algorithm optimizer for HVAC setpoint scheduling.
    
    Optimizes 24-hour setpoint profiles to minimize cost while
    maintaining thermal comfort.
    """
    
    def __init__(
        self,
        surrogate: BuildingSurrogateWrapper,
        config: OptimizationConfig = None,
        tariff: ElectricityTariff = None
    ):
        self.surrogate = surrogate
        self.config = config or OptimizationConfig()
        self.tariff = tariff or ElectricityTariff()
        self.comfort_calc = ThermalComfortCalculator(self.config)
        
        # Setup DEAP
        self._setup_deap()
        
    def _setup_deap(self):
        """Configure DEAP genetic algorithm."""
        # Clear any existing creator classes
        if hasattr(creator, 'FitnessMin'):
            del creator.FitnessMin
        if hasattr(creator, 'Individual'):
            del creator.Individual
            
        # Create fitness and individual types
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        # Attribute generator (setpoint values)
        self.toolbox.register(
            "attr_setpoint",
            random.uniform,
            self.config.min_setpoint,
            self.config.max_setpoint
        )
        
        # Individual (24-hour setpoint profile)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_setpoint,
            n=self.config.horizon_hours
        )
        
        # Population
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )
        
        # Genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=0,
            sigma=1.0,
            indpb=0.2
        )
        self.toolbox.register(
            "select",
            tools.selTournament,
            tournsize=self.config.tournament_size
        )
    
    def _evaluate(
        self,
        individual: List[float],
        weather_data: pd.DataFrame,
        is_weekday: bool = True
    ) -> Tuple[float,]:
        """
        Evaluate fitness of a setpoint schedule.
        
        Args:
            individual: 24-hour setpoint profile
            weather_data: Weather conditions
            is_weekday: Whether it's a weekday (for tariff)
            
        Returns:
            Tuple with single objective value
        """
        setpoints = np.array(individual)
        
        # Clamp setpoints to valid range
        setpoints = np.clip(
            setpoints,
            self.config.min_setpoint,
            self.config.max_setpoint
        )
        
        # Simulate building response
        energy, indoor_temps = self.surrogate.simulate_day(setpoints, weather_data)
        
        # Calculate energy cost
        total_cost = 0.0
        for h in range(len(energy)):
            rate = self.tariff.get_rate(h, is_weekday)
            total_cost += energy[h] * rate
        
        # Calculate comfort penalty
        total_comfort_penalty = 0.0
        for h in range(len(indoor_temps)):
            outdoor_temp = weather_data['outdoor_temp'].iloc[h]
            rh = weather_data.get('relative_humidity', pd.Series([50]*24)).iloc[h]
            
            pmv = self.comfort_calc.calculate_pmv(
                indoor_temps[h],
                outdoor_temp,
                rh
            )
            total_comfort_penalty += self.comfort_calc.comfort_penalty(pmv)
        
        # Combined objective
        objective = total_cost + self.config.comfort_weight * total_comfort_penalty
        
        return (objective,)
    
    def optimize(
        self,
        weather_data: pd.DataFrame,
        is_weekday: bool = True,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run GA optimization for a single day.
        
        Args:
            weather_data: 24-hour weather forecast
            is_weekday: Whether it's a weekday
            verbose: Print progress
            
        Returns:
            Optimized setpoint profile and statistics
        """
        import time
        start_time = time.time()
        
        # Register evaluation function with current context
        self.toolbox.register(
            "evaluate",
            self._evaluate,
            weather_data=weather_data,
            is_weekday=is_weekday
        )
        
        # Create initial population
        pop = self.toolbox.population(n=self.config.population_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        
        # Run evolution
        pop, logbook = algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=self.config.crossover_prob,
            mutpb=self.config.mutation_prob,
            ngen=self.config.generations,
            stats=stats,
            verbose=verbose
        )
        
        # Get best individual
        best = tools.selBest(pop, k=1)[0]
        best_setpoints = np.clip(
            np.array(best),
            self.config.min_setpoint,
            self.config.max_setpoint
        )
        
        computation_time = time.time() - start_time
        
        return best_setpoints, {
            'logbook': logbook,
            'computation_time': computation_time,
            'best_fitness': best.fitness.values[0]
        }
    
    def run_baseline(
        self,
        weather_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run baseline fixed setpoint controller.
        
        Returns:
            setpoints, energy, indoor_temps
        """
        baseline_setpoints = np.full(
            self.config.horizon_hours,
            self.config.baseline_setpoint
        )
        energy, indoor_temps = self.surrogate.simulate_day(
            baseline_setpoints,
            weather_data
        )
        return baseline_setpoints, energy, indoor_temps


class MultiObjectiveGAOptimizer(GAOptimizer):
    """
    Multi-objective GA for Pareto analysis of cost vs comfort.
    """
    
    def _setup_deap(self):
        """Configure DEAP for multi-objective optimization."""
        if hasattr(creator, 'FitnessMulti'):
            del creator.FitnessMulti
        if hasattr(creator, 'IndividualMulti'):
            del creator.IndividualMulti
            
        # Multi-objective: minimize cost, minimize discomfort
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        
        self.toolbox.register(
            "attr_setpoint",
            random.uniform,
            self.config.min_setpoint,
            self.config.max_setpoint
        )
        
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.IndividualMulti,
            self.toolbox.attr_setpoint,
            n=self.config.horizon_hours
        )
        
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )
        
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=0,
            sigma=1.0,
            indpb=0.2
        )
        self.toolbox.register("select", tools.selNSGA2)
    
    def _evaluate_multi(
        self,
        individual: List[float],
        weather_data: pd.DataFrame,
        is_weekday: bool = True
    ) -> Tuple[float, float]:
        """Multi-objective evaluation: (cost, discomfort)."""
        setpoints = np.clip(
            np.array(individual),
            self.config.min_setpoint,
            self.config.max_setpoint
        )
        
        energy, indoor_temps = self.surrogate.simulate_day(setpoints, weather_data)
        
        # Objective 1: Energy cost
        total_cost = sum(
            energy[h] * self.tariff.get_rate(h, is_weekday)
            for h in range(len(energy))
        )
        
        # Objective 2: Discomfort hours
        discomfort = 0.0
        for h in range(len(indoor_temps)):
            outdoor_temp = weather_data['outdoor_temp'].iloc[h]
            rh = weather_data.get('relative_humidity', pd.Series([50]*24)).iloc[h]
            pmv = self.comfort_calc.calculate_pmv(indoor_temps[h], outdoor_temp, rh)
            if not self.comfort_calc.is_comfortable(pmv):
                discomfort += 1.0
        
        return (total_cost, discomfort)
    
    def optimize_pareto(
        self,
        weather_data: pd.DataFrame,
        is_weekday: bool = True,
        verbose: bool = False
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Generate Pareto front of solutions.
        
        Returns:
            List of (setpoints, cost, discomfort) tuples
        """
        self.toolbox.register(
            "evaluate",
            self._evaluate_multi,
            weather_data=weather_data,
            is_weekday=is_weekday
        )
        
        pop = self.toolbox.population(n=self.config.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Evolution
        for gen in range(self.config.generations):
            offspring = algorithms.varAnd(
                pop,
                self.toolbox,
                cxpb=self.config.crossover_prob,
                mutpb=self.config.mutation_prob
            )
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            pop = self.toolbox.select(pop + offspring, k=self.config.population_size)
        
        # Extract Pareto front
        pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        
        results = []
        for ind in pareto_front:
            setpoints = np.clip(
                np.array(ind),
                self.config.min_setpoint,
                self.config.max_setpoint
            )
            results.append((setpoints, ind.fitness.values[0], ind.fitness.values[1]))
        
        # Sort by cost
        results.sort(key=lambda x: x[1])
        
        return results


def run_day_optimization(
    surrogate: BuildingSurrogateWrapper,
    weather_data: pd.DataFrame,
    config: OptimizationConfig = None,
    tariff: ElectricityTariff = None,
    multi_objective: bool = False,
    verbose: bool = True
) -> OptimizationResult:
    """
    Run complete day-ahead optimization.
    
    Args:
        surrogate: Building surrogate model wrapper
        weather_data: 24-hour weather forecast
        config: Optimization configuration
        tariff: Electricity tariff
        multi_objective: Whether to compute Pareto front
        verbose: Print progress
        
    Returns:
        OptimizationResult with all metrics
    """
    import time
    start_time = time.time()
    
    config = config or OptimizationConfig()
    tariff = tariff or ElectricityTariff()
    is_weekday = True  # Assume weekday for this example
    
    # Single-objective optimization
    optimizer = GAOptimizer(surrogate, config, tariff)
    optimized_setpoints, opt_stats = optimizer.optimize(
        weather_data, is_weekday, verbose=verbose
    )
    
    # Run baseline
    baseline_setpoints, baseline_energy, baseline_temps = optimizer.run_baseline(
        weather_data
    )
    
    # Simulate optimized
    optimized_energy, optimized_temps = surrogate.simulate_day(
        optimized_setpoints, weather_data
    )
    
    # Calculate costs
    optimized_cost = sum(
        optimized_energy[h] * tariff.get_rate(h, is_weekday)
        for h in range(24)
    )
    baseline_cost = sum(
        baseline_energy[h] * tariff.get_rate(h, is_weekday)
        for h in range(24)
    )
    
    # Count comfort violations
    comfort_calc = ThermalComfortCalculator(config)
    
    def count_violations(temps, weather):
        violations = 0
        for h in range(len(temps)):
            outdoor_temp = weather['outdoor_temp'].iloc[h]
            rh = weather.get('relative_humidity', pd.Series([50]*24)).iloc[h]
            pmv = comfort_calc.calculate_pmv(temps[h], outdoor_temp, rh)
            if not comfort_calc.is_comfortable(pmv):
                violations += 1
        return violations
    
    opt_violations = count_violations(optimized_temps, weather_data)
    base_violations = count_violations(baseline_temps, weather_data)
    
    # Pareto front (optional)
    pareto_front = None
    if multi_objective:
        mo_optimizer = MultiObjectiveGAOptimizer(surrogate, config, tariff)
        pareto_front = mo_optimizer.optimize_pareto(
            weather_data, is_weekday, verbose=False
        )
    
    computation_time = time.time() - start_time
    
    return OptimizationResult(
        optimized_setpoints=optimized_setpoints,
        baseline_setpoints=baseline_setpoints,
        optimized_energy=optimized_energy,
        baseline_energy=baseline_energy,
        optimized_cost=optimized_cost,
        baseline_cost=baseline_cost,
        optimized_comfort_violations=opt_violations,
        baseline_comfort_violations=base_violations,
        optimized_indoor_temps=optimized_temps,
        baseline_indoor_temps=baseline_temps,
        pareto_front=pareto_front,
        computation_time=computation_time
    )


def create_optimization_table() -> pd.DataFrame:
    """
    Create Table 3: Objective Function and Optimization Constraints.
    """
    data = [
        {
            'Parameter': 'Objective Function',
            'Description': 'J = C_energy + w · D_comfort',
            'Value / Constraint': 'Trade-off between energy cost and discomfort'
        },
        {
            'Parameter': 'Decision Variable',
            'Description': 'HVAC Setpoint T_set',
            'Value / Constraint': '19°C ≤ T_set ≤ 26°C'
        },
        {
            'Parameter': 'Comfort Metric',
            'Description': 'PMV-based comfort band',
            'Value / Constraint': '-0.5 ≤ PMV ≤ +0.5'
        },
        {
            'Parameter': 'Algorithm',
            'Description': 'Genetic Algorithm',
            'Value / Constraint': 'Population = 50, Generations = 100'
        },
        {
            'Parameter': 'Time Horizon',
            'Description': 'Prediction Window',
            'Value / Constraint': '24 hours (day-ahead optimization)'
        }
    ]
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Optimization Module Test")
    print("\nTable 3: Objective Function and Optimization Constraints")
    print(create_optimization_table().to_string(index=False))
