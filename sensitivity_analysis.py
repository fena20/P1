"""
Sensitivity Analysis: Vary parameters and analyze impacts
Generates Figure 4: Sensitivity heatmap
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from config import *
from optimization_simulation import rule_based_baseline, simple_mpc, rl_optimized

def sensitivity_analysis(data: pd.DataFrame, dl_predictions: Dict, building_id: int):
    """
    Perform sensitivity analysis by varying key parameters
    """
    print("Performing sensitivity analysis...")
    
    building_data = data[data['building_id'] == building_id].copy()
    if len(building_data) < 100:
        return None
    
    if len(building_data) > 200:
        building_data = building_data.head(200)
    
    # Parameter ranges
    occupant_density_factors = [0.5, 0.75, 1.0, 1.25, 1.5]  # Multiplier for occupancy
    weather_uncertainty = [0.0, 0.5, 1.0, 1.5, 2.0]  # Temperature noise (degrees C)
    comfort_thresholds = [5.0, 7.5, 10.0, 12.5, 15.0]  # PPD thresholds
    
    results = []
    
    # Baseline (reference)
    baseline_result = rule_based_baseline(building_data, building_id)
    baseline_energy = baseline_result['total_energy']
    
    # Vary occupant density
    print("Varying occupant density...")
    for density_factor in occupant_density_factors:
        modified_data = building_data.copy()
        modified_data['meter_reading'] *= density_factor  # Scale energy with occupancy
        
        result = rl_optimized(modified_data, dl_predictions, building_id)
        energy_savings = ((baseline_energy * density_factor - result['total_energy']) / 
                         (baseline_energy * density_factor)) * 100
        
        results.append({
            'parameter': 'Occupant Density',
            'value': density_factor,
            'energy_savings_pct': energy_savings,
            'avg_ppd': result['avg_ppd']
        })
    
    # Vary weather uncertainty
    print("Varying weather uncertainty...")
    for uncertainty in weather_uncertainty:
        modified_data = building_data.copy()
        modified_data['air_temperature'] += np.random.normal(0, uncertainty, len(modified_data))
        
        result = rl_optimized(modified_data, dl_predictions, building_id)
        energy_savings = ((baseline_energy - result['total_energy']) / baseline_energy) * 100
        
        results.append({
            'parameter': 'Weather Uncertainty',
            'value': uncertainty,
            'energy_savings_pct': energy_savings,
            'avg_ppd': result['avg_ppd']
        })
    
    # Vary comfort threshold
    print("Varying comfort threshold...")
    global COMFORT_PENALTY_THRESHOLD
    original_threshold = COMFORT_PENALTY_THRESHOLD
    
    for threshold in comfort_thresholds:
        COMFORT_PENALTY_THRESHOLD = threshold
        
        result = rl_optimized(building_data, dl_predictions, building_id)
        energy_savings = ((baseline_energy - result['total_energy']) / baseline_energy) * 100
        
        results.append({
            'parameter': 'Comfort Threshold',
            'value': threshold,
            'energy_savings_pct': energy_savings,
            'avg_ppd': result['avg_ppd']
        })
    
    COMFORT_PENALTY_THRESHOLD = original_threshold
    
    results_df = pd.DataFrame(results)
    return results_df

def generate_sensitivity_heatmap(results_df: pd.DataFrame):
    """Generate Figure 4: Sensitivity analysis heatmap"""
    plt.style.use(FIG_STYLE)
    
    # Pivot data for heatmap
    pivot_energy = results_df.pivot_table(
        values='energy_savings_pct',
        index='parameter',
        columns='value',
        aggfunc='mean'
    )
    
    pivot_ppd = results_df.pivot_table(
        values='avg_ppd',
        index='parameter',
        columns='value',
        aggfunc='mean'
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Energy savings heatmap
    sns.heatmap(pivot_energy, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Energy Savings (%)'}, ax=axes[0])
    axes[0].set_title('Energy Savings Sensitivity Analysis', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Parameter Value', fontsize=12)
    axes[0].set_ylabel('Parameter', fontsize=12)
    
    # Comfort (PPD) heatmap
    sns.heatmap(pivot_ppd, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Average PPD (%)'}, ax=axes[1])
    axes[1].set_title('Comfort Sensitivity Analysis', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Parameter Value', fontsize=12)
    axes[1].set_ylabel('Parameter', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig4.png", dpi=DPI, bbox_inches='tight')
    print(f"Figure 4 saved to {FIGURES_DIR}/fig4.png")
    plt.close()

def generate_pareto_front(data: pd.DataFrame, dl_predictions: Dict, building_id: int):
    """
    Generate Pareto front for multi-objective optimization (energy vs comfort)
    Figure 5: Pareto front
    """
    print("Generating Pareto front...")
    
    building_data = data[data['building_id'] == building_id].copy()
    if len(building_data) > 200:
        building_data = building_data.head(200)
    
    # Vary comfort threshold to explore Pareto front
    comfort_thresholds = np.linspace(5, 20, 20)
    energy_values = []
    comfort_values = []
    
    global COMFORT_PENALTY_THRESHOLD
    original_threshold = COMFORT_PENALTY_THRESHOLD
    
    for threshold in comfort_thresholds:
        COMFORT_PENALTY_THRESHOLD = threshold
        result = rl_optimized(building_data, dl_predictions, building_id)
        
        energy_values.append(result['total_energy'])
        comfort_values.append(result['avg_ppd'])
    
    COMFORT_PENALTY_THRESHOLD = original_threshold
    
    # Baseline for reference
    baseline_result = rule_based_baseline(building_data, building_id)
    baseline_energy = baseline_result['total_energy']
    baseline_ppd = baseline_result['avg_ppd']
    
    # Plot Pareto front
    plt.style.use(FIG_STYLE)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Sort by energy for proper Pareto front visualization
    sorted_indices = np.argsort(energy_values)
    sorted_energy = np.array(energy_values)[sorted_indices]
    sorted_comfort = np.array(comfort_values)[sorted_indices]
    
    # Find Pareto-optimal points
    pareto_mask = np.ones(len(sorted_energy), dtype=bool)
    for i in range(len(sorted_energy)):
        for j in range(len(sorted_energy)):
            if (sorted_energy[j] < sorted_energy[i] and sorted_comfort[j] <= sorted_comfort[i]) or \
               (sorted_energy[j] <= sorted_energy[i] and sorted_comfort[j] < sorted_comfort[i]):
                pareto_mask[i] = False
                break
    
    pareto_energy = sorted_energy[pareto_mask]
    pareto_comfort = sorted_comfort[pareto_mask]
    
    # Plot all points
    ax.scatter(energy_values, comfort_values, alpha=0.5, s=50, label='All Solutions', color='gray')
    
    # Plot Pareto front
    pareto_indices = np.argsort(pareto_energy)
    ax.plot(pareto_energy[pareto_indices], pareto_comfort[pareto_indices], 
           'r-', linewidth=2, label='Pareto Front', marker='o', markersize=8)
    
    # Plot baseline
    ax.scatter([baseline_energy], [baseline_ppd], s=200, marker='*', 
              color='gold', label='Baseline', zorder=5, edgecolors='black', linewidths=2)
    
    # Plot proposed method (using default threshold)
    proposed_result = rl_optimized(building_data, dl_predictions, building_id)
    ax.scatter([proposed_result['total_energy']], [proposed_result['avg_ppd']], 
              s=200, marker='D', color='green', label='Proposed Method', 
              zorder=5, edgecolors='black', linewidths=2)
    
    ax.set_xlabel('Energy Consumption (kWh)', fontsize=12)
    ax.set_ylabel('Average PPD (%)', fontsize=12)
    ax.set_title('Multi-Objective Optimization: Energy vs Comfort', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add comfort threshold line
    ax.axhline(y=COMFORT_PENALTY_THRESHOLD, color='orange', linestyle='--', 
              alpha=0.7, label=f'Comfort Threshold ({COMFORT_PENALTY_THRESHOLD}%)')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig5.png", dpi=DPI, bbox_inches='tight')
    print(f"Figure 5 saved to {FIGURES_DIR}/fig5.png")
    plt.close()

if __name__ == "__main__":
    # Load data
    test_data = pd.read_csv(f"{DATA_DIR}/test_processed.csv", parse_dates=['timestamp'])
    
    # Load DL predictions
    try:
        dl_preds = np.load(f"{OUTPUT_DIR}/dl_predictions.npy", allow_pickle=True).item()
    except:
        print("DL predictions not found. Using baseline predictions.")
        dl_preds = {
            'energy_preds': test_data['meter_reading'].values,
            'comfort_preds': test_data['ppd'].values
        }
    
    # Select a building for sensitivity analysis
    building_ids = test_data['building_id'].unique()
    if len(building_ids) > 0:
        sample_building = building_ids[0]
        
        # Sensitivity analysis
        sensitivity_results = sensitivity_analysis(test_data, dl_preds, sample_building)
        if sensitivity_results is not None:
            generate_sensitivity_heatmap(sensitivity_results)
            sensitivity_results.to_csv(f"{OUTPUT_DIR}/sensitivity_analysis.csv", index=False)
        
        # Pareto front
        generate_pareto_front(test_data, dl_preds, sample_building)
