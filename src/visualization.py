"""
Publication-Quality Visualizations

This module generates figures for the research paper:
1. Framework diagram
2. Daily optimization profiles
3. Pareto front (cost vs comfort)
4. Cross-building performance comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

sns.set_style("whitegrid")
sns.set_palette("colorblind")


class PublicationVisualizer:
    """Generate publication-quality figures."""
    
    def __init__(self):
        self.results_path = Path("/workspace/results")
        self.figures_path = Path("/workspace/figures")
        self.figures_path.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.load_results()
    
    def load_results(self):
        """Load optimization and model training results."""
        
        print("Loading results...")
        
        # Load optimization results
        opt_file = self.results_path / "optimization_results.json"
        if opt_file.exists():
            with open(opt_file, 'r') as f:
                self.optimization_results = json.load(f)
            print(f"  Loaded optimization results for {len(self.optimization_results)} buildings")
        else:
            self.optimization_results = {}
            print("  No optimization results found")
        
        # Load model training results  
        model_file = self.results_path / "model_training_results.json"
        if model_file.exists():
            with open(model_file, 'r') as f:
                self.model_results = json.load(f)
            print(f"  Loaded model training results")
        else:
            self.model_results = {}
            print("  No model training results found")
    
    def plot_framework_diagram(self):
        """
        Figure 1: Surrogate-Assisted Optimization Framework Schematic
        """
        print("\nGenerating Figure 1: Framework Diagram...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.axis('off')
        
        # Define boxes
        boxes = {
            'bdg2': {'xy': (0.1, 0.7), 'width': 0.25, 'height': 0.2, 
                     'text': 'BDG2 Dataset\n(1,636 buildings)\n• Hourly meter data\n• Weather data\n• Building metadata',
                     'color': '#E8F4F8'},
            'preprocessing': {'xy': (0.1, 0.4), 'width': 0.25, 'height': 0.15,
                             'text': 'Data Pre-processing\n• Filter residential\n• Clean & normalize\n• Train/val/test split',
                             'color': '#D4E6F1'},
            'surrogate': {'xy': (0.45, 0.5), 'width': 0.25, 'height': 0.3,
                         'text': 'Surrogate Models\n• LSTM (temporal)\n• XGBoost (ensemble)\n\nInputs: Weather, Time\nOutput: Energy prediction',
                         'color': '#FADBD8'},
            'ga': {'xy': (0.8, 0.5), 'width': 0.15, 'height': 0.3,
                  'text': 'GA Optimizer\n\nObjective:\nCost + w·Comfort\n\nConstraints:\n19°C ≤ T ≤ 26°C',
                  'color': '#D5F4E6'},
            'output': {'xy': (0.8, 0.15), 'width': 0.15, 'height': 0.15,
                      'text': 'Optimized\nSetpoint Schedule\n(24 hours)',
                      'color': '#FCF3CF'}
        }
        
        # Draw boxes
        for key, box in boxes.items():
            rect = plt.Rectangle(box['xy'], box['width'], box['height'],
                               facecolor=box['color'], edgecolor='black',
                               linewidth=1.5, zorder=1)
            ax.add_patch(rect)
            
            # Add text
            text_x = box['xy'][0] + box['width'] / 2
            text_y = box['xy'][1] + box['height'] / 2
            ax.text(text_x, text_y, box['text'],
                   ha='center', va='center', fontsize=9, zorder=2)
        
        # Draw arrows
        arrows = [
            ((0.225, 0.7), (0.225, 0.55)),  # BDG2 -> preprocessing
            ((0.35, 0.475), (0.45, 0.65)),  # preprocessing -> surrogate
            ((0.7, 0.65), (0.8, 0.65)),  # surrogate -> GA
            ((0.875, 0.5), (0.875, 0.3)),  # GA -> output
        ]
        
        for arrow in arrows:
            ax.annotate('', xy=arrow[1], xytext=arrow[0],
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Add labels
        ax.text(0.5, 0.95, 'Surrogate-Assisted HVAC Optimization Framework',
               ha='center', va='top', fontsize=14, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Save
        output_file = self.figures_path / "fig1_framework_diagram.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {output_file}")
    
    def plot_daily_profiles(self, building_id: str = None):
        """
        Figure 2: Daily Optimization Profiles
        Shows baseline vs optimized setpoints, energy, and cost over 24 hours
        """
        print("\nGenerating Figure 2: Daily Optimization Profiles...")
        
        # Use first available building if not specified
        if building_id is None:
            building_id = list(self.optimization_results.keys())[0]
        
        if building_id not in self.optimization_results:
            print(f"  No results for {building_id}")
            return
        
        results = self.optimization_results[building_id]
        baseline = results['baseline']['hourly_details']
        optimized = results['optimized']['hourly_details']
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        
        hours = [h['hour'] for h in baseline]
        
        # Subplot 1: Outdoor temperature
        ax = axes[0]
        temps = [h['T_outdoor'] for h in baseline]
        ax.plot(hours, temps, 'k-', linewidth=2, label='Outdoor Temperature')
        ax.set_ylabel('Temperature (°C)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Day-Ahead Optimization Results: {building_id}', fontweight='bold')
        
        # Subplot 2: Setpoint schedules
        ax = axes[1]
        baseline_setpoints = [h['setpoint'] for h in baseline]
        optimized_setpoints = [h['setpoint'] for h in optimized]
        
        ax.step(hours, baseline_setpoints, 'b-', linewidth=2, where='post',
               label='Baseline (constant)', alpha=0.7)
        ax.step(hours, optimized_setpoints, 'r-', linewidth=2, where='post',
               label='GA-optimized', alpha=0.7)
        ax.set_ylabel('Setpoint (°C)')
        ax.set_ylim(18, 27)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Shade peak/off-peak pricing periods
        for h in range(0, 24):
            if 8 <= h < 20:  # Peak hours
                ax.axvspan(h, h+1, alpha=0.1, color='orange')
        ax.text(14, 26.5, 'Peak pricing', fontsize=8, ha='center', color='orange')
        
        # Subplot 3: Energy consumption
        ax = axes[2]
        baseline_energy = [h['energy'] for h in baseline]
        optimized_energy = [h['energy'] for h in optimized]
        
        ax.bar(hours, baseline_energy, width=0.4, align='edge', 
              label='Baseline', alpha=0.7, color='blue')
        ax.bar([h+0.4 for h in hours], optimized_energy, width=0.4, align='edge',
              label='Optimized', alpha=0.7, color='red')
        ax.set_ylabel('Energy (kWh)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Subplot 4: Cumulative cost
        ax = axes[3]
        baseline_cost_cumul = np.cumsum([h['cost'] for h in baseline])
        optimized_cost_cumul = np.cumsum([h['cost'] for h in optimized])
        
        ax.plot(hours, baseline_cost_cumul, 'b-', linewidth=2, label='Baseline', marker='o')
        ax.plot(hours, optimized_cost_cumul, 'r-', linewidth=2, label='Optimized', marker='s')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Cumulative Cost ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add savings annotation
        final_savings = baseline_cost_cumul[-1] - optimized_cost_cumul[-1]
        savings_pct = results['savings']['cost_pct']
        ax.text(12, max(baseline_cost_cumul) * 0.5,
               f'Total savings: ${final_savings:.2f} ({savings_pct:.1f}%)',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save
        output_file = self.figures_path / f"fig2_daily_profiles_{building_id}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {output_file}")
    
    def plot_pareto_front(self):
        """
        Figure 3: Pareto Front (Cost vs Comfort)
        Note: Currently shows single-objective results; would need multi-run for true Pareto
        """
        print("\nGenerating Figure 3: Pareto Front Concept...")
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # For demonstration, create a conceptual Pareto front
        # In a full implementation, this would come from running optimization with different weights
        
        # Collect single points from our results
        points_x = []  # comfort penalty
        points_y = []  # cost
        labels = []
        
        for building_id, results in self.optimization_results.items():
            if 'optimized' in results:
                points_x.append(results['optimized']['comfort_penalty'])
                points_y.append(results['optimized']['total_cost'])
                labels.append(building_id.split('_')[0] + '_' + building_id.split('_')[1])
        
        # Create conceptual Pareto curve
        # Simulate range from cost-optimal to comfort-optimal
        x_pareto = np.linspace(0, max(points_x) * 1.5, 50)
        # Inverse relationship: lower comfort penalty means higher cost
        y_pareto = min(points_y) + (max(points_y) - min(points_y)) * np.exp(-x_pareto * 0.5)
        
        # Plot Pareto front
        ax.plot(x_pareto, y_pareto, 'b--', linewidth=2, alpha=0.5, label='Pareto Front (conceptual)')
        
        # Plot actual optimization results
        ax.scatter(points_x, points_y, s=100, c='red', marker='o', 
                  label='GA-optimized solutions', zorder=5, edgecolors='black', linewidth=1.5)
        
        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, (points_x[i], points_y[i]),
                       xytext=(10, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Comfort Penalty (hours × deviation)', fontsize=11)
        ax.set_ylabel('Daily Energy Cost ($)', fontsize=11)
        ax.set_title('Trade-off Between Energy Cost and Thermal Comfort', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        ax.annotate('Cost-optimal\n(comfort relaxed)', xy=(x_pareto[-1], y_pareto[-1]),
                   xytext=(50, 20), textcoords='offset points',
                   fontsize=9, arrowprops=dict(arrowstyle='->', color='blue'))
        ax.annotate('Comfort-optimal\n(higher cost)', xy=(x_pareto[0], y_pareto[0]),
                   xytext=(-80, 20), textcoords='offset points',
                   fontsize=9, arrowprops=dict(arrowstyle='->', color='blue'))
        
        plt.tight_layout()
        
        # Save
        output_file = self.figures_path / "fig3_pareto_front.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {output_file}")
    
    def plot_cross_building_comparison(self):
        """
        Figure 4: Cross-Building Performance Comparison
        """
        print("\nGenerating Figure 4: Cross-Building Comparison...")
        
        # Extract comparison data
        buildings = []
        energy_savings = []
        cost_savings = []
        
        for building_id, results in self.optimization_results.items():
            if 'savings' in results:
                buildings.append(building_id.replace('_', '\n'))
                energy_savings.append(results['savings']['energy_pct'])
                cost_savings.append(results['savings']['cost_pct'])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        x = np.arange(len(buildings))
        width = 0.7
        
        # Energy savings
        ax = axes[0]
        bars = ax.bar(x, energy_savings, width, color='steelblue', alpha=0.8)
        ax.set_ylabel('Energy Savings (%)', fontsize=11)
        ax.set_title('Energy Consumption Reduction', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(buildings, rotation=45, ha='right', fontsize=8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Cost savings
        ax = axes[1]
        bars = ax.bar(x, cost_savings, width, color='coral', alpha=0.8)
        ax.set_ylabel('Cost Savings (%)', fontsize=11)
        ax.set_title('Energy Cost Reduction', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(buildings, rotation=45, ha='right', fontsize=8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('GA-Optimized HVAC Control: Cross-Building Performance',
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        output_file = self.figures_path / "fig4_cross_building_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {output_file}")
    
    def plot_model_performance(self):
        """
        Figure 5: Surrogate Model Performance Comparison
        """
        print("\nGenerating Figure 5: Model Performance...")
        
        if not self.model_results:
            print("  No model results available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract model performance data
        buildings = []
        lstm_r2 = []
        xgb_r2 = []
        lstm_cvrmse = []
        xgb_cvrmse = []
        
        for building_id in self.model_results.get('lstm', {}).keys():
            buildings.append(building_id.replace('_', '\n'))
            
            # LSTM
            if 'val_metrics' in self.model_results['lstm'].get(building_id, {}):
                lstm_r2.append(self.model_results['lstm'][building_id]['val_metrics']['r2'])
                lstm_cvrmse.append(self.model_results['lstm'][building_id]['val_metrics']['cvrmse'])
            else:
                lstm_r2.append(0)
                lstm_cvrmse.append(100)
            
            # XGBoost
            if 'val_metrics' in self.model_results['xgboost'].get(building_id, {}):
                xgb_r2.append(self.model_results['xgboost'][building_id]['val_metrics']['r2'])
                xgb_cvrmse.append(self.model_results['xgboost'][building_id]['val_metrics']['cvrmse'])
            else:
                xgb_r2.append(0)
                xgb_cvrmse.append(100)
        
        x = np.arange(len(buildings))
        width = 0.35
        
        # R² comparison
        ax = axes[0]
        ax.bar(x - width/2, lstm_r2, width, label='LSTM', color='skyblue', alpha=0.8)
        ax.bar(x + width/2, xgb_r2, width, label='XGBoost', color='lightcoral', alpha=0.8)
        ax.set_ylabel('R² Score (Validation)', fontsize=11)
        ax.set_title('Model Prediction Accuracy', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(buildings, rotation=45, ha='right', fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # CV(RMSE) comparison
        ax = axes[1]
        ax.bar(x - width/2, lstm_cvrmse, width, label='LSTM', color='skyblue', alpha=0.8)
        ax.bar(x + width/2, xgb_cvrmse, width, label='XGBoost', color='lightcoral', alpha=0.8)
        ax.set_ylabel('CV(RMSE) % (Validation)', fontsize=11)
        ax.set_title('Model Prediction Error', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(buildings, rotation=45, ha='right', fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Surrogate Model Performance Comparison',
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        output_file = self.figures_path / "fig5_model_performance.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {output_file}")


def main():
    """Generate all publication figures."""
    
    print("="*80)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("="*80)
    
    visualizer = PublicationVisualizer()
    
    # Generate all figures
    visualizer.plot_framework_diagram()
    
    # Generate daily profiles for multiple buildings
    for building_id in list(visualizer.optimization_results.keys())[:3]:
        visualizer.plot_daily_profiles(building_id)
    
    visualizer.plot_pareto_front()
    visualizer.plot_cross_building_comparison()
    visualizer.plot_model_performance()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"All figures saved to /workspace/figures/")
    print("\nGenerated figures:")
    figures_path = Path("/workspace/figures")
    for fig in sorted(figures_path.glob("*.png")):
        print(f"  - {fig.name}")


if __name__ == "__main__":
    main()
