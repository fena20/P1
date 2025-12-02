"""
Visualization Module for Publication-Quality Figures.

Generates figures for Applied Energy publication:
- Figure 1: Framework schematic
- Figure 2: Daily optimization profiles
- Figure 3: Pareto front analysis
- Figure 4: Cross-building performance comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from config import VisualizationConfig, FIGURES_DIR
from optimization import OptimizationResult


# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5
})


class FigureGenerator:
    """Generator for publication-quality figures."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.colors = self.config.colors
        
    def figure1_framework_schematic(
        self,
        save_path: Path = None
    ) -> plt.Figure:
        """
        Create Figure 1: Surrogate-Assisted Optimization Framework.
        
        Shows data flow from BDG2 through surrogate training to GA optimization.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis('off')
        
        # Color scheme
        data_color = '#E8F4F8'
        model_color = '#FFF3E0'
        opt_color = '#F3E5F5'
        arrow_color = '#455A64'
        
        # ===== Left Section: BDG2 Data =====
        # Main box
        data_box = FancyBboxPatch(
            (0.3, 0.5), 2.4, 4.0,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=data_color,
            edgecolor='#0277BD',
            linewidth=2
        )
        ax.add_patch(data_box)
        ax.text(1.5, 4.3, 'BDG2 Dataset', fontsize=12, fontweight='bold',
                ha='center', va='center', color='#01579B')
        
        # Sub-components
        components = [
            (1.5, 3.5, 'Electricity Meters\n(3,053 meters)', '#4FC3F7'),
            (1.5, 2.5, 'Weather Data\n(19 sites)', '#4DD0E1'),
            (1.5, 1.5, 'Building Metadata\n(1,636 buildings)', '#4DB6AC')
        ]
        
        for x, y, text, color in components:
            box = FancyBboxPatch(
                (0.5, y-0.35), 2.0, 0.7,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=color,
                edgecolor='white',
                linewidth=1
            )
            ax.add_patch(box)
            ax.text(x, y, text, fontsize=8, ha='center', va='center')
        
        # ===== Middle Section: Surrogate Model =====
        model_box = FancyBboxPatch(
            (3.5, 0.5), 2.8, 4.0,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=model_color,
            edgecolor='#E65100',
            linewidth=2
        )
        ax.add_patch(model_box)
        ax.text(4.9, 4.3, 'Surrogate Model', fontsize=12, fontweight='bold',
                ha='center', va='center', color='#E65100')
        
        # Model components
        model_components = [
            (4.9, 3.5, 'Data Preprocessing\n& Feature Engineering', '#FFCC80'),
            (4.9, 2.5, 'LSTM / XGBoost\nTraining', '#FFB74D'),
            (4.9, 1.5, 'Model Validation\n& Testing', '#FFA726')
        ]
        
        for x, y, text, color in model_components:
            box = FancyBboxPatch(
                (3.7, y-0.35), 2.4, 0.7,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=color,
                edgecolor='white',
                linewidth=1
            )
            ax.add_patch(box)
            ax.text(x, y, text, fontsize=8, ha='center', va='center')
        
        # ===== Right Section: Optimization =====
        opt_box = FancyBboxPatch(
            (7.0, 0.5), 2.6, 4.0,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=opt_color,
            edgecolor='#7B1FA2',
            linewidth=2
        )
        ax.add_patch(opt_box)
        ax.text(8.3, 4.3, 'GA Optimizer', fontsize=12, fontweight='bold',
                ha='center', va='center', color='#7B1FA2')
        
        # Optimization components
        opt_components = [
            (8.3, 3.5, 'Setpoint Schedule\nGeneration', '#CE93D8'),
            (8.3, 2.5, 'Fitness Evaluation\n(Cost + Comfort)', '#BA68C8'),
            (8.3, 1.5, 'Optimal Control\nSchedule', '#AB47BC')
        ]
        
        for x, y, text, color in opt_components:
            box = FancyBboxPatch(
                (7.2, y-0.35), 2.2, 0.7,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=color,
                edgecolor='white',
                linewidth=1
            )
            ax.add_patch(box)
            ax.text(x, y, text, fontsize=8, ha='center', va='center')
        
        # ===== Arrows =====
        arrow_style = dict(
            arrowstyle='->,head_width=0.3,head_length=0.2',
            color=arrow_color,
            lw=2
        )
        
        # Data to Model
        ax.annotate('', xy=(3.4, 2.5), xytext=(2.8, 2.5),
                    arrowprops=arrow_style)
        
        # Model to Optimization
        ax.annotate('', xy=(6.9, 2.5), xytext=(6.4, 2.5),
                    arrowprops=arrow_style)
        
        # Feedback loop
        ax.annotate('', xy=(6.4, 1.8), xytext=(6.9, 1.8),
                    arrowprops=dict(arrowstyle='<-,head_width=0.2,head_length=0.15',
                                    color=arrow_color, lw=1.5,
                                    connectionstyle='arc3,rad=-0.3'))
        ax.text(6.65, 1.2, 'Evaluation\nLoop', fontsize=7, ha='center',
                style='italic', color=arrow_color)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = FIGURES_DIR / f'figure1_framework.{self.config.figure_format}'
        fig.savefig(save_path, dpi=self.config.figure_dpi, 
                    facecolor='white', edgecolor='none')
        
        return fig
    
    def figure2_daily_optimization(
        self,
        result: OptimizationResult,
        weather_data: pd.DataFrame,
        save_path: Path = None
    ) -> plt.Figure:
        """
        Create Figure 2: Daily Optimization Profile.
        
        Three-panel figure showing:
        - Top: Outdoor temperature
        - Middle: Setpoint comparison (baseline vs optimized)
        - Bottom: Energy consumption comparison
        """
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        hours = np.arange(24)
        
        # ===== Panel 1: Outdoor Temperature =====
        ax1 = axes[0]
        ax1.plot(hours, weather_data['outdoor_temp'].values[:24],
                 color=self.colors['accent'], linewidth=2, marker='o',
                 markersize=4, label='Outdoor Temperature')
        ax1.fill_between(hours, weather_data['outdoor_temp'].values[:24],
                         alpha=0.2, color=self.colors['accent'])
        ax1.set_ylabel('Temperature (°C)')
        ax1.legend(loc='upper right')
        ax1.set_title('(a) Outdoor Temperature Profile', loc='left', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # ===== Panel 2: Setpoint Schedules =====
        ax2 = axes[1]
        ax2.step(hours, result.baseline_setpoints, where='mid',
                 color=self.colors['baseline'], linewidth=2,
                 linestyle='--', label='Baseline (Fixed 23°C)')
        ax2.step(hours, result.optimized_setpoints, where='mid',
                 color=self.colors['optimized'], linewidth=2,
                 label='Optimized Schedule')
        
        # Add price periods
        ax2_twin = ax2.twinx()
        prices = []
        from config import ElectricityTariff
        tariff = ElectricityTariff()
        for h in hours:
            prices.append(tariff.get_rate(h, True))
        ax2_twin.fill_between(hours, prices, alpha=0.1, color='green',
                              step='mid', label='Electricity Price')
        ax2_twin.set_ylabel('Price ($/kWh)', color='green')
        ax2_twin.tick_params(axis='y', labelcolor='green')
        ax2_twin.set_ylim(0, 0.35)
        
        ax2.set_ylabel('Setpoint (°C)')
        ax2.set_ylim(18, 27)
        ax2.legend(loc='upper left')
        ax2.set_title('(b) HVAC Setpoint Schedules', loc='left', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Highlight pre-cooling period
        peak_start, peak_end = 14, 20
        ax2.axvspan(peak_start, peak_end, alpha=0.15, color='red',
                    label='Peak Price Period')
        
        # ===== Panel 3: Energy Consumption =====
        ax3 = axes[2]
        width = 0.35
        x = hours
        
        bars1 = ax3.bar(x - width/2, result.baseline_energy,
                        width, label='Baseline', color=self.colors['baseline'],
                        alpha=0.8, edgecolor='white')
        bars2 = ax3.bar(x + width/2, result.optimized_energy,
                        width, label='Optimized', color=self.colors['optimized'],
                        alpha=0.8, edgecolor='white')
        
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Energy (kWh)')
        ax3.legend(loc='upper right')
        ax3.set_title('(c) Hourly Energy Consumption', loc='left', fontweight='bold')
        ax3.set_xticks(range(0, 24, 2))
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add summary statistics
        energy_savings = (sum(result.baseline_energy) - sum(result.optimized_energy)) / sum(result.baseline_energy) * 100
        cost_savings = (result.baseline_cost - result.optimized_cost) / result.baseline_cost * 100
        
        summary_text = (f"Energy Savings: {energy_savings:.1f}%\n"
                       f"Cost Savings: {cost_savings:.1f}%\n"
                       f"Comfort Violations: {result.optimized_comfort_violations} vs "
                       f"{result.baseline_comfort_violations} (baseline)")
        
        ax3.text(0.98, 0.95, summary_text,
                 transform=ax3.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = FIGURES_DIR / f'figure2_daily_optimization.{self.config.figure_format}'
        fig.savefig(save_path, dpi=self.config.figure_dpi,
                    facecolor='white', edgecolor='none')
        
        return fig
    
    def figure3_pareto_front(
        self,
        pareto_solutions: List[Tuple[np.ndarray, float, float]],
        baseline_cost: float,
        baseline_discomfort: float,
        save_path: Path = None
    ) -> plt.Figure:
        """
        Create Figure 3: Pareto Front - Cost vs Comfort Trade-off.
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Extract Pareto points
        costs = [sol[1] for sol in pareto_solutions]
        discomforts = [sol[2] for sol in pareto_solutions]
        
        # Plot Pareto front
        sorted_indices = np.argsort(costs)
        sorted_costs = np.array(costs)[sorted_indices]
        sorted_discomforts = np.array(discomforts)[sorted_indices]
        
        # Pareto frontier line
        ax.plot(sorted_discomforts, sorted_costs,
                color=self.colors['pareto'], linewidth=2,
                linestyle='-', label='Pareto Front', zorder=3)
        
        # Scatter points
        scatter = ax.scatter(discomforts, costs,
                            c=range(len(costs)), cmap='plasma',
                            s=60, alpha=0.7, edgecolor='white',
                            linewidth=0.5, zorder=4)
        
        # Baseline point
        ax.scatter([baseline_discomfort], [baseline_cost],
                   color=self.colors['baseline'], s=150, marker='X',
                   edgecolor='black', linewidth=1.5,
                   label='Baseline Controller', zorder=5)
        
        # Best cost point
        best_cost_idx = np.argmin(costs)
        ax.scatter([discomforts[best_cost_idx]], [costs[best_cost_idx]],
                   color='#4CAF50', s=120, marker='*',
                   edgecolor='black', linewidth=1,
                   label='Minimum Cost Solution', zorder=5)
        
        # Best comfort point
        best_comfort_idx = np.argmin(discomforts)
        ax.scatter([discomforts[best_comfort_idx]], [costs[best_comfort_idx]],
                   color='#2196F3', s=120, marker='s',
                   edgecolor='black', linewidth=1,
                   label='Maximum Comfort Solution', zorder=5)
        
        # Add utopia point reference
        ax.axhline(y=min(costs) * 0.95, color='gray', linestyle=':',
                   alpha=0.5, linewidth=1)
        ax.axvline(x=min(discomforts) * 0.95, color='gray', linestyle=':',
                   alpha=0.5, linewidth=1)
        
        # Labels and formatting
        ax.set_xlabel('Discomfort Hours', fontsize=11)
        ax.set_ylabel('Daily Energy Cost ($)', fontsize=11)
        ax.set_title('Pareto-Optimal Solutions: Cost vs Comfort Trade-off',
                     fontsize=12, fontweight='bold')
        
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for solution progression
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Solution Index', fontsize=9)
        
        # Annotations for improvement
        if baseline_cost > max(costs):
            improvement = (baseline_cost - min(costs)) / baseline_cost * 100
            ax.annotate(f'{improvement:.0f}% cost\nreduction possible',
                        xy=(baseline_discomfort * 0.7, (baseline_cost + min(costs)) / 2),
                        fontsize=9, ha='center',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = FIGURES_DIR / f'figure3_pareto_front.{self.config.figure_format}'
        fig.savefig(save_path, dpi=self.config.figure_dpi,
                    facecolor='white', edgecolor='none')
        
        return fig
    
    def figure4_cross_building_performance(
        self,
        results_df: pd.DataFrame,
        save_path: Path = None
    ) -> plt.Figure:
        """
        Create Figure 4: Cross-Building and Cross-Climate Performance.
        
        Args:
            results_df: DataFrame with columns:
                - building_id
                - climate_zone
                - prediction_rmse
                - energy_savings_pct
                - cost_savings_pct
                - comfort_improvement_pct
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Color palette for climate zones
        climate_colors = {
            'Hot-Humid': '#FF5722',
            'Mixed-Humid': '#FF9800',
            'Cold': '#2196F3',
            'Very-Cold': '#3F51B5'
        }
        
        # ===== Panel (a): Prediction RMSE by Building =====
        ax1 = axes[0, 0]
        x_pos = range(len(results_df))
        colors = [climate_colors.get(cz, '#9E9E9E') for cz in results_df['climate_zone']]
        bars1 = ax1.bar(x_pos, results_df['prediction_rmse'],
                        color=colors, edgecolor='white', alpha=0.8)
        ax1.set_xlabel('Building')
        ax1.set_ylabel('RMSE (kWh)')
        ax1.set_title('(a) Prediction Accuracy by Building', loc='left', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'B{i+1}' for i in x_pos], rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # ===== Panel (b): Energy Savings by Climate =====
        ax2 = axes[0, 1]
        climate_groups = results_df.groupby('climate_zone')['energy_savings_pct']
        climate_means = climate_groups.mean()
        climate_stds = climate_groups.std()
        
        x_climate = range(len(climate_means))
        climate_colors_list = [climate_colors.get(cz, '#9E9E9E') for cz in climate_means.index]
        
        bars2 = ax2.bar(x_climate, climate_means.values,
                        yerr=climate_stds.values, capsize=5,
                        color=climate_colors_list, edgecolor='white', alpha=0.8)
        ax2.set_xlabel('Climate Zone')
        ax2.set_ylabel('Energy Savings (%)')
        ax2.set_title('(b) Energy Savings by Climate Zone', loc='left', fontweight='bold')
        ax2.set_xticks(x_climate)
        ax2.set_xticklabels(climate_means.index, rotation=30, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=results_df['energy_savings_pct'].mean(), color='red',
                    linestyle='--', linewidth=1.5, label='Overall Mean')
        ax2.legend()
        
        # ===== Panel (c): Cost vs Comfort Improvement =====
        ax3 = axes[1, 0]
        for cz in results_df['climate_zone'].unique():
            mask = results_df['climate_zone'] == cz
            ax3.scatter(results_df.loc[mask, 'cost_savings_pct'],
                       results_df.loc[mask, 'comfort_improvement_pct'],
                       c=climate_colors.get(cz, '#9E9E9E'),
                       s=80, alpha=0.7, edgecolor='white',
                       label=cz)
        
        ax3.set_xlabel('Cost Savings (%)')
        ax3.set_ylabel('Comfort Improvement (%)')
        ax3.set_title('(c) Cost vs Comfort Trade-off', loc='left', fontweight='bold')
        ax3.legend(loc='lower right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Add quadrant labels
        ax3.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
        ax3.axvline(x=15, color='gray', linestyle=':', alpha=0.5)
        
        # ===== Panel (d): Box Plot Summary =====
        ax4 = axes[1, 1]
        metrics = ['energy_savings_pct', 'cost_savings_pct']
        metric_labels = ['Energy\nSavings', 'Cost\nSavings']
        
        box_data = [results_df[m].values for m in metrics]
        bp = ax4.boxplot(box_data, patch_artist=True, labels=metric_labels)
        
        colors_box = [self.colors['primary'], self.colors['secondary']]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Percentage (%)')
        ax4.set_title('(d) Overall Performance Summary', loc='left', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add mean values as text
        for i, (m, label) in enumerate(zip(metrics, metric_labels)):
            mean_val = results_df[m].mean()
            ax4.text(i + 1, mean_val + 2, f'μ={mean_val:.1f}%',
                     ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = FIGURES_DIR / f'figure4_cross_building.{self.config.figure_format}'
        fig.savefig(save_path, dpi=self.config.figure_dpi,
                    facecolor='white', edgecolor='none')
        
        return fig
    
    def create_results_table(
        self,
        results: List[Dict],
        save_path: Path = None
    ) -> pd.DataFrame:
        """
        Create Table 4: Comparative Results.
        """
        data = []
        for r in results:
            data.append({
                'Performance Metric': 'Total Energy (kWh)',
                'Baseline Controller': f"{r['baseline_energy']:.0f}",
                'Proposed AI Optimizer': f"{r['optimized_energy']:.0f}",
                'Improvement (%)': f"{r['energy_improvement']:.1f}% ↓"
            })
            data.append({
                'Performance Metric': 'Energy Cost ($)',
                'Baseline Controller': f"{r['baseline_cost']:.2f}",
                'Proposed AI Optimizer': f"{r['optimized_cost']:.2f}",
                'Improvement (%)': f"{r['cost_improvement']:.1f}% ↓"
            })
            data.append({
                'Performance Metric': 'Comfort Violation (hours)',
                'Baseline Controller': f"{r['baseline_violations']}",
                'Proposed AI Optimizer': f"{r['optimized_violations']}",
                'Improvement (%)': f"{r['comfort_improvement']:.1f}% ↓"
            })
            data.append({
                'Performance Metric': 'Computational Time (s)',
                'Baseline Controller': f"{r['physics_time']:.1f} (Physics Sim)",
                'Proposed AI Optimizer': f"{r['surrogate_time']:.1f} (Surrogate)",
                'Improvement (%)': f"{r['time_improvement']:.1f}% ↓"
            })
        
        df = pd.DataFrame(data)
        
        if save_path is None:
            save_path = FIGURES_DIR / 'table4_results.csv'
        df.to_csv(save_path, index=False)
        
        return df


def generate_all_figures(
    results: List[OptimizationResult],
    weather_data: pd.DataFrame,
    cross_building_df: pd.DataFrame,
    config: VisualizationConfig = None
) -> Dict[str, plt.Figure]:
    """
    Generate all publication figures.
    
    Returns:
        Dictionary mapping figure names to Figure objects.
    """
    generator = FigureGenerator(config)
    figures = {}
    
    print("Generating Figure 1: Framework Schematic...")
    figures['figure1'] = generator.figure1_framework_schematic()
    
    if results and len(results) > 0:
        print("Generating Figure 2: Daily Optimization Profile...")
        figures['figure2'] = generator.figure2_daily_optimization(
            results[0], weather_data
        )
        
        if results[0].pareto_front:
            print("Generating Figure 3: Pareto Front...")
            figures['figure3'] = generator.figure3_pareto_front(
                results[0].pareto_front,
                results[0].baseline_cost,
                results[0].baseline_comfort_violations
            )
    
    if cross_building_df is not None and len(cross_building_df) > 0:
        print("Generating Figure 4: Cross-Building Performance...")
        figures['figure4'] = generator.figure4_cross_building_performance(
            cross_building_df
        )
    
    print(f"\nAll figures saved to: {FIGURES_DIR}")
    return figures


if __name__ == "__main__":
    # Generate sample Figure 1 (no data needed)
    print("Testing visualization module...")
    generator = FigureGenerator()
    fig1 = generator.figure1_framework_schematic()
    print(f"Figure 1 saved to {FIGURES_DIR}")
    plt.close('all')
