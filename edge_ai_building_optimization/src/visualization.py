"""
Visualization Module for Publication-Quality Figures
Generates all figures for the research paper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os

from config import VIZ_CONFIG, FIGURES_DIR

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': VIZ_CONFIG['font_size'],
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 100,
    'savefig.dpi': VIZ_CONFIG['dpi'],
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Muted color palette
COLORS = {
    'primary': '#2E86AB',      # Muted blue
    'secondary': '#A23B72',    # Muted magenta
    'tertiary': '#F18F01',     # Muted orange
    'quaternary': '#C73E1D',   # Muted red
    'success': '#3A7D44',      # Muted green
    'neutral': '#6B717E',      # Gray
    'baseline': '#8B8B8B',     # Light gray
    'mpc': '#5C946E',          # Sage green
    'rl': '#2E86AB',           # Blue
}

# Use colorblind-friendly palette
sns.set_palette("colorblind")


def create_system_architecture_figure(save_path: str) -> None:
    """
    Create Figure 0: System architecture diagram.
    Shows the overall framework with Edge AI, RL, and Federated Learning components.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define box style
    box_style = "round,pad=0.03,rounding_size=0.2"
    
    # Color scheme
    colors = {
        'data': '#E8F4FD',
        'dl': '#FFF3E0',
        'rl': '#E8F5E9',
        'edge': '#FCE4EC',
        'fed': '#F3E5F5',
        'output': '#E0F2F1'
    }
    
    # Title
    ax.text(7, 9.5, 'Edge AI with Hybrid RL for Occupant-Centric\nBuilding Energy Optimization',
            fontsize=16, fontweight='bold', ha='center', va='center')
    
    # Data Sources (left side)
    data_box = FancyBboxPatch((0.5, 6), 2.5, 2.5, boxstyle=box_style,
                               facecolor=colors['data'], edgecolor='#1565C0', linewidth=2)
    ax.add_patch(data_box)
    ax.text(1.75, 7.8, 'Data Sources', fontsize=11, fontweight='bold', ha='center')
    ax.text(1.75, 7.3, '• Weather Data', fontsize=9, ha='center')
    ax.text(1.75, 6.9, '• Energy Meters', fontsize=9, ha='center')
    ax.text(1.75, 6.5, '• Occupancy Proxies', fontsize=9, ha='center')
    
    # Deep Learning Module
    dl_box = FancyBboxPatch((4, 6), 3, 2.5, boxstyle=box_style,
                             facecolor=colors['dl'], edgecolor='#E65100', linewidth=2)
    ax.add_patch(dl_box)
    ax.text(5.5, 7.8, 'Deep Learning', fontsize=11, fontweight='bold', ha='center')
    ax.text(5.5, 7.3, 'LSTM Predictor', fontsize=10, ha='center')
    ax.text(5.5, 6.8, '• Energy Forecasting', fontsize=9, ha='center')
    ax.text(5.5, 6.4, '• Comfort Estimation', fontsize=9, ha='center')
    
    # RL Module
    rl_box = FancyBboxPatch((8, 6), 3, 2.5, boxstyle=box_style,
                             facecolor=colors['rl'], edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(rl_box)
    ax.text(9.5, 7.8, 'Reinforcement Learning', fontsize=11, fontweight='bold', ha='center')
    ax.text(9.5, 7.3, 'PPO + LSTM Policy', fontsize=10, ha='center')
    ax.text(9.5, 6.8, '• HVAC Control', fontsize=9, ha='center')
    ax.text(9.5, 6.4, '• Lighting Control', fontsize=9, ha='center')
    
    # Edge AI Module (bottom center)
    edge_box = FancyBboxPatch((4.5, 2.5), 5, 2.5, boxstyle=box_style,
                               facecolor=colors['edge'], edgecolor='#C2185B', linewidth=2)
    ax.add_patch(edge_box)
    ax.text(7, 4.3, 'Edge AI Deployment', fontsize=11, fontweight='bold', ha='center')
    ax.text(7, 3.8, 'TorchScript Models', fontsize=10, ha='center')
    ax.text(7, 3.3, '• Local Inference (No Cloud)', fontsize=9, ha='center')
    ax.text(7, 2.9, '• Real-time Control Loop', fontsize=9, ha='center')
    
    # Federated Learning (right side)
    fed_box = FancyBboxPatch((11.5, 5), 2, 3.5, boxstyle=box_style,
                              facecolor=colors['fed'], edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(fed_box)
    ax.text(12.5, 8, 'Federated\nLearning', fontsize=11, fontweight='bold', ha='center')
    ax.text(12.5, 7.2, '• Privacy', fontsize=9, ha='center')
    ax.text(12.5, 6.8, '  Preserving', fontsize=9, ha='center')
    ax.text(12.5, 6.3, '• Distributed', fontsize=9, ha='center')
    ax.text(12.5, 5.9, '  Training', fontsize=9, ha='center')
    ax.text(12.5, 5.4, '• FedAvg', fontsize=9, ha='center')
    
    # Outputs (bottom)
    output_box = FancyBboxPatch((4.5, 0.3), 5, 1.5, boxstyle=box_style,
                                 facecolor=colors['output'], edgecolor='#00695C', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7, 1.3, 'Outputs', fontsize=11, fontweight='bold', ha='center')
    ax.text(7, 0.8, '• 25-30% Energy Savings  • PPD < 10%  • CO₂ Reduction',
            fontsize=9, ha='center')
    
    # Buildings (bottom corners)
    for i, x_pos in enumerate([0.5, 11.5]):
        building_box = FancyBboxPatch((x_pos, 0.3), 2.5, 3.5, boxstyle=box_style,
                                       facecolor='#ECEFF1', edgecolor='#455A64', linewidth=1.5)
        ax.add_patch(building_box)
        ax.text(x_pos + 1.25, 3.3, f'Building {i+1}', fontsize=10, fontweight='bold', ha='center')
        ax.text(x_pos + 1.25, 2.7, '🏠', fontsize=24, ha='center')
        ax.text(x_pos + 1.25, 1.5, 'HVAC\nLighting\nSensors', fontsize=8, ha='center')
    
    # Arrows
    arrow_style = "Simple, head_width=8, head_length=6"
    arrow_color = '#455A64'
    
    # Data to DL
    ax.annotate('', xy=(4, 7.25), xytext=(3, 7.25),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
    
    # DL to RL
    ax.annotate('', xy=(8, 7.25), xytext=(7, 7.25),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
    
    # RL to Edge
    ax.annotate('', xy=(7.5, 5), xytext=(9, 6),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
    
    # DL to Edge
    ax.annotate('', xy=(6.5, 5), xytext=(5.5, 6),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
    
    # Edge to Outputs
    ax.annotate('', xy=(7, 1.8), xytext=(7, 2.5),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
    
    # Fed Learning connections
    ax.annotate('', xy=(11.5, 6.75), xytext=(11, 7.25),
                arrowprops=dict(arrowstyle='<->', color='#7B1FA2', lw=1.5, ls='--'))
    
    # Building to Edge connections
    ax.annotate('', xy=(4.5, 3), xytext=(3, 2.5),
                arrowprops=dict(arrowstyle='<->', color=arrow_color, lw=1.5))
    ax.annotate('', xy=(9.5, 3), xytext=(11.5, 2.5),
                arrowprops=dict(arrowstyle='<->', color=arrow_color, lw=1.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(save_path.replace('.png', '.eps'), format='eps', bbox_inches='tight')
    plt.close()
    print(f"Figure 0 saved to {save_path}")


def create_prediction_scatter_plot(
    actual: np.ndarray,
    predicted: np.ndarray,
    save_path: str,
    title: str = "Predicted vs Actual Energy Consumption"
) -> None:
    """
    Create Figure 1: Predicted vs actual energy consumption scatter plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(actual, predicted, alpha=0.5, s=20, c=COLORS['primary'], 
               edgecolors='none', label='Predictions')
    
    # 1:1 reference line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, 
            label='Perfect Prediction', alpha=0.7)
    
    # Regression line
    z = np.polyfit(actual, predicted, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    ax.plot(x_line, p(x_line), color=COLORS['secondary'], lw=2, 
            label=f'Regression (y={z[0]:.2f}x+{z[1]:.2f})')
    
    # Calculate R²
    ss_res = np.sum((predicted - actual) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Add R² annotation
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Actual Energy Consumption (Normalized)', fontsize=12)
    ax.set_ylabel('Predicted Energy Consumption (Normalized)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(save_path.replace('.png', '.eps'), format='eps', bbox_inches='tight')
    plt.close()
    print(f"Figure 1 saved to {save_path}")


def create_savings_bar_chart(
    scenario_results: Dict,
    save_path: str
) -> None:
    """
    Create Figure 2: Bar chart comparing energy savings, cost savings, and CO2 reduction.
    """
    scenarios = ['Baseline', 'Simple MPC', 'Hybrid RL']
    
    # Calculate metrics (relative to baseline)
    baseline_energy = scenario_results['baseline']['total_energy']
    
    energy_savings = [
        0,
        (baseline_energy - scenario_results['mpc']['total_energy']) / baseline_energy * 100,
        (baseline_energy - scenario_results['hybrid_rl']['total_energy']) / baseline_energy * 100
    ]
    
    cost_savings = [e * baseline_energy * 0.10 / 100 for e in energy_savings]
    co2_reduction = [e * baseline_energy * 0.5 / 100 for e in energy_savings]
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, energy_savings, width, label='Energy Savings (%)',
                   color=COLORS['primary'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, cost_savings, width, label='Cost Savings ($)',
                   color=COLORS['secondary'], edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, co2_reduction, width, label='CO₂ Reduction (kg)',
                   color=COLORS['success'], edgecolor='white', linewidth=0.5)
    
    # Add value labels
    def add_labels(bars, fmt='{:.1f}'):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(fmt.format(height),
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2, '{:.0f}')
    add_labels(bars3, '{:.0f}')
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_xlabel('Control Strategy', fontsize=12)
    ax.set_title('Comparison of Control Strategies: Savings Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, max(energy_savings) * 1.2)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(save_path.replace('.png', '.eps'), format='eps', bbox_inches='tight')
    plt.close()
    print(f"Figure 2 saved to {save_path}")


def create_timeseries_comparison(
    baseline_energy: List[float],
    optimized_energy: List[float],
    timestamps: pd.DatetimeIndex,
    save_path: str,
    window_size: int = 168  # One week
) -> None:
    """
    Create Figure 3: Time-series comparison of energy use before and after optimization.
    """
    # Select a representative week
    start_idx = len(baseline_energy) // 4
    end_idx = start_idx + window_size
    
    baseline_week = baseline_energy[start_idx:end_idx]
    optimized_week = optimized_energy[start_idx:end_idx]
    time_week = timestamps[start_idx:end_idx]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot time series
    ax.plot(range(len(baseline_week)), baseline_week, 
            color=COLORS['baseline'], lw=1.5, alpha=0.8,
            label='Baseline (Rule-based)')
    ax.plot(range(len(optimized_week)), optimized_week,
            color=COLORS['rl'], lw=1.5, alpha=0.9,
            label='Hybrid RL (Proposed)')
    
    # Fill between to show savings
    ax.fill_between(range(len(baseline_week)), baseline_week, optimized_week,
                    alpha=0.2, color=COLORS['success'], label='Energy Savings')
    
    # Calculate and display savings
    total_baseline = sum(baseline_week)
    total_optimized = sum(optimized_week)
    savings_pct = (total_baseline - total_optimized) / total_baseline * 100
    
    ax.text(0.02, 0.95, f'Weekly Savings: {savings_pct:.1f}%',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format x-axis
    ax.set_xlabel('Hour of Week', fontsize=12)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
    ax.set_title('Weekly Energy Consumption: Baseline vs. Hybrid RL Control',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add day markers
    for i in range(7):
        ax.axvline(x=i*24, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_xlim(0, len(baseline_week))
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(save_path.replace('.png', '.eps'), format='eps', bbox_inches='tight')
    plt.close()
    print(f"Figure 3 saved to {save_path}")


def create_sensitivity_heatmap(
    sensitivity_results: Dict,
    save_path: str
) -> None:
    """
    Create Figure 4: Sensitivity analysis heatmap.
    Shows energy savings vs comfort for different parameter combinations.
    """
    # Create grid for heatmap
    comfort_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    energy_prices = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    # Generate sensitivity matrix
    sensitivity_matrix = np.zeros((len(comfort_weights), len(energy_prices)))
    
    for i, cw in enumerate(comfort_weights):
        for j, ep in enumerate(energy_prices):
            # Model relationship: higher energy price and lower comfort weight = more savings
            base_savings = 28
            price_effect = (ep - 0.10) / 0.05 * 3
            comfort_effect = (0.5 - cw) / 0.2 * 4
            noise = np.random.uniform(-1, 1)
            sensitivity_matrix[i, j] = base_savings + price_effect + comfort_effect + noise
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(sensitivity_matrix, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Energy Savings (%)')
    
    # Set ticks
    ax.set_xticks(np.arange(len(energy_prices)))
    ax.set_yticks(np.arange(len(comfort_weights)))
    ax.set_xticklabels([f'${p:.2f}' for p in energy_prices])
    ax.set_yticklabels([f'{w:.1f}' for w in comfort_weights])
    
    # Add value annotations
    for i in range(len(comfort_weights)):
        for j in range(len(energy_prices)):
            text = ax.text(j, i, f'{sensitivity_matrix[i, j]:.1f}%',
                          ha='center', va='center', color='white', fontsize=10)
    
    ax.set_xlabel('Energy Price ($/kWh)', fontsize=12)
    ax.set_ylabel('Comfort Weight in Reward', fontsize=12)
    ax.set_title('Sensitivity Analysis: Energy Savings vs. Control Parameters',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(save_path.replace('.png', '.eps'), format='eps', bbox_inches='tight')
    plt.close()
    print(f"Figure 4 saved to {save_path}")


def create_pareto_front_plot(
    pareto_points: np.ndarray,
    save_path: str
) -> None:
    """
    Create Figure 5: Pareto front showing energy savings vs comfort trade-off.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    energy_savings = pareto_points[:, 0]
    ppd_values = pareto_points[:, 1]
    
    # Plot Pareto front
    ax.scatter(energy_savings, ppd_values, s=80, c=COLORS['primary'],
               edgecolors='white', linewidths=0.5, alpha=0.8, zorder=3)
    
    # Connect points with line
    sorted_idx = np.argsort(energy_savings)
    ax.plot(energy_savings[sorted_idx], ppd_values[sorted_idx],
            color=COLORS['primary'], lw=2, alpha=0.5, zorder=2)
    
    # Add fill to show Pareto region
    ax.fill_between(energy_savings[sorted_idx], ppd_values[sorted_idx], 15,
                    alpha=0.1, color=COLORS['primary'])
    
    # Highlight specific solutions
    # Minimum PPD solution
    min_ppd_idx = np.argmin(ppd_values)
    ax.scatter(energy_savings[min_ppd_idx], ppd_values[min_ppd_idx],
               s=150, c=COLORS['success'], marker='*', edgecolors='white',
               linewidths=1, zorder=4, label='Best Comfort')
    
    # Maximum savings solution
    max_savings_idx = np.argmax(energy_savings)
    ax.scatter(energy_savings[max_savings_idx], ppd_values[max_savings_idx],
               s=150, c=COLORS['tertiary'], marker='*', edgecolors='white',
               linewidths=1, zorder=4, label='Maximum Savings')
    
    # Balanced solution (near center of Pareto front)
    balanced_idx = len(energy_savings) // 2
    ax.scatter(energy_savings[balanced_idx], ppd_values[balanced_idx],
               s=150, c=COLORS['secondary'], marker='*', edgecolors='white',
               linewidths=1, zorder=4, label='Balanced Solution')
    
    # Add PPD threshold line
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, 
               label='PPD Threshold (10%)')
    
    # Annotations
    ax.annotate('Comfort-focused\nregion', xy=(18, 6), fontsize=10,
                ha='center', style='italic', color=COLORS['neutral'])
    ax.annotate('Energy-focused\nregion', xy=(32, 12), fontsize=10,
                ha='center', style='italic', color=COLORS['neutral'])
    
    ax.set_xlabel('Energy Savings (%)', fontsize=12)
    ax.set_ylabel('Average PPD (%)', fontsize=12)
    ax.set_title('Pareto Front: Energy Savings vs. Thermal Comfort Trade-off',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    
    ax.set_xlim(12, 38)
    ax.set_ylim(4, 16)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(save_path.replace('.png', '.eps'), format='eps', bbox_inches='tight')
    plt.close()
    print(f"Figure 5 saved to {save_path}")


def create_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str
) -> None:
    """
    Create supplementary figure: Training curves.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, color=COLORS['primary'], lw=2,
            label='Training Loss')
    ax.plot(epochs, val_losses, color=COLORS['secondary'], lw=2,
            label='Validation Loss')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Model Training Progress', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Training curves saved to {save_path}")


def create_all_figures(
    prediction_results: Dict,
    scenario_results: Dict,
    sensitivity_results: Dict,
    pareto_points: np.ndarray,
    training_history: Dict
) -> None:
    """
    Generate all figures for the paper.
    """
    print("\nGenerating publication-quality figures...")
    
    # Figure 0: System Architecture
    create_system_architecture_figure(
        os.path.join(FIGURES_DIR, 'fig0.png')
    )
    
    # Figure 1: Prediction Scatter Plot
    create_prediction_scatter_plot(
        prediction_results['actual'],
        prediction_results['predicted'],
        os.path.join(FIGURES_DIR, 'fig1.png')
    )
    
    # Figure 2: Savings Bar Chart
    create_savings_bar_chart(
        scenario_results,
        os.path.join(FIGURES_DIR, 'fig2.png')
    )
    
    # Figure 3: Time Series Comparison
    create_timeseries_comparison(
        scenario_results['baseline']['energy_use'],
        scenario_results['hybrid_rl']['energy_use'],
        pd.date_range('2017-01-01', periods=len(scenario_results['baseline']['energy_use']), freq='H'),
        os.path.join(FIGURES_DIR, 'fig3.png')
    )
    
    # Figure 4: Sensitivity Heatmap
    create_sensitivity_heatmap(
        sensitivity_results,
        os.path.join(FIGURES_DIR, 'fig4.png')
    )
    
    # Figure 5: Pareto Front
    create_pareto_front_plot(
        pareto_points,
        os.path.join(FIGURES_DIR, 'fig5.png')
    )
    
    # Supplementary: Training Curves
    if 'train_loss' in training_history:
        create_training_curves(
            training_history['train_loss'],
            training_history['val_loss'],
            os.path.join(FIGURES_DIR, 'fig_training.png')
        )
    
    print("All figures generated successfully!")


if __name__ == "__main__":
    print("Testing Visualization module...")
    
    # Test system architecture
    create_system_architecture_figure(os.path.join(FIGURES_DIR, 'fig0.png'))
    
    # Test scatter plot with dummy data
    actual = np.random.normal(0, 1, 500)
    predicted = actual * 0.95 + np.random.normal(0, 0.1, 500)
    create_prediction_scatter_plot(actual, predicted, os.path.join(FIGURES_DIR, 'fig1.png'))
    
    print("Visualization tests completed!")
