"""
Generate System Architecture Diagram (Figure 0)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

from config import *

def generate_system_architecture_diagram():
    """Generate Figure 0: System Architecture Diagram"""
    plt.style.use(FIG_STYLE)
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    color_edge = '#4A90E2'
    color_cloud = '#7B68EE'
    color_dl = '#50C878'
    color_rl = '#FF6B6B'
    color_data = '#FFD700'
    
    # Title
    ax.text(5, 9.5, 'Edge AI with Hybrid RL and Deep Learning Architecture', 
           ha='center', fontsize=16, fontweight='bold')
    
    # Edge Devices Layer
    edge_box = FancyBboxPatch((0.5, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                              facecolor=color_edge, edgecolor='black', linewidth=2)
    ax.add_patch(edge_box)
    ax.text(1.75, 8.25, 'Edge AI\nDevices', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='white')
    
    # Local Inference
    inference_box = Rectangle((0.7, 7.2), 1.6, 0.6, facecolor='white', edgecolor='black')
    ax.add_patch(inference_box)
    ax.text(1.5, 7.5, 'Local Inference', ha='center', va='center', fontsize=9)
    
    # Deep Learning Component
    dl_box = FancyBboxPatch((3.5, 6.5), 2.5, 2, boxstyle="round,pad=0.1", 
                           facecolor=color_dl, edgecolor='black', linewidth=2)
    ax.add_patch(dl_box)
    ax.text(4.75, 8.25, 'Deep Learning\nPrediction Model', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='white')
    
    # LSTM/Transformer
    lstm_box = Rectangle((3.7, 7.5), 2.1, 0.8, facecolor='white', edgecolor='black')
    ax.add_patch(lstm_box)
    ax.text(4.75, 7.9, 'LSTM/Transformer', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Outputs
    ax.text(3.7, 7.1, 'Energy Prediction', ha='left', fontsize=9)
    ax.text(3.7, 6.8, 'Comfort (PPD) Prediction', ha='left', fontsize=9)
    
    # Reinforcement Learning Component
    rl_box = FancyBboxPatch((6.5, 6.5), 2.5, 2, boxstyle="round,pad=0.1", 
                           facecolor=color_rl, edgecolor='black', linewidth=2)
    ax.add_patch(rl_box)
    ax.text(7.75, 8.25, 'Reinforcement\nLearning Agent', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='white')
    
    # PPO with LSTM
    ppo_box = Rectangle((6.7, 7.5), 2.1, 0.8, facecolor='white', edgecolor='black')
    ax.add_patch(ppo_box)
    ax.text(7.75, 7.9, 'PPO + LSTM Policy', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Actions
    ax.text(6.7, 7.1, 'HVAC Setpoint', ha='left', fontsize=9)
    ax.text(6.7, 6.8, 'Lighting Control', ha='left', fontsize=9)
    
    # Data Sources
    data_box = FancyBboxPatch((1, 4.5), 2, 1.5, boxstyle="round,pad=0.1", 
                              facecolor=color_data, edgecolor='black', linewidth=2)
    ax.add_patch(data_box)
    ax.text(2, 5.75, 'BDG2 Dataset', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    ax.text(2, 5.3, 'Weather Data', ha='center', fontsize=9)
    ax.text(2, 5.0, 'Building Metadata', ha='center', fontsize=9)
    ax.text(2, 4.7, 'Meter Readings', ha='center', fontsize=9)
    
    # Federated Learning
    fed_box = FancyBboxPatch((4, 4.5), 2, 1.5, boxstyle="round,pad=0.1", 
                             facecolor='#FFA500', edgecolor='black', linewidth=2)
    ax.add_patch(fed_box)
    ax.text(5, 5.75, 'Federated\nLearning', ha='center', va='center', 
           fontsize=11, fontweight='bold', color='white')
    ax.text(5, 5.0, 'Privacy-Preserving', ha='center', fontsize=9, color='white')
    ax.text(5, 4.7, 'Distributed Training', ha='center', fontsize=9, color='white')
    
    # Cloud/Aggregation
    cloud_box = FancyBboxPatch((7, 4.5), 2, 1.5, boxstyle="round,pad=0.1", 
                               facecolor=color_cloud, edgecolor='black', linewidth=2)
    ax.add_patch(cloud_box)
    ax.text(8, 5.75, 'Model\nAggregation', ha='center', va='center', 
           fontsize=11, fontweight='bold', color='white')
    ax.text(8, 5.0, 'FedAvg Algorithm', ha='center', fontsize=9, color='white')
    
    # Arrows: Data to DL
    arrow1 = FancyArrowPatch((2, 6), (4.75, 7.5), arrowstyle='->', 
                            lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow1)
    ax.text(3.2, 6.8, 'Training', ha='center', fontsize=8, style='italic')
    
    # Arrows: DL to RL
    arrow2 = FancyArrowPatch((6, 7.5), (6.5, 7.5), arrowstyle='->', 
                            lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow2)
    ax.text(6.25, 7.8, 'Predictions', ha='center', fontsize=8, style='italic')
    
    # Arrows: RL to Edge
    arrow3 = FancyArrowPatch((7.75, 6.5), (2.5, 7), arrowstyle='->', 
                            lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow3)
    ax.text(5, 6.2, 'Deploy Model', ha='center', fontsize=8, style='italic')
    
    # Arrows: Edge to Federated
    arrow4 = FancyArrowPatch((2.5, 7), (4.5, 5.5), arrowstyle='->', 
                            lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow4)
    ax.text(3.3, 6.0, 'Model Updates', ha='center', fontsize=8, style='italic')
    
    # Arrows: Federated to Cloud
    arrow5 = FancyArrowPatch((6, 5.25), (7, 5.25), arrowstyle='->', 
                            lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow5)
    
    # Output/Results
    results_box = FancyBboxPatch((3, 2), 4, 1.5, boxstyle="round,pad=0.1", 
                                 facecolor='#90EE90', edgecolor='black', linewidth=2)
    ax.add_patch(results_box)
    ax.text(5, 3.25, 'Optimization Results', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(4, 2.7, '• Energy Savings: 20-30%', ha='left', fontsize=9)
    ax.text(4, 2.4, '• Comfort: PPD < 10%', ha='left', fontsize=9)
    ax.text(6, 2.7, '• Cost Reduction', ha='left', fontsize=9)
    ax.text(6, 2.4, '• CO₂ Reduction', ha='left', fontsize=9)
    
    # Arrow: RL to Results
    arrow6 = FancyArrowPatch((7.75, 6.5), (5, 3.5), arrowstyle='->', 
                            lw=2, color='black', mutation_scale=20)
    ax.add_patch(arrow6)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color_edge, label='Edge AI'),
        mpatches.Patch(facecolor=color_dl, label='Deep Learning'),
        mpatches.Patch(facecolor=color_rl, label='Reinforcement Learning'),
        mpatches.Patch(facecolor='#FFA500', label='Federated Learning'),
        mpatches.Patch(facecolor=color_data, label='Data Sources')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig0.png", dpi=DPI, bbox_inches='tight')
    print(f"Figure 0 (System Architecture) saved to {FIGURES_DIR}/fig0.png")
    plt.close()

if __name__ == "__main__":
    generate_system_architecture_diagram()
