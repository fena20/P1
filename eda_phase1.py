"""
Phase 1: Exploratory Data Analysis (EDA) for Building Energy Systems
Building Data Genome Project 2 (BDG2) Dataset Analysis
Target: Applied Energy Manuscript

This script performs comprehensive EDA on electricity meter data from BDG2,
focusing on temporal patterns, correlations, and data quality assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Create output directories
Path('figures').mkdir(exist_ok=True)
Path('tables').mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)

print("=" * 80)
print("Phase 1: Exploratory Data Analysis (EDA)")
print("Building Data Genome Project 2 (BDG2) - Electricity Meters")
print("=" * 80)
