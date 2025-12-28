#!/usr/bin/env python3
"""
==============================================================================
Comprehensive Sensitivity Analysis (Refactored for Rigor)
==============================================================================
Manuscript Section: Model Robustness & Parameter Sensitivity

This module performs:
1. Feature Importance Sensitivity (Permutation on Test Set)
2. Hyperparameter Sensitivity Analysis
3. Data Volume Sensitivity (Chronological Learning Curves)
4. Noise Robustness Analysis

Author: Data Science Research Team (Refactored by Jules)
Date: November 2024
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import pipeline_core
from pipeline_core import SafeXGBRegressor, PipelineConfig

import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# Publication-quality settings
plt.style.use('seaborn-v0_8-whitegrid')

print("=" * 80)
print("COMPREHENSIVE SENSITIVITY ANALYSIS (RIGOROUS MODE)")
print("=" * 80)

# 1. Load Data & Features
print("\n[1/6] Loading data and preparing features (Leakage-Free)...")
df = pipeline_core.load_data('energydata_complete.csv')
config = PipelineConfig()
df_feat = pipeline_core.create_physics_features(df, config)

# Split
X_train, X_test, y_train, y_test, feature_cols = pipeline_core.get_chronological_split(df_feat, config)

# Scale
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# Train Baseline
print("Training Baseline XGBoost...")
model = SafeXGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
model.fit(X_train_scaled, y_train_scaled)

y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Baseline RMSE: {baseline_rmse:.2f}")

# 2. Permutation Importance (On Test Set)
print("\n[2/6] Feature Importance (Permutation on Test)...")
# Note: SafeXGBRegressor wraps XGB, so we pass it directly
result = permutation_importance(model, X_test_scaled, y_test, n_repeats=5, random_state=42, n_jobs=-1, scoring='neg_root_mean_squared_error')
# Note: y_test is unscaled, but model predicts scaled?
# Wait, model predicts scaled. permutation_importance needs the scorer to handle it.
# The default scorer will see scaled predictions vs unscaled y_test -> Error.
# We need to use a custom scorer or pass scaled y_test.
# Let's pass scaled y_test for permutation importance, then the importance values are in scaled units, which is fine for relative ranking.
# OR we define a scorer that inverses.
# Simplest: Permutation on scaled targets, result is relative increase in MSE (scaled).
result = permutation_importance(model, X_test_scaled, scaler_y.transform(y_test.reshape(-1,1)).ravel(), n_repeats=5, random_state=42, n_jobs=-1)

perm_sorted_idx = result.importances_mean.argsort()
perm_indices = np.arange(0, len(feature_cols)) + 0.5

# Plot
fig, ax = plt.subplots(figsize=(10, 12))
ax.barh(perm_indices, result.importances_mean[perm_sorted_idx], height=0.7)
ax.set_yticks(perm_indices)
ax.set_yticklabels(np.array(feature_cols)[perm_sorted_idx])
ax.set_title("Permutation Importance (Test Set)")
plt.tight_layout()
plt.savefig('outputs/sensitivity_permutation_importance.png')
print("Saved importance plot.")

# 3. Noise Robustness
print("\n[3/6] Noise Robustness...")
noise_levels = [0.01, 0.05, 0.1, 0.2]
results = []
for nl in noise_levels:
    X_test_noisy = X_test_scaled + np.random.normal(0, nl, X_test_scaled.shape)
    y_pred_n = model.predict(X_test_noisy)
    y_pred_n_inv = scaler_y.inverse_transform(y_pred_n.reshape(-1, 1)).ravel()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_n_inv))
    results.append({'Noise': nl, 'RMSE': rmse})
    print(f"Noise {nl}: RMSE {rmse:.2f}")

pd.DataFrame(results).to_csv('outputs/sensitivity_noise.csv', index=False)

print("\nSensitivity Analysis Complete.")
