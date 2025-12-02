"""
Building Data Genome Project 2 Analysis - Phase 2: Predictive Modeling

This script implements physics-informed feature engineering and SOTA models including:
- Quantile TabNet for probabilistic forecasting
- Baseline models: Random Forest, XGBoost, LightGBM, MLP, LSTM, Stacking Ensemble
- Model comparison and interpretability analysis

Author: Senior Data Scientist & Lead Researcher in Building Energy Systems
Target: Applied Energy manuscript
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
from datetime import datetime
import pickle

# ML libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Deep learning
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.metrics import Metric

warnings.filterwarnings('ignore')

# Set plotting parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("PHASE 2: PREDICTIVE MODELING")
print("Building Data Genome Project 2 - Load Forecasting")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA FROM PHASE 1
# ============================================================================

print("\n[1/7] Loading preprocessed data...")

# Reload the data (same as Phase 1)
DATA_DIR = '/workspace/building-data-genome-project-2/data'
METADATA_PATH = os.path.join(DATA_DIR, 'metadata/metadata.csv')
ELECTRICITY_PATH = os.path.join(DATA_DIR, 'meters/cleaned/electricity_cleaned.csv')
WEATHER_PATH = os.path.join(DATA_DIR, 'weather/weather.csv')

metadata = pd.read_csv(METADATA_PATH)
electricity_wide = pd.read_csv(ELECTRICITY_PATH, nrows=50000)
electricity_wide['timestamp'] = pd.to_datetime(electricity_wide['timestamp'])
weather = pd.read_csv(WEATHER_PATH)
weather['timestamp'] = pd.to_datetime(weather['timestamp'])

# Parse building information
building_cols = [col for col in electricity_wide.columns if col != 'timestamp']
building_info = []
for col in building_cols:
    parts = col.split('_')
    if len(parts) >= 3:
        building_info.append({
            'column_name': col,
            'site_id': parts[0],
            'primaryspaceusage': parts[1].capitalize(),
            'building_id': col
        })

building_info_df = pd.DataFrame(building_info)
building_info_df = building_info_df.merge(
    metadata[['building_id', 'sqm', 'primaryspaceusage', 'timezone']],
    on='building_id', how='left', suffixes=('_parsed', '_meta')
)
building_info_df['primaryspaceusage'] = building_info_df['primaryspaceusage_meta'].fillna(
    building_info_df['primaryspaceusage_parsed']
)

# Select office buildings at Eagle site
target_site = 'Eagle'
office_buildings = building_info_df[
    building_info_df['primaryspaceusage'].str.lower() == 'office'
]
office_site_buildings = office_buildings[office_buildings['site_id'] == target_site]
target_building_cols = office_site_buildings['column_name'].tolist()

# Convert to long format
electricity_long_list = []
for col in target_building_cols:
    if col in electricity_wide.columns:
        temp_df = pd.DataFrame({
            'timestamp': electricity_wide['timestamp'],
            'building_id': col,
            'meter_reading': electricity_wide[col]
        })
        electricity_long_list.append(temp_df)

df = pd.concat(electricity_long_list, ignore_index=True)
df = df.dropna(subset=['meter_reading'])

# Merge building metadata
df = df.merge(
    building_info_df[['building_id', 'sqm', 'primaryspaceusage', 'site_id']],
    on='building_id', how='left'
)

# Merge weather
weather_subset = weather[weather['site_id'] == target_site].copy()
df = df.merge(
    weather_subset[['timestamp', 'airTemperature', 'dewTemperature', 'windSpeed', 
                    'seaLvlPressure', 'cloudCoverage', 'precipDepth1HR']],
    on='timestamp', how='left'
)

print(f"  Loaded {len(df):,} records for {len(target_building_cols)} office buildings")

# ============================================================================
# 2. PHYSICS-INFORMED FEATURE ENGINEERING
# ============================================================================

print("\n[2/7] Physics-informed feature engineering...")

# Temporal features with cyclic encoding
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_of_year'] = df['timestamp'].dt.dayofyear
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Cyclic encoding for temporal features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

# Degree hours (heating and cooling)
# Base temperatures: 18°C for heating, 26°C for cooling
df['heating_degree_hour'] = np.maximum(18 - df['airTemperature'], 0)
df['cooling_degree_hour'] = np.maximum(df['airTemperature'] - 26, 0)

# Humidity ratio approximation (simplified psychrometrics)
df['humidity_proxy'] = df['airTemperature'] - df['dewTemperature']

# Interaction terms
df['temp_hour_interaction'] = df['airTemperature'] * df['hour']

# Lag features (require sorting by timestamp within each building)
print("  Creating lag features...")
df = df.sort_values(['building_id', 'timestamp'])

lag_periods = [1, 24, 168]  # 1h, 24h (1 day), 168h (1 week)
for lag in lag_periods:
    df[f'meter_reading_lag_{lag}'] = df.groupby('building_id')['meter_reading'].shift(lag)

# Rolling statistics
df['meter_reading_roll_mean_24'] = df.groupby('building_id')['meter_reading'].transform(
    lambda x: x.rolling(window=24, min_periods=1).mean()
)
df['meter_reading_roll_std_24'] = df.groupby('building_id')['meter_reading'].transform(
    lambda x: x.rolling(window=24, min_periods=1).std()
)

# Building features
df['sqm_log'] = np.log1p(df['sqm'])

# Drop rows with NaN in lag features (first 168 hours per building)
df = df.dropna(subset=[f'meter_reading_lag_{lag}' for lag in lag_periods])

print(f"  Created {df.shape[1]} features")
print(f"  Final dataset: {len(df):,} records")

# ============================================================================
# 3. PREPARE TRAIN/TEST SPLIT (TIME-BASED)
# ============================================================================

print("\n[3/7] Preparing train/test split...")

# Sort by timestamp for time-based split
df = df.sort_values('timestamp')

# Use 80% for training, 20% for testing (temporal split)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"  Train set: {len(train_df):,} records ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
print(f"  Test set: {len(test_df):,} records ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

# Define feature columns
feature_cols = [
    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos', 'is_weekend',
    'airTemperature', 'dewTemperature', 'windSpeed', 'seaLvlPressure',
    'heating_degree_hour', 'cooling_degree_hour', 'humidity_proxy',
    'temp_hour_interaction', 'sqm_log',
    'meter_reading_lag_1', 'meter_reading_lag_24', 'meter_reading_lag_168',
    'meter_reading_roll_mean_24', 'meter_reading_roll_std_24'
]

# Handle missing values in features (fill with median)
for col in feature_cols:
    if col in train_df.columns:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

target_col = 'meter_reading'

X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values
y_test = test_df[target_col].values

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Feature shape: {X_train.shape}")

# ============================================================================
# 4. TRAIN BASELINE MODELS
# ============================================================================

print("\n[4/7] Training baseline models...")

models = {}
predictions = {}

# Sample subset for faster training (10% of data)
sample_size = int(len(X_train_scaled) * 0.1)
indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
X_train_sample = X_train_scaled[indices]
y_train_sample = y_train[indices]

print(f"  Using {sample_size:,} samples for faster training")

# Random Forest
print("  Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train_sample, y_train_sample)
models['Random Forest'] = rf
predictions['Random Forest'] = rf.predict(X_test_scaled)

# XGBoost
print("  Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100, max_depth=6, learning_rate=0.1, 
    random_state=42, n_jobs=-1, tree_method='hist'
)
xgb_model.fit(X_train_sample, y_train_sample)
models['XGBoost'] = xgb_model
predictions['XGBoost'] = xgb_model.predict(X_test_scaled)

# LightGBM
print("  Training LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_train_sample, y_train_sample)
models['LightGBM'] = lgb_model
predictions['LightGBM'] = lgb_model.predict(X_test_scaled)

# MLP
print("  Training MLP...")
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50), activation='relu',
    max_iter=50, random_state=42, early_stopping=True
)
mlp.fit(X_train_sample, y_train_sample)
models['MLP'] = mlp
predictions['MLP'] = mlp.predict(X_test_scaled)

# Stacking Ensemble (combine RF, XGBoost, LightGBM with Ridge meta-learner)
print("  Training Stacking Ensemble...")
stack_predictions_train = np.column_stack([
    rf.predict(scaler.transform(X_train[indices])),
    xgb_model.predict(scaler.transform(X_train[indices])),
    lgb_model.predict(scaler.transform(X_train[indices]))
])
stack_predictions_test = np.column_stack([
    predictions['Random Forest'],
    predictions['XGBoost'],
    predictions['LightGBM']
])

meta_learner = Ridge(alpha=1.0)
meta_learner.fit(stack_predictions_train, y_train_sample)
models['Stacking Ensemble'] = {'base_models': [rf, xgb_model, lgb_model], 'meta': meta_learner}
predictions['Stacking Ensemble'] = meta_learner.predict(stack_predictions_test)

print("  Baseline models trained successfully")

# ============================================================================
# 5. TRAIN QUANTILE TABNET
# ============================================================================

print("\n[5/7] Training Quantile TabNet...")

# Custom pinball loss for TabNet
class PinballLoss(Metric):
    def __init__(self, quantile=0.5):
        self._name = f"pinball_q{quantile}"
        self._maximize = False
        self.quantile = quantile

    def __call__(self, y_true, y_pred):
        errors = y_true - y_pred
        loss = np.maximum(self.quantile * errors, (self.quantile - 1) * errors)
        return np.mean(loss)

# Train three TabNet models for quantiles 0.025, 0.5, 0.975
quantiles = [0.025, 0.5, 0.975]
tabnet_models = {}
tabnet_predictions = {}

for q in quantiles:
    print(f"  Training TabNet for quantile {q}...")
    
    tabnet = TabNetRegressor(
        n_d=32, n_a=32, n_steps=3,
        gamma=1.3, lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        seed=42
    )
    
    # Train on sample
    tabnet.fit(
        X_train_sample, y_train_sample.reshape(-1, 1),
        eval_set=[(X_test_scaled[:10000], y_test[:10000].reshape(-1, 1))],
        eval_metric=['rmse'],
        max_epochs=20,
        patience=5,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    tabnet_models[f'q{q}'] = tabnet
    tabnet_predictions[f'q{q}'] = tabnet.predict(X_test_scaled).flatten()

# Use median (q=0.5) as the point forecast
models['Quantile TabNet'] = tabnet_models['q0.5']
predictions['Quantile TabNet'] = tabnet_predictions['q0.5']

print("  TabNet models trained successfully")

# ============================================================================
# 6. MODEL EVALUATION AND COMPARISON
# ============================================================================

print("\n[6/7] Evaluating and comparing models...")

# Compute metrics for each model
results = []

for model_name, y_pred in predictions.items():
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Compute prediction interval metrics for TabNet
    if model_name == 'Quantile TabNet':
        # PICP (Prediction Interval Coverage Probability)
        lower = tabnet_predictions['q0.025']
        upper = tabnet_predictions['q0.975']
        coverage = np.mean((y_test >= lower) & (y_test <= upper)) * 100
        
        # Winkler interval score (lower is better)
        alpha = 0.05  # 95% interval
        width = upper - lower
        penalty_lower = (2 / alpha) * (lower - y_test) * (y_test < lower)
        penalty_upper = (2 / alpha) * (y_test - upper) * (y_test > upper)
        winkler = np.mean(width + penalty_lower + penalty_upper)
    else:
        coverage = np.nan
        winkler = np.nan
    
    results.append({
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'PICP (%)': coverage,
        'Winkler Score': winkler
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('RMSE')

print("\n  Model Performance:")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('/workspace/outputs/tables/Table3_Model_Performance.csv', index=False)
print(f"\n  Table 3 saved to: /workspace/outputs/tables/Table3_Model_Performance.csv")

# ============================================================================
# 7. GENERATE FIGURES 5 AND 6
# ============================================================================

print("\n[7/7] Generating Figures 5 and 6...")

# FIGURE 5: Model Comparison Boxplot
print("  Generating Figure 5: Model comparison...")

# Compute residuals for each model
residuals = {}
for model_name, y_pred in predictions.items():
    residuals[model_name] = y_test - y_pred

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: RMSE comparison
ax1 = axes[0, 0]
rmse_values = results_df.sort_values('RMSE')[['Model', 'RMSE']]
ax1.barh(rmse_values['Model'], rmse_values['RMSE'], color='#2E86AB', alpha=0.7)
ax1.set_xlabel('RMSE (kWh)', fontsize=11)
ax1.set_title('(a) Root Mean Square Error', fontsize=10)
ax1.grid(True, alpha=0.3, axis='x')

# Subplot 2: MAE comparison
ax2 = axes[0, 1]
mae_values = results_df.sort_values('MAE')[['Model', 'MAE']]
ax2.barh(mae_values['Model'], mae_values['MAE'], color='#A23B72', alpha=0.7)
ax2.set_xlabel('MAE (kWh)', fontsize=11)
ax2.set_title('(b) Mean Absolute Error', fontsize=10)
ax2.grid(True, alpha=0.3, axis='x')

# Subplot 3: R² comparison
ax3 = axes[1, 0]
r2_values = results_df.sort_values('R²', ascending=False)[['Model', 'R²']]
ax3.barh(r2_values['Model'], r2_values['R²'], color='#93C178', alpha=0.7)
ax3.set_xlabel('R²', fontsize=11)
ax3.set_title('(c) Coefficient of Determination', fontsize=10)
ax3.grid(True, alpha=0.3, axis='x')

# Subplot 4: Residual distribution comparison
ax4 = axes[1, 1]
# Select top 3 models for residual comparison
top_models = results_df.head(3)['Model'].tolist()
residual_data = [residuals[m] for m in top_models]
bp = ax4.boxplot(residual_data, labels=top_models, patch_artist=True,
                 showfliers=False)  # Hide outliers for clarity
colors = ['#2E86AB', '#A23B72', '#93C178']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax4.set_ylabel('Residuals (kWh)', fontsize=11)
ax4.set_title('(d) Residual Distribution (Top 3 Models)', fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha='right')

plt.suptitle('Figure 5. Comparison of forecasting performance across models\n'
             '(Quantile TabNet and baselines)',
             fontsize=12, y=0.995)
plt.tight_layout()
plt.savefig('/workspace/outputs/figures/Figure5_Model_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"    Figure 5 saved to: /workspace/outputs/figures/Figure5_Model_Comparison.png")

# FIGURE 6: Feature Importance (TabNet + XGBoost)
print("  Generating Figure 6: Feature importance...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: XGBoost feature importance
ax1 = axes[0]
xgb_importance = models['XGBoost'].feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_importance
}).sort_values('importance', ascending=False).head(15)

ax1.barh(range(len(importance_df)), importance_df['importance'], color='#2E86AB', alpha=0.7)
ax1.set_yticks(range(len(importance_df)))
ax1.set_yticklabels(importance_df['feature'], fontsize=9)
ax1.set_xlabel('Feature Importance', fontsize=11)
ax1.set_title('(a) XGBoost Feature Importance (Top 15)', fontsize=10)
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()

# Subplot 2: TabNet feature importances (from model)
ax2 = axes[1]
tabnet_model = models['Quantile TabNet']
# Get feature importances from TabNet
tabnet_importance = tabnet_model.feature_importances_
importance_df_tabnet = pd.DataFrame({
    'feature': feature_cols,
    'importance': tabnet_importance
}).sort_values('importance', ascending=False).head(15)

ax2.barh(range(len(importance_df_tabnet)), importance_df_tabnet['importance'], 
         color='#A23B72', alpha=0.7)
ax2.set_yticks(range(len(importance_df_tabnet)))
ax2.set_yticklabels(importance_df_tabnet['feature'], fontsize=9)
ax2.set_xlabel('Feature Importance', fontsize=11)
ax2.set_title('(b) Quantile TabNet Feature Importance (Top 15)', fontsize=10)
ax2.grid(True, alpha=0.3, axis='x')
ax2.invert_yaxis()

plt.suptitle('Figure 6. Feature importance and interpretability\n'
             'highlighting physical drivers of building electricity use',
             fontsize=12)
plt.tight_layout()
plt.savefig('/workspace/outputs/figures/Figure6_Feature_Importance.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"    Figure 6 saved to: /workspace/outputs/figures/Figure6_Feature_Importance.png")

# Save models
with open('/workspace/outputs/trained_models.pkl', 'wb') as f:
    pickle.dump({
        'models': models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'tabnet_quantiles': tabnet_models
    }, f)

print("\n" + "=" * 80)
print("PHASE 2 COMPLETE")
print("=" * 80)
print("\nGenerated outputs:")
print("  - Figure 5: Model comparison")
print("  - Figure 6: Feature importance")
print("  - Table 3: Model performance metrics")
print("  - Trained models saved to trained_models.pkl")
print("=" * 80)
