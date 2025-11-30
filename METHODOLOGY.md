# Detailed Methodology Documentation

## Physics-Informed Machine Learning Framework for Building Energy Prediction and Multi-Objective Optimization

**Document Version**: 1.0  
**Last Updated**: November 2024  
**Purpose**: Complete methodology description for academic manuscript

---

## Table of Contents

1. [Introduction & Research Objectives](#1-introduction--research-objectives)
2. [Dataset Description & Preprocessing](#2-dataset-description--preprocessing)
3. [Physics-Informed Feature Engineering](#3-physics-informed-feature-engineering)
4. [Machine Learning Models](#4-machine-learning-models)
5. [Multi-Objective Optimization](#5-multi-objective-optimization)
6. [Generalization & Transfer Learning](#6-generalization--transfer-learning)
7. [Sensitivity Analysis](#7-sensitivity-analysis)
8. [Experimental Setup & Reproducibility](#8-experimental-setup--reproducibility)
9. [Statistical Validation](#9-statistical-validation)
10. [Limitations & Future Work](#10-limitations--future-work)

---

## 1. Introduction & Research Objectives

### 1.1 Problem Statement

Building energy consumption accounts for approximately 40% of global energy use and 36% of CO₂ emissions (IEA, 2023). Accurate prediction and optimization of building energy consumption is essential for:

- **Demand-side management**: Enabling grid operators to balance supply and demand
- **Retrofit decision-making**: Quantifying potential savings from efficiency improvements
- **Thermal comfort optimization**: Balancing energy efficiency with occupant well-being
- **Net-zero pathways**: Supporting decarbonization goals through evidence-based interventions

### 1.2 Research Questions

This study addresses four primary research questions:

1. **RQ1**: How can physics-informed feature engineering improve the predictive accuracy of machine learning models for building energy consumption?

2. **RQ2**: Does the Quantile TabNet architecture offer advantages over traditional ensemble methods in terms of accuracy and interpretability?

3. **RQ3**: What energy savings potential exists through multi-objective optimization of thermal setpoints while maintaining occupant comfort?

4. **RQ4**: How well do models trained on one building generalize to different buildings, climates, and usage patterns?

### 1.3 Research Contributions

| Contribution | Description | Section |
|-------------|-------------|---------|
| **C1** | Physics-informed feature engineering framework | §3 |
| **C2** | Quantile TabNet for uncertainty-aware prediction | §4.2 |
| **C3** | NSGA-II + TOPSIS for energy-comfort optimization | §5 |
| **C4** | Universal adapter for cross-building transfer | §6 |
| **C5** | Comprehensive sensitivity analysis framework | §7 |

---

## 2. Dataset Description & Preprocessing

### 2.1 Primary Dataset: UCI Appliances Energy Prediction

#### 2.1.1 Data Source
- **Origin**: Candanedo et al. (2017), Energy and Buildings
- **Location**: Low-energy passive house, Stambruges, Belgium
- **Building Type**: Residential, 220 m², 4 occupants
- **Construction**: Passive house certified (U < 0.1 W/m²K)

#### 2.1.2 Data Characteristics

| Characteristic | Value |
|---------------|-------|
| Time Period | January 11 - May 27, 2016 |
| Duration | 137 days |
| Sampling Interval | 10 minutes |
| Total Observations | 19,735 |
| Missing Values | 0 (0%) |
| Features | 28 original + 39 engineered = 67 total |

#### 2.1.3 Sensor Configuration

The dataset includes measurements from a ZigBee wireless sensor network:

```
Indoor Sensors (9 zones):
├── T1, RH_1: Kitchen (cooking area)
├── T2, RH_2: Living Room (main activity space)
├── T3, RH_3: Laundry Room (appliance cluster)
├── T4, RH_4: Office (home workspace)
├── T5, RH_5: Bathroom (wet room)
├── T6, RH_6: Building North Exterior
├── T7, RH_7: Ironing Room (utility)
├── T8, RH_8: Teenager Room (bedroom)
└── T9, RH_9: Parents Room (master bedroom)

Weather Station (Chièvres Airport, ~24 km):
├── T_out: Outdoor temperature (°C)
├── RH_out: Outdoor humidity (%)
├── Press_mm_hg: Atmospheric pressure (mmHg)
├── Windspeed: Wind speed (m/s)
├── Visibility: Visibility (km)
└── Tdewpoint: Dew point temperature (°C)

Energy Meters:
├── Appliances: Appliances energy (Wh) - TARGET
└── lights: Lighting energy (Wh)
```

### 2.2 Data Quality Assessment

#### 2.2.1 Missing Value Analysis
```
Total missing values: 0
Completeness rate: 100%
```

#### 2.2.2 Physical Validity Checks
- Temperature range: [-6.1°C, 29.9°C] ✓ (valid for Belgian climate)
- Humidity range: [1%, 100%] ✓ (within physical bounds)
- Energy values: [10, 1080] Wh ✓ (non-negative)

#### 2.2.3 Statistical Properties of Target Variable

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Mean | 97.7 Wh | Average 10-min consumption |
| Median | 60.0 Wh | Right-skewed distribution |
| Std Dev | 102.5 Wh | High variability |
| Skewness | 3.39 | Strongly right-skewed |
| Kurtosis | 13.66 | Leptokurtic (heavy tails) |
| CV | 104.9% | High relative variability |

**Implication**: The right-skewed distribution with heavy tails suggests that log-transformation or robust regression methods may improve model performance.

### 2.3 External Validation Datasets

#### 2.3.1 Kitakyushu Campus Building (Japan)
- **Type**: Commercial/educational
- **Period**: 2002-2011
- **Resolution**: Hourly
- **Purpose**: Cross-domain validation

#### 2.3.2 Synthetic European Building
- **Type**: Residential
- **Climate**: Central European
- **Purpose**: Controlled generalization testing

---

## 3. Physics-Informed Feature Engineering

### 3.1 Rationale

Traditional machine learning approaches treat building energy prediction as a purely data-driven problem. However, incorporating domain knowledge from building physics and thermodynamics can:

1. Improve model interpretability
2. Enhance generalization to unseen conditions
3. Reduce required training data
4. Ensure physically plausible predictions

### 3.2 Feature Categories

#### 3.2.1 Cyclical Temporal Encoding

**Problem**: Linear encoding of time (e.g., hour = 0, 1, ..., 23) creates artificial discontinuities (hour 23 appears far from hour 0).

**Solution**: Sinusoidal encoding preserves cyclical nature:

$$
\text{hour}_{\sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right)
$$

$$
\text{hour}_{\cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)
$$

Similarly for day of week (period = 7) and month (period = 12).

**Implementation**:
```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
```

#### 3.2.2 Thermodynamic Features

**a) Dew Point Temperature (Tdp)**

The Magnus-Tetens formula approximates dew point:

$$
T_{dp} = \frac{b \cdot \alpha(T, RH)}{a - \alpha(T, RH)}
$$

where:
$$
\alpha(T, RH) = \frac{a \cdot T}{b + T} + \ln\left(\frac{RH}{100}\right)
$$

with constants $a = 17.27$ and $b = 237.7°C$.

**Physical Significance**: Dew point indicates moisture content and affects both thermal comfort and HVAC load.

**b) Thermal Gradient (ΔT)**

$$
\Delta T = \bar{T}_{indoor} - T_{outdoor}
$$

where $\bar{T}_{indoor}$ is the average of all indoor temperature sensors.

**Physical Significance**: Directly proportional to heat loss/gain through the building envelope (Fourier's law).

**c) Heat Index (Apparent Temperature)**

Simplified Rothfusz regression:

$$
HI = c_1 + c_2 T + c_3 RH + c_4 T \cdot RH + c_5 T^2 + c_6 RH^2 + ...
$$

**Physical Significance**: Combines temperature and humidity to represent perceived thermal comfort.

#### 3.2.3 Lag Features (Thermal Inertia)

Building thermal mass creates temporal dependencies:

$$
E(t) = f(E(t-1), E(t-2), ..., E(t-k), X(t))
$$

**Implementation**:
```python
for lag in [1, 2, 3, 6]:  # 10min, 20min, 30min, 1hour
    df[f'Appliances_lag{lag}'] = df['Appliances'].shift(lag)
    df[f'T_indoor_lag{lag}'] = df['T_indoor_avg'].shift(lag)
```

**Justification**: Building thermal time constants typically range from 1-6 hours, making lag features up to t-6 (1 hour at 10-min resolution) physically meaningful.

#### 3.2.4 Rolling Statistics

Capture smoothed trends and variability:

```python
for window in [6, 12]:  # 1-hour, 2-hour windows
    df[f'E_roll{window}_mean'] = df['Appliances'].rolling(window).mean()
    df[f'E_roll{window}_std'] = df['Appliances'].rolling(window).std()
```

#### 3.2.5 Interaction Features

Cross-feature interactions capture non-linear physics:

```python
df['T_RH_interaction'] = df['T_indoor_avg'] * df['RH_indoor_avg']
df['DeltaT_wind'] = df['DeltaT'] * df['Windspeed']  # Convective heat loss
```

### 3.3 Feature Engineering Summary

| Category | Features | Count | Physical Basis |
|----------|----------|-------|----------------|
| Cyclical Temporal | hour_sin/cos, dow_sin/cos, month_sin/cos | 6 | Periodicity preservation |
| Thermodynamic | Tdp, ΔT, HeatIndex | 12 | Heat transfer laws |
| Lag | E_lag1-6, T_lag1-6 | 12 | Thermal inertia |
| Rolling | mean, std (1h, 2h) | 8 | Trend smoothing |
| Interaction | T×RH, ΔT×wind | 4 | Non-linear coupling |
| **Total Engineered** | | **42** | |

---

## 4. Machine Learning Models

### 4.1 Model Selection Rationale

| Model | Strengths | Limitations | Role in Study |
|-------|-----------|-------------|---------------|
| **Quantile TabNet** | Interpretability, uncertainty | Complexity | SOTA benchmark |
| **Stacking Ensemble** | Robustness, accuracy | Black-box | Primary predictor |
| XGBoost | Fast, accurate | Overfitting risk | Base learner |
| LightGBM | Memory efficient | Less interpretable | Base learner |
| Random Forest | Stable | Less accurate | Baseline |
| MLP | Non-linear | Requires tuning | Deep learning baseline |

### 4.2 Quantile TabNet Architecture

#### 4.2.1 Architecture Overview

TabNet (Arik & Pfister, 2021) uses sequential attention for sparse feature selection:

```
Input → BN → [Step 1] → [Step 2] → ... → [Step N] → Output
              ↓           ↓                 ↓
           Attention   Attention        Attention
              ↓           ↓                 ↓
           Mask 1      Mask 2           Mask N
```

#### 4.2.2 Key Components

**a) Sparse Attention Mechanism**

At each decision step $i$:

$$
M[i] = \text{sparsemax}(P[i-1] \cdot h_i(a[i-1]))
$$

where $P[i-1]$ is the prior scale (feature reuse penalty) and $h_i$ is a learnable mapping.

**b) Feature Transformer**

$$
d[i] = \sum_{j=1}^{N_a} \text{ReLU}(BN(f_{c,j}^{[i]}(M[i] \odot x)))
$$

**c) Quantile Output Layer**

For uncertainty quantification, we predict three quantiles:

$$
\hat{y}_{q} = g_q(d_{agg}), \quad q \in \{0.025, 0.5, 0.975\}
$$

#### 4.2.3 Pinball Loss Function

$$
\mathcal{L}_q(y, \hat{y}_q) = \begin{cases}
q(y - \hat{y}_q) & \text{if } y \geq \hat{y}_q \\
(1-q)(\hat{y}_q - y) & \text{if } y < \hat{y}_q
\end{cases}
$$

Total loss:
$$
\mathcal{L} = \frac{1}{3} \sum_{q \in \{0.025, 0.5, 0.975\}} \mathcal{L}_q(y, \hat{y}_q)
$$

#### 4.2.4 Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| N_steps | 3 | Balance complexity/speed |
| N_a, N_d | 64 | Feature dimension |
| γ (relaxation) | 1.5 | Moderate feature reuse |
| Batch size | 256 | GPU memory optimization |
| Epochs | 100 | Early stopping enabled |
| Learning rate | 0.02 | Adam optimizer default |

### 4.3 Stacking Ensemble Architecture

#### 4.3.1 Two-Level Architecture

```
Level 0 (Base Learners):
├── XGBoost: Gradient boosting with regularization
└── LightGBM: Histogram-based gradient boosting

Level 1 (Meta-Learner):
└── Ridge Regression: Linear combination with L2 penalty
```

#### 4.3.2 Training Procedure

1. Split training data into K folds (K=3)
2. For each base learner:
   - Train on K-1 folds
   - Generate out-of-fold predictions
3. Stack predictions as meta-features
4. Train Ridge meta-learner on stacked predictions

#### 4.3.3 Mathematical Formulation

Final prediction:
$$
\hat{y}_{stack} = \alpha_0 + \alpha_1 \hat{y}_{XGB} + \alpha_2 \hat{y}_{LGBM}
$$

where $\alpha_i$ are learned by Ridge regression with regularization:

$$
\min_{\alpha} \|y - X_{meta}\alpha\|_2^2 + \lambda\|\alpha\|_2^2
$$

### 4.4 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Scale-dependent error |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Robust to outliers |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | Variance explained |
| PICP | $\frac{1}{n}\sum\mathbb{1}[y_i \in [\hat{y}_{0.025}, \hat{y}_{0.975}]]$ | Interval coverage |
| Winkler Score | $\bar{w} + \frac{2}{\alpha}\bar{p}$ | Interval sharpness |

---

## 5. Multi-Objective Optimization

### 5.1 Problem Formulation

#### 5.1.1 Decision Variables

Temperature setpoints for 8 thermal zones:
$$
\mathbf{x} = [T_1^{set}, T_2^{set}, ..., T_8^{set}], \quad T_i^{set} \in [18, 26]°C
$$

#### 5.1.2 Objective Functions

**Objective 1: Minimize Energy Consumption**

$$
f_1(\mathbf{x}) = E_{baseline} \cdot \left(1 - \eta_{setback} - \eta_{zone} - \eta_{schedule}\right)
$$

where:
- $\eta_{setback}$: Savings from temperature setback (5% per °C)
- $\eta_{zone}$: Savings from zone differentiation
- $\eta_{schedule}$: Savings from occupancy-based scheduling

**Objective 2: Minimize Thermal Discomfort**

$$
f_2(\mathbf{x}) = \frac{1}{N_{zones}} \sum_{i=1}^{N_{zones}} |PMV_i(T_i^{set})|
$$

Predicted Mean Vote (PMV) approximation:
$$
PMV \approx 0.4 \cdot \frac{T_{set} - T_{comfort}}{3}
$$

where $T_{comfort} = 21°C$.

#### 5.1.3 Constraints

$$
18°C \leq T_i^{set} \leq 26°C \quad \forall i \in \{1,...,8\}
$$

### 5.2 NSGA-II Algorithm

#### 5.2.1 Algorithm Parameters

| Parameter | Value |
|-----------|-------|
| Population size | 200 |
| Generations | 150 |
| Crossover | SBX (η=15, prob=0.9) |
| Mutation | Polynomial (η=20) |
| Selection | Binary tournament |

#### 5.2.2 Algorithm Steps

```
1. Initialize population P₀ randomly
2. Evaluate objectives f₁, f₂ for each individual
3. FOR generation = 1 to N_gen:
   a. Non-dominated sorting → Fronts F₁, F₂, ...
   b. Crowding distance assignment within fronts
   c. Selection using tournament (rank + crowding)
   d. Crossover (SBX) and mutation (PM)
   e. Combine parents and offspring
   f. Select next generation using NSGA-II ranking
4. Return Pareto-optimal set
```

### 5.3 TOPSIS Decision Making

#### 5.3.1 Method Overview

TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) selects a balanced solution from the Pareto front.

#### 5.3.2 Steps

1. **Normalize** objective values:
$$
r_{ij} = \frac{f_{ij}}{\sqrt{\sum_i f_{ij}^2}}
$$

2. **Weight** normalized values:
$$
v_{ij} = w_j \cdot r_{ij}
$$

3. **Identify** ideal ($A^+$) and anti-ideal ($A^-$) solutions

4. **Calculate** distances:
$$
D_i^+ = \sqrt{\sum_j (v_{ij} - v_j^+)^2}
$$

5. **Compute** relative closeness:
$$
C_i = \frac{D_i^-}{D_i^+ + D_i^-}
$$

6. **Select** solution with maximum $C_i$

#### 5.3.3 Weight Selection

Default: Equal weights (w₁ = w₂ = 0.5)

For energy-prioritized scenarios: w₁ = 0.6, w₂ = 0.4

---

## 6. Generalization & Transfer Learning

### 6.1 Universal Schema Adapter

#### 6.1.1 Problem

External datasets use different:
- Column naming conventions
- Sensor configurations
- Temporal resolutions

#### 6.1.2 Solution: `align_to_stacking_schema()`

```python
def align_to_stacking_schema(new_df, required_features, 
                              column_mappings, target_freq='10T'):
    """
    Aligns external datasets to trained model schema.
    
    Steps:
    1. Standardize column names (lowercase, underscores)
    2. Apply semantic column mappings
    3. Resample to target frequency
    4. Impute missing features using physics-based heuristics
    5. Validate output schema
    """
```

#### 6.1.3 Imputation Strategies

| Missing Feature | Imputation Method |
|-----------------|-------------------|
| Indoor temperature | Mean of available T sensors |
| Indoor humidity | Mean of available RH sensors |
| Outdoor temperature | Use T6 (building exterior) or weather API |
| Pressure | Standard atmospheric (760 mmHg) |
| Dew point | Magnus formula from T and RH |

### 6.2 Zero-Shot Transfer

**Scenario**: Apply pre-trained model directly to new building without retraining.

**Expected Outcome**: Degraded performance due to:
- Different building characteristics
- Different occupant behavior
- Different climate

**Metric**: Performance degradation relative to baseline

### 6.3 Few-Shot Adaptation

**Scenario**: Rapid deployment with minimal calibration data.

**Protocol**:
1. Take first 10% of new data as calibration set
2. Retrain only the meta-learner (Ridge regression)
3. Keep base learners frozen
4. Test on remaining 90%

**Rationale**: Base learners capture general energy patterns; meta-learner adapts to building-specific bias.

---

## 7. Sensitivity Analysis

### 7.1 Feature Importance Analysis

#### 7.1.1 Permutation Importance

Algorithm:
1. Compute baseline performance
2. For each feature:
   - Shuffle feature values
   - Recompute performance
   - Record performance drop
3. Rank features by importance

#### 7.1.2 Feature Group Ablation

Systematically remove feature groups and measure impact:

| Group Removed | RMSE Impact |
|---------------|-------------|
| Lag Features | +49.4% |
| Temporal | +3.7% |
| Indoor Temperature | +1.1% |
| Outdoor Weather | +0.9% |

**Finding**: Lag features are critical; removal increases RMSE by ~50%.

### 7.2 Hyperparameter Sensitivity

Test performance across hyperparameter ranges:

| Parameter | Range Tested | Optimal | Sensitivity |
|-----------|--------------|---------|-------------|
| n_estimators | [50, 300] | 150 | Low |
| max_depth | [3, 10] | 6 | Medium |
| learning_rate | [0.01, 0.3] | 0.1 | High |

### 7.3 Noise Robustness

Model performance under input perturbation:

| Noise Level (σ) | RMSE Degradation |
|-----------------|------------------|
| 1% | +0.2% |
| 5% | +1.4% |
| 10% | +6.3% |
| 20% | +19.5% |

**Finding**: Model maintains acceptable performance (<10% degradation) with up to 10% input noise.

### 7.4 Temporal Resolution Sensitivity

| Resolution | Samples | R² |
|------------|---------|-----|
| 10 min | 19,730 | 0.760 |
| 30 min | 6,574 | 0.675 |
| 1 hour | 3,285 | 0.668 |
| 4 hours | 818 | 0.533 |

**Finding**: Finer resolution preserves information and improves prediction.

---

## 8. Experimental Setup & Reproducibility

### 8.1 Hardware Configuration

- CPU: Intel Xeon / AMD EPYC (multi-core)
- RAM: 32 GB minimum
- GPU: Optional (CUDA-compatible for TabNet)

### 8.2 Software Environment

```
Python 3.8+
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.1.0
torch==2.0.0
pytorch-tabnet==4.1.0
pymoo==0.6.0
matplotlib==3.7.0
seaborn==0.12.0
```

### 8.3 Random Seed Protocol

```python
import numpy as np
import torch
import random

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
```

### 8.4 Data Split Protocol

- Training: 75%
- Testing: 25%
- Validation (TabNet): 20% of training
- Cross-validation: 3-fold (Stacking)

---

## 9. Statistical Validation

### 9.1 Significance Testing

Paired t-test comparing TabNet vs. Stacking:

$$
t = \frac{\bar{d}}{s_d / \sqrt{n}}
$$

where $\bar{d}$ is the mean difference in RMSE across folds.

### 9.2 Confidence Intervals

95% CI for RMSE:

$$
\text{CI} = \bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{n}}
$$

### 9.3 Cross-Validation Results

| Model | RMSE (mean ± std) | p-value vs. TabNet |
|-------|-------------------|-------------------|
| TabNet | 36.91 ± 0.74 | - |
| Stacking | 42.46 ± 0.85 | < 0.001 |
| XGBoost | 43.76 ± 1.31 | < 0.001 |

---

## 10. Limitations & Future Work

### 10.1 Limitations

1. **Single climate zone**: Training data from Belgian maritime climate
2. **Residential focus**: Limited commercial building validation
3. **Static optimization**: Does not consider real-time adaptation
4. **PMV simplification**: Uses approximate comfort model

### 10.2 Future Research Directions

1. **Multi-climate training**: Include diverse climate zones
2. **Real-time MPC**: Model Predictive Control integration
3. **Reinforcement learning**: Adaptive setpoint optimization
4. **Federated learning**: Privacy-preserving multi-building training

---

## Appendix A: Complete Feature List

| # | Feature | Type | Source |
|---|---------|------|--------|
| 1 | T1 | Original | ZigBee sensor |
| 2 | RH_1 | Original | ZigBee sensor |
| ... | ... | ... | ... |
| 67 | Appliances_roll12_std | Engineered | Rolling window |

## Appendix B: Mathematical Notation

| Symbol | Description | Units |
|--------|-------------|-------|
| $E$ | Energy consumption | Wh |
| $T$ | Temperature | °C |
| $RH$ | Relative humidity | % |
| $\Delta T$ | Thermal gradient | °C |
| $PMV$ | Predicted Mean Vote | - |
| $\hat{y}$ | Predicted value | Wh |

---

**Document End**

*For questions or clarifications, contact: [author@institution.edu]*
