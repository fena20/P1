"""
Multi-Objective Optimization Framework for Building Energy Management
=====================================================================
Target Journal: Applied Energy (Q1)

This script implements:
1. Advanced preprocessing with lag features for thermal inertia modeling
2. Surrogate modeling with baseline and proposed stacking ensemble
3. Rigorous validation (5-fold CV, statistical significance testing)
4. Model interpretability (SHAP)
5. Multi-objective optimization (NSGA-II)
6. Comprehensive visualization and reporting

Author: Senior Data Scientist & Energy Researcher
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import lightgbm as lgb
import optuna
from scipy.stats import wilcoxon
import shap
import warnings
warnings.filterwarnings('ignore')

# For NSGA-II optimization
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.visualization.scatter import Scatter
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("Warning: pymoo not available. NSGA-II optimization will use a simplified implementation.")

# Set random seeds for reproducibility
np.random.seed(42)
RANDOM_STATE = 42

# ============================================================================
# SECTION 1: DATA LOADING & PREPROCESSING
# ============================================================================

def load_data(url="https://raw.githubusercontent.com/Fateme9977/P2/main/energydata_complete.csv"):
    """
    Load energy consumption dataset from GitHub repository.
    
    Scientific Justification: Direct loading from source ensures reproducibility
    and access to the latest version of the dataset used in the original study.
    """
    print("Loading data from GitHub repository...")
    try:
        df = pd.read_csv(url)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading from URL: {e}")
        print("Attempting to load from local file...")
        df = pd.read_csv("energydata_complete.csv")
        return df


def preprocess_data(df):
    """
    Comprehensive data preprocessing including datetime handling and feature engineering.
    
    Scientific Justification: Proper preprocessing ensures data quality and enables
    the model to capture temporal patterns essential for building energy prediction.
    """
    print("\n" + "="*80)
    print("SECTION 1: DATA PREPROCESSING & FEATURE ENGINEERING")
    print("="*80)
    
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # Handle datetime if present
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        # Extract temporal features (scientific justification: captures seasonal patterns)
        data['hour'] = data['date'].dt.hour
        data['day_of_week'] = data['date'].dt.dayofweek
        data['month'] = data['date'].dt.month
        data = data.drop('date', axis=1)
    
    # Identify target variable
    if 'Appliances' not in data.columns:
        raise ValueError("Target variable 'Appliances' not found in dataset")
    
    # Separate features and target
    y = data['Appliances'].values
    X = data.drop('Appliances', axis=1)
    
    # Remove any non-numeric columns that might cause issues
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    print(f"Original features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    
    return X, y, data


def create_lag_features(X, lag_periods=[1, 2], feature_subset=None):
    """
    Create lag features for temperature and humidity variables.
    
    Scientific Justification: Thermal inertia is a fundamental property of buildings.
    Lag features (t-1, t-2) capture the delayed response of energy consumption to
    temperature/humidity changes, which is critical for accurate energy prediction
    in building systems (standard practice in Applied Energy publications).
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    lag_periods : list
        List of lag periods to create (e.g., [1, 2] for t-1 and t-2)
    feature_subset : list
        Specific features to create lags for (None = auto-detect temp/humidity)
    """
    print("\nCreating lag features for thermal inertia modeling...")
    
    X_lagged = X.copy()
    
    # Auto-detect temperature and humidity features if not specified
    if feature_subset is None:
        # Common patterns for temperature and humidity features
        temp_features = [col for col in X.columns if any(x in col.lower() 
                        for x in ['temp', 't_out', 't_', 'temperature'])]
        humidity_features = [col for col in X.columns if any(x in col.lower() 
                            for x in ['rh', 'humidity', 'hum'])]
        feature_subset = temp_features + humidity_features
        
        # If no temp/humidity features found, use first few numeric features
        if len(feature_subset) == 0:
            print("Warning: No temperature/humidity features detected. Using first 5 numeric features for lag creation.")
            feature_subset = list(X.select_dtypes(include=[np.number]).columns[:5])
    
    if len(feature_subset) == 0:
        print("Warning: No features selected for lag creation. Skipping lag features.")
        return X_lagged
    
    print(f"Creating lag features for {len(feature_subset)} features: {feature_subset[:5]}...")
    
    # Create lag features
    for lag in lag_periods:
        for feature in feature_subset:
            if feature in X.columns:
                X_lagged[f'{feature}_lag_{lag}'] = X[feature].shift(lag)
    
    # Remove rows with NaN values created by lagging
    X_lagged = X_lagged.bfill().ffill()
    
    # Also shift target if needed (we'll handle this in the main function)
    print(f"Features after lag creation: {X_lagged.shape[1]}")
    print(f"Added {len(feature_subset) * len(lag_periods)} lag features")
    
    return X_lagged


# ============================================================================
# SECTION 2: SURROGATE MODELING
# ============================================================================

class StackingEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Stacking Ensemble Regressor combining XGBoost, LightGBM, and Extra Trees.
    
    Scientific Justification: Stacking ensembles leverage the complementary strengths
    of different algorithms. XGBoost excels at handling complex interactions,
    LightGBM provides fast and accurate gradient boosting, and Extra Trees offers
    robust variance reduction. The meta-learner (Linear Regression) learns optimal
    combinations, often achieving superior performance compared to individual models
    (common approach in top-tier energy journals).
    """
    
    def __init__(self, n_estimators_xgb=100, n_estimators_lgb=100, 
                 n_estimators_et=100, random_state=42):
        self.n_estimators_xgb = n_estimators_xgb
        self.n_estimators_lgb = n_estimators_lgb
        self.n_estimators_et = n_estimators_et
        self.random_state = random_state
        
        # Base models
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=n_estimators_xgb,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )
        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=n_estimators_lgb,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1
        )
        self.et_model = ExtraTreesRegressor(
            n_estimators=n_estimators_et,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Meta-learner
        from sklearn.linear_model import LinearRegression
        self.meta_learner = LinearRegression()
        
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit the stacking ensemble using out-of-fold predictions."""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        meta_features = np.zeros((X.shape[0], 3))
        
        # Train base models and generate meta-features
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx], \
                                       X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train base models
            self.xgb_model.fit(X_train_fold, y_train_fold)
            self.lgb_model.fit(X_train_fold, y_train_fold)
            self.et_model.fit(X_train_fold, y_train_fold)
            
            # Generate predictions for meta-features
            meta_features[val_idx, 0] = self.xgb_model.predict(X_val_fold)
            meta_features[val_idx, 1] = self.lgb_model.predict(X_val_fold)
            meta_features[val_idx, 2] = self.et_model.predict(X_val_fold)
        
        # Retrain base models on full dataset
        self.xgb_model.fit(X, y)
        self.lgb_model.fit(X, y)
        self.et_model.fit(X, y)
        
        # Train meta-learner
        self.meta_learner.fit(meta_features, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Generate predictions using the stacking ensemble."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get base model predictions
        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        et_pred = self.et_model.predict(X)
        
        # Stack predictions
        meta_features = np.column_stack([xgb_pred, lgb_pred, et_pred])
        
        # Meta-learner prediction
        return self.meta_learner.predict(meta_features)


def train_baseline_models(X_train, y_train, X_test, y_test):
    """
    Train baseline models: Random Forest and SVR.
    
    Scientific Justification: These models serve as benchmarks to demonstrate
    the improvement achieved by the proposed stacking ensemble. RF is a robust
    tree-based method, while SVR captures non-linear relationships, representing
    different modeling paradigms common in energy prediction literature.
    """
    print("\n" + "="*80)
    print("SECTION 2: SURROGATE MODELING - BASELINE MODELS")
    print("="*80)
    
    models = {}
    results = {}
    
    # Random Forest Baseline
    print("\nTraining Random Forest baseline...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=15
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    models['Random Forest'] = rf_model
    results['Random Forest'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'R2': r2_score(y_test, rf_pred),
        'MAE': mean_absolute_error(y_test, rf_pred),
        'predictions': rf_pred
    }
    print(f"RF - RMSE: {results['Random Forest']['RMSE']:.4f}, "
          f"R2: {results['Random Forest']['R2']:.4f}")
    
    # SVR Baseline (requires scaling)
    print("\nTraining SVR baseline...")
    scaler_svr = StandardScaler()
    X_train_scaled = scaler_svr.fit_transform(X_train)
    X_test_scaled = scaler_svr.transform(X_test)
    
    svr_model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    svr_model.fit(X_train_scaled, y_train)
    svr_pred = svr_model.predict(X_test_scaled)
    
    models['SVR'] = (svr_model, scaler_svr)
    results['SVR'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, svr_pred)),
        'R2': r2_score(y_test, svr_pred),
        'MAE': mean_absolute_error(y_test, svr_pred),
        'predictions': svr_pred
    }
    print(f"SVR - RMSE: {results['SVR']['RMSE']:.4f}, "
          f"R2: {results['SVR']['R2']:.4f}")
    
    return models, results


def optimize_stacking_model(X_train, y_train, n_trials=50):
    """
    Hyperparameter optimization for stacking ensemble using Optuna.
    
    Scientific Justification: Systematic hyperparameter tuning ensures optimal
    model performance. Optuna uses Tree-structured Parzen Estimator (TPE) for
    efficient Bayesian optimization, which is more effective than grid search
    for high-dimensional hyperparameter spaces (standard practice in ML research).
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION (Optuna)")
    print("="*80)
    
    def objective(trial):
        n_estimators_xgb = trial.suggest_int('n_estimators_xgb', 50, 300)
        n_estimators_lgb = trial.suggest_int('n_estimators_lgb', 50, 300)
        n_estimators_et = trial.suggest_int('n_estimators_et', 50, 300)
        
        model = StackingEnsembleRegressor(
            n_estimators_xgb=n_estimators_xgb,
            n_estimators_lgb=n_estimators_lgb,
            n_estimators_et=n_estimators_et,
            random_state=RANDOM_STATE
        )
        
        # Use cross-validation for robust evaluation
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, cv=kf, 
                                scoring='neg_mean_squared_error', n_jobs=1)
        return -scores.mean()  # Return positive RMSE
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest hyperparameters: {study.best_params}")
    print(f"Best CV RMSE: {np.sqrt(study.best_value):.4f}")
    
    # Train final model with best hyperparameters
    best_model = StackingEnsembleRegressor(
        n_estimators_xgb=study.best_params['n_estimators_xgb'],
        n_estimators_lgb=study.best_params['n_estimators_lgb'],
        n_estimators_et=study.best_params['n_estimators_et'],
        random_state=RANDOM_STATE
    )
    best_model.fit(X_train, y_train)
    
    return best_model, study.best_params


# ============================================================================
# SECTION 3: RIGOROUS VALIDATION
# ============================================================================

def cross_validation_robustness(model, X, y, n_splits=5):
    """
    Perform 5-fold cross-validation to assess model robustness.
    
    Scientific Justification: Cross-validation provides a robust estimate of
    model performance that is not sensitive to specific train/test splits.
    Reporting mean and standard deviation of RMSE demonstrates the stability
    and generalizability of the proposed method (essential for Q1 journal standards).
    """
    print("\n" + "="*80)
    print("SECTION 3: RIGOROUS VALIDATION - 5-FOLD CROSS-VALIDATION")
    print("="*80)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create a fresh model instance for each fold
        if isinstance(model, StackingEnsembleRegressor):
            fold_model = StackingEnsembleRegressor(
                n_estimators_xgb=model.n_estimators_xgb,
                n_estimators_lgb=model.n_estimators_lgb,
                n_estimators_et=model.n_estimators_et,
                random_state=RANDOM_STATE
            )
        else:
            fold_model = type(model)(**model.get_params())
        
        fold_model.fit(X_train_fold, y_train_fold)
        y_pred_fold = fold_model.predict(X_val_fold)
        rmse_fold = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
        cv_scores.append(rmse_fold)
        print(f"Fold {fold}: RMSE = {rmse_fold:.4f}")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"\nCross-Validation Results:")
    print(f"Mean RMSE: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Std RMSE: {cv_std:.4f}")
    
    return cv_scores, cv_mean, cv_std


def statistical_significance_test(y_true, y_pred_baseline, y_pred_proposed):
    """
    Perform Wilcoxon Signed-Rank Test for statistical significance.
    
    Scientific Justification: The Wilcoxon test is a non-parametric statistical
    test that determines if the improvement of the proposed method over the
    baseline is statistically significant (p < 0.05). This provides rigorous
    evidence of model superiority beyond simple metric comparisons (required
    for top-tier journal publications).
    """
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TEST (Wilcoxon Signed-Rank)")
    print("="*80)
    
    # Calculate prediction errors
    errors_baseline = np.abs(y_true - y_pred_baseline)
    errors_proposed = np.abs(y_true - y_pred_proposed)
    
    # Perform Wilcoxon test
    statistic, p_value = wilcoxon(errors_baseline, errors_proposed, 
                                  alternative='greater')
    
    print(f"\nWilcoxon Signed-Rank Test Results:")
    print(f"Test Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("✓ Statistically significant improvement (p < 0.05)")
        print("  The proposed model significantly outperforms the baseline.")
    else:
        print("✗ No statistically significant improvement (p >= 0.05)")
    
    return statistic, p_value


# ============================================================================
# SECTION 4: MODEL INTERPRETABILITY (XAI)
# ============================================================================

def shap_interpretability(model, X_train, X_test, feature_names=None):
    """
    SHAP (SHapley Additive exPlanations) analysis for model interpretability.
    
    Scientific Justification: SHAP values provide a unified framework for
    explaining model predictions by quantifying each feature's contribution.
    This is crucial for understanding which building parameters (e.g., T_out,
    RH_1) drive energy consumption, enabling actionable insights for energy
    management (increasingly required in Applied Energy publications).
    """
    print("\n" + "="*80)
    print("SECTION 4: MODEL INTERPRETABILITY (SHAP)")
    print("="*80)
    
    # Use a sample for faster computation
    X_sample = X_test.iloc[:100] if hasattr(X_test, 'iloc') else X_test[:100]
    
    # For stacking ensemble, use the XGBoost component for SHAP (most interpretable)
    if isinstance(model, StackingEnsembleRegressor):
        explainer_model = model.xgb_model
    else:
        explainer_model = model
    
    print("Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(explainer_model)
    shap_values = explainer.shap_values(X_sample)
    
    # Generate summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title("SHAP Summary Plot: Feature Importance for Energy Consumption", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    print("SHAP summary plot saved as 'shap_summary_plot.png'")
    
    # Calculate mean absolute SHAP values for feature importance
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'Feature': X_sample.columns if hasattr(X_sample, 'columns') else 
                   [f'Feature_{i}' for i in range(X_sample.shape[1])],
        'Mean |SHAP Value|': mean_shap
    }).sort_values('Mean |SHAP Value|', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return shap_values, feature_importance


# ============================================================================
# SECTION 5: MULTI-OBJECTIVE OPTIMIZATION (NSGA-II)
# ============================================================================

class EnergyOptimizationProblem(Problem):
    """
    Multi-objective optimization problem for building energy management.
    
    Scientific Justification: NSGA-II is a well-established evolutionary algorithm
    for multi-objective optimization, widely used in energy system optimization
    (Applied Energy, Energy & Buildings). It efficiently explores the Pareto front
    representing trade-offs between energy consumption and thermal comfort.
    """
    
    def __init__(self, surrogate_model, X_train, feature_names, 
                 temp_features=None, rh_features=None, 
                 ideal_temp=21.0, ideal_rh=50.0, w1=1.0, w2=1.0):
        """
        Initialize the optimization problem.
        
        Parameters:
        -----------
        surrogate_model : trained model
            Surrogate model for energy prediction
        X_train : DataFrame
            Training data for context
        feature_names : list
            Names of all features
        temp_features : list
            Indices/names of temperature features
        rh_features : list
            Indices/names of humidity features
        ideal_temp : float
            Ideal temperature (°C)
        ideal_rh : float
            Ideal relative humidity (%)
        w1, w2 : float
            Weights for discomfort calculation
        """
        self.surrogate_model = surrogate_model
        self.X_train = X_train
        self.feature_names = feature_names if isinstance(feature_names, list) else list(feature_names)
        self.ideal_temp = ideal_temp
        self.ideal_rh = ideal_rh
        self.w1 = w1
        self.w2 = w2
        
        # Identify temperature and humidity feature indices
        if temp_features is None:
            temp_features = [i for i, name in enumerate(self.feature_names) 
                           if any(x in name.lower() for x in ['temp', 't_out', 't_', 'temperature'])]
        if rh_features is None:
            rh_features = [i for i, name in enumerate(self.feature_names) 
                         if any(x in name.lower() for x in ['rh', 'humidity', 'hum'])]
        
        self.temp_indices = temp_features[:2] if len(temp_features) >= 2 else temp_features  # Use first 2 temp features
        self.rh_indices = rh_features[:2] if len(rh_features) >= 2 else rh_features  # Use first 2 RH features
        
        # Decision variables: thermostat setpoints (T1, T2) and optionally RH setpoints
        n_vars = len(self.temp_indices) + len(self.rh_indices)
        
        # Get bounds from training data
        temp_bounds = []
        rh_bounds = []
        
        if hasattr(X_train, 'iloc'):
            for idx in self.temp_indices:
                col_values = X_train.iloc[:, idx] if isinstance(idx, int) else X_train[idx]
                temp_bounds.append((max(18, col_values.min()), min(26, col_values.max())))
            for idx in self.rh_indices:
                col_values = X_train.iloc[:, idx] if isinstance(idx, int) else X_train[idx]
                rh_bounds.append((max(30, col_values.min()), min(70, col_values.max())))
        else:
            for idx in self.temp_indices:
                temp_bounds.append((max(18, X_train[:, idx].min()), min(26, X_train[:, idx].max())))
            for idx in self.rh_indices:
                rh_bounds.append((max(30, X_train[:, idx].min()), min(70, X_train[:, idx].max())))
        
        # Define problem bounds
        xl = [b[0] for b in temp_bounds] + [b[0] for b in rh_bounds]
        xu = [b[1] for b in temp_bounds] + [b[1] for b in rh_bounds]
        
        super().__init__(
            n_var=n_vars,
            n_obj=2,
            n_constr=0,
            xl=np.array(xl),
            xu=np.array(xu)
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives for a population of solutions.
        
        Objective 1: Minimize Energy Consumption (using surrogate model)
        Objective 2: Minimize Thermal Discomfort
        """
        n_pop = X.shape[0]
        energy_consumption = np.zeros(n_pop)
        discomfort = np.zeros(n_pop)
        
        # Create feature vectors for prediction
        if hasattr(self.X_train, 'iloc'):
            # Use median values from training data as baseline
            baseline = self.X_train.median().values
        else:
            baseline = np.median(self.X_train, axis=0)
        
        for i in range(n_pop):
            # Create modified feature vector
            feature_vector = baseline.copy()
            
            # Update decision variables (thermostat setpoints)
            n_temp = len(self.temp_indices)
            for j, temp_idx in enumerate(self.temp_indices):
                feature_vector[temp_idx] = X[i, j]
            for j, rh_idx in enumerate(self.rh_indices):
                feature_vector[rh_idx] = X[i, n_temp + j]
            
            # Objective 1: Predict energy consumption
            feature_df = pd.DataFrame([feature_vector], columns=self.feature_names)
            energy_consumption[i] = self.surrogate_model.predict(feature_df)[0]
            
            # Objective 2: Calculate discomfort
            temp_values = [X[i, j] for j in range(n_temp)]
            rh_values = [X[i, n_temp + j] for j in range(len(self.rh_indices))]
            
            temp_discomfort = sum([self.w1 * abs(t - self.ideal_temp) for t in temp_values])
            rh_discomfort = sum([self.w2 * abs(rh - self.ideal_rh) for rh in rh_values])
            discomfort[i] = temp_discomfort + rh_discomfort
        
        out["F"] = np.column_stack([energy_consumption, discomfort])


def nsga2_optimization(surrogate_model, X_train, feature_names, 
                      n_gen=50, pop_size=100):
    """
    Perform NSGA-II multi-objective optimization.
    
    Scientific Justification: NSGA-II efficiently finds the Pareto-optimal set
    of solutions representing the best trade-offs between energy consumption
    and thermal comfort. This enables decision-makers to select solutions based
    on their priorities (eco-centric, comfort-centric, or balanced).
    """
    print("\n" + "="*80)
    print("SECTION 5: MULTI-OBJECTIVE OPTIMIZATION (NSGA-II)")
    print("="*80)
    
    if not PYMOO_AVAILABLE:
        print("Warning: pymoo not available. Using simplified optimization...")
        return simplified_optimization(surrogate_model, X_train, feature_names)
    
    # Create optimization problem
    problem = EnergyOptimizationProblem(
        surrogate_model=surrogate_model,
        X_train=X_train,
        feature_names=feature_names
    )
    
    # Initialize NSGA-II algorithm
    algorithm = NSGA2(pop_size=pop_size)
    
    print(f"Running NSGA-II optimization ({n_gen} generations, population size: {pop_size})...")
    res = minimize(problem, algorithm, ('n_gen', n_gen), verbose=False, seed=RANDOM_STATE)
    
    # Extract Pareto front
    pareto_front = res.F
    pareto_solutions = res.X
    
    print(f"\nOptimization completed!")
    print(f"Pareto front solutions: {len(pareto_front)}")
    print(f"Energy range: [{pareto_front[:, 0].min():.2f}, {pareto_front[:, 0].max():.2f}]")
    print(f"Discomfort range: [{pareto_front[:, 1].min():.2f}, {pareto_front[:, 1].max():.2f}]")
    
    return pareto_front, pareto_solutions, problem


def simplified_optimization(surrogate_model, X_train, feature_names):
    """
    Simplified optimization when pymoo is not available.
    Uses random sampling to approximate Pareto front.
    """
    print("Using simplified optimization approach...")
    
    problem = EnergyOptimizationProblem(
        surrogate_model=surrogate_model,
        X_train=X_train,
        feature_names=feature_names
    )
    
    # Random sampling
    n_samples = 1000
    X_random = np.random.uniform(problem.xl, problem.xu, (n_samples, problem.n_var))
    
    out = {}
    problem._evaluate(X_random, out)
    objectives = out["F"]
    
    # Find non-dominated solutions (simple Pareto filter)
    pareto_mask = np.ones(n_samples, dtype=bool)
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                if (objectives[j, 0] <= objectives[i, 0] and 
                    objectives[j, 1] <= objectives[i, 1] and
                    (objectives[j, 0] < objectives[i, 0] or 
                     objectives[j, 1] < objectives[i, 1])):
                    pareto_mask[i] = False
                    break
    
    pareto_front = objectives[pareto_mask]
    pareto_solutions = X_random[pareto_mask]
    
    print(f"Found {len(pareto_front)} Pareto-optimal solutions")
    
    return pareto_front, pareto_solutions, problem


def find_knee_point(pareto_front):
    """
    Find the knee point (balanced solution) on the Pareto front.
    
    Scientific Justification: The knee point represents the solution with the
    best trade-off, where a small improvement in one objective requires a
    large sacrifice in the other. This is identified using the maximum
    perpendicular distance from the line connecting extreme points.
    """
    # Normalize objectives
    energy_norm = (pareto_front[:, 0] - pareto_front[:, 0].min()) / \
                  (pareto_front[:, 0].max() - pareto_front[:, 0].min() + 1e-10)
    discomfort_norm = (pareto_front[:, 1] - pareto_front[:, 1].min()) / \
                      (pareto_front[:, 1].max() - pareto_front[:, 1].min() + 1e-10)
    
    # Find extreme points
    eco_idx = np.argmin(energy_norm)  # Minimum energy
    comfort_idx = np.argmin(discomfort_norm)  # Minimum discomfort
    
    # Line connecting extremes
    line_vec = np.array([energy_norm[comfort_idx] - energy_norm[eco_idx],
                        discomfort_norm[comfort_idx] - discomfort_norm[eco_idx]])
    line_vec = line_vec / (np.linalg.norm(line_vec) + 1e-10)
    
    # Calculate perpendicular distances
    distances = []
    for i in range(len(pareto_front)):
        point_vec = np.array([energy_norm[i] - energy_norm[eco_idx],
                             discomfort_norm[i] - discomfort_norm[eco_idx]])
        # Perpendicular distance
        proj = np.dot(point_vec, line_vec)
        perp_vec = point_vec - proj * line_vec
        distances.append(np.linalg.norm(perp_vec))
    
    knee_idx = np.argmax(distances)
    return knee_idx


# ============================================================================
# SECTION 6: VISUALIZATION & REPORTING
# ============================================================================

def plot_pareto_front(pareto_front, pareto_solutions=None, problem=None):
    """
    Visualize the Pareto front with highlighted solutions.
    
    Scientific Justification: Visualization of the Pareto front enables clear
    understanding of the energy-comfort trade-off, facilitating informed
    decision-making for building energy management strategies.
    """
    print("\n" + "="*80)
    print("SECTION 6: VISUALIZATION & REPORTING")
    print("="*80)
    
    plt.figure(figsize=(12, 8))
    
    # Plot Pareto front
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], 
               alpha=0.6, s=50, label='Pareto Front', color='blue')
    
    # Find and highlight key solutions
    eco_idx = np.argmin(pareto_front[:, 0])  # Minimum energy
    comfort_idx = np.argmin(pareto_front[:, 1])  # Minimum discomfort
    knee_idx = find_knee_point(pareto_front)  # Balanced solution
    
    plt.scatter(pareto_front[eco_idx, 0], pareto_front[eco_idx, 1],
               s=300, marker='*', color='green', label='Eco-Centric', zorder=5)
    plt.scatter(pareto_front[comfort_idx, 0], pareto_front[comfort_idx, 1],
               s=300, marker='*', color='red', label='Comfort-Centric', zorder=5)
    plt.scatter(pareto_front[knee_idx, 0], pareto_front[knee_idx, 1],
               s=300, marker='*', color='orange', label='Balanced (Knee Point)', zorder=5)
    
    plt.xlabel('Energy Consumption (Wh)', fontsize=12, fontweight='bold')
    plt.ylabel('Thermal Discomfort Index', fontsize=12, fontweight='bold')
    plt.title('Pareto Front: Energy vs. Thermal Discomfort Trade-off', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pareto_front.png', dpi=300, bbox_inches='tight')
    print("Pareto front plot saved as 'pareto_front.png'")
    
    # Print solution details
    print("\n" + "-"*80)
    print("KEY SOLUTIONS ON PARETO FRONT:")
    print("-"*80)
    print(f"\n1. Eco-Centric Solution:")
    print(f"   Energy: {pareto_front[eco_idx, 0]:.2f} Wh")
    print(f"   Discomfort: {pareto_front[eco_idx, 1]:.2f}")
    
    print(f"\n2. Comfort-Centric Solution:")
    print(f"   Energy: {pareto_front[comfort_idx, 0]:.2f} Wh")
    print(f"   Discomfort: {pareto_front[comfort_idx, 1]:.2f}")
    
    print(f"\n3. Balanced (Knee Point) Solution:")
    print(f"   Energy: {pareto_front[knee_idx, 0]:.2f} Wh")
    print(f"   Discomfort: {pareto_front[knee_idx, 1]:.2f}")
    
    if pareto_solutions is not None:
        print(f"\nDecision Variables (Balanced Solution):")
        if problem is not None:
            n_temp = len(problem.temp_indices)
            print(f"   Temperature setpoints: {pareto_solutions[knee_idx, :n_temp]}")
            print(f"   Humidity setpoints: {pareto_solutions[knee_idx, n_temp:]}")
    
    plt.show()


def create_comparison_table(results_dict):
    """
    Create a comprehensive comparison table of all models.
    
    Scientific Justification: A clear comparison table enables readers to
    quickly assess model performance across multiple metrics, supporting
    the claim of improved performance by the proposed method.
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    
    comparison_data = []
    for model_name, metrics in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'RMSE': f"{metrics['RMSE']:.4f}",
            'R²': f"{metrics['R2']:.4f}",
            'MAE': f"{metrics['MAE']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Save to CSV
    comparison_df.to_csv('model_comparison.csv', index=False)
    print("\nComparison table saved as 'model_comparison.csv'")
    
    return comparison_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function orchestrating the entire workflow.
    """
    print("="*80)
    print("MULTI-OBJECTIVE OPTIMIZATION FRAMEWORK FOR BUILDING ENERGY MANAGEMENT")
    print("Target Journal: Applied Energy (Q1)")
    print("="*80)
    
    # Step 1: Load and preprocess data
    df = load_data()
    X, y, data = preprocess_data(df)
    
    # Create lag features for thermal inertia
    X_lagged = create_lag_features(X, lag_periods=[1, 2])
    
    # After filling NaNs, data lengths should match
    # Ensure alignment (drop first rows where lag features are less reliable)
    max_lag = max([1, 2])  # Maximum lag period
    y_aligned = y[max_lag:]  # Drop first max_lag rows from target
    X_lagged = X_lagged.iloc[max_lag:].reset_index(drop=True)
    
    # Ensure same length
    min_len = min(len(X_lagged), len(y_aligned))
    X_lagged = X_lagged.iloc[:min_len]
    y_aligned = y_aligned[:min_len]
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_lagged, y_aligned, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 2: Train baseline models
    baseline_models, baseline_results = train_baseline_models(
        X_train, y_train, X_test, y_test
    )
    
    # Step 3: Optimize and train proposed stacking model
    print("\n" + "="*80)
    print("TRAINING PROPOSED STACKING ENSEMBLE MODEL")
    print("="*80)
    stacking_model, best_params = optimize_stacking_model(
        X_train, y_train, n_trials=30  # Reduced for faster execution
    )
    
    # Evaluate proposed model
    stacking_pred = stacking_model.predict(X_test)
    stacking_results = {
        'RMSE': np.sqrt(mean_squared_error(y_test, stacking_pred)),
        'R2': r2_score(y_test, stacking_pred),
        'MAE': mean_absolute_error(y_test, stacking_pred),
        'predictions': stacking_pred
    }
    print(f"\nProposed Stacking Model - RMSE: {stacking_results['RMSE']:.4f}, "
          f"R2: {stacking_results['R2']:.4f}")
    
    # Combine all results
    all_results = {**baseline_results, 'Proposed Stacking': stacking_results}
    
    # Step 4: Cross-validation robustness check
    cv_scores, cv_mean, cv_std = cross_validation_robustness(
        stacking_model, X_lagged, y_aligned, n_splits=5
    )
    
    # Step 5: Statistical significance test
    best_baseline_name = min(baseline_results.keys(), 
                            key=lambda x: baseline_results[x]['RMSE'])
    best_baseline_pred = baseline_results[best_baseline_name]['predictions']
    
    statistic, p_value = statistical_significance_test(
        y_test, best_baseline_pred, stacking_pred
    )
    
    # Step 6: SHAP interpretability
    shap_values, feature_importance = shap_interpretability(
        stacking_model, X_train, X_test, feature_names=X_train.columns
    )
    
    # Step 7: Multi-objective optimization
    pareto_front, pareto_solutions, problem = nsga2_optimization(
        stacking_model, X_train, X_train.columns, n_gen=30, pop_size=50
    )
    
    # Step 8: Visualization
    plot_pareto_front(pareto_front, pareto_solutions, problem)
    comparison_table = create_comparison_table(all_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - shap_summary_plot.png: SHAP feature importance visualization")
    print("  - pareto_front.png: Pareto front with key solutions")
    print("  - model_comparison.csv: Performance comparison table")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
