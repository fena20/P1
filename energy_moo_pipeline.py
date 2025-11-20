#!/usr/bin/env python3
"""
Comprehensive Multi-Objective Optimization workflow for building energy management.

The script follows a publication-grade methodology aligned with Applied Energy (Q1) expectations:
1. Advanced preprocessing with lagged thermal features to encode building inertia.
2. Baseline surrogate models (RF/SVR) and a tuned stacking ensemble (XGBoost, LightGBM, Extra Trees).
3. 5-fold cross-validation for robustness and Wilcoxon Signed-Rank testing for statistical significance.
4. SHAP-based interpretability to quantify feature contributions.
5. NSGA-II multi-objective optimization that balances energy consumption against thermal discomfort.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_SEED = 42
DATA_URL = "https://raw.githubusercontent.com/Fateme9977/P2/master/energydata_complete.csv"
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def _import_or_install(package: str, import_path: str | None = None, attr: str | None = None):
    """Imports a package, installing it on the fly if necessary (keeps workflow reproducible)."""
    module_name = import_path or package
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        module = importlib.import_module(module_name)
    if attr:
        return getattr(module, attr)
    return module


optuna = _import_or_install("optuna")
shap = _import_or_install("shap")
XGBRegressor = _import_or_install("xgboost", "xgboost", "XGBRegressor")
LGBMRegressor = _import_or_install("lightgbm", "lightgbm", "LGBMRegressor")
NSGA2 = _import_or_install("pymoo", "pymoo.algorithms.moo.nsga2", "NSGA2")
Problem = _import_or_install("pymoo", "pymoo.core.problem", "Problem")
minimize = _import_or_install("pymoo", "pymoo.optimize", "minimize")
get_termination = _import_or_install("pymoo", "pymoo.termination", "get_termination")


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Ensures deterministic behavior across numpy/sklearn for replicability (journal standard)."""
    np.random.seed(seed)


def load_energy_data(url: str = DATA_URL) -> pd.DataFrame:
    """Loads the open-source dataset and parses the time stamp."""
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    return df


def engineer_features(
    df: pd.DataFrame,
    target: str = "Appliances",
    lag_steps: Tuple[int, ...] = (1, 2),
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Creates calendar-derived features and lagged thermal/humidity signals to capture thermal inertia,
    a critical phenomenon in building physics.
    """
    engineered = df.copy()
    engineered["hour"] = engineered["date"].dt.hour
    engineered["dayofweek"] = engineered["date"].dt.dayofweek
    engineered["month"] = engineered["date"].dt.month
    engineered["is_weekend"] = (engineered["dayofweek"] >= 5).astype(int)

    thermal_cols = [c for c in engineered.columns if c.startswith(("T", "RH"))]
    for col in thermal_cols:
        for lag in lag_steps:
            engineered[f"{col}_lag{lag}"] = engineered[col].shift(lag)

    engineered.dropna(inplace=True)
    y = engineered[target].copy()
    X = engineered.drop(columns=[target, "date"])
    return X, y


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Time-ordered split prevents leakage from future to past (essential for energy forecasting)."""
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def build_baseline_models() -> Dict[str, Pipeline]:
    """Defines reproducible baseline surrogates to benchmark the proposed architecture."""
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "SVR": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svr",
                    SVR(
                        kernel="rbf",
                        C=10.0,
                        epsilon=0.1,
                        gamma="scale",
                    ),
                ),
            ]
        ),
    }
    return models


def evaluate_regressor(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Fits and evaluates a regressor while reporting RMSE/MAE/R2 (standard surrogate metrics)."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "RMSE": mean_squared_error(y_test, preds, squared=False),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
    }


def tune_stacking_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 30,
) -> Tuple[StackingRegressor, optuna.Study]:
    """Tunes the stacking ensemble with Optuna to balance bias-variance under limited trials."""
    X_array = X_train.to_numpy(dtype=np.float32)
    y_array = y_train.to_numpy(dtype=np.float32)
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    def build_stack(params: Dict[str, float]) -> StackingRegressor:
        xgb = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_estimators=int(params["xgb_n_estimators"]),
            max_depth=int(params["xgb_max_depth"]),
            learning_rate=params["xgb_lr"],
            subsample=params["xgb_subsample"],
            colsample_bytree=params["xgb_colsample"],
            reg_alpha=params["xgb_reg_alpha"],
            reg_lambda=params["xgb_reg_lambda"],
        )
        lgbm = LGBMRegressor(
            random_state=RANDOM_SEED,
            n_estimators=int(params["lgbm_n_estimators"]),
            num_leaves=int(params["lgbm_num_leaves"]),
            learning_rate=params["lgbm_lr"],
            subsample=params["lgbm_subsample"],
            colsample_bytree=params["lgbm_colsample"],
            min_child_samples=int(params["lgbm_min_child"]),
            reg_lambda=params["lgbm_reg_lambda"],
        )
        extra = ExtraTreesRegressor(
            n_estimators=int(params["et_n_estimators"]),
            max_depth=int(params["et_max_depth"]),
            min_samples_leaf=int(params["et_min_samples_leaf"]),
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        meta = RidgeCV(alphas=np.logspace(-3, 3, 13))
        stack = StackingRegressor(
            estimators=[
                ("xgb", xgb),
                ("lgbm", lgbm),
                ("extra", extra),
            ],
            final_estimator=meta,
            passthrough=True,
            n_jobs=-1,
        )
        return stack

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 300, 900),
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
            "xgb_lr": trial.suggest_float("xgb_lr", 0.01, 0.2, log=True),
            "xgb_subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "xgb_colsample": trial.suggest_float("xgb_colsample", 0.6, 1.0),
            "xgb_reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-5, 1.0, log=True),
            "xgb_reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-3, 5.0, log=True),
            "lgbm_n_estimators": trial.suggest_int("lgbm_n_estimators", 300, 900),
            "lgbm_num_leaves": trial.suggest_int("lgbm_num_leaves", 16, 256),
            "lgbm_lr": trial.suggest_float("lgbm_lr", 0.01, 0.2, log=True),
            "lgbm_subsample": trial.suggest_float("lgbm_subsample", 0.6, 1.0),
            "lgbm_colsample": trial.suggest_float("lgbm_colsample", 0.6, 1.0),
            "lgbm_min_child": trial.suggest_int("lgbm_min_child", 5, 60),
            "lgbm_reg_lambda": trial.suggest_float("lgbm_reg_lambda", 1e-3, 10.0, log=True),
            "et_n_estimators": trial.suggest_int("et_n_estimators", 300, 900),
            "et_max_depth": trial.suggest_int("et_max_depth", 6, 20),
            "et_min_samples_leaf": trial.suggest_int("et_min_samples_leaf", 1, 10),
        }
        model = build_stack(params)
        rmses = []
        for tr_idx, val_idx in cv.split(X_array):
            model.fit(X_array[tr_idx], y_array[tr_idx])
            preds = model.predict(X_array[val_idx])
            rmses.append(mean_squared_error(y_array[val_idx], preds, squared=False))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize", study_name="stacking_opt")
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    best_model = build_stack(study.best_params)
    best_model.fit(X_array, y_array)
    return best_model, study


def cross_validate_rmse(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    folds: int = 5,
) -> Tuple[float, float]:
    """5-fold CV underpins robustness claims (mean ± std RMSE)."""
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    X_array = X.to_numpy(dtype=np.float32)
    y_array = y.to_numpy(dtype=np.float32)
    rmses = []
    for tr_idx, val_idx in cv.split(X_array):
        model.fit(X_array[tr_idx], y_array[tr_idx])
        preds = model.predict(X_array[val_idx])
        rmses.append(mean_squared_error(y_array[val_idx], preds, squared=False))
    return float(np.mean(rmses)), float(np.std(rmses))


def wilcoxon_signed_rank(
    errors_baseline: np.ndarray,
    errors_proposed: np.ndarray,
) -> float:
    """Non-parametric Wilcoxon test validates statistical superiority (p < 0.05)."""
    stat, p_value = wilcoxon(errors_baseline, errors_proposed, alternative="greater")
    return float(p_value)


def compute_shap_summary(
    model,
    X_sample: pd.DataFrame,
    output_path: Path,
) -> None:
    """SHAP summary plot highlights the drivers of appliance energy use."""
    explainer = shap.Explainer(model.predict, X_sample)
    shap_values = explainer(X_sample)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


@dataclass
class OptimizationSetup:
    model: StackingRegressor
    feature_template: pd.Series
    control_t_cols: List[str]
    control_rh_cols: List[str]
    column_groups: Dict[str, List[str]]


class BuildingControlProblem(Problem):
    """NSGA-II optimization problem linking surrogate predictions with discomfort penalties."""

    def __init__(self, setup: OptimizationSetup):
        self.setup = setup
        n_var = len(setup.control_t_cols) + len(setup.control_rh_cols)
        temp_bounds = (18.0, 26.0)  # comfort-aware operational bounds
        rh_bounds = (30.0, 70.0)    # humidity control limits
        xl = np.array(
            [temp_bounds[0]] * len(setup.control_t_cols)
            + [rh_bounds[0]] * len(setup.control_rh_cols),
            dtype=np.float64,
        )
        xu = np.array(
            [temp_bounds[1]] * len(setup.control_t_cols)
            + [rh_bounds[1]] * len(setup.control_rh_cols),
            dtype=np.float64,
        )
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, X: np.ndarray, out: Dict, *args, **kwargs) -> None:
        features = []
        for candidate in X:
            temp_values = candidate[: len(self.setup.control_t_cols)]
            rh_values = candidate[len(self.setup.control_t_cols):]
            sample = self.setup.feature_template.copy()

            for value, col in zip(temp_values, self.setup.control_t_cols):
                for feat in self.setup.column_groups.get(col, []):
                    sample[feat] = value
            for value, col in zip(rh_values, self.setup.control_rh_cols):
                for feat in self.setup.column_groups.get(col, []):
                    sample[feat] = value

            features.append(sample.to_numpy(dtype=np.float64))

        features = np.vstack(features)
        energy = self.setup.model.predict(features)
        temp_vals = X[:, : len(self.setup.control_t_cols)]
        rh_vals = X[:, len(self.setup.control_t_cols):]
        discomfort = 0.6 * np.mean(np.abs(temp_vals - 21.0), axis=1) + 0.4 * np.mean(
            np.abs(rh_vals - 50.0), axis=1
        )
        out["F"] = np.column_stack([energy, discomfort])


def prepare_optimization(setup_model: StackingRegressor, X_ref: pd.DataFrame) -> OptimizationSetup:
    """Creates a steady-state feature template and grouping for NSGA-II."""
    feature_template = X_ref.median()
    control_t_cols = [col for col in ["T1", "T2", "T3", "T4"] if col in X_ref.columns]
    control_rh_cols = [col for col in ["RH_1", "RH_2", "RH_3"] if col in X_ref.columns]
    column_groups: Dict[str, List[str]] = {}
    for col in control_t_cols + control_rh_cols:
        column_groups[col] = [c for c in X_ref.columns if c.startswith(col)]
    return OptimizationSetup(
        model=setup_model,
        feature_template=feature_template,
        control_t_cols=control_t_cols,
        control_rh_cols=control_rh_cols,
        column_groups=column_groups,
    )


def run_nsga2(setup: OptimizationSetup, pop_size: int = 60, n_gen: int = 80):
    """Executes NSGA-II to reveal Pareto-optimal trade-offs."""
    problem = BuildingControlProblem(setup)
    algorithm = NSGA2(pop_size=pop_size, eliminate_duplicates=True)
    termination = get_termination("n_gen", n_gen)
    result = minimize(problem, algorithm, termination, seed=RANDOM_SEED, verbose=False)
    return result


def plot_pareto_front(result, output_path: Path) -> Dict[str, Dict[str, float]]:
    """Visualizes the Pareto frontier while highlighting eco/comfort/knee solutions."""
    F = result.F
    X = result.X
    energy = F[:, 0]
    discomfort = F[:, 1]

    eco_idx = int(np.argmin(energy))
    comfort_idx = int(np.argmin(discomfort))
    norm_energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-9)
    norm_discomfort = (discomfort - discomfort.min()) / (discomfort.max() - discomfort.min() + 1e-9)
    knee_idx = int(np.argmin((norm_energy**2 + norm_discomfort**2)))

    selected = {
        "Eco-Centric": {"energy": energy[eco_idx], "discomfort": discomfort[eco_idx], "decision": X[eco_idx]},
        "Comfort-Centric": {
            "energy": energy[comfort_idx],
            "discomfort": discomfort[comfort_idx],
            "decision": X[comfort_idx],
        },
        "Knee-Point": {"energy": energy[knee_idx], "discomfort": discomfort[knee_idx], "decision": X[knee_idx]},
    }

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=energy, y=discomfort, color="steelblue", label="Pareto Set")
    markers = {"Eco-Centric": "o", "Comfort-Centric": "s", "Knee-Point": "D"}
    colors = {"Eco-Centric": "darkgreen", "Comfort-Centric": "orange", "Knee-Point": "crimson"}
    for label, attrs in selected.items():
        plt.scatter(attrs["energy"], attrs["discomfort"], color=colors[label], marker=markers[label], s=120, label=label)

    plt.xlabel("Predicted Energy (Wh)")
    plt.ylabel("Discomfort Index")
    plt.title("NSGA-II Pareto Front: Energy vs. Discomfort")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return selected


def main() -> None:
    set_global_seed()
    print("Loading dataset...")
    df = load_energy_data()
    print(f"Dataset loaded with shape: {df.shape}")

    print("Engineering features (lagging thermal signals)...")
    X, y = engineer_features(df)
    print(f"Feature matrix: {X.shape}, Target vector: {y.shape}")

    print("Performing temporal train/test split...")
    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y)

    print("Training baseline models...")
    baseline_models = build_baseline_models()
    metrics_table = []
    predictions = {}
    for name, model in baseline_models.items():
        metrics = evaluate_regressor(model, X_train, y_train, X_test, y_test)
        metrics["Model"] = name
        metrics_table.append(metrics)
        predictions[name] = model.predict(X_test)
        print(f"{name} -> RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, R2: {metrics['R2']:.3f}")

    print("Tuning stacking ensemble with Optuna...")
    stacked_model, study = tune_stacking_model(X_train, y_train)
    stack_metrics = evaluate_regressor(stacked_model, X_train, y_train, X_test, y_test)
    stack_metrics["Model"] = "Stacking-Ensemble"
    metrics_table.append(stack_metrics)
    predictions["Stacking-Ensemble"] = stacked_model.predict(X_test)
    print(
        f"Stacking-Ensemble -> RMSE: {stack_metrics['RMSE']:.2f}, "
        f"MAE: {stack_metrics['MAE']:.2f}, R2: {stack_metrics['R2']:.3f}"
    )

    print("Running 5-fold cross-validation on the proposed model...")
    cv_mean, cv_std = cross_validate_rmse(stacked_model, X_train, y_train)
    print(f"5-Fold CV RMSE: {cv_mean:.2f} ± {cv_std:.2f}")

    print("Conducting Wilcoxon Signed-Rank Test...")
    best_baseline_name = min(
        [m for m in metrics_table if m["Model"] != "Stacking-Ensemble"], key=lambda d: d["RMSE"]
    )["Model"]
    baseline_errors = np.abs(y_test.to_numpy() - predictions[best_baseline_name])
    proposed_errors = np.abs(y_test.to_numpy() - predictions["Stacking-Ensemble"])
    p_value = wilcoxon_signed_rank(baseline_errors, proposed_errors)
    print(f"Wilcoxon p-value (baseline={best_baseline_name}): {p_value:.4f}")

    print("Computing SHAP values on a stratified subset...")
    shap_sample = X_train.sample(n=min(500, len(X_train)), random_state=RANDOM_SEED)
    shap_path = PLOTS_DIR / "shap_summary.png"
    compute_shap_summary(stacked_model, shap_sample, shap_path)
    print(f"SHAP summary plot saved to: {shap_path}")

    print("Preparing NSGA-II optimization problem...")
    optimization_setup = prepare_optimization(stacked_model, X_train)
    nsga_result = run_nsga2(optimization_setup)
    pareto_path = PLOTS_DIR / "pareto_front.png"
    highlighted = plot_pareto_front(nsga_result, pareto_path)
    print(f"Pareto front saved to: {pareto_path}")

    metrics_df = pd.DataFrame(metrics_table).set_index("Model").sort_values("RMSE")
    print("\nModel Performance Summary (lower RMSE is better):")
    print(metrics_df.round(3))
    print(f"\n5-Fold CV RMSE (Proposed): {cv_mean:.3f} ± {cv_std:.3f}")
    print(f"Wilcoxon p-value vs {best_baseline_name}: {p_value:.4f}")

    highlight_df = pd.DataFrame(
        {
            label: {
                "Predicted Energy (Wh)": attrs["energy"],
                "Discomfort Index": attrs["discomfort"],
            }
            for label, attrs in highlighted.items()
        }
    ).T
    print("\nRepresentative Operating Points from Pareto Frontier:")
    print(highlight_df.round(3))

    print("\nWorkflow complete. Figures stored in the 'plots' directory.")


if __name__ == "__main__":
    main()
