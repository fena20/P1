"""
Comprehensive multi-objective optimization workflow for building energy management.

The script follows the workflow requested for an Applied Energy submission:
1. Advanced preprocessing with lag features capturing thermal inertia.
2. Baseline (RF, SVR) and proposed stacking surrogate (XGBoost + LightGBM + ExtraTrees).
3. Hyperparameter tuning via Optuna and robustness checks with 5-fold TimeSeriesSplit CV.
4. Statistical significance via Wilcoxon Signed-Rank test.
5. SHAP-based interpretability.
6. NSGA-II optimization (pymoo) to jointly minimize energy use and thermal discomfort.
7. Visualization of Pareto front with annotated operating strategies.

Author: GPT-5.1 Codex (Senior Data Scientist & Energy Researcher persona)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # Ensures plots can be rendered in headless environments

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_URL = "https://raw.githubusercontent.com/Fateme9977/P2/master/energydata_complete.csv"
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
CONTROL_FEATURES = ["T1", "T2", "T3", "RH_1", "RH_2"]
CONTROL_BOUNDS = {
    "T1": (18.0, 26.0),
    "T2": (18.0, 26.0),
    "T3": (18.0, 26.0),
    "RH_1": (30.0, 70.0),
    "RH_2": (30.0, 70.0),
}
DISCOMFORT_WEIGHTS = {"temperature": 0.7, "humidity": 0.3}
IDEAL_TEMPERATURE = 21.0
IDEAL_RH = 50.0


def ensure_directories() -> None:
    """Create required output directories."""
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """
    Load and preprocess the energy dataset.

    Scientific justification:
        - The dataset is timestamped; sorting by datetime preserves temporal causality.
        - Lag features approximate thermal inertia, which is essential for building physics.
    """
    df = pd.read_csv(DATA_URL, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")

    # Engineer lag features for temperature/humidity signals driving thermal dynamics
    lag_cols = [
        "T1",
        "T2",
        "T3",
        "T_out",
        "RH_1",
        "RH_2",
        "RH_3",
        "RH_out",
    ]
    for col in lag_cols:
        for lag in (1, 2):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df = df.dropna()
    return df


def time_based_split(
    df: pd.DataFrame, target: str, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronologically split the dataset to avoid look-ahead bias."""
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    return X_train, X_test, y_train, y_test


def build_baseline_models() -> Dict[str, Pipeline]:
    """Instantiate baseline regressors with scientifically grounded defaults."""
    svr_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svr",
                SVR(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    epsilon=0.1,
                ),
            ),
        ]
    )
    rf = RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
    )
    return {"RandomForest": rf, "SVR": svr_pipeline}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def evaluate_baselines(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Fit baseline models and collect performance metrics and errors."""
    records = []
    residuals = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        residuals[name] = y_test.values - preds
        metrics = compute_metrics(y_test, preds)
        metrics["model"] = name
        records.append(metrics)
    metrics_df = pd.DataFrame(records).set_index("model")
    return metrics_df, residuals


def build_stacking_model(params: Dict[str, float]) -> StackingRegressor:
    """Assemble stacking regressor given sampled hyperparameters."""
    xgb = XGBRegressor(
        n_estimators=int(params["xgb_n_estimators"]),
        max_depth=int(params["xgb_max_depth"]),
        learning_rate=params["xgb_lr"],
        subsample=params["xgb_subsample"],
        colsample_bytree=params["xgb_colsample"],
        min_child_weight=params["xgb_min_child"],
        reg_lambda=params["xgb_lambda"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    lgbm = LGBMRegressor(
        n_estimators=int(params["lgbm_n_estimators"]),
        num_leaves=int(params["lgbm_num_leaves"]),
        learning_rate=params["lgbm_lr"],
        subsample=params["lgbm_subsample"],
        colsample_bytree=params["lgbm_colsample"],
        min_child_samples=int(params["lgbm_min_child"]),
        reg_lambda=params["lgbm_lambda"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    extratrees = ExtraTreesRegressor(
        n_estimators=int(params["etr_n_estimators"]),
        max_depth=int(params["etr_max_depth"]),
        min_samples_split=int(params["etr_min_split"]),
        min_samples_leaf=int(params["etr_min_leaf"]),
        max_features=params["etr_max_features"],
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    meta_regressor = ExtraTreesRegressor(
        n_estimators=int(params["meta_n_estimators"]),
        max_depth=int(params["meta_max_depth"]),
        min_samples_leaf=int(params["meta_min_leaf"]),
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    stacking_model = StackingRegressor(
        estimators=[
            ("xgb", xgb),
            ("lgbm", lgbm),
            ("etr", extratrees),
        ],
        final_estimator=meta_regressor,
        n_jobs=-1,
        passthrough=False,
        cv=5,
    )
    return stacking_model


def tune_stacking_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 30,
) -> Tuple[StackingRegressor, optuna.study.Study]:
    """
    Use Optuna to minimize RMSE via 5-fold TimeSeriesSplit.

    Scientific justification:
        - TS-aware CV prevents leakage across folds for temporal data.
        - Optuna efficiently explores high-dimensional hyperparameter spaces.
    """
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 200, 800),
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
            "xgb_lr": trial.suggest_float("xgb_lr", 0.01, 0.2, log=True),
            "xgb_subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "xgb_colsample": trial.suggest_float("xgb_colsample", 0.6, 1.0),
            "xgb_min_child": trial.suggest_float("xgb_min_child", 1.0, 10.0),
            "xgb_lambda": trial.suggest_float("xgb_lambda", 1e-3, 10.0, log=True),
            "lgbm_n_estimators": trial.suggest_int("lgbm_n_estimators", 200, 800),
            "lgbm_num_leaves": trial.suggest_int("lgbm_num_leaves", 16, 128),
            "lgbm_lr": trial.suggest_float("lgbm_lr", 0.01, 0.2, log=True),
            "lgbm_subsample": trial.suggest_float("lgbm_subsample", 0.6, 1.0),
            "lgbm_colsample": trial.suggest_float("lgbm_colsample", 0.6, 1.0),
            "lgbm_min_child": trial.suggest_int("lgbm_min_child", 5, 60),
            "lgbm_lambda": trial.suggest_float("lgbm_lambda", 1e-3, 10.0, log=True),
            "etr_n_estimators": trial.suggest_int("etr_n_estimators", 200, 800),
            "etr_max_depth": trial.suggest_int("etr_max_depth", 6, 20),
            "etr_min_split": trial.suggest_int("etr_min_split", 2, 10),
            "etr_min_leaf": trial.suggest_int("etr_min_leaf", 1, 5),
            "etr_max_features": trial.suggest_float("etr_max_features", 0.3, 0.9),
            "meta_n_estimators": trial.suggest_int("meta_n_estimators", 200, 600),
            "meta_max_depth": trial.suggest_int("meta_max_depth", 6, 20),
            "meta_min_leaf": trial.suggest_int("meta_min_leaf", 1, 5),
        }
        model = build_stacking_model(params)
        rmse_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmse_scores.append(mean_squared_error(y_val, preds, squared=False))
        return float(np.mean(rmse_scores))

    study = optuna.create_study(direction="minimize", study_name="stacking_surrogate")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)
    best_model = build_stacking_model(study.best_params)
    best_model.fit(X_train, y_train)
    return best_model, study


def robustness_cv(
    model: StackingRegressor, X: pd.DataFrame, y: pd.Series
) -> Tuple[float, float]:
    """Compute mean and std RMSE across 5-fold TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model_clone = StackingRegressor(
            estimators=model.estimators,
            final_estimator=model.final_estimator,
            n_jobs=-1,
            passthrough=False,
            cv=5,
        )
        model_clone.fit(X_tr, y_tr)
        preds = model_clone.predict(X_val)
        rmse_scores.append(mean_squared_error(y_val, preds, squared=False))
    rmse_scores = np.array(rmse_scores)
    return float(rmse_scores.mean()), float(rmse_scores.std())


def run_wilcoxon_test(
    baseline_errors: np.ndarray, proposed_errors: np.ndarray
) -> Tuple[float, float]:
    """Wilcoxon Signed-Rank test on absolute errors."""
    stat, p_value = wilcoxon(np.abs(baseline_errors), np.abs(proposed_errors))
    return float(stat), float(p_value)


def run_shap_analysis(
    model: StackingRegressor, X_train: pd.DataFrame, output_path: Path, sample_size: int = 500
) -> None:
    """
    Kernel SHAP over a representative sample to rank drivers of energy use.

    Scientific justification:
        - Tree-based surrogates capture nonlinearities; SHAP summarizes feature contributions,
          aligning with transparency expectations of Applied Energy reviewers.
    """
    sample = shap.sample(X_train, sample_size, random_state=RANDOM_STATE)
    explainer = shap.Explainer(model.predict, sample)
    shap_values = explainer(sample)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, plot_type="dot", show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


@dataclass
class OptimizationArtifact:
    population: np.ndarray
    objectives: np.ndarray
    labels: Dict[str, Dict[str, float]]


def prepare_feature_template(df: pd.DataFrame) -> pd.Series:
    """
    Create a representative operating point for optimization.
    We use the median to stay robust against outliers.
    """
    return df.median()


def create_feature_vector(
    template: pd.Series, control_values: Dict[str, float]
) -> pd.Series:
    vec = template.copy()
    for feature, value in control_values.items():
        vec[feature] = value
        for lag in (1, 2):
            lag_col = f"{feature}_lag{lag}"
            if lag_col in vec.index:
                vec[lag_col] = value  # steady-state assumption for controllable setpoints
    return vec


class EnergyComfortProblem(ElementwiseProblem):
    """NSGA-II optimization problem definition."""

    def __init__(
        self,
        surrogate,
        template: pd.Series,
        control_features: List[str],
        bounds: Dict[str, Tuple[float, float]],
    ):
        self.surrogate = surrogate
        self.template = template
        self.control_features = control_features
        xl = [bounds[f][0] for f in control_features]
        xu = [bounds[f][1] for f in control_features]
        super().__init__(n_var=len(control_features), n_obj=2, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        control_values = {feat: val for feat, val in zip(self.control_features, x)}
        feature_vec = create_feature_vector(self.template, control_values)
        energy = float(self.surrogate.predict(feature_vec.to_frame().T)[0])
        temps = [control_values[f] for f in self.control_features if f.startswith("T")]
        rhs = [control_values[f] for f in self.control_features if f.startswith("RH")]
        temp_discomfort = np.mean([abs(t - IDEAL_TEMPERATURE) for t in temps])
        rh_discomfort = np.mean([abs(r - IDEAL_RH) for r in rhs])
        discomfort = (
            DISCOMFORT_WEIGHTS["temperature"] * temp_discomfort
            + DISCOMFORT_WEIGHTS["humidity"] * rh_discomfort
        )
        out["F"] = [energy, discomfort]


def run_nsga_optimization(
    surrogate: StackingRegressor,
    template: pd.Series,
    output_path: Path,
    population_size: int = 80,
    n_generations: int = 120,
) -> OptimizationArtifact:
    """
    Execute NSGA-II and return Pareto-optimal solutions with annotations.

    Scientific justification:
        - NSGA-II is a benchmark evolutionary algorithm for Pareto optimization in energy systems.
        - Population-based search captures the non-convex trade-offs between comfort and energy.
    """
    problem = EnergyComfortProblem(
        surrogate=surrogate,
        template=template,
        control_features=CONTROL_FEATURES,
        bounds=CONTROL_BOUNDS,
    )
    algorithm = NSGA2(pop_size=population_size)
    termination = get_termination("n_gen", n_generations)
    result = minimize(problem, algorithm, termination, seed=RANDOM_STATE, verbose=False)

    F = result.F
    pop = result.X
    # Identify eco-centric (min energy), comfort-centric (min discomfort), and knee (closest to ideal)
    energy_idx = np.argmin(F[:, 0])
    comfort_idx = np.argmin(F[:, 1])
    ideal = F.min(axis=0)
    norm_F = (F - ideal) / (F.max(axis=0) - ideal + 1e-8)
    distances = np.linalg.norm(norm_F, axis=1)
    knee_idx = np.argmin(distances)

    labels = {
        "Eco-centric": {"index": int(energy_idx), "energy": F[energy_idx, 0], "discomfort": F[energy_idx, 1]},
        "Comfort-centric": {
            "index": int(comfort_idx),
            "energy": F[comfort_idx, 0],
            "discomfort": F[comfort_idx, 1],
        },
        "Knee": {"index": int(knee_idx), "energy": F[knee_idx, 0], "discomfort": F[knee_idx, 1]},
    }

    plt.figure(figsize=(8, 6))
    plt.scatter(F[:, 0], F[:, 1], c="lightgray", label="Pareto Set")
    colors = {"Eco-centric": "green", "Comfort-centric": "blue", "Knee": "orange"}
    for label, info in labels.items():
        idx = info["index"]
        plt.scatter(F[idx, 0], F[idx, 1], s=120, color=colors[label], label=f"{label}")
        plt.annotate(
            label,
            (F[idx, 0], F[idx, 1]),
            textcoords="offset points",
            xytext=(5, -10),
            ha="left",
            fontsize=9,
        )
    plt.xlabel("Predicted Energy Use (Wh)")
    plt.ylabel("Thermal Discomfort Index")
    plt.title("NSGA-II Pareto Front: Energy vs. Discomfort")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return OptimizationArtifact(population=pop, objectives=F, labels=labels)


def main() -> None:
    ensure_directories()
    df = load_dataset()
    target = "Appliances"
    X_train, X_test, y_train, y_test = time_based_split(df, target)

    # === Baseline models ===
    baseline_models = build_baseline_models()
    baseline_metrics, baseline_residuals = evaluate_baselines(
        baseline_models, X_train, y_train, X_test, y_test
    )

    # === Proposed stacking surrogate with Optuna tuning ===
    stacking_model, study = tune_stacking_with_optuna(X_train, y_train, n_trials=25)
    stacking_preds = stacking_model.predict(X_test)
    stacking_metrics = compute_metrics(y_test, stacking_preds)
    stacking_metrics["model"] = "Stacking (Proposed)"

    # Aggregate metrics
    stacking_row = pd.Series(stacking_metrics).to_frame().T.set_index("model")
    metrics_table = pd.concat([baseline_metrics, stacking_row], axis=0)
    metrics_path = OUTPUT_DIR / "model_performance.csv"
    metrics_table.to_csv(metrics_path)

    # === Robustness check via 5-fold CV ===
    cv_mean_rmse, cv_std_rmse = robustness_cv(stacking_model, X_train, y_train)

    # === Statistical significance test ===
    best_baseline_name = baseline_metrics["RMSE"].idxmin()
    stat, p_value = run_wilcoxon_test(
        baseline_residuals[best_baseline_name], y_test.values - stacking_preds
    )

    # === SHAP interpretability ===
    shap_path = OUTPUT_DIR / "shap_summary.png"
    run_shap_analysis(stacking_model, X_train, shap_path)

    # === Multi-objective optimization via NSGA-II ===
    template = prepare_feature_template(df.drop(columns=[target]))
    pareto_path = OUTPUT_DIR / "pareto_front.png"
    moo_artifact = run_nsga_optimization(stacking_model, template, pareto_path)

    # Persist NSGA solutions for reproducibility
    pd.DataFrame(
        moo_artifact.population, columns=CONTROL_FEATURES
    ).assign(energy=moo_artifact.objectives[:, 0], discomfort=moo_artifact.objectives[:, 1]).to_csv(
        OUTPUT_DIR / "pareto_solutions.csv", index=False
    )

    summary = {
        "metrics_table": str(metrics_path),
        "cv_rmse_mean": cv_mean_rmse,
        "cv_rmse_std": cv_std_rmse,
        "wilcoxon_stat": stat,
        "wilcoxon_p_value": p_value,
        "best_baseline": best_baseline_name,
        "shap_plot": str(shap_path),
        "pareto_plot": str(pareto_path),
        "pareto_labels": moo_artifact.labels,
        "optuna_best_params": study.best_params,
    }
    summary_path = OUTPUT_DIR / "summary.json"
    pd.Series(summary).to_json(summary_path, indent=2)
    print("=== Experiment Summary ===")
    print(pd.Series(summary))
    print("\nDetailed metrics saved to:", metrics_path)
    print("SHAP plot saved to:", shap_path)
    print("Pareto plot saved to:", pareto_path)


if __name__ == "__main__":
    main()
