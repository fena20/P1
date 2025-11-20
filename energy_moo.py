#!/usr/bin/env python3
"""
Comprehensive Multi-Objective Optimization framework for building energy management.

Author: GPT-5.1 Codex acting as Senior Data Scientist & Energy Researcher
Target journal standard: Applied Energy (Q1)
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import optuna
import shap
from scipy.stats import wilcoxon

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/Fateme9977/P2/master/energydata_complete.csv"
DATA_PATH = Path("data/energydata_complete.csv")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42
LAG_FEATURES = ["T1", "T2", "T3", "T_out", "RH_1", "RH_2", "RH_out"]
TEST_SIZE = 0.2
N_OPTUNA_TRIALS = 30
CV_FOLDS = 5  # Scientific justification: 5-fold CV balances bias-variance and is standard in Applied Energy-grade analyses.


def ensure_data() -> Path:
    """Download the dataset if it is not already present."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        import urllib.request

        print("[INFO] Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print(f"[INFO] Dataset downloaded to {DATA_PATH}")
    else:
        print("[INFO] Dataset already available.")
    return DATA_PATH


def load_and_preprocess(path: Path) -> pd.DataFrame:
    """Load the dataset and perform feature engineering."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset with shape: {df.shape}")

    # Convert date column and extract temporal features to capture occupancy and load cycles.
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Lag features to encode thermal inertia, aligning with building physics literature.
    for col in LAG_FEATURES:
        for lag in [1, 2]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df = df.dropna().reset_index(drop=True)
    print(f"[INFO] Dataset after lag feature creation: {df.shape}")
    return df


def train_test_split_time(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronological split to respect temporal ordering and avoid leakage."""
    feature_cols = [c for c in df.columns if c not in {"date", "Appliances"}]
    X = df[feature_cols]
    y = df["Appliances"]

    split_idx = int(len(df) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[INFO] Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """Fit a model and compute performance metrics."""
    print(f"[INFO] Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"[INFO] {name} RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")
    return {
        "model": model,
        "name": name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "preds": preds,
    }


def build_baseline_models(X_train, y_train, X_test, y_test) -> List[Dict[str, Any]]:
    """Train RF and SVR baselines."""
    baselines = []

    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    baselines.append(evaluate_model("RandomForest", rf, X_train, y_train, X_test, y_test))

    svr_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=50, epsilon=0.5, gamma="auto")),
        ]
    )
    baselines.append(evaluate_model("SVR", svr_pipeline, X_train, y_train, X_test, y_test))

    return baselines


def build_stacking_model(trial: optuna.Trial):
    """Construct stacking ensemble with trial-defined hyperparameters."""
    xgb = XGBRegressor(
        n_estimators=trial.suggest_int("xgb_n_estimators", 200, 600),
        max_depth=trial.suggest_int("xgb_max_depth", 3, 8),
        learning_rate=trial.suggest_float("xgb_eta", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("xgb_colsample", 0.6, 1.0),
        reg_lambda=trial.suggest_float("xgb_lambda", 1e-3, 10, log=True),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    lgbm = LGBMRegressor(
        n_estimators=trial.suggest_int("lgbm_n_estimators", 200, 800),
        num_leaves=trial.suggest_int("lgbm_num_leaves", 15, 120),
        learning_rate=trial.suggest_float("lgbm_lr", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("lgbm_subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("lgbm_colsample", 0.6, 1.0),
        reg_alpha=trial.suggest_float("lgbm_alpha", 1e-3, 5, log=True),
        reg_lambda=trial.suggest_float("lgbm_lambda", 1e-3, 5, log=True),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    extra = ExtraTreesRegressor(
        n_estimators=trial.suggest_int("extra_n_estimators", 400, 1000),
        max_depth=trial.suggest_int("extra_max_depth", 10, 40),
        min_samples_split=trial.suggest_int("extra_min_samples_split", 2, 10),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    meta = RidgeCV(alphas=np.logspace(-3, 3, 13))

    stacking = StackingRegressor(
        estimators=[
            ("xgb", xgb),
            ("lgbm", lgbm),
            ("extra", extra),
        ],
        final_estimator=meta,
        passthrough=True,
        n_jobs=-1,
    )

    return stacking


def tune_stacking(X_train, y_train):
    """Hyperparameter tuning via Optuna with RMSE-focused objective."""

    def objective(trial: optuna.Trial) -> float:
        model = build_stacking_model(trial)
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
        )
        mean_score = scores.mean()
        return -mean_score  # Optuna minimizes

    study = optuna.create_study(direction="minimize", study_name="stacking_rmse")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    print(f"[INFO] Best stacking RMSE (CV): {study.best_value:.3f}")
    print(f"[INFO] Best params: {json.dumps(study.best_params, indent=2)}")

    best_trial = optuna.trial.FixedTrial(study.best_params)
    best_model = build_stacking_model(best_trial)
    best_model.fit(X_train, y_train)

    return best_model, study


def cross_validate_model(model, X, y) -> Tuple[float, float]:
    """Compute 5-fold CV RMSE mean and std for robustness reporting."""
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )
    rmse_scores = -scores
    return rmse_scores.mean(), rmse_scores.std()


def statistical_significance(baseline_errors, stack_errors) -> float:
    """Wilcoxon Signed-Rank test between prediction errors."""
    stat, p_value = wilcoxon(baseline_errors, stack_errors)
    print(f"[INFO] Wilcoxon test statistic: {stat:.3f}, p-value: {p_value:.5f}")
    return p_value


def generate_shap_summary(model: StackingRegressor, X_train: pd.DataFrame):
    """Produce SHAP summary plot using the Extra Trees base learner."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    base_model = model.named_estimators_["extra"]
    explainer = shap.TreeExplainer(base_model)
    sample = X_train.sample(n=min(2000, len(X_train)), random_state=RANDOM_STATE)
    shap_values = explainer.shap_values(sample)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, show=False)
    shap_path = OUTPUT_DIR / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(shap_path, dpi=300)
    plt.close()
    print(f"[INFO] SHAP summary plot saved to {shap_path}")


class EnergyComfortProblem(Problem):
    """Custom NSGA-II problem leveraging the surrogate model."""

    def __init__(self, surrogate, feature_template: pd.Series):
        self.surrogate = surrogate
        self.template = feature_template
        xl = np.array([18.0, 18.0, 30.0])  # bounds for T1, T2, RH_1
        xu = np.array([26.0, 26.0, 70.0])
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        features = np.tile(self.template.values, (X.shape[0], 1))
        feature_df = pd.DataFrame(features, columns=self.template.index)
        feature_df["T1"] = X[:, 0]
        feature_df["T2"] = X[:, 1]
        feature_df["RH_1"] = X[:, 2]
        # Align lagged features with steady-state assumption.
        for lag in [1, 2]:
            feature_df[f"T1_lag{lag}"] = X[:, 0]
            feature_df[f"T2_lag{lag}"] = X[:, 1]
            feature_df[f"RH_1_lag{lag}"] = X[:, 2]

        energy = self.surrogate.predict(feature_df)
        discomfort = (
            np.abs(feature_df["T1"] - 21)
            + np.abs(feature_df["T2"] - 21)
            + 0.5 * np.abs(feature_df["RH_1"] - 50)
        )
        out["F"] = np.column_stack([energy, discomfort])


def run_nsga2(surrogate, X_reference: pd.Series):
    """Execute NSGA-II optimization."""
    problem = EnergyComfortProblem(surrogate, X_reference)
    algorithm = NSGA2(pop_size=120, eliminate_duplicates=True)
    termination = get_termination("n_gen", 100)
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=RANDOM_STATE,
        verbose=False,
    )

    pareto_F = result.F
    pareto_X = result.X
    print(f"[INFO] NSGA-II completed with {pareto_F.shape[0]} Pareto-optimal points.")
    return pareto_X, pareto_F


def identify_pareto_solutions(pareto_X, pareto_F):
    """Select eco-centric, comfort-centric, and knee-point solutions."""
    eco_idx = np.argmin(pareto_F[:, 0])
    comfort_idx = np.argmin(pareto_F[:, 1])

    min_energy, max_energy = pareto_F[:, 0].min(), pareto_F[:, 0].max()
    min_discomfort, max_discomfort = pareto_F[:, 1].min(), pareto_F[:, 1].max()
    normalized = np.column_stack(
        [
            (pareto_F[:, 0] - min_energy) / (max_energy - min_energy + 1e-9),
            (pareto_F[:, 1] - min_discomfort) / (max_discomfort - min_discomfort + 1e-9),
        ]
    )
    knee_idx = np.argmin(np.linalg.norm(normalized - 0, axis=1))

    return {
        "Eco-Centric": {"setpoints": pareto_X[eco_idx], "objectives": pareto_F[eco_idx]},
        "Comfort-Centric": {"setpoints": pareto_X[comfort_idx], "objectives": pareto_F[comfort_idx]},
        "Knee-Point": {"setpoints": pareto_X[knee_idx], "objectives": pareto_F[knee_idx]},
    }


def plot_pareto(pareto_F, highlighted):
    """Visualize Pareto front."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(pareto_F[:, 0], pareto_F[:, 1], c="skyblue", label="Pareto Front")
    colors = {"Eco-Centric": "green", "Comfort-Centric": "red", "Knee-Point": "orange"}
    for label, data in highlighted.items():
        energy, discomfort = data["objectives"]
        plt.scatter(energy, discomfort, c=colors[label], label=label, edgecolor="k", s=120)
        plt.annotate(label, (energy, discomfort), textcoords="offset points", xytext=(5, 5))

    plt.xlabel("Predicted Energy (Wh)")
    plt.ylabel("Discomfort Index")
    plt.title("NSGA-II Pareto Front: Energy vs. Thermal Discomfort")
    plt.legend()
    plt.grid(alpha=0.3)
    pareto_path = OUTPUT_DIR / "pareto_front.png"
    plt.tight_layout()
    plt.savefig(pareto_path, dpi=300)
    plt.close()
    print(f"[INFO] Pareto front plot saved to {pareto_path}")


def main():
    ensure_data()
    df = load_and_preprocess(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split_time(df)

    baselines = build_baseline_models(X_train, y_train, X_test, y_test)

    stacking_model, study = tune_stacking(X_train, y_train)
    stacking_results = evaluate_model("StackingEnsemble", stacking_model, X_train, y_train, X_test, y_test)

    cv_mean, cv_std = cross_validate_model(stacking_model, X_train, y_train)
    print(f"[INFO] 5-Fold CV RMSE (Stacking): mean={cv_mean:.3f}, std={cv_std:.3f}")

    models_summary = baselines + [stacking_results]
    metrics_df = pd.DataFrame(
        [
            {"Model": res["name"], "RMSE": res["rmse"], "MAE": res["mae"], "R2": res["r2"]}
            for res in models_summary
        ]
    ).sort_values(by="RMSE")
    print("\n=== Model Performance Summary ===")
    print(metrics_df.to_string(index=False))

    best_baseline = min(baselines, key=lambda x: x["rmse"])
    baseline_errors = np.abs(y_test.values - best_baseline["preds"])
    stack_errors = np.abs(y_test.values - stacking_results["preds"])
    p_value = statistical_significance(baseline_errors, stack_errors)

    generate_shap_summary(stacking_model, X_train)

    feature_template = X_train.mean()
    pareto_X, pareto_F = run_nsga2(stacking_model, feature_template)
    highlighted = identify_pareto_solutions(pareto_X, pareto_F)
    plot_pareto(pareto_F, highlighted)

    solution_rows = []
    for label, data in highlighted.items():
        solution_rows.append(
            {
                "Solution": label,
                "T1_set (°C)": data["setpoints"][0],
                "T2_set (°C)": data["setpoints"][1],
                "RH1_set (%)": data["setpoints"][2],
                "Pred_Energy (Wh)": data["objectives"][0],
                "Discomfort": data["objectives"][1],
            }
        )
    pareto_df = pd.DataFrame(solution_rows)
    print("\n=== Representative Pareto Solutions ===")
    print(pareto_df.to_string(index=False))

    report = {
        "cv_rmse_mean": cv_mean,
        "cv_rmse_std": cv_std,
        "wilcoxon_p_value": p_value,
        "model_metrics": metrics_df.to_dict(orient="records"),
        "pareto_solutions": solution_rows,
        "optuna_best_params": study.best_params,
    }
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    report_path = OUTPUT_DIR / "summary_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Summary report saved to {report_path}")


if __name__ == "__main__":
    main()
