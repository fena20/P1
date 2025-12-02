"""Data preparation for BDG2 / ASHRAE Great Energy Predictor III.

This module:
- Loads `building_metadata.csv`, `weather_train.csv`, and `train.csv` from `data/`.
- Filters for residential buildings (`primary_use == 'Lodging/residential'`).
- Merges energy, weather, and metadata.
- Performs basic cleaning, feature engineering, and normalization.
- Extracts simple occupancy proxies from diurnal patterns.
- Splits data into 2016 (train/val) and 2017 (test).
- Generates a summary statistics table as LaTeX (`tables/table1.tex`).

If the real dataset is not available, the module can fall back to generating a
small synthetic dataset with a compatible schema so that the rest of the
pipeline can still be executed for demonstration and debugging.
"""

from __future__ import annotations

import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tables")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def _load_csv_or_none(filename: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _generate_synthetic_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a small synthetic dataset mimicking the ASHRAE schema.

    This is used when the real Kaggle files are not available so that the
    rest of the pipeline (models, RL, plots) can still run end-to-end.
    """

    n_buildings = 10
    building_ids = np.arange(n_buildings)
    metadata = pd.DataFrame(
        {
            "building_id": building_ids,
            "primary_use": ["Lodging/residential"] * n_buildings,
            "square_feet": np.random.randint(300, 2000, size=n_buildings),
            "year_built": np.random.randint(1970, 2015, size=n_buildings),
            "floor_count": np.random.randint(1, 5, size=n_buildings),
        }
    )

    date_range = pd.date_range("2016-01-01", "2017-12-31 23:00:00", freq="H")
    n_hours = len(date_range)

    weather = pd.DataFrame(
        {
            "timestamp": np.tile(date_range, n_buildings),
            "site_id": np.repeat(building_ids, n_hours),
            "air_temperature": 10
            + 10 * np.sin(2 * np.pi * (np.tile(np.arange(n_hours), n_buildings) % 24) / 24),
            "cloud_coverage": np.random.randint(0, 9, size=n_buildings * n_hours),
            "dew_temperature": 5
            + 5 * np.cos(2 * np.pi * (np.tile(np.arange(n_hours), n_buildings) % 24) / 24),
        }
    )

    # Simple synthetic meter readings driven by temperature and time of day
    hours = np.tile(np.arange(n_hours), n_buildings)
    base_load = 5 + 2 * np.sin(2 * np.pi * (hours - 6) / 24).clip(min=0)
    temp_effect = 0.3 * (weather["air_temperature"].values - 15).clip(min=0)
    noise = np.random.normal(scale=0.5, size=n_buildings * n_hours)

    train = pd.DataFrame(
        {
            "building_id": np.repeat(building_ids, n_hours),
            "meter": 0,
            "timestamp": weather["timestamp"],
            "meter_reading": (base_load + temp_effect + noise).clip(min=0.1),
            "site_id": np.repeat(building_ids, n_hours),
        }
    )

    return metadata, weather, train


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Load raw BDG2/ASHRAE data or fall back to synthetic.

    Returns
    -------
    building_metadata, weather_train, train, is_synthetic
    """

    building_metadata = _load_csv_or_none("building_metadata.csv")
    weather_train = _load_csv_or_none("weather_train.csv")
    train = _load_csv_or_none("train.csv")

    if building_metadata is None or weather_train is None or train is None:
        # Fallback to synthetic data
        building_metadata, weather_train, train = _generate_synthetic_dataset()
        is_synthetic = True
    else:
        is_synthetic = False

    return building_metadata, weather_train, train, is_synthetic


def preprocess_data() -> Dict[str, pd.DataFrame]:
    """Prepare merged residential dataset and train/test splits.

    Returns a dictionary of key DataFrames and writes processed CSVs and
    LaTeX Table 1 with summary statistics.
    """

    bmeta, weather, train, is_synthetic = load_raw_data()

    # Ensure timestamp is datetime
    for df in (weather, train):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Filter to electricity meter (0) when available
    if "meter" in train.columns:
        train = train[train["meter"] == 0].copy()

    # Filter residential buildings
    if "primary_use" in bmeta.columns:
        residential_ids = bmeta.loc[
            bmeta["primary_use"].str.contains("Lodging/residential", case=False, na=False),
            "building_id",
        ].unique()
        train = train[train["building_id"].isin(residential_ids)].copy()
        bmeta = bmeta[bmeta["building_id"].isin(residential_ids)].copy()

    # Merge train with metadata
    df = train.merge(bmeta, on="building_id", how="left")

    # Merge with weather; align on site_id if available, otherwise building_id
    if "site_id" in df.columns and "site_id" in weather.columns:
        df = df.merge(weather, on=["site_id", "timestamp"], how="left")
    else:
        df = df.merge(weather, on="timestamp", how="left")

    # Basic cleaning: handle missing numerical features
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Feature engineering: time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Occupancy proxy: normalized diurnal pattern per building
    df["meter_reading_log1p"] = np.log1p(df["meter_reading"])
    diurnal = (
        df.groupby(["building_id", "hour"]) ["meter_reading_log1p"].mean().rename("diurnal_load")
    )
    df = df.join(diurnal, on=["building_id", "hour"])

    # Simple normalization of continuous features
    cont_features = [
        col
        for col in [
            "square_feet",
            "floor_count",
            "air_temperature",
            "dew_temperature",
            "cloud_coverage",
            "meter_reading_log1p",
            "diurnal_load",
        ]
        if col in df.columns
    ]

    for col in cont_features:
        mean = df[col].mean()
        std = df[col].std() or 1.0
        df[f"{col}_norm"] = (df[col] - mean) / std

    # Train/test split by year (2016 -> train/val, 2017 -> test)
    df["year"] = df["timestamp"].dt.year
    train_mask = df["year"] == 2016
    test_mask = df["year"] == 2017

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    # Save processed data
    df.to_csv(os.path.join(PROCESSED_DIR, "bdg2_residential_all.csv"), index=False)
    df_train.to_csv(os.path.join(PROCESSED_DIR, "bdg2_residential_train_2016.csv"), index=False)
    df_test.to_csv(os.path.join(PROCESSED_DIR, "bdg2_residential_test_2017.csv"), index=False)

    # Generate summary statistics table (Table 1)
    summary_cols = [
        "meter_reading",
        "air_temperature",
        "dew_temperature",
        "cloud_coverage",
        "square_feet",
        "floor_count",
    ]
    summary_cols = [c for c in summary_cols if c in df.columns]

    summary = df[summary_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    summary.index.name = "variable"

    latex_table_path = os.path.join(TABLES_DIR, "table1.tex")
    with open(latex_table_path, "w") as f:
        f.write(summary.to_latex(float_format=lambda x: f"{x:0.3f}"))

    return {
        "raw_metadata": bmeta,
        "raw_weather": weather,
        "raw_train": train,
        "merged_all": df,
        "train_2016": df_train,
        "test_2017": df_test,
        "is_synthetic": is_synthetic,
        "summary": summary,
    }


if __name__ == "__main__":
    results = preprocess_data()
    print("Preprocessing completed.")
    print(f"Synthetic data used: {results['is_synthetic']}")
    print("Train 2016 shape:", results["train_2016"].shape)
    print("Test 2017 shape:", results["test_2017"].shape)
    print("Summary statistics saved to tables/table1.tex")
