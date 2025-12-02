"""LSTM-based deep learning model for energy prediction and comfort proxies.

This module:
- Loads processed residential BDG2 data from `data/processed`.
- Builds a PyTorch LSTM model to predict next-step energy consumption.
- Approximates PMV/PPD comfort indices from simple thermal variables.
- Trains with MSE loss on energy and comfort proxies.
- Evaluates R^2, MAE, and RMSE on the 2017 test set.
- Generates Figure 1: predicted vs. actual energy scatter plot.

The code assumes that numpy, pandas, torch, matplotlib, and seaborn are
installed in the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import Dataset, DataLoader

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _load_processed() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(DATA_DIR, "bdg2_residential_train_2016.csv")
    test_path = os.path.join(DATA_DIR, "bdg2_residential_test_2017.csv")
    train = pd.read_csv(train_path, parse_dates=["timestamp"]) if os.path.exists(train_path) else None
    test = pd.read_csv(test_path, parse_dates=["timestamp"]) if os.path.exists(test_path) else None
    if train is None or test is None:
        raise FileNotFoundError(
            "Processed data not found. Run src/data_preparation.py first to generate processed CSVs."
        )
    return train, test


def approximate_pmv(air_temp_c: np.ndarray, rel_humidity: np.ndarray | None = None) -> np.ndarray:
    """Very simple PMV-like comfort proxy.

    This is *not* a full ISO 7730 implementation but a surrogate that
    penalizes deviations from a neutral temperature around 22°C.
    """

    t_neutral = 22.0
    if rel_humidity is None:
        rel_humidity = np.full_like(air_temp_c, 50.0)

    temp_dev = air_temp_c - t_neutral
    pmv = -0.2 * temp_dev  # arbitrary scaling for illustration
    return pmv


def pmv_to_ppd(pmv: np.ndarray) -> np.ndarray:
    """Convert PMV proxy to PPD using ISO 7730 functional form.

    PPD = 100 - 95 * exp(-0.03353 PMV^4 - 0.2179 PMV^2)
    """

    return 100 - 95 * np.exp(-0.03353 * np.power(pmv, 4) - 0.2179 * np.power(pmv, 2))


class EnergySequenceDataset(Dataset):
    """Sequence dataset for next-step energy prediction."""

    def __init__(self, df: pd.DataFrame, seq_len: int = 24):
        self.seq_len = seq_len

        # Sort by building and time to create meaningful sequences
        df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

        # Feature selection: normalized continuous variables and time features
        feature_cols = [
            col
            for col in df.columns
            if col.endswith("_norm") or col in ["hour", "dayofweek", "month"]
        ]
        self.features = df[feature_cols].astype("float32").values
        self.targets = df["meter_reading"].astype("float32").values

        # Pre-compute PMV from air temperature
        if "air_temperature" in df.columns:
            self.pmv = approximate_pmv(df["air_temperature"].values.astype("float32"))
            self.ppd = pmv_to_ppd(self.pmv)
        else:
            self.pmv = np.zeros_like(self.targets)
            self.ppd = np.full_like(self.targets, 50.0)

        self.feature_dim = self.features.shape[1]

    def __len__(self) -> int:
        return max(0, len(self.targets) - self.seq_len)

    def __getitem__(self, idx: int):
        x = self.features[idx : idx + self.seq_len]
        y_energy = self.targets[idx + self.seq_len]
        y_pmv = self.pmv[idx + self.seq_len]
        y_ppd = self.ppd[idx + self.seq_len]
        return (
            torch.from_numpy(x),
            torch.tensor([y_energy], dtype=torch.float32),
            torch.tensor([y_pmv], dtype=torch.float32),
            torch.tensor([y_ppd], dtype=torch.float32),
        )


class LSTMEnergyModel(nn.Module):
    """LSTM model predicting energy and comfort proxies."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_energy = nn.Linear(hidden_dim, 1)
        self.fc_pmv = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        energy = self.fc_energy(h_last)
        pmv = self.fc_pmv(h_last)
        return energy, pmv


@dataclass
class TrainingConfig:
    seq_len: int = 24
    batch_size: int = 32
    hidden_dim: int = 64
    num_layers: int = 2
    lr: float = 1e-3
    epochs: int = 5  # keep modest for demonstration; can increase to 50 for full runs
    lambda_pmv: float = 0.1  # weight for PMV loss relative to energy
    device: str = "cpu"


def train_model(config: TrainingConfig) -> Tuple[LSTMEnergyModel, dict]:
    train_df, test_df = _load_processed()

    train_ds = EnergySequenceDataset(train_df, seq_len=config.seq_len)
    test_ds = EnergySequenceDataset(test_df, seq_len=config.seq_len)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    model = LSTMEnergyModel(input_dim=train_ds.feature_dim, hidden_dim=config.hidden_dim, num_layers=config.num_layers)
    model.to(config.device)

    criterion_energy = nn.MSELoss()
    criterion_pmv = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history = {"train_loss": []}

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, y_energy, y_pmv, _ in train_loader:
            xb = xb.to(config.device)
            y_energy = y_energy.to(config.device)
            y_pmv = y_pmv.to(config.device)

            optimizer.zero_grad()
            pred_energy, pred_pmv = model(xb)
            loss_energy = criterion_energy(pred_energy, y_energy)
            loss_pmv = criterion_pmv(pred_pmv, y_pmv)
            loss = loss_energy + config.lambda_pmv * loss_pmv
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_ds)
        history["train_loss"].append(epoch_loss)
        print(f"Epoch {epoch+1}/{config.epochs} - train loss: {epoch_loss:.4f}")

    metrics = evaluate_model(model, test_loader, config)
    return model, {"history": history, "metrics": metrics}


def evaluate_model(model: LSTMEnergyModel, loader: DataLoader, config: TrainingConfig) -> dict:
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for xb, y_energy, _, _ in loader:
            xb = xb.to(config.device)
            y_energy = y_energy.to(config.device)

            pred_energy, _ = model(xb)
            all_true.append(y_energy.cpu().numpy().ravel())
            all_pred.append(pred_energy.cpu().numpy().ravel())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    print(f"Test R^2: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    # Figure 1: predicted vs actual
    sns.set_style("whitegrid")
    plt.style.use("seaborn-v0_8-paper")
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.3, edgecolor="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="Ideal")
    plt.xlabel("Actual energy consumption (kWh)")
    plt.ylabel("Predicted energy consumption (kWh)")
    plt.title("Figure 1: Predicted vs. actual energy consumption")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig1_pred_vs_actual.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return {"r2": r2, "mae": mae, "rmse": rmse, "fig1_path": fig_path}


if __name__ == "__main__":
    cfg = TrainingConfig()
    model, outputs = train_model(cfg)
    metrics = outputs["metrics"]
    print("Metrics:", metrics)
