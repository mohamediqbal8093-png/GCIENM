from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-8


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    denom = np.maximum(np.abs(y_true), EPS)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred):
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), EPS)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def nrmse(y_true, y_pred):
    data_range = float(np.max(y_true) - np.min(y_true))
    if data_range < EPS:
        return float("nan")
    return rmse(y_true, y_pred) / data_range


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: true={y_true.shape}, pred={y_pred.shape}")
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "NRMSE": nrmse(y_true, y_pred),
    }


def evaluate_prediction_csv(path: str):
    df = pd.read_csv(path)
    true_cols = sorted([c for c in df.columns if c.startswith("true_")])
    pred_cols = sorted([c for c in df.columns if c.startswith("pred_")])
    if not true_cols or len(true_cols) != len(pred_cols):
        raise ValueError("Prediction CSV must contain paired true_<h> and pred_<h> columns.")
    y_true = df[true_cols].to_numpy()
    y_pred = df[pred_cols].to_numpy()
    return compute_metrics(y_true, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    metrics = evaluate_prediction_csv(args.predictions)
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
