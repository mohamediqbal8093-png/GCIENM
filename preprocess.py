from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class Standardizer:
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform_column(self, x: np.ndarray, column_index: int) -> np.ndarray:
        return x * self.std_[column_index] + self.mean_[column_index]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(cfg: dict) -> None:
    split = cfg["split"]
    fractions = [
        float(split["train_fraction"]),
        float(split["validation_fraction"]),
        float(split["test_fraction"]),
    ]
    if not np.isclose(sum(fractions), 1.0, atol=1e-6):
        raise ValueError("train_fraction + validation_fraction + test_fraction must equal 1.")
    if int(split["lookback"]) < 1 or int(split["horizon"]) < 1:
        raise ValueError("lookback and horizon must be positive.")
    if int(split["n_folds"]) < 1:
        raise ValueError("n_folds must be positive.")


def load_dataframe(cfg: dict) -> pd.DataFrame:
    data_cfg = cfg["data"]
    path = Path(data_cfg["csv_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Configure data.csv_path with the exact SCADA CSV used by the study."
        )
    df = pd.read_csv(path)
    ts = data_cfg["timestamp_column"]
    required = [ts] + list(dict.fromkeys(data_cfg["feature_columns"] + [data_cfg["target_column"]]))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df[ts] = pd.to_datetime(df[ts], errors="coerce")
    df = df.dropna(subset=[ts]).sort_values(ts).drop_duplicates(ts, keep="last").reset_index(drop=True)

    numeric_cols = list(dict.fromkeys(data_cfg["feature_columns"] + [data_cfg["target_column"]]))
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rule = data_cfg.get("resample_rule")
    if rule:
        df = (
            df.set_index(ts)[numeric_cols]
            .resample(rule)
            .mean()
            .reset_index()
        )

    limit = data_cfg.get("interpolation_limit", None)
    df = df.set_index(ts)
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="time", limit=limit, limit_direction="both"
    )
    df = df.reset_index()

    if bool(data_cfg.get("drop_remaining_nan", True)):
        df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    if len(df) < int(cfg["split"]["lookback"]) + int(cfg["split"]["horizon"]) + 10:
        raise ValueError("Dataset is too short for configured lookback/horizon.")
    return df


def make_supervised_arrays(
    df: pd.DataFrame,
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = cfg["data"]["feature_columns"]
    target_col = cfg["data"]["target_column"]
    lookback = int(cfg["split"]["lookback"])
    horizon = int(cfg["split"]["horizon"])

    x_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df[target_col].to_numpy(dtype=np.float32)

    xs, ys, origins = [], [], []
    for end in range(lookback - 1, len(df) - horizon):
        start = end - lookback + 1
        future_start = end + 1
        future_end = future_start + horizon
        xs.append(x_raw[start : end + 1])
        ys.append(y_raw[future_start:future_end])
        origins.append(end)

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(origins, dtype=np.int64),
    )


def chronological_holdout_indices(n_samples: int, cfg: dict) -> Dict[str, np.ndarray]:
    split = cfg["split"]
    n_train = int(np.floor(n_samples * float(split["train_fraction"])))
    n_val = int(np.floor(n_samples * float(split["validation_fraction"])))
    n_test = n_samples - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError("Split creates an empty train/validation/test subset.")
    return {
        "train": np.arange(0, n_train),
        "val": np.arange(n_train, n_train + n_val),
        "test": np.arange(n_train + n_val, n_samples),
    }


def walk_forward_folds(n_samples: int, cfg: dict) -> List[Dict[str, np.ndarray]]:
    """Create non-overlapping chronological validation/test blocks with expanding training."""
    n_folds = int(cfg["split"]["n_folds"])
    min_train_fraction = float(cfg["split"].get("min_train_fraction", 0.50))
    min_train = max(1, int(np.floor(n_samples * min_train_fraction)))
    remainder = n_samples - min_train
    if remainder < 2 * n_folds:
        return [chronological_holdout_indices(n_samples, cfg)]

    block = remainder // (2 * n_folds)
    folds = []
    for fold in range(n_folds):
        train_end = min_train + fold * block
        val_start = train_end
        val_end = min(val_start + block, n_samples)
        test_start = val_end
        test_end = min(test_start + block, n_samples)
        if test_end <= test_start:
            break
        folds.append({
            "train": np.arange(0, train_end),
            "val": np.arange(val_start, val_end),
            "test": np.arange(test_start, test_end),
        })
    if not folds:
        folds = [chronological_holdout_indices(n_samples, cfg)]
    return folds


def fit_transform_fold(
    x: np.ndarray,
    y: np.ndarray,
    indices: Dict[str, np.ndarray],
    cfg: dict,
):
    train_idx = indices["train"]
    feature_cols = cfg["data"]["feature_columns"]
    target_col = cfg["data"]["target_column"]
    if target_col not in feature_cols:
        raise ValueError(
            "For consistent inverse scaling, target_column must be present in feature_columns."
        )
    target_feature_idx = feature_cols.index(target_col)

    # Fit scaler using only raw feature values contained in training windows.
    train_values = x[train_idx].reshape(-1, x.shape[-1])
    scaler = Standardizer.fit(train_values)

    x_scaled = scaler.transform(x)
    y_mean = scaler.mean_[target_feature_idx]
    y_std = scaler.std_[target_feature_idx]
    y_scaled = ((y - y_mean) / y_std).astype(np.float32)

    fold = {}
    for key, idx in indices.items():
        fold[key] = (
            x_scaled[idx],
            y_scaled[idx],
            y[idx].astype(np.float32),
            idx,
        )
    return fold, scaler, target_feature_idx


def prepare_all_folds(cfg: dict):
    validate_config(cfg)
    df = load_dataframe(cfg)
    x, y, origins = make_supervised_arrays(df, cfg)
    folds = walk_forward_folds(len(x), cfg)
    prepared = []
    for fold_id, indices in enumerate(folds):
        data, scaler, target_idx = fit_transform_fold(x, y, indices, cfg)
        prepared.append({
            "fold_id": fold_id,
            "data": data,
            "scaler": scaler,
            "target_feature_index": target_idx,
            "origins": origins,
        })
    return df, prepared


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df, folds = prepare_all_folds(cfg)
    print(f"Rows after cleaning: {len(df)}")
    print(f"Prepared folds: {len(folds)}")
    for f in folds:
        d = f["data"]
        print(
            f"fold={f['fold_id']} train={len(d['train'][0])} "
            f"val={len(d['val'][0])} test={len(d['test'][0])}"
        )
