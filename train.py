from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baselines import build_baseline
from evaluate import compute_metrics
from gcienm import model_from_config
from preprocess import load_config, prepare_all_folds


MODEL_NAMES = ["gcienm", "cnn", "tcn", "rnn", "gru", "lstm", "transformer"]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def resolve_device(cfg: dict):
    requested = str(cfg["training"].get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_loader(x, y, batch_size, shuffle, num_workers=0):
    ds = TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    outputs = []
    targets = []
    for xb, yb in loader:
        xb = xb.to(device)
        outputs.append(model(xb).cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(outputs), np.concatenate(targets)


def fit_model(model, train_loader, val_loader, cfg, device, epochs_override=None):
    model.to(device)
    t = cfg["training"]
    epochs = int(epochs_override or t["epochs"])
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(t["learning_rate"]),
        weight_decay=float(t.get("weight_decay", 0.0)),
    )
    criterion = nn.MSELoss()
    patience = int(t["patience"])
    clip = float(t.get("gradient_clip", 0.0))

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        val_pred, val_true = predict(model, val_loader, device)
        val_loss = float(np.mean((val_pred - val_true) ** 2))
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})

        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_val


def build_model(name, cfg, n_features):
    if name == "gcienm":
        return model_from_config(cfg, n_features)
    return build_baseline(name, cfg, n_features)


def train_one(
    cfg: dict,
    model_name: str,
    fold_bundle: dict,
    seed: int,
    save_artifacts: bool = True,
    epochs_override=None,
):
    seed_everything(seed)
    device = resolve_device(cfg)
    data = fold_bundle["data"]
    batch = int(cfg["training"]["batch_size"])
    workers = int(cfg["training"].get("num_workers", 0))

    train_x, train_y_scaled, _, _ = data["train"]
    val_x, val_y_scaled, _, _ = data["val"]
    test_x, test_y_scaled, test_y_raw, test_idx = data["test"]

    train_loader = make_loader(train_x, train_y_scaled, batch, True, workers)
    val_loader = make_loader(val_x, val_y_scaled, batch, False, workers)
    test_loader = make_loader(test_x, test_y_scaled, batch, False, workers)

    model = build_model(model_name, cfg, train_x.shape[-1])
    model, history, best_val = fit_model(
        model, train_loader, val_loader, cfg, device, epochs_override=epochs_override
    )

    pred_scaled, _ = predict(model, test_loader, device)
    scaler = fold_bundle["scaler"]
    target_i = fold_bundle["target_feature_index"]
    pred_raw = scaler.inverse_transform_column(pred_scaled, target_i)
    metrics = compute_metrics(test_y_raw, pred_raw)

    fold_id = int(fold_bundle["fold_id"])
    record = {
        "model": model_name,
        "fold": fold_id,
        "seed": seed,
        **metrics,
        "best_val_mse_scaled": best_val,
        "epochs_ran": int(history["epoch"].iloc[-1]),
    }

    if save_artifacts:
        out_dir = Path(cfg["project"].get("output_dir", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model": model_name,
                "fold": fold_id,
                "seed": seed,
                "config": cfg,
            },
            out_dir / f"best_model_{model_name}_fold{fold_id}_seed{seed}.pt",
        )
        history.to_csv(
            out_dir / f"history_{model_name}_fold{fold_id}_seed{seed}.csv",
            index=False,
        )

        pred_df = {"sample_index": test_idx}
        for h in range(test_y_raw.shape[1]):
            pred_df[f"true_{h+1:03d}"] = test_y_raw[:, h]
            pred_df[f"pred_{h+1:03d}"] = pred_raw[:, h]
        pd.DataFrame(pred_df).to_csv(
            out_dir / f"predictions_{model_name}_fold{fold_id}_seed{seed}.csv",
            index=False,
        )

    return record


def run_model(cfg: dict, model_name: str):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}")
    _, folds = prepare_all_folds(cfg)
    records = []
    for seed in cfg["training"]["seeds"]:
        for fold in folds:
            print(f"[train] model={model_name} fold={fold['fold_id']} seed={seed}")
            records.append(train_one(cfg, model_name, fold, int(seed)))
    out = pd.DataFrame(records)
    out_path = Path(cfg["project"].get("output_dir", ".")) / f"results_{model_name}.csv"
    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    args = parser.parse_args()
    cfg = load_config(args.config)
    df = run_model(cfg, args.model)
    print(df.groupby("model")[["MAE", "RMSE", "MAPE"]].agg(["mean", "std"]))
