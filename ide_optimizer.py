from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from preprocess import load_config, prepare_all_folds
from train import build_model, fit_model, make_loader, resolve_device, seed_everything


def _space_items(cfg):
    return list(cfg["ide"]["search_space"].items())


def vector_to_params(vector, cfg):
    params = {}
    for value, (name, choices) in zip(vector, _space_items(cfg)):
        idx = int(np.clip(round(float(value)), 0, len(choices) - 1))
        params[name] = choices[idx]
    return params


def apply_params(cfg, params):
    out = copy.deepcopy(cfg)
    m = out["model"]
    t = out["training"]

    if "hidden_dim" in params:
        m["hidden_dim"] = int(params["hidden_dim"])
    if "kernel_size" in params:
        m["kernel_size"] = int(params["kernel_size"])
    if "dilation_base" in params:
        base = int(params["dilation_base"])
        m["dilation_rates"] = [base, base * 2, base * 4]
    if "num_heads" in params:
        heads = int(params["num_heads"])
        hidden = int(m["hidden_dim"])
        # Repair head count to a divisor of hidden_dim.
        valid = [h for h in [1, 2, 4, 8, 16] if h <= hidden and hidden % h == 0]
        m["num_heads"] = min(valid, key=lambda h: abs(h - heads))
    if "dropout" in params:
        m["dropout"] = float(params["dropout"])
    if "learning_rate" in params:
        t["learning_rate"] = float(params["learning_rate"])
    return out


def evaluate_candidate(base_cfg, params, fold, seed):
    cfg = apply_params(base_cfg, params)
    seed_everything(seed)
    device = resolve_device(cfg)
    data = fold["data"]
    batch = int(cfg["training"]["batch_size"])
    train_x, train_y, _, _ = data["train"]
    val_x, val_y, _, _ = data["val"]

    train_loader = make_loader(train_x, train_y, batch, True)
    val_loader = make_loader(val_x, val_y, batch, False)
    model = build_model("gcienm", cfg, train_x.shape[-1])
    _, _, best_val = fit_model(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        epochs_override=int(cfg["ide"]["evaluation_epochs"]),
    )
    return float(best_val)


def optimize(cfg):
    ide = cfg["ide"]
    _, folds = prepare_all_folds(cfg)
    fold = folds[0]  # validation-only search; final evaluation uses all folds separately.
    seed = int(cfg["training"]["seeds"][0])
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    dims = len(_space_items(cfg))
    lows = np.zeros(dims)
    highs = np.asarray([len(v) - 1 for _, v in _space_items(cfg)], dtype=float)

    npop = int(ide["population_size"])
    generations = int(ide["generations"])
    pop = rng.uniform(lows, highs, size=(npop, dims))
    fitness = np.asarray([
        evaluate_candidate(cfg, vector_to_params(ind, cfg), fold, seed)
        for ind in pop
    ])

    cr = float(ide["initial_cr"])
    mu = float(ide.get("chaos_mu", 4.0))
    f_min = float(ide["mutation_min"])
    f_max = float(ide["mutation_max"])
    history = []

    for g in range(generations):
        best_idx = int(np.argmin(fitness))
        best = pop[best_idx].copy()

        # Generation-dependent mutation factor; higher exploration early,
        # stronger exploitation later.
        progress = g / max(generations - 1, 1)
        f = f_max - (f_max - f_min) * progress

        # Logistic chaotic crossover update.
        cr = mu * cr * (1.0 - cr)
        cr = float(np.clip(cr, 0.05, 0.95))

        for i in range(npop):
            choices = [j for j in range(npop) if j != i]
            r1, r2 = rng.choice(choices, size=2, replace=False)
            mutant = best + f * (pop[r1] - pop[r2])
            mutant = np.clip(mutant, lows, highs)

            mask = rng.random(dims) < cr
            mask[rng.integers(0, dims)] = True
            trial = np.where(mask, mutant, pop[i])
            trial = np.clip(trial, lows, highs)
            params = vector_to_params(trial, cfg)
            trial_fit = evaluate_candidate(cfg, params, fold, seed)
            if trial_fit <= fitness[i]:
                pop[i] = trial
                fitness[i] = trial_fit

        best_idx = int(np.argmin(fitness))
        best_params = vector_to_params(pop[best_idx], cfg)
        history.append({
            "generation": g + 1,
            "best_validation_mse": float(fitness[best_idx]),
            "mutation_factor": float(f),
            "crossover_rate": float(cr),
            "parameters": best_params,
        })
        print(
            f"[IDE] generation={g+1}/{generations} "
            f"best_val_mse={fitness[best_idx]:.6g} params={best_params}"
        )

    best_idx = int(np.argmin(fitness))
    result = {
        "best_validation_mse": float(fitness[best_idx]),
        "best_parameters": vector_to_params(pop[best_idx], cfg),
        "history": history,
    }
    out_dir = Path(cfg["project"].get("output_dir", "."))
    with open(out_dir / "ide_best.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = optimize(cfg)
    print(json.dumps(result, indent=2))
