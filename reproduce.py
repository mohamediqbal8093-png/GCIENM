from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from ide_optimizer import apply_params, optimize
from preprocess import load_config, prepare_all_folds
from sensitivity import run_sensitivity
from statistics import run_statistics
from train import MODEL_NAMES, train_one


def run_full(cfg, skip_ide=False, skip_sensitivity=False, skip_statistics=False):
    out_dir = Path(cfg["project"].get("output_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)

    active_cfg = copy.deepcopy(cfg)

    if active_cfg["ide"].get("enabled", True) and not skip_ide:
        print("\n=== IDE optimization ===")
        ide_result = optimize(active_cfg)
        active_cfg = apply_params(active_cfg, ide_result["best_parameters"])
        with open(out_dir / "resolved_config.json", "w", encoding="utf-8") as f:
            json.dump(active_cfg, f, indent=2)
    else:
        print("\n=== IDE optimization skipped ===")

    print("\n=== Preparing common chronological folds ===")
    _, folds = prepare_all_folds(active_cfg)

    print("\n=== Fair repeated-run comparison ===")
    records = []
    for model in MODEL_NAMES:
        for seed in active_cfg["training"]["seeds"]:
            for fold in folds:
                print(
                    f"[reproduce] model={model} fold={fold['fold_id']} seed={seed}"
                )
                rec = train_one(
                    active_cfg, model, fold, int(seed), save_artifacts=True
                )
                records.append(rec)

    results = pd.DataFrame(records)
    result_path = out_dir / "repeated_results.csv"
    results.to_csv(result_path, index=False)

    print("\n=== Aggregate metrics ===")
    aggregate = (
        results.groupby("model")[["MAE", "RMSE", "MAPE", "sMAPE", "NRMSE"]]
        .agg(["mean", "std"])
    )
    print(aggregate)
    aggregate.to_csv(out_dir / "aggregate_results.csv")

    if not skip_statistics:
        print("\n=== Statistical validation ===")
        summary, pairwise, friedman = run_statistics(
            result_path,
            metric="RMSE",
            reference="gcienm",
            output_dir=out_dir,
        )
        print(summary.to_string(index=False))
        print("Friedman:", friedman)
        print(pairwise.to_string(index=False))

    if not skip_sensitivity:
        print("\n=== Sensitivity and ablation analysis ===")
        run_sensitivity(active_cfg)

    print("\nReproduction pipeline completed.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-ide", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-statistics", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_full(
        cfg,
        skip_ide=args.skip_ide,
        skip_sensitivity=args.skip_sensitivity,
        skip_statistics=args.skip_statistics,
    )
