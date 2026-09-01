from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from preprocess import load_config, prepare_all_folds
from train import train_one


def _run_variant(base_cfg, label, modifier):
    cfg = copy.deepcopy(base_cfg)
    modifier(cfg)
    _, folds = prepare_all_folds(cfg)
    seed = int(cfg["training"]["seeds"][0])
    # Use all folds, one fixed seed, to limit sensitivity-study cost while retaining chronology.
    records = []
    for fold in folds:
        rec = train_one(
            cfg, "gcienm", fold, seed,
            save_artifacts=False,
        )
        rec["sensitivity_case"] = label
        records.append(rec)
    return records


def run_sensitivity(cfg):
    all_records = []
    sens = cfg["sensitivity"]

    for value in sens.get("kernel_size", []):
        all_records += _run_variant(
            cfg,
            f"kernel_size={value}",
            lambda c, v=value: c["model"].__setitem__("kernel_size", int(v)),
        )

    for value in sens.get("dilation_base", []):
        def set_dilation(c, v=value):
            b = int(v)
            c["model"]["dilation_rates"] = [b, b * 2, b * 4]
        all_records += _run_variant(cfg, f"dilation_base={value}", set_dilation)

    for value in sens.get("lookback", []):
        all_records += _run_variant(
            cfg,
            f"lookback={value}",
            lambda c, v=value: c["split"].__setitem__("lookback", int(v)),
        )

    for mode in sens.get("attention_mode", []):
        all_records += _run_variant(
            cfg,
            f"attention_mode={mode}",
            lambda c, v=mode: c["model"].__setitem__("attention_mode", str(v)),
        )

    for ablation in sens.get("ablations", []):
        def apply_ablation(c, name=ablation):
            c["model"]["use_atm"] = name != "no_atm"
            c["model"]["use_mvpnn"] = name != "no_mvpnn"
            c["model"]["use_dilated_conv"] = name != "no_dilated_conv"
        all_records += _run_variant(cfg, f"ablation={ablation}", apply_ablation)

    df = pd.DataFrame(all_records)
    out = Path(cfg["project"].get("output_dir", ".")) / "sensitivity_results.csv"
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = run_sensitivity(cfg)
    print(
        result.groupby("sensitivity_case")[["MAE", "RMSE", "MAPE"]]
        .agg(["mean", "std"])
        .to_string()
    )
