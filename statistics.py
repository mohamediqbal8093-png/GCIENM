from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(values, confidence=0.95, n_boot=5000, seed=1234):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def rank_biserial_from_paired_differences(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    if len(diff) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    pos = ranks[diff > 0].sum()
    neg = ranks[diff < 0].sum()
    denom = pos + neg
    return float((pos - neg) / denom) if denom else 0.0


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted


def aggregate_summary(df, metric="RMSE"):
    rows = []
    for model, g in df.groupby("model"):
        vals = g[metric].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(vals)
        rows.append({
            "model": model,
            "metric": metric,
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)),
            "ci95_low": lo,
            "ci95_high": hi,
        })
    return pd.DataFrame(rows).sort_values("mean")


def aligned_table(df, metric):
    pivot = df.pivot_table(
        index=["fold", "seed"], columns="model", values=metric, aggfunc="mean"
    ).dropna(axis=0)
    return pivot


def friedman_test(df, metric="RMSE"):
    pivot = aligned_table(df, metric)
    if pivot.shape[1] < 3 or pivot.shape[0] < 2:
        return {"statistic": np.nan, "p_value": np.nan, "n_blocks": len(pivot)}
    arrays = [pivot[c].to_numpy() for c in pivot.columns]
    stat, p = stats.friedmanchisquare(*arrays)
    ranks = pivot.rank(axis=1, method="average", ascending=True).mean(axis=0)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_blocks": int(len(pivot)),
        "average_ranks": ranks.to_dict(),
    }


def pairwise_against(df, reference="gcienm", metric="RMSE"):
    pivot = aligned_table(df, metric)
    if reference not in pivot.columns:
        raise ValueError(f"Reference model '{reference}' is absent.")
    rows = []
    for model in pivot.columns:
        if model == reference:
            continue
        ref = pivot[reference].to_numpy()
        other = pivot[model].to_numpy()
        diff = other - ref  # positive means reference has lower error
        if np.allclose(diff, 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        rows.append({
            "reference": reference,
            "comparison": model,
            "metric": metric,
            "n_pairs": len(diff),
            "wilcoxon_statistic": float(stat),
            "p_value": float(p),
            "median_difference_other_minus_reference": float(np.median(diff)),
            "rank_biserial_effect": rank_biserial_from_paired_differences(diff),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["holm_adjusted_p"] = holm_adjust(out["p_value"].to_numpy())
    return out


def run_statistics(path, metric="RMSE", reference="gcienm", output_dir="."):
    df = pd.read_csv(path)
    required = {"model", "fold", "seed", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")

    summary = aggregate_summary(df, metric)
    friedman = friedman_test(df, metric)
    pairwise = pairwise_against(df, reference, metric)

    output_dir = Path(output_dir)
    summary.to_csv(output_dir / "statistics_summary.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_statistics.csv", index=False)
    pd.DataFrame([{
        "metric": metric,
        "friedman_statistic": friedman["statistic"],
        "friedman_p_value": friedman["p_value"],
        "n_blocks": friedman["n_blocks"],
    }]).to_csv(output_dir / "friedman_statistics.csv", index=False)

    if "average_ranks" in friedman:
        pd.DataFrame(
            [{"model": k, "average_rank": v} for k, v in friedman["average_ranks"].items()]
        ).sort_values("average_rank").to_csv(
            output_dir / "average_ranks.csv", index=False
        )
    return summary, pairwise, friedman


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--metric", default="RMSE")
    parser.add_argument("--reference", default="gcienm")
    args = parser.parse_args()
    summary, pairwise, friedman = run_statistics(
        args.results, args.metric, args.reference
    )
    print(summary.to_string(index=False))
    print("\nFriedman:", friedman)
    print("\nPairwise:")
    print(pairwise.to_string(index=False))
