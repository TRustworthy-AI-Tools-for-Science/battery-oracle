"""Smoke tests for the tune_plots module: synthetic sweep_results.csv +
calibration_summary.json in, three PNGs out. No PyBaMM involved.
"""
import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from battery_oracle.tune_plots import (
    _pareto_frontier,
    plot_tune_oracle_summary,
)


def _synthetic_results_df(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "kinetics_scale":       rng.uniform(0.1, 0.5, n),
        "sei_rate_scale":       rng.uniform(0.01, 1.0, n),
        "dead_li_decay_scale":  rng.uniform(0.1, 1000.0, n),
        "plating_rate_scale":   rng.uniform(0.01, 10.0, n),
        "oracle_arc_ratio":     rng.uniform(0.3, 1.5, n),
        "oracle_r1_growth_pct": rng.uniform(1.0, 30.0, n),
        "implied_eol_cycle":    rng.uniform(20.0, 120.0, n),
        "score":                rng.uniform(0.1, 5.0, n),
        "trial_number":         np.arange(n),
    })


def _synthetic_summary(df):
    best_row = df.loc[df["score"].idxmin()]
    return {
        "dataset": "synthetic",
        "preset": "accelerated",
        "cell_id": "C01",
        "n_trials": len(df),
        "n_cycles": 10,
        "crate_sensitivity_min": 3.0,
        "eol_target_cycles": 55.0,
        "real_targets": {"mean_arc_ratio": 0.85, "r1_growth_pct": 12.0},
        "real_crate2_slope": {"slope_mAh_per_mA": 0.08, "ci_lo": 0.02,
                              "ci_hi": 0.14, "n_cells": 10, "n_pairs": 900},
        "best": {**best_row.to_dict(), "oracle_slope_mAh_per_mA": 0.07},
        "best_score": float(best_row["score"]),
        "drift_result": None,
    }


def test_pareto_frontier_staircase():
    x = np.array([1.0, 2.0, 3.0, 0.5])
    y = np.array([4.0, 4.5, 0.5, 5.0])
    idx = set(_pareto_frontier(x, y))
    # (2.0, 4.5) is dominated by (1.0, 4.0): higher on both axes → excluded.
    assert idx == {3, 0, 2}


def test_plot_tune_oracle_summary_writes_three_pngs(tmp_path):
    df = _synthetic_results_df()
    csv = tmp_path / "sweep_results.csv"
    df.to_csv(csv, index=False)
    summary_json = tmp_path / "calibration_summary.json"
    with open(summary_json, "w") as f:
        json.dump(_synthetic_summary(df), f)

    out = tmp_path / "plots"
    paths = plot_tune_oracle_summary(csv, summary_json, out)
    assert set(paths) == {"pareto", "history", "alignment"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0


def test_plot_tune_oracle_summary_degrades_without_score(tmp_path):
    # Older sweeps predate the engine recording "score"/"trial_number" —
    # plots should still render (uncolored scatter, row-order history).
    df = _synthetic_results_df().drop(columns=["score", "trial_number"])
    csv = tmp_path / "sweep_results.csv"
    df.to_csv(csv, index=False)
    summary = _synthetic_summary(_synthetic_results_df())
    summary_json = tmp_path / "calibration_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f)

    paths = plot_tune_oracle_summary(csv, summary_json, tmp_path / "plots")
    for p in paths.values():
        assert p.exists()
