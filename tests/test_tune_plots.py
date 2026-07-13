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
    plot_eis_comparison,
    plot_tune_oracle_summary,
)


def _synthetic_eis_data(n_freq=40, n_panels=2):
    """Mimic collect_eis_comparison output: a real (large-R0) + oracle (small-R0)
    spectrum per panel, so the plot must normalise by R0 to overlay them."""
    freq = np.logspace(-2, 4, n_freq)
    w = 2 * np.pi * freq

    def semicircle(r_ohmic, r_ct, tau):
        # simple R0 + (R_ct || C) Nyquist arc
        z = r_ohmic + r_ct / (1 + 1j * w * tau)
        return z.real, -z.imag, r_ohmic

    panels = []
    for k in range(n_panels):
        re_o, nim_o, r0_o = semicircle(0.01 * (1 + 0.1 * k), 0.005, 1e-2)   # oracle: small R0
        re_r, nim_r, r0_r = semicircle(0.16 * (1 + 0.1 * k), 0.09, 1e-2)    # real: ~16x larger R0
        panels.append({
            "cycle_label": str(k),
            "oracle": {"z_re": re_o, "z_neg_im": nim_o, "r_ohmic": r0_o},
            "real":   {"z_re": re_r, "z_neg_im": nim_r, "r_ohmic": r0_r},
        })
    return {"circuit": "R1-[R2,P3]-[R4,P5]", "frequencies": freq, "panels": panels}


def test_plot_eis_comparison_writes_png(tmp_path):
    out = tmp_path / "oracle_tuning_eis.png"
    fig = plot_eis_comparison(_synthetic_eis_data(), save_path=out)
    assert out.exists() and out.stat().st_size > 0
    # one Nyquist panel per requested cycle
    assert len(fig.axes) == 2


def test_plot_eis_comparison_handles_no_eis(tmp_path):
    # capacity-only calibration: collect_eis_comparison returns None -> placeholder,
    # not a crash.
    out = tmp_path / "eis_none.png"
    for payload in (None, {"panels": []}):
        fig = plot_eis_comparison(payload, save_path=out)
        assert out.exists()
        assert len(fig.axes) == 1


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
