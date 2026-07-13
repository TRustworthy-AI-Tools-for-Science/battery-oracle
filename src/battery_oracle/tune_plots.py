"""Plots for the oracle calibration engine (:mod:`battery_oracle.tune`): how the
Optuna search traded off its co-optimized objectives (Pareto front), whether it
converged (optimisation history), and how well the winning candidate matches the
real reference cell (alignment summary).

Consumes the engine's two machine-readable outputs — ``sweep_results.csv``
(all trial rows, incl. ``score``/``trial_number``) and
``calibration_summary.json`` (:func:`battery_oracle.tune.write_calibration_summary`) —
so plots can be regenerated without re-running the search, via the
``battery-oracle-tune-plot`` CLI (:func:`main`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._plotting import SLIPSTREAM_COLORS


def _pareto_frontier(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated points for 2D minimization (lower x, lower y).

    Sorts by x, then keeps a point only if its y is a new minimum seen so far
    — the standard staircase construction for a 2-objective Pareto frontier.
    """
    order = np.argsort(x)
    frontier = []
    best_y = np.inf
    for i in order:
        if y[i] < best_y:
            frontier.append(i)
            best_y = y[i]
    return np.array(frontier, dtype=int)


def plot_pareto_front(results_df: pd.DataFrame, real_targets: dict,
                       best: dict | None = None, save_path=None):
    """Pareto front of the two primary co-optimized objectives.

    X/Y are each trial's relative error (%) against the real reference cell
    for EIS arc-shape ((R3+R5)/R1) and R1 growth rate — the two metrics
    that jointly determine the arc/R1 "validated" status in the written
    config. Point color encodes the full combined score (which also folds in
    the EOL and C-rate terms), so a point that looks good on arc/R1 alone but
    scores poorly is still visible.

    Parameters
    ----------
    results_df : pd.DataFrame
        Loaded from the calibration run's sweep_results.csv.
    real_targets : dict
        From calibration_summary.json — needs "mean_arc_ratio", "r1_growth_pct".
    best : dict or None
        The winning trial's row (calibration_summary.json's "best"), highlighted
        with a star if given.
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    arc_real = real_targets.get("mean_arc_ratio")
    r1_real = real_targets.get("r1_growth_pct")

    d = results_df.dropna(subset=["oracle_arc_ratio", "oracle_r1_growth_pct"]).copy()

    fig, ax = plt.subplots(figsize=(3.5, 3.5), layout="constrained")

    if not arc_real or not r1_real or abs(r1_real) < 1e-3 or d.empty:
        ax.text(0.5, 0.5, "Not enough data for a Pareto front\n(missing real targets or trials)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    d["arc_err_pct"] = (d["oracle_arc_ratio"] / arc_real - 1.0).abs() * 100.0
    d["r1_err_pct"] = (d["oracle_r1_growth_pct"] / r1_real - 1.0).abs() * 100.0

    has_score = "score" in d.columns and d["score"].notna().any()
    sc = ax.scatter(
        d["arc_err_pct"], d["r1_err_pct"],
        c=d["score"] if has_score else SLIPSTREAM_COLORS[0],
        cmap="viridis_r" if has_score else None,
        s=22, alpha=0.85, edgecolors="none", zorder=2,
    )
    if has_score:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Combined score (lower = better)")

    frontier_idx = _pareto_frontier(d["arc_err_pct"].to_numpy(), d["r1_err_pct"].to_numpy())
    if len(frontier_idx) > 0:
        fr = d.iloc[frontier_idx].sort_values("arc_err_pct")
        ax.plot(fr["arc_err_pct"], fr["r1_err_pct"], "-o",
                color=SLIPSTREAM_COLORS[4], markersize=4, linewidth=1.2,
                fillstyle="none", label="Pareto frontier", zorder=3)

    if best is not None:
        barc, br1 = best.get("oracle_arc_ratio"), best.get("oracle_r1_growth_pct")
        if barc is not None and br1 is not None:
            b_arc_err = abs(barc / arc_real - 1.0) * 100.0
            b_r1_err = abs(br1 / r1_real - 1.0) * 100.0
            ax.scatter([b_arc_err], [b_r1_err], marker="*", s=180,
                       color=SLIPSTREAM_COLORS[5], edgecolors="black", linewidths=0.6,
                       zorder=4, label="Selected calibration")

    ax.set_xlabel("EIS arc-shape error, |Δ(R3+R5)/R1| (%)")
    ax.set_ylabel("R1 growth-rate error (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_optimization_history(results_df: pd.DataFrame, save_path=None):
    """Combined score vs trial number, with the running best-so-far overlay.

    The standard Bayesian-optimisation convergence plot: shows whether the
    search actually improved over its n_trials budget or plateaued early.

    Parameters
    ----------
    results_df : pd.DataFrame
        Loaded from the calibration run's sweep_results.csv (needs "score";
        uses "trial_number" if present, else row order).
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if "score" in results_df.columns:
        d = results_df.dropna(subset=["score"]).copy()
    else:
        # Sweeps predating the engine's score/trial_number recording
        d = results_df.iloc[0:0].copy()
    if "trial_number" in d.columns:
        d = d.sort_values("trial_number")
        x = d["trial_number"]
    else:
        x = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(3.5, 2.625), layout="constrained")
    if d.empty:
        ax.text(0.5, 0.5, "No finite-score trials to plot", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    ax.plot(x, d["score"], "o", color=SLIPSTREAM_COLORS[0], markersize=4,
            alpha=0.6, fillstyle="none", label="Trial score")
    running_best = d["score"].cummin()
    ax.plot(x, running_best, "-", color=SLIPSTREAM_COLORS[4], linewidth=1.5,
            label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Combined score (lower = better)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_alignment_summary(summary: dict, save_path=None):
    """Real vs achieved comparison for each calibration target metric.

    One panel per metric the winning candidate was scored against: EIS arc
    ratio, R1 growth rate, implied EOL cycle, and (if the slope probe ran)
    the C_rate_2 fade-slope, with the real multi-cell 95% bootstrap CI shown
    as an error bar. Panels are omitted (not left blank) when a metric wasn't
    computed for this run (e.g. --skip-crate2-slope).

    Parameters
    ----------
    summary : dict
        calibration_summary.json's content (real_targets, best,
        real_crate2_slope, eol_target_cycles).
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    real_targets = summary.get("real_targets", {}) or {}
    best = summary.get("best", {}) or {}
    real_c2 = summary.get("real_crate2_slope")
    eol_target = summary.get("eol_target_cycles")

    panels = []  # (title, real, achieved, ci_lo, ci_hi)
    arc_real, arc_ach = real_targets.get("mean_arc_ratio"), best.get("oracle_arc_ratio")
    if arc_real is not None and arc_ach is not None:
        panels.append(("EIS arc ratio\n(R3+R5)/R1 (unitless)", arc_real, arc_ach, None, None))
    r1_real, r1_ach = real_targets.get("r1_growth_pct"), best.get("oracle_r1_growth_pct")
    if r1_real is not None and r1_ach is not None:
        panels.append(("R1 growth (%)", r1_real, r1_ach, None, None))
    eol_ach = best.get("implied_eol_cycle")
    if eol_target is not None and eol_ach is not None:
        panels.append(("Implied EOL (cycles)", eol_target, eol_ach, None, None))
    if real_c2 is not None:
        c2_ach = best.get("oracle_slope_mAh_per_mA")
        panels.append(("C_rate_2 slope\nd(fade)/d(C_rate_2) (mAh/mA)",
                        real_c2.get("slope_mAh_per_mA"), c2_ach,
                        real_c2.get("ci_lo"), real_c2.get("ci_hi")))

    n = max(len(panels), 1)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                              layout="constrained")
    axes_flat = np.atleast_1d(axes).ravel()

    if not panels:
        axes_flat[0].text(0.5, 0.5, "No aligned metrics available", ha="center",
                           va="center", transform=axes_flat[0].transAxes, fontsize=9)
        axes_flat[0].set_xticks([]); axes_flat[0].set_yticks([])

    for ax, (title, real, achieved, ci_lo, ci_hi) in zip(axes_flat, panels):
        x = [0, 1]
        heights = [real, achieved]
        colors = [SLIPSTREAM_COLORS[4], SLIPSTREAM_COLORS[0]]
        yerr = [[0, 0], [0, 0]]
        if ci_lo is not None and ci_hi is not None:
            yerr[0] = [real - ci_lo, ci_hi - real]
        ax.bar(x, heights, color=colors, width=0.6,
               yerr=np.array(yerr).T if ci_lo is not None else None,
               capsize=3, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(["Real", "Achieved"])
        ax.set_ylabel(title)
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes_flat[len(panels):]:
        ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_eis_comparison(eis_data: dict | None, save_path=None):
    """Nyquist overlay of the oracle's synthesized EIS vs. the ground-truth EIS.

    For the winning calibration candidate (data from
    :func:`battery_oracle.tune.collect_eis_comparison`): one panel per
    representative cycle (first + last), each overlaying the real cell's spectrum
    — reconstructed from that cycle's cached ECM — on the oracle's synthesized
    charge-state spectrum.

    Both curves are normalised by their ohmic resistance R0 (Re(Z) at the
    high-frequency limit). The oracle's 5 Ah PyBaMM cell has a ~16x smaller
    absolute impedance than a real coin cell, so only the R0-normalised arc shape
    — exactly what the arc-ratio metric scores — is comparable; the absolute R0
    (Ohm) of each curve is annotated in the legend so the scale is not lost.

    Parameters
    ----------
    eis_data : dict or None
        ``{"circuit", "frequencies", "panels": [{cycle_label, oracle, real}, ...]}``
        from collect_eis_comparison. ``None`` (capacity-only cache: no EIS) renders
        an explanatory placeholder instead of failing.
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    panels = (eis_data or {}).get("panels") or []
    n = max(len(panels), 1)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), layout="constrained",
                             squeeze=False)
    axes_flat = axes.ravel()

    if not panels:
        axes_flat[0].text(
            0.5, 0.5,
            "No EIS to compare\n(capacity-only calibration:\ndataset ships no EIS)",
            ha="center", va="center", transform=axes_flat[0].transAxes, fontsize=9)
        axes_flat[0].set_xticks([]); axes_flat[0].set_yticks([])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    series = (
        # key,     label,                    color,               line/marker style
        ("real",   "Ground truth (real ECM)", SLIPSTREAM_COLORS[4],
         dict(linestyle="-", marker="", linewidth=1.6)),
        ("oracle", "Oracle (synthesized)",    SLIPSTREAM_COLORS[0],
         dict(linestyle="none", marker="o", markersize=4, fillstyle="none")),
    )
    for ax, panel in zip(axes_flat, panels):
        for key, label, color, style in series:
            s = panel[key]
            r0 = s["r_ohmic"] if s.get("r_ohmic") and abs(s["r_ohmic"]) > 1e-12 else 1.0
            ax.plot(np.asarray(s["z_re"]) / r0, np.asarray(s["z_neg_im"]) / r0,
                    color=color, label=f"{label}  (R0={s['r_ohmic']:.3g} Ω)", **style)
        ax.set_title(f"Cycle {panel['cycle_label']}", fontsize=9)
        ax.set_xlabel("Re(Z) / R0")
        ax.set_ylabel("−Im(Z) / R0")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=7, loc="upper left")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_tune_oracle_summary(results_csv, summary_json, out_dir) -> dict[str, Path]:
    """Load a calibration run's outputs and write all three tuning plots to out_dir.

    Parameters
    ----------
    results_csv : str or Path
        Path to sweep_results.csv.
    summary_json : str or Path
        Path to calibration_summary.json.
    out_dir : str or Path
        Directory to write the PNGs into (created if missing).

    Returns
    -------
    dict[str, Path]
        Keys "pareto", "history", "alignment" mapped to the written PNG paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    with open(summary_json) as f:
        summary = json.load(f)

    paths = {
        "pareto": out_dir / "oracle_tuning_pareto.png",
        "history": out_dir / "oracle_tuning_history.png",
        "alignment": out_dir / "oracle_tuning_alignment.png",
    }
    fig_pareto = plot_pareto_front(df, summary.get("real_targets", {}) or {},
                                    summary.get("best"), save_path=paths["pareto"])
    fig_history = plot_optimization_history(df, save_path=paths["history"])
    fig_alignment = plot_alignment_summary(summary, save_path=paths["alignment"])
    for fig in (fig_pareto, fig_history, fig_alignment):
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# CLI: regenerate the calibration plots without re-running the search
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    """``battery-oracle-tune-plot``: regenerate Pareto/history/alignment plots.

    Calibration runs auto-generate these after a normal run; this CLI exists to
    regenerate them later — after tweaking a plot's styling, or from an older
    run whose sweep_results.csv/calibration_summary.json are still on disk —
    without paying for another multi-hour Optuna search.
    """
    p = argparse.ArgumentParser(
        description="Regenerate the oracle-calibration Pareto/history/alignment "
                    "plots from an existing sweep_results.csv + calibration_summary.json."
    )
    p.add_argument("--results-dir", required=True,
                   help="Directory containing sweep_results.csv and "
                        "calibration_summary.json (the calibration run's results dir).")
    p.add_argument("--results-csv", default=None,
                   help="Override path to sweep_results.csv (default: "
                        "<results-dir>/sweep_results.csv).")
    p.add_argument("--summary-json", default=None,
                   help="Override path to calibration_summary.json (default: "
                        "<results-dir>/calibration_summary.json).")
    p.add_argument("--out-dir", default=None,
                   help="Where to write the PNGs (default: --results-dir, same "
                        "place the calibration run itself writes them).")
    args = p.parse_args(argv)

    results_dir = Path(args.results_dir)
    results_csv = Path(args.results_csv) if args.results_csv else results_dir / "sweep_results.csv"
    summary_json = Path(args.summary_json) if args.summary_json else results_dir / "calibration_summary.json"
    out_dir = Path(args.out_dir) if args.out_dir else results_dir

    if not results_csv.exists():
        raise FileNotFoundError(
            f"{results_csv} not found — run a calibration first (or point "
            "--results-csv at an existing sweep_results.csv)."
        )
    if not summary_json.exists():
        raise FileNotFoundError(
            f"{summary_json} not found. This file is written by calibration runs "
            "that call write_calibration_summary; older runs predating it only "
            "have sweep_results.csv (no 'score' column) and config_oracle*.yml — "
            "re-run the calibration to produce a compatible calibration_summary.json."
        )

    paths = plot_tune_oracle_summary(results_csv, summary_json, out_dir)
    print("Wrote:")
    for name, path in paths.items():
        print(f"  {name:10s} {path}")


if __name__ == "__main__":
    main()
