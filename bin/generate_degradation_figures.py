#!/usr/bin/env python3
"""Generate the mechanistic-degradation figures embedded in docs/degradation.md.

Runs the same protocol through each degradation preset (nominal / accelerated /
severe) for ``--cycles`` oracle calls and records the per-mechanism state the
oracle exposes in its history (cumulative capacity loss to SEI, plating and
SEI-on-cracks; SEI and dead-lithium film thicknesses; SOH). Writes PNGs and a
JSON data record into ``docs/_static/degradation/`` (git-whitelisted).

Manual dev tool — not imported by the package, not run in CI. Core install only
(uses the analytic Randles-stub ECM, so no AutoEIS is required).

    uv run python bin/generate_degradation_figures.py --cycles 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / "docs" / "_static" / "degradation"
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from battery_oracle import OracleFailure, PyBaMMOracle  # noqa: E402
from battery_oracle._plotting import label_axes, slipstream  # noqa: E402
from battery_oracle.oracle import _randles_stub_ecm  # noqa: E402

plt.rcParams.update({"font.family": "serif", "figure.dpi": 120,
                     "axes.grid": True, "grid.alpha": 0.3})

PRESETS = ("nominal", "accelerated", "severe")
PRESET_COLOR = {"nominal": slipstream(0.15), "accelerated": slipstream(0.55),
                "severe": slipstream(0.9)}

# Moderately aggressive protocol (1.5C first-stage charge after the x25 capacity
# scaling onto the 5 Ah Chen2020 cell) so the plating pathway in the accelerated /
# severe presets is actually exercised; nominal (SEI-only) stays C-rate-insensitive.
PROTOCOL = np.array([300.0, 150.0, 0.25, 0.25, 200.0, 1.0], dtype=np.float64)

HIST_KEYS = [
    "end_soh", "capacity_ah",
    "cumulative_sei_loss_ah", "cumulative_plating_ah", "cumulative_crack_sei_ah",
    "cumulative_lli_total_ah", "sei_thickness_nm", "dead_li_nm", "lam_pct",
]


def run_preset(preset: str, n_cycles: int) -> dict:
    oracle = PyBaMMOracle(ecm_model_fn=_randles_stub_ecm, degradation_preset=preset,
                          capacity_check=False)
    oracle.reset()
    rec: dict = {k: [] for k in HIST_KEYS}
    rec["eol_cycle"] = None
    for i in range(n_cycles):
        try:
            oracle(PROTOCOL)
        except OracleFailure as exc:
            print(f"  [{preset}] EOL/solver stop at cycle {i}: {exc}")
            rec["eol_cycle"] = i
            break
        h = oracle._history[-1]
        for k in HIST_KEYS:
            v = h.get(k)
            rec[k].append(float(v) if v is not None and np.isfinite(v) else float("nan"))
    return rec


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = {}
    for preset in PRESETS:
        print(f"[{preset}] running {args.cycles} cycles ...")
        data[preset] = run_preset(preset, args.cycles)

    (outdir / "degradation_data.json").write_text(
        json.dumps({"protocol_mA_h": PROTOCOL.tolist(), "data": data}, indent=2) + "\n"
    )

    # --- Fig 1: SOH trajectories --------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for preset in PRESETS:
        soh = data[preset]["end_soh"]
        ax.plot(range(1, len(soh) + 1), soh, "o-", color=PRESET_COLOR[preset], label=preset)
    ax.set_xlabel("oracle call (cycle)")
    ax.set_ylabel("state of health  $Q/Q_0$")
    ax.set_title("SOH trajectory by degradation preset")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "degradation_soh.png", bbox_inches="tight")
    plt.close(fig)

    # --- Fig 2: cumulative capacity loss per mechanism, one panel per preset -
    mech = [("cumulative_sei_loss_ah", "SEI"), ("cumulative_plating_ah", "plating (net)"),
            ("cumulative_crack_sei_ah", "SEI-on-cracks"),
            ("cumulative_lli_total_ah", "total LLI")]
    fig, axes = plt.subplots(1, len(PRESETS), figsize=(4.0 * len(PRESETS), 3.4), sharey=True)
    mcols = [slipstream(x) for x in (0.1, 0.45, 0.7, 0.95)]
    for ax, preset in zip(axes, PRESETS):
        for (key, lab), c in zip(mech, mcols):
            y = data[preset][key]
            ax.plot(range(1, len(y) + 1), y, "o-", ms=3, color=c, label=lab)
        ax.set_xlabel("cycle")
        ax.set_title(preset)
    axes[0].set_ylabel("cumulative capacity loss [A.h]")
    axes[-1].legend(fontsize=7)
    label_axes(axes)
    fig.tight_layout()
    fig.savefig(outdir / "degradation_mechanisms.png", bbox_inches="tight")
    plt.close(fig)

    # --- Fig 3: interphase film thicknesses ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for preset in PRESETS:
        c = PRESET_COLOR[preset]
        sei = data[preset]["sei_thickness_nm"]
        dead = data[preset]["dead_li_nm"]
        axes[0].plot(range(1, len(sei) + 1), sei, "o-", ms=3, color=c, label=preset)
        axes[1].plot(range(1, len(dead) + 1), dead, "o-", ms=3, color=c, label=preset)
    axes[0].set_xlabel("cycle"); axes[0].set_ylabel("SEI thickness [nm]")
    axes[0].set_title("X-averaged SEI film growth")
    axes[1].set_xlabel("cycle"); axes[1].set_ylabel("dead-Li thickness [nm]")
    axes[1].set_title("Dead-lithium film accumulation")
    axes[1].legend(fontsize=8)
    label_axes(axes)
    fig.tight_layout()
    fig.savefig(outdir / "degradation_films.png", bbox_inches="tight")
    plt.close(fig)

    for f in ("degradation_soh.png", "degradation_mechanisms.png",
              "degradation_films.png", "degradation_data.json"):
        print("wrote", (outdir / f).relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
