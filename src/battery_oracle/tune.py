"""Oracle calibration ("tune-oracle") engine.

Calibrates :class:`PyBaMMOracle` hyperparameters — kinetics_scale,
sei_rate_scale, dead_li_decay_scale, plating_rate_scale (+ optional EIS drift) —
against a real cell's measured EIS/capacity behaviour, via Optuna Bayesian
optimisation over a 4-D log-space.

Dataset-agnostic by design: everything here operates on a plain cache dict
(measured ECM-per-cycle + capacity + protocols) and precomputed real target
metrics. Loading that cache from a specific dataset (e.g. jones2022) is the
caller's job. The :func:`main` CLI (``battery-oracle-tune``) reads the cache + targets
from JSON so any battery's data can drive it.

Two calibration modes, selected automatically by the cache contents (see
:func:`score_candidate`):

* **EIS/ECM** — the cache carries per-cycle ``ecm_charge``/``ecm_discharge``; the
  fit targets the arc-ratio and R1-growth signatures (jones2022 path).
* **Capacity-fade** — the cache carries per-cycle ``real_soh`` but null ECMs (the
  CALCE / Oxford / MATR datasets ship capacity/cycling, no EIS); the fit targets
  the mean per-cycle SOH-loss rate instead. ``kinetics_scale`` is left near its
  chemistry default in this mode (no charge-transfer-arc signal to constrain it).

Public API
----------
calibrate_oracle    — run the BO; returns best params + all trial results
calibrate_drift     — fit eis_drift_scale to a real linKK low/high ratio
write_oracle_config — write a config_oracle_*.yml with calibration provenance
compute_real_targets, score_candidate, run_oracle_candidate, ... — engine pieces

Cache schema (JSON-serialisable)
--------------------------------
{
  "cell_id": str,
  "real_cell_capacity_mah": float,
  "cycles": ["9", "10", ...],              # string keys
  "first_real_capacity_mah": float,
  "data": {"9": {"protocol": [c1,c2,d1,d2,D,dd], "real_capacity_mah": float,
                 "real_soh": float, "ecm_charge": [...9...]|null,
                 "ecm_discharge": [...9...]|null}, ...}
}
"""
from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import logging
import math
import multiprocessing
import shutil
import tempfile
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from battery_oracle._circuit import (
    _param_labels_from_circuit,
    randles_pairs_from_circuit,
)
from battery_oracle._eis.kk import linkk_residuals
from battery_oracle.experiment import load_default_ecm_circuit, load_oracle_config
from battery_oracle.oracle import OracleFailure, PyBaMMOracle

log = logging.getLogger(__name__)

# Calibration always runs on SPMe, regardless of the experiment-time model
# (DFN is reserved for the final fidelity check of shortlisted protocols — it is
# far too slow for the hundreds of oracle calls a calibration sweep makes, #5).
# Pinned explicitly here so a future change to PyBaMMOracle's default `model`
# can't silently pull DFN into the calibration inner loop.
CALIBRATION_MODEL = "SPMe"


def _resolve_circuit(circuit: str | None) -> str:
    """Return *circuit*, or the default ECM circuit loaded from the YAML config.

    The tune engine does not assume any ECM structure: the circuit (and hence the
    ECM parameter layout the arc-ratio / R1-growth metrics use) is loaded from
    ``config_oracle_defaults.yml`` (``ecm.circuit``) unless the caller overrides it.
    """
    return circuit or load_default_ecm_circuit()


def _ecm_indices(circuit: str) -> tuple[int, list[int]]:
    """Positions of the ohmic resistor and the arc resistors within an ECM vector.

    Derived from the circuit string (no hard-coded 9-element assumption): the ohmic
    R is the resistor not inside any parallel ``[R, CPE]`` arc; the arc Rs are the
    resistors inside the brackets. Returns ``(ohmic_index, [arc_r_indices])`` into
    the ECM parameter vector whose labels are ``_param_labels_from_circuit(circuit)``.
    """
    labels = _param_labels_from_circuit(circuit)
    pairs = randles_pairs_from_circuit(circuit)
    arc_rs = [r for r, _ in pairs]
    ohmic = next((l for l in labels if l.startswith("R") and l not in arc_rs), labels[0])
    return labels.index(ohmic), [labels.index(r) for r in arc_rs]


def _reconstruct_ecm_spectrum(circuit: str, params, frequencies: np.ndarray) -> np.ndarray:
    """Complex impedance Z(f) of *circuit* at *params*, evaluated on *frequencies*.

    ``params`` is an ECM parameter vector in :func:`_param_labels_from_circuit`
    order — the layout the cache's ``ecm_charge``/``ecm_discharge`` and the oracle's
    ``ecm_params_*`` both use. Used to turn a stored ECM back into a Nyquist curve
    for the oracle-vs-real EIS plot. Maps the vector to AutoEIS's own label order
    defensively (the two orderings are documented to match). Requires the ``[tune]``
    extra (AutoEIS); raises if unavailable, so callers guard with try/except.
    """
    import autoeis as ae

    values = dict(zip(_param_labels_from_circuit(circuit),
                      np.asarray(params, dtype=float)))
    ordered = np.array([values[lbl] for lbl in ae.parser.get_parameter_labels(circuit)],
                       dtype=float)
    circuit_fn = ae.utils.generate_circuit_fn(circuit)
    return np.asarray(circuit_fn(np.asarray(frequencies, dtype=float), ordered),
                      dtype=complex)


def _eol_target_cycles_from_range(range_str: str | None) -> float | None:
    """Parse a ``"lo-hi"`` cycle-count range string (e.g. ``"40-70"``) to its midpoint."""
    if not range_str:
        return None
    try:
        lo, hi = str(range_str).split("-")
        return (float(lo) + float(hi)) / 2.0
    except ValueError:
        return None


def _eol_target_cycles(preset: str, oracle_cfg: dict | None = None) -> float | None:
    """Midpoint of config_oracle_defaults.yml's ``target_eol_cycles_at_1c`` range for *preset*.

    Used to penalize candidates whose implied EOL cycle count is wildly off the
    physically reasonable range — without this the BO can fit arc_ratio/r1_growth
    within the scoring window while reaching real EOL in <10 cycles or >500 cycles.
    Sourced from the YAML (single source of truth shared with oracle.py's
    ``_build_degradation_config`` defaults) rather than a hand-duplicated dict.
    """
    oracle_cfg = oracle_cfg if oracle_cfg is not None else load_oracle_config()
    preset_constants = (oracle_cfg.get("degradation", {}) or {}).get("preset_constants", {}) or {}
    return _eol_target_cycles_from_range((preset_constants.get(preset) or {}).get("target_eol_cycles_at_1c"))


# ---------------------------------------------------------------------------
# Real calibration targets (operate on the cache dict; no dataset access)
# ---------------------------------------------------------------------------

def _real_soh_series(cache: dict) -> list[float]:
    """Per-cycle real SOH across ``cache["cycles"]``, for the capacity-fade target.

    Prefers each cycle's ``real_soh``; falls back to ``real_capacity_mah`` divided
    by the cache's reference capacity (``first_real_capacity_mah`` else
    ``real_cell_capacity_mah``). Cycles carrying neither are skipped. The order
    follows ``cache["cycles"]`` so the series is directly comparable to the oracle
    SOH history replayed over the same cycles in :func:`run_oracle_candidate`.
    """
    data = cache["data"]
    ref_cap = cache.get("first_real_capacity_mah") or cache.get("real_cell_capacity_mah")
    series: list[float] = []
    for cyc in cache["cycles"]:
        entry = data[cyc]
        soh = entry.get("real_soh")
        if soh is None:
            cap = entry.get("real_capacity_mah")
            if cap is not None and ref_cap:
                soh = float(cap) / float(ref_cap)
        if soh is not None:
            series.append(float(soh))
    return series


def compute_real_targets(cache: dict, circuit: str | None = None) -> dict[str, float | None]:
    """Extract arc-ratio, R1-growth, and capacity-fade targets from the real cache.

    Parameters
    ----------
    cache : dict
        Measured cache (``ecm_charge`` / ``ecm_discharge`` and/or ``real_soh`` per
        cycle + protocols).
    circuit : str, optional
        ECM circuit string defining the parameter layout of the cached vectors.
        No structure is assumed: the ohmic/arc-resistor positions are derived from
        this circuit. ``None`` loads the default from ``config_oracle_defaults.yml``
        (``ecm.circuit``); pass ``cache.get("circuit")`` if the cache carries one.

    Returns
    -------
    dict
        Mapping with three keys: ``mean_arc_ratio`` — mean (sum arc R)/(ohmic R)
        across all cycles and states (or ``None``); ``r1_growth_pct`` —
        ``(R1[last]/R1[first] - 1) * 100`` from the charge ECM (or ``None``); and
        ``soh_fade_per_cycle`` — mean per-cycle real-SOH loss over the window (or
        ``None``). The two ECM targets come back ``None`` for an EIS-less cache
        (``ecm_charge``/``ecm_discharge`` all null, as with CALCE/Oxford/MATR),
        leaving ``soh_fade_per_cycle`` as the sole health signal. Outliers
        (ohmic R < 1e-6) are excluded as AutoEIS blowup artefacts.
    """
    circuit = _resolve_circuit(circuit or cache.get("circuit"))
    ohmic_i, arc_is = _ecm_indices(circuit)
    data = cache["data"]
    cycles = cache["cycles"]

    arc_ratios: list[float] = []
    r1_first = r1_last = None

    for cyc in cycles:
        entry = data[cyc]
        for state_key in ("ecm_charge", "ecm_discharge"):
            ecm = entry.get(state_key)
            if ecm is None:
                continue
            r1 = ecm[ohmic_i]
            if r1 < 1e-6:
                continue
            arc_ratios.append(sum(ecm[i] for i in arc_is) / r1)

        # R1 growth from charge state only (consistent with oracle history key)
        ecm_c = entry.get("ecm_charge")
        if ecm_c is not None and ecm_c[ohmic_i] >= 1e-6:
            if r1_first is None:
                r1_first = ecm_c[ohmic_i]
            r1_last = ecm_c[ohmic_i]

    r1_growth = None
    if r1_first is not None and r1_last is not None:
        r1_growth = (r1_last / r1_first - 1.0) * 100.0

    # Capacity-fade target: mean per-cycle SOH loss across the cache window. For
    # datasets that ship capacity/cycling but no EIS (CALCE / Oxford / MATR) this
    # is the only health signal — their caches carry per-cycle real_soh with null
    # ECMs, so the two ECM targets above are None and this drives the fit. Computed
    # over the same cycles run_oracle_candidate replays, so it is directly
    # comparable to that function's oracle_soh_fade_per_cycle.
    soh_series = _real_soh_series(cache)
    soh_fade = None
    if len(soh_series) >= 2:
        fade = (soh_series[0] - soh_series[-1]) / (len(soh_series) - 1)
        if fade > 0:
            soh_fade = float(fade)

    return {
        "mean_arc_ratio": float(np.mean(arc_ratios)) if arc_ratios else None,
        "r1_growth_pct":  r1_growth,
        "soh_fade_per_cycle": soh_fade,
    }


# ---------------------------------------------------------------------------
# Run an oracle candidate + C-rate probes (operate on cache + a fresh oracle)
# ---------------------------------------------------------------------------

def run_oracle_candidate(
    cache: dict,
    kinetics_scale: float,
    sei_rate_scale: float,
    dead_li_decay_scale: float,
    plating_rate_scale: float,
    preset: str,
    capacity_check: bool,
    circuit: str | None = None,
    calibration_model: str = CALIBRATION_MODEL,
    chemistry: str = "Chen2020",
) -> dict[str, Any]:
    """Replay cached protocols through one oracle candidate and return metrics.

    ``circuit`` sets the oracle's ECM structure (and the ohmic/arc-resistor
    positions the arc-ratio / R1-growth metrics read). It is not assumed: ``None``
    loads the default from ``config_oracle_defaults.yml`` (``ecm.circuit``).
    ``calibration_model`` pins the reduced-order model (SPMe) for the sweep;
    ``chemistry`` selects the PyBaMM parameter set (#14).
    """
    circuit = _resolve_circuit(circuit)
    ohmic_i, arc_is = _ecm_indices(circuit)
    oracle = PyBaMMOracle(
        model=calibration_model,
        chemistry=chemistry,
        degradation_preset=preset,
        capacity_check=capacity_check,
        real_cell_capacity_mah=float(cache["real_cell_capacity_mah"]),
        kinetics_scale=kinetics_scale,
        sei_rate_scale=sei_rate_scale,
        dead_li_decay_scale=dead_li_decay_scale,
        plating_rate_scale=plating_rate_scale,
        circuit=circuit,   # ECM layout matches the index-based reads below
    )
    oracle.reset()

    arc_ratios: list[float] = []
    r1_first = r1_last = None
    oracle_failure = False
    n_completed = 0
    soh_history: list[float] = []

    for cyc in cache["cycles"]:
        protocol = np.array(cache["data"][cyc]["protocol"], dtype=np.float64)
        try:
            oracle(protocol)
        except OracleFailure as exc:
            log.warning("  Oracle EOL at cycle %s: %s", cyc, exc)
            oracle_failure = True
            break
        n_completed += 1

        h = oracle._history[-1]
        soh = h.get("end_soh")
        if soh is not None:
            soh_history.append(float(soh))

        for state_key in ("ecm_params_charge", "ecm_params_discharge"):
            params = h.get(state_key)
            if params is None:
                continue
            r1 = params[ohmic_i]
            if r1 < 1e-6:
                continue
            arc_ratios.append(sum(params[i] for i in arc_is) / r1)

        # R1 growth from charge params
        p_c = h.get("ecm_params_charge")
        if p_c is not None and p_c[ohmic_i] >= 1e-6:
            if r1_first is None:
                r1_first = float(p_c[ohmic_i])
            r1_last = float(p_c[ohmic_i])

    r1_growth = None
    if r1_first is not None and r1_last is not None:
        r1_growth = (r1_last / r1_first - 1.0) * 100.0

    # Mean per-cycle SOH loss over the replayed window — the oracle-side match to
    # compute_real_targets' soh_fade_per_cycle (same formula, same cycles), and the
    # metric the capacity-fade calibration term fits against.
    oracle_soh_fade_per_cycle = None
    if len(soh_history) >= 2:
        oracle_soh_fade_per_cycle = (soh_history[0] - soh_history[-1]) / (len(soh_history) - 1)

    # Implied EOL cycle: if the oracle actually failed within the window, that IS
    # the EOL cycle. Otherwise extrapolate from the mean per-cycle capacity loss
    # rate observed over the window to the eol_capacity_fraction threshold (0.80
    # SOH). Linear extrapolation is rough but enough to catch order-of-magnitude
    # errors either way.
    implied_eol_cycle = None
    if oracle_failure:
        implied_eol_cycle = float(n_completed)
    elif oracle_soh_fade_per_cycle is not None and oracle_soh_fade_per_cycle > 1e-6:
        implied_eol_cycle = 0.20 / oracle_soh_fade_per_cycle

    return {
        "kinetics_scale":       kinetics_scale,
        "sei_rate_scale":       sei_rate_scale,
        "dead_li_decay_scale":  dead_li_decay_scale,
        "plating_rate_scale":   plating_rate_scale,
        "oracle_arc_ratio":     float(np.mean(arc_ratios)) if arc_ratios else None,
        "oracle_r1_growth_pct": r1_growth,
        "oracle_soh_fade_per_cycle": oracle_soh_fade_per_cycle,
        "oracle_failure":       oracle_failure,
        "n_cycles_completed":   n_completed,
        "implied_eol_cycle":    implied_eol_cycle,
    }


def collect_eis_comparison(
    cache: dict,
    best: dict,
    *,
    preset: str = "accelerated",
    capacity_check: bool = True,
    circuit: str | None = None,
    calibration_model: str = CALIBRATION_MODEL,
    chemistry: str = "Chen2020",
    max_panels: int = 2,
) -> dict | None:
    """Collect oracle-vs-ground-truth EIS spectra for the winning candidate.

    Rebuilds one oracle with *best*'s scales, replays the cache's protocols, and
    pairs the oracle's synthesized charge-state spectrum at representative cycles
    (first + last with a real ECM) against the real cell's spectrum reconstructed
    from the cached ECM parameters — same circuit, same frequency grid. Both are
    returned in the raw ``(freq, Re, -Im)`` convention plus their ohmic R, so the
    plotter ({func}`battery_oracle.tune_plots.plot_eis_comparison`) can normalise.

    Returns ``None`` when there is nothing to compare: a capacity-only (EIS-less)
    cache whose ECMs are all null — the CALCE/Oxford/MATR case — so the automatic
    plot is a graceful no-op there. AutoEIS/oracle failures propagate to the
    caller, which guards them.

    Note: this replays the cache once more (a single confirmation run, negligible
    beside the ``n_trials`` full replays the search already paid for). It needs a
    live oracle, so — unlike the Pareto/history/alignment plots — it is produced at
    calibration time and is not regenerable from the CSV/JSON sidecars alone.
    """
    circuit = _resolve_circuit(circuit or cache.get("circuit"))
    data = cache["data"]
    ecm_cycles = [c for c in cache["cycles"] if data[c].get("ecm_charge") is not None]
    if not ecm_cycles:
        log.info("EIS comparison skipped: cache has no ECM data (capacity-only mode).")
        return None

    # First + last cycle carrying a real ECM (shows arc growth over ageing). Their
    # positions in cache["cycles"] index 1:1 into the oracle history replayed below.
    sel_labels = ([ecm_cycles[0]] if len(ecm_cycles) == 1
                  else [ecm_cycles[0], ecm_cycles[-1]])[:max_panels]
    sel_pos = {lbl: cache["cycles"].index(lbl) for lbl in sel_labels}
    max_pos = max(sel_pos.values())

    oracle = PyBaMMOracle(
        model=calibration_model, chemistry=chemistry, degradation_preset=preset,
        capacity_check=capacity_check,
        real_cell_capacity_mah=float(cache["real_cell_capacity_mah"]),
        kinetics_scale=best["kinetics_scale"], sei_rate_scale=best["sei_rate_scale"],
        dead_li_decay_scale=best["dead_li_decay_scale"],
        plating_rate_scale=best["plating_rate_scale"], circuit=circuit,
    )
    oracle.reset()
    for i, cyc in enumerate(cache["cycles"]):
        try:
            oracle(np.array(data[cyc]["protocol"], dtype=np.float64))
        except OracleFailure as exc:
            log.warning("EIS comparison: oracle EOL at cycle %s before all panels "
                        "collected: %s", cyc, exc)
            break
        if i >= max_pos:
            break

    freq = np.asarray(oracle.frequencies, dtype=float)
    hf = int(np.argmax(freq))
    panels = []
    for lbl in sel_labels:
        pos = sel_pos[lbl]
        if pos >= len(oracle._history):
            continue
        h = oracle._history[pos]
        oz_re = np.asarray(h["Z_charge_real"], dtype=float)
        oz_nim = np.asarray(h["Z_charge_neg_imag"], dtype=float)
        try:
            Zr = _reconstruct_ecm_spectrum(circuit, data[lbl]["ecm_charge"], freq)
        except Exception as exc:
            log.warning("EIS comparison: real-spectrum reconstruction failed for "
                        "cycle %s: %s", lbl, exc)
            continue
        panels.append({
            "cycle_label": str(lbl),
            "oracle": {"z_re": oz_re, "z_neg_im": oz_nim, "r_ohmic": float(oz_re[hf])},
            "real":   {"z_re": Zr.real, "z_neg_im": -Zr.imag,
                       "r_ohmic": float(Zr.real[hf])},
        })
    if not panels:
        return None
    return {"circuit": circuit, "frequencies": freq, "panels": panels}


def run_crate_sensitivity_probe(
    cache: dict,
    kinetics_scale: float,
    sei_rate_scale: float,
    dead_li_decay_scale: float,
    plating_rate_scale: float,
    preset: str,
    capacity_check: bool,
    probe_cycles: int = 4,
    low_c_mult: float = 1.0,
    high_c_mult: float = 8.0,
    circuit: str | None = None,
    chemistry: str = "Chen2020",
) -> dict[str, Any]:
    """Run two short oracle simulations at low vs high C-rate; return loss rates.

    Holds the cached median protocol fixed except for C_rate_1 (index 0 of the
    6-D action vector), runs *probe_cycles* cycles at each level from a fresh
    cell, and reports the ratio of mean per-cycle SOH loss. A fresh oracle is
    built per level so neither run's cell state leaks into the other. ``circuit``
    (``None`` -> loaded from the YAML config) sets the oracle's ECM structure.
    """
    circuit = _resolve_circuit(circuit)
    protocols = np.array([cache["data"][c]["protocol"] for c in cache["cycles"]], dtype=np.float64)
    base_protocol = np.median(protocols, axis=0)
    real_cap_mah = float(cache["real_cell_capacity_mah"])
    low_ma  = low_c_mult * real_cap_mah
    high_ma = high_c_mult * real_cap_mah

    def _mean_loss_per_cycle(c_rate_ma: float) -> float | None:
        oracle = PyBaMMOracle(
            model=CALIBRATION_MODEL,
            chemistry=chemistry,
            degradation_preset=preset,
            capacity_check=capacity_check,
            real_cell_capacity_mah=real_cap_mah,
            kinetics_scale=kinetics_scale,
            sei_rate_scale=sei_rate_scale,
            dead_li_decay_scale=dead_li_decay_scale,
            plating_rate_scale=plating_rate_scale,
            circuit=circuit,
        )
        oracle.reset()
        proto = base_protocol.copy()
        proto[0] = c_rate_ma
        soh = [1.0]
        for _ in range(probe_cycles):
            try:
                oracle(proto)
            except OracleFailure as exc:
                log.warning("  [crate-probe @ %.0f mA] EOL/solver failure after %d cycle(s): %s",
                            c_rate_ma, len(soh) - 1, exc)
                break
            soh.append(float(oracle._history[-1]["end_soh"]))
        if len(soh) < 2:
            return None
        return (soh[0] - soh[-1]) / (len(soh) - 1)

    loss_low  = _mean_loss_per_cycle(low_ma)
    loss_high = _mean_loss_per_cycle(high_ma)
    ratio = None
    if loss_low is not None and loss_low > 1e-9 and loss_high is not None:
        ratio = loss_high / loss_low

    return {
        "crate_loss_low":          loss_low,
        "crate_loss_high":         loss_high,
        "crate_sensitivity_ratio": ratio,
        "crate_low_ma":            low_ma,
        "crate_high_ma":           high_ma,
    }


def run_crate2_slope_probe(
    cache: dict,
    kinetics_scale: float,
    sei_rate_scale: float,
    dead_li_decay_scale: float,
    plating_rate_scale: float,
    preset: str,
    capacity_check: bool,
    probe_cycles: int = 5,
    c2_levels_mult: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    circuit: str | None = None,
    chemistry: str = "Chen2020",
) -> dict[str, Any]:
    """Oracle-side analogue of the real multi-cell C_rate_2 slope.

    Fits the oracle's OWN slope d(capacity_fade_mAh)/d(C_rate_2_mA) from
    *c2_levels_mult* synthetic protocols (varying only C_rate_2, index 1 of the
    6-D action vector, holding the cache's median protocol otherwise fixed), in
    the SAME units as the real-data slope so the two compare directly. A fresh
    oracle is built per level. ``circuit`` (``None`` -> loaded from the YAML
    config) sets the oracle's ECM structure.
    """
    circuit = _resolve_circuit(circuit)
    protocols = np.array([cache["data"][c]["protocol"] for c in cache["cycles"]], dtype=np.float64)
    base_protocol = np.median(protocols, axis=0)
    real_cap_mah = float(cache["real_cell_capacity_mah"])
    c2_levels_ma = [m * real_cap_mah for m in c2_levels_mult]

    fade_mAh: list[float] = []
    for c2_ma in c2_levels_ma:
        oracle = PyBaMMOracle(
            model=CALIBRATION_MODEL,
            chemistry=chemistry,
            degradation_preset=preset,
            capacity_check=capacity_check,
            real_cell_capacity_mah=real_cap_mah,
            kinetics_scale=kinetics_scale,
            sei_rate_scale=sei_rate_scale,
            dead_li_decay_scale=dead_li_decay_scale,
            plating_rate_scale=plating_rate_scale,
            circuit=circuit,
        )
        oracle.reset()
        proto = base_protocol.copy()
        proto[1] = c2_ma
        soh = [1.0]
        for _ in range(probe_cycles):
            try:
                oracle(proto)
            except OracleFailure as exc:
                log.warning("  [crate2-slope-probe @ C2=%.0f mA] EOL/solver failure "
                            "after %d cycle(s): %s", c2_ma, len(soh) - 1, exc)
                break
            soh.append(float(oracle._history[-1]["end_soh"]))
        if len(soh) < 2:
            fade_mAh.append(float("nan"))
            continue
        mean_loss_frac_per_cycle = (soh[0] - soh[-1]) / (len(soh) - 1)
        fade_mAh.append(mean_loss_frac_per_cycle * real_cap_mah)

    c2_arr = np.asarray(c2_levels_ma)
    fade_arr = np.asarray(fade_mAh)
    valid = np.isfinite(fade_arr)
    slope = None
    if valid.sum() >= 2 and len(np.unique(c2_arr[valid])) >= 2:
        slope = float(np.polyfit(c2_arr[valid], fade_arr[valid], 1)[0])

    return {
        "oracle_slope_mAh_per_mA": slope,
        "c2_levels_mA":            c2_levels_ma,
        "fade_mAh":                fade_mAh,
    }


def slope_match_error(oracle_slope: float | None, real_slope: dict) -> float:
    """Error term for how well *oracle_slope* matches the real bootstrap CI.

    Zero inside [ci_lo, ci_hi]; outside, grows linearly in units of CI
    half-widths past the nearer edge. inf if the oracle slope is unavailable.
    """
    if oracle_slope is None or not np.isfinite(oracle_slope):
        return float("inf")
    lo, hi = real_slope["ci_lo"], real_slope["ci_hi"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 0.0  # bootstrap CI unavailable -- don't penalize on missing data
    if lo <= oracle_slope <= hi:
        return 0.0
    half_width = max((hi - lo) / 2.0, 1e-12)
    center = (hi + lo) / 2.0
    return abs(oracle_slope - center) / half_width - 1.0


def score_candidate(
    result: dict, real_targets: dict, preset: str = "accelerated",
    crate_sensitivity_min: float = 3.0,
    real_crate2_slope: dict | None = None,
) -> float:
    """Combined relative error (lower = better). Returns inf if nothing can be fit.

    Two calibration modes, selected by which real targets the cache provides:

    * **EIS/ECM cache** — ``mean_arc_ratio`` and/or ``r1_growth_pct`` present:
      arc-ratio relative error + R1-growth relative error + an EOL-rate log-ratio
      *plausibility* anchor to the preset's documented cycle-life midpoint (a
      constant, not a data fit). This is the original scoring, unchanged.
    * **EIS-less cache** — no ECM target (CALCE/Oxford/MATR ship capacity/cycling
      but no EIS): the capacity-fade term (log-ratio of oracle vs real per-cycle
      SOH loss, from ``real_soh``) is the sole health signal, and replaces the
      preset EOL anchor.

    A term whose real target IS present but whose oracle value is missing scores
    inf (that candidate failed to produce a comparable metric). The legacy
    C_rate_1 sensitivity term (skipped when the probe wasn't run) and the C_rate_2
    slope-matching term (see :func:`slope_match_error`) are added in both modes.
    """
    real_arc = real_targets.get("mean_arc_ratio")
    real_r1 = real_targets.get("r1_growth_pct")
    real_fade = real_targets.get("soh_fade_per_cycle")
    have_ecm_target = bool(real_arc) or bool(real_r1 and abs(real_r1) > 1e-3)

    terms: list[float] = []
    if have_ecm_target:
        if real_arc:
            oracle_arc = result.get("oracle_arc_ratio")
            terms.append(abs(oracle_arc / real_arc - 1.0)
                         if oracle_arc is not None else float("inf"))
        if real_r1 and abs(real_r1) > 1e-3:
            oracle_r1 = result.get("oracle_r1_growth_pct")
            terms.append(abs(oracle_r1 / real_r1 - 1.0)
                         if oracle_r1 is not None else float("inf"))
        eol_target = _eol_target_cycles(preset)
        implied_eol = result.get("implied_eol_cycle")
        terms.append(abs(math.log(implied_eol / eol_target))
                     if eol_target and implied_eol and implied_eol > 0 else float("inf"))
    elif real_fade and real_fade > 0:
        oracle_fade = result.get("oracle_soh_fade_per_cycle")
        terms.append(abs(math.log(oracle_fade / real_fade))
                     if oracle_fade and oracle_fade > 0 else float("inf"))
    else:
        # No ECM target and no measured capacity fade — nothing to fit against.
        return float("inf")

    ratio = result.get("crate_sensitivity_ratio")
    if ratio is not None and ratio > 0:
        terms.append(max(0.0, math.log(crate_sensitivity_min / ratio)))
    elif not result.get("crate_probe_skipped", False):
        terms.append(float("inf"))
    if real_crate2_slope is not None:
        terms.append(slope_match_error(
            result.get("oracle_slope_mAh_per_mA"), real_crate2_slope))

    return sum(terms)


# ---------------------------------------------------------------------------
# EIS non-stationarity drift calibration (oracle-side; real target passed in)
# ---------------------------------------------------------------------------

def _linkk_lowhigh_ratio(freq, Z, c: float = 0.85, max_M: int = 50) -> float:
    """linKK low/high-freq residual ratio — the non-stationarity signature.

    A spectrum that drifts during the sweep has KK residuals elevated at low
    frequency; returns mean|res|(low tercile) / mean|res|(high tercile). The
    linKK call itself (input sanitizing, NumPy-2.x fix, stdout suppression)
    lives in ``_eis/kk.py``'s :func:`linkk_residuals`.

    ``c``/``max_M`` default to the same values as ``_eis/kk.py``'s
    ``linkk_rmse`` (config_oracle_defaults.yml's ``eis.linkk`` section).
    """
    out = linkk_residuals(freq, Z, c=c, max_M=max_M)
    if out is None:
        return float("nan")
    f, res_real, res_imag = out
    res = np.sqrt(res_real ** 2 + res_imag ** 2)
    q1, q2 = np.quantile(f, 1 / 3), np.quantile(f, 2 / 3)
    lo, hi = res[f < q1].mean(), res[f > q2].mean()
    return float(lo / max(hi, 1e-12))


def calibrate_drift(
    *,
    real_lowhigh: float,
    kinetics_scale: float,
    sei_rate_scale: float,
    dead_li_decay_scale: float,
    plating_rate_scale: float,
    preset: str,
    capacity_check: bool,
    real_cap_mah: float,
    rest_s: float = 1200.0,
    tau_relax_s: float = 600.0,
    n_periods: float = 4.0,
    probe_cycles: int = 3,
    scale_grid: list[float] | None = None,
    chemistry: str = "Chen2020",
) -> dict[str, Any]:
    """Fit ``eis_drift_scale`` to match a real cell's linKK low/high-freq ratio.

    The real target ratio *real_lowhigh* is supplied by the caller (compute it
    from real reference spectra with :func:`_linkk_lowhigh_ratio`), keeping this
    routine dataset-free. For each candidate ``drift_scale`` a fresh oracle is
    built at ``rest_s`` and the same ratio measured on its spectra; the closest
    match is returned, along with the ratio at a very long rest (rest-response).

    KNOWN LIMITATION (default OFF). Only the ``v0(t)`` drift-leakage part of
    Hallemans et al. 2023 Eq (43) is modelled and integer-period single-sine
    extraction rejects slow drift, so the oracle ratio barely moves with
    ``drift_scale`` and normally cannot reach the real target; this typically
    returns drift_scale≈0. Retained for provenance.
    """
    if scale_grid is None:
        scale_grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]

    def _oracle_ratio(drift_scale: float, rest: float) -> float:
        oracle = PyBaMMOracle(
            model=CALIBRATION_MODEL,
            chemistry=chemistry,
            degradation_preset=preset,
            capacity_check=capacity_check,
            real_cell_capacity_mah=real_cap_mah,
            kinetics_scale=kinetics_scale,
            sei_rate_scale=sei_rate_scale,
            dead_li_decay_scale=dead_li_decay_scale,
            plating_rate_scale=plating_rate_scale,
            rest_s=rest,
            eis_drift_scale=drift_scale,
            eis_drift_tau_s=tau_relax_s,
            eis_drift_n_periods=n_periods,
        )
        oracle.reset()
        proto = np.array([real_cap_mah, 0.75 * real_cap_mah, 0.16, 0.08, real_cap_mah, 0.38])
        ratios = []
        for _ in range(probe_cycles):
            try:
                oracle(proto)
            except OracleFailure:
                break
            r = _linkk_lowhigh_ratio(oracle.frequencies, oracle._last_Z)
            if np.isfinite(r):
                ratios.append(r)
        return float(np.median(ratios)) if ratios else float("nan")

    grid = []
    for s in scale_grid:
        ratio = _oracle_ratio(s, rest_s)
        grid.append({"drift_scale": s, "oracle_lowhigh": ratio})
        log.info("  [drift-cal] drift_scale=%.3g -> oracle low/high linKK=%.2f (real target %.2f)",
                 s, ratio, real_lowhigh)

    finite = [g for g in grid if np.isfinite(g["oracle_lowhigh"])]
    if finite and np.isfinite(real_lowhigh):
        best = min(finite, key=lambda g: abs(g["oracle_lowhigh"] - real_lowhigh))
    else:
        best = {"drift_scale": 0.0, "oracle_lowhigh": grid[0]["oracle_lowhigh"]}
        log.warning("  [drift-cal] no finite match (real=%.3g); leaving drift disabled", real_lowhigh)

    wellrested = (_oracle_ratio(best["drift_scale"], 1e6)
                  if best["drift_scale"] > 0.0 else best["oracle_lowhigh"])

    return {
        "drift_scale": best["drift_scale"],
        "drift_tau_s": tau_relax_s,
        "drift_n_periods": n_periods,
        "real_lowhigh": real_lowhigh,
        "oracle_lowhigh": best["oracle_lowhigh"],
        "oracle_lowhigh_wellrested": wellrested,
        "grid": grid,
    }


# ---------------------------------------------------------------------------
# Bayesian-optimisation driver
# ---------------------------------------------------------------------------

def _make_sampler(sampler: str, seed: int):
    """TPE (default) or GP Optuna sampler. Shared by the sequential and parallel
    drivers so both search the space the same way."""
    import optuna
    if sampler == "gp":
        return optuna.samplers.GPSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed, n_startup_trials=8)


def _split_trials(n_trials: int, n_workers: int) -> list[int]:
    """Divide *n_trials* across *n_workers* as evenly as possible (sums to n_trials)."""
    base, extra = divmod(max(n_trials, 0), max(n_workers, 1))
    return [base + (1 if i < extra else 0) for i in range(n_workers)]


def _native(obj):
    """Recursively convert numpy scalars/arrays to JSON-serialisable Python types.

    Parallel trial results travel back to the parent through Optuna
    ``set_user_attr`` (JSON-encoded in the RDB storage), which rejects numpy types.
    """
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _suggest_scales(trial, ranges: dict) -> tuple[float, float, float, float]:
    """The four log-space scale suggestions — identical in both drivers."""
    return (
        trial.suggest_float("kinetics_scale",      ranges["ks_min"],  ranges["ks_max"],  log=True),
        trial.suggest_float("sei_rate_scale",      ranges["srs_min"], ranges["srs_max"], log=True),
        trial.suggest_float("dead_li_decay_scale", ranges["dds_min"], ranges["dds_max"], log=True),
        trial.suggest_float("plating_rate_scale",  ranges["prs_min"], ranges["prs_max"], log=True),
    )


def _evaluate_candidate(cache: dict, ks: float, srs: float, dds: float, prs: float,
                        cfg: dict) -> tuple[dict, float]:
    """Replay one candidate through the oracle (+ optional probes) and score it.

    Single source of truth for a trial's evaluation, shared by the sequential
    driver and the parallel workers so both produce identical results/scores.
    ``cfg`` bundles the (picklable) run configuration built in :func:`calibrate_oracle`.
    """
    t0 = time.time()
    result = run_oracle_candidate(cache, ks, srs, dds, prs, cfg["preset"],
                                  cfg["capacity_check"], circuit=cfg["circuit"],
                                  chemistry=cfg["chemistry"])
    if cfg["skip_crate_probe"]:
        result["crate_probe_skipped"] = True
        result["crate_sensitivity_ratio"] = None
    else:
        result.update(run_crate_sensitivity_probe(
            cache, ks, srs, dds, prs, cfg["preset"], cfg["capacity_check"],
            probe_cycles=cfg["crate_probe_cycles"], low_c_mult=cfg["crate_probe_low_c"],
            high_c_mult=cfg["crate_probe_high_c"], circuit=cfg["circuit"],
            chemistry=cfg["chemistry"]))
        result["crate_probe_skipped"] = False
    if not cfg["skip_crate2_slope"]:
        result.update(run_crate2_slope_probe(
            cache, ks, srs, dds, prs, cfg["preset"], cfg["capacity_check"],
            probe_cycles=cfg["crate2_probe_cycles"], c2_levels_mult=tuple(cfg["crate2_levels"]),
            circuit=cfg["circuit"], chemistry=cfg["chemistry"]))
    result["runtime_s"] = round(time.time() - t0, 1)
    score = score_candidate(result, cfg["real_targets"], cfg["preset"],
                            cfg["crate_sensitivity_min"], cfg["real_crate2_slope"])
    result["score"] = score if math.isfinite(score) else None
    return result, score


def _parallel_worker(study_name: str, storage_url: str, n_trials_worker: int,
                     seed: int, sampler: str, ranges: dict, cache: dict,
                     cfg: dict) -> None:
    """One worker PROCESS: load the shared study and run ``n_trials_worker`` trials.

    Runs in a spawned subprocess so AutoEIS's numpyro/JAX global + tracing state is
    process-local and never shared across concurrent trials — the reason this
    engine parallelises by process, not thread. (Threaded ``n_jobs`` silently
    corrupted ~1/5 of ECM fits into the Randles-stub fallback via numpyro
    param-store collisions and JAX tracer escapes.) A subprocess can't append to
    the parent's list, so each trial hands its result back through the ``result``
    user-attr in the shared RDB.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = optuna.storages.RDBStorage(
        storage_url, engine_kwargs={"connect_args": {"timeout": 60}})
    study = optuna.load_study(study_name=study_name, storage=storage,
                              sampler=_make_sampler(sampler, seed))

    def objective(trial: "optuna.Trial") -> float:
        ks, srs, dds, prs = _suggest_scales(trial, ranges)
        result, score = _evaluate_candidate(cache, ks, srs, dds, prs, cfg)
        result["trial_number"] = trial.number
        trial.set_user_attr("result", _native(result))
        if not math.isfinite(score):
            raise optuna.TrialPruned()
        return score

    study.optimize(objective, n_trials=n_trials_worker, n_jobs=1, show_progress_bar=False)


def _calibrate_oracle_parallel(cache, real_targets, *, n_trials, n_jobs, sampler,
                               seed, ranges, cfg, warm_start_ks, warm_start_srs):
    """Process-parallel calibration: N worker processes over a shared Optuna RDB.

    Threads are unsafe here (see :func:`_parallel_worker`); each worker is a
    separate spawned process with its own interpreter and JAX state, so concurrent
    trials cannot collide. Trial *order* across workers is non-deterministic
    (inherent to parallel BO), but every fit is race-free and comes from the same
    AutoEIS fitter as the real targets.
    """
    import optuna

    n_workers = max(1, min(n_jobs, n_trials))
    counts = _split_trials(n_trials, n_workers)
    tmp_dir = Path(tempfile.mkdtemp(prefix="botune_"))
    storage_url = f"sqlite:///{tmp_dir / 'study.db'}"
    study_name = f"tune_{uuid.uuid4().hex[:8]}"
    storage = optuna.storages.RDBStorage(
        storage_url, engine_kwargs={"connect_args": {"timeout": 60}})
    study = optuna.create_study(study_name=study_name, storage=storage,
                                direction="minimize", sampler=_make_sampler(sampler, seed))

    if warm_start_ks and warm_start_srs:
        warm_pairs = list(itertools.product(warm_start_ks, warm_start_srs))
        log.info("Warm-starting BO with %d grid point(s): ks=%s  srs=%s",
                 len(warm_pairs), warm_start_ks, warm_start_srs)
        for ks_init, srs_init in warm_pairs:
            study.enqueue_trial({"kinetics_scale": ks_init, "sei_rate_scale": srs_init,
                                 "dead_li_decay_scale": 1.0, "plating_rate_scale": 1.0})

    log.info("Starting process-parallel Optuna BO [%d worker process(es), n_trials=%d, "
             "sampler=%s, preset=%s]", n_workers, n_trials, sampler, cfg["preset"])
    ctx = multiprocessing.get_context("spawn")
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers,
                                                    mp_context=ctx) as ex:
            futures = [
                ex.submit(_parallel_worker, study_name, storage_url, counts[i],
                          seed + i + 1, sampler, ranges, cache, cfg)
                for i in range(n_workers) if counts[i] > 0
            ]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()   # re-raise any worker exception in the parent
        results = [dict(t.user_attrs["result"])
                   for t in study.get_trials(deepcopy=False)
                   if "result" in t.user_attrs]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not results:
        raise RuntimeError(
            "Process-parallel calibration produced no scored trials (every candidate "
            "failed or was pruned). Re-run with n_jobs=1 to see per-trial logs.")
    scored = sorted(
        [(r, score_candidate(r, real_targets, cfg["preset"],
                             cfg["crate_sensitivity_min"], cfg["real_crate2_slope"]))
         for r in results],
        key=lambda x: x[1])
    best, best_score = scored[0]
    log.info("Process-parallel BO done: %d trials collected, best score %.3f",
             len(results), best_score)
    return {"best": best, "best_score": best_score, "results": results, "scored": scored}


def calibrate_oracle(
    cache: dict,
    real_targets: dict,
    *,
    preset: str = "accelerated",
    ks_min: float = 0.10, ks_max: float = 0.50,
    srs_min: float = 0.01, srs_max: float = 1.0,
    dds_min: float = 0.1, dds_max: float = 1000.0,
    prs_min: float = 0.01, prs_max: float = 10.0,
    n_trials: int = 35,
    n_jobs: int = 1,
    sampler: str = "tpe",
    seed: int = 42,
    capacity_check: bool = True,
    skip_crate_probe: bool = True,
    crate_probe_cycles: int = 4,
    crate_probe_low_c: float = 1.0,
    crate_probe_high_c: float = 8.0,
    skip_crate2_slope: bool = False,
    crate2_probe_cycles: int = 5,
    crate2_levels: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    crate_sensitivity_min: float = 3.0,
    real_crate2_slope: dict | None = None,
    warm_start_ks: list[float] | None = None,
    warm_start_srs: list[float] | None = None,
    circuit: str | None = None,
    chemistry: str = "Chen2020",
) -> dict[str, Any]:
    """Run the Optuna BO over (kinetics, sei_rate, dead_li_decay, plating)_scale.

    Pure engine: operates on *cache* + precomputed *real_targets* (and optional
    *real_crate2_slope*); no dataset access. Returns a dict with ``best``,
    ``best_score``, ``results`` (all trials) and ``scored`` (ranked).

    ``circuit`` is the ECM structure the candidate oracles fit and the metrics
    read; no layout is assumed. ``None`` loads the default from the YAML config
    (``config_oracle_defaults.yml`` ``ecm.circuit``) — pass ``cache.get("circuit")``
    when the cache was featurized with a specific circuit.

    ``n_jobs`` > 1 runs trials in parallel **processes** (not threads): each worker
    is a spawned subprocess with its own interpreter, coordinating through a shared
    temporary Optuna RDB. Threads are unsafe because AutoEIS's numpyro/JAX inference
    keeps global + tracing state that races across threads — under the old thread
    backend ~1 in 5 ECM fits silently fell back to the Randles stub (numpyro
    param-store collisions, JAX tracer escapes), biasing the score and breaking
    reproducibility. Each process worker gets its own JAX state, so concurrent
    trials cannot collide; the trade-offs are per-worker JAX/PyBaMM import warmup
    and non-deterministic trial *order* (inherent to any parallel BO). ``n_jobs=1``
    (default) uses the fast in-memory single-process path with full per-trial logs.
    """
    import optuna

    circuit = _resolve_circuit(circuit or cache.get("circuit"))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Everything a trial needs to evaluate + score a candidate, bundled so it is
    # picklable and used verbatim by both the sequential objective and the parallel
    # workers (keeps the two drivers from diverging).
    cfg = {
        "preset": preset, "circuit": circuit, "chemistry": chemistry,
        "capacity_check": capacity_check,
        "skip_crate_probe": skip_crate_probe, "crate_probe_cycles": crate_probe_cycles,
        "crate_probe_low_c": crate_probe_low_c, "crate_probe_high_c": crate_probe_high_c,
        "skip_crate2_slope": skip_crate2_slope, "crate2_probe_cycles": crate2_probe_cycles,
        "crate2_levels": tuple(crate2_levels), "crate_sensitivity_min": crate_sensitivity_min,
        "real_crate2_slope": real_crate2_slope, "real_targets": real_targets,
    }
    ranges = {"ks_min": ks_min, "ks_max": ks_max, "srs_min": srs_min, "srs_max": srs_max,
              "dds_min": dds_min, "dds_max": dds_max, "prs_min": prs_min, "prs_max": prs_max}

    if n_jobs and n_jobs > 1:
        return _calibrate_oracle_parallel(
            cache, real_targets, n_trials=n_trials, n_jobs=n_jobs, sampler=sampler,
            seed=seed, ranges=ranges, cfg=cfg,
            warm_start_ks=warm_start_ks, warm_start_srs=warm_start_srs)

    # ── Sequential (single-process) path: identical search, full per-trial logs ──
    results: list[dict] = []

    def objective(trial: "optuna.Trial") -> float:
        ks, srs, dds, prs = _suggest_scales(trial, ranges)
        log.info(
            "── Trial %d/%d: ks=%.4f  srs=%.4f  dds=%.4f  prs=%.4f ──",
            trial.number + 1, n_trials, ks, srs, dds, prs,
        )
        result, score = _evaluate_candidate(cache, ks, srs, dds, prs, cfg)
        result["trial_number"] = trial.number
        results.append(result)
        ratio = result.get("crate_sensitivity_ratio")
        oracle_slope = result.get("oracle_slope_mAh_per_mA")
        log.info(
            "  arc_ratio=%s (real=%s)  r1_growth=%s (real=%s)  eol_cycle~%s (target=%s)  "
            "fade/cyc=%s (real=%s)  crate_ratio=%s (min=%.1f)  c2_slope=%s (real_CI=%s)  score=%.3f  %.0fs",
            f"{result['oracle_arc_ratio']:.2f}" if result["oracle_arc_ratio"] is not None else "n/a",
            f"{real_targets['mean_arc_ratio']:.2f}" if real_targets["mean_arc_ratio"] else "?",
            f"{result['oracle_r1_growth_pct']:.1f}%" if result["oracle_r1_growth_pct"] is not None else "n/a",
            f"{real_targets['r1_growth_pct']:.1f}%" if real_targets["r1_growth_pct"] else "?",
            f"{result['implied_eol_cycle']:.0f}" if result["implied_eol_cycle"] is not None else "n/a",
            _eol_target_cycles(preset) or "?",
            f"{result['oracle_soh_fade_per_cycle']:.4g}" if result.get("oracle_soh_fade_per_cycle") else "n/a",
            f"{real_targets['soh_fade_per_cycle']:.4g}" if real_targets.get("soh_fade_per_cycle") else "?",
            f"{ratio:.2f}" if ratio is not None else "n/a",
            crate_sensitivity_min,
            f"{oracle_slope:.5f}" if oracle_slope is not None else "n/a",
            f"[{real_crate2_slope['ci_lo']:.5f}, {real_crate2_slope['ci_hi']:.5f}]" if real_crate2_slope else "n/a",
            score if math.isfinite(score) else float("nan"),
            result["runtime_s"],
        )
        if not math.isfinite(score):
            raise optuna.TrialPruned()
        return score

    study = optuna.create_study(direction="minimize", sampler=_make_sampler(sampler, seed))

    # Warm-start: enqueue grid points from warm_start_ks / warm_start_srs
    if warm_start_ks and warm_start_srs:
        warm_pairs = list(itertools.product(warm_start_ks, warm_start_srs))
        log.info("Warm-starting BO with %d grid point(s): ks=%s  srs=%s",
                 len(warm_pairs), warm_start_ks, warm_start_srs)
        for ks_init, srs_init in warm_pairs:
            study.enqueue_trial({
                "kinetics_scale":      ks_init,
                "sei_rate_scale":      srs_init,
                "dead_li_decay_scale": 1.0,
                "plating_rate_scale":  1.0,
            })

    log.info("Starting Optuna BO [sampler=%s, n_trials=%d, n_jobs=1, preset=%s]",
             sampler, n_trials, preset)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    scored = sorted(
        [(r, score_candidate(r, real_targets, preset, crate_sensitivity_min, real_crate2_slope))
         for r in results],
        key=lambda x: x[1],
    )
    best, best_score = scored[0]
    return {"best": best, "best_score": best_score, "results": results, "scored": scored}


def print_summary(
    scored: list[tuple[dict, float]],
    best: dict,
    real_targets: dict,
    *,
    dataset: str,
    preset: str,
    cell_id: str,
    sampler: str,
    crate_sensitivity_min: float,
    real_crate2_slope: dict | None,
) -> None:
    """Print the ranked top-15 calibration trials (shared by both CLIs)."""
    arc_real_str = f"{real_targets['mean_arc_ratio']:.2f}" if real_targets["mean_arc_ratio"] else "n/a"
    r1_real_str  = f"{real_targets['r1_growth_pct']:.1f}%" if real_targets["r1_growth_pct"]  else "n/a"
    print(f"\n{'='*75}")
    print(f"  Calibration BO [{sampler.upper()}] — dataset={dataset}  preset={preset}  cell={cell_id}")
    eol_target = _eol_target_cycles(preset)
    eol_target_str = f"{eol_target:.0f}" if eol_target else "n/a"
    print(f"  Real targets:  arc_ratio={arc_real_str}   r1_growth={r1_real_str}   eol_cycle~{eol_target_str}")
    print(f"{'='*75}")
    header = (f"{'ks':>8}  {'srs':>8}  {'dds':>8}  {'prs':>8}  {'arc_ratio':>10}  "
               f"{'r1_growth%':>11}  {'eol_cyc':>8}  {'crate_x':>8}  {'c2_slope':>10}  "
               f"{'score':>8}  {'eol?':>5}")
    print(header)
    print("-" * len(header))
    best_row = best
    for r, sc in scored[:15]:
        eol = "yes" if r["oracle_failure"] else "no"
        marker = " <<" if r is best_row else ""
        eol_cyc = r.get("implied_eol_cycle")
        crate_x = r.get("crate_sensitivity_ratio")
        c2_slope = r.get("oracle_slope_mAh_per_mA")
        print(
            f"{r['kinetics_scale']:>8.4f}  {r['sei_rate_scale']:>8.4f}  {r['dead_li_decay_scale']:>8.2f}"
            f"  {r.get('plating_rate_scale', 1.0):>8.4f}"
            f"  {r['oracle_arc_ratio'] or float('nan'):>10.2f}"
            f"  {r['oracle_r1_growth_pct'] or float('nan'):>10.1f}%"
            f"  {eol_cyc if eol_cyc is not None else float('nan'):>8.0f}"
            f"  {crate_x if crate_x is not None else float('nan'):>8.2f}"
            f"  {c2_slope if c2_slope is not None else float('nan'):>10.5f}"
            f"  {sc:>8.3f}  {eol:>5}{marker}"
        )
    if len(scored) > 15:
        print(f"  ... ({len(scored) - 15} more trials not shown)")
    print()


# ---------------------------------------------------------------------------
# Write calibration config
# ---------------------------------------------------------------------------

def write_oracle_config(
    output_path: Path,
    dataset: str,
    preset: str,
    cell_id: str,
    n_cycles: int,
    best: dict[str, Any],
    real_targets: dict,
    all_results: list[dict],
    crate_sensitivity_min: float = 3.0,
    real_crate2_slope: dict | None = None,
    drift_result: dict | None = None,
    chemistry: str = "Chen2020",
) -> None:
    """Write YAML config with calibration provenance to *output_path*."""
    output_path = Path(output_path)
    oracle_defaults = load_oracle_config()
    _od_cycling = oracle_defaults.get("cycling", {}) or {}
    _od_eis = oracle_defaults.get("eis", {}) or {}
    _od_degradation = oracle_defaults.get("degradation", {}) or {}
    today = date.today().isoformat()
    ks_swept  = sorted({r["kinetics_scale"]       for r in all_results})
    srs_swept = sorted({r["sei_rate_scale"]        for r in all_results})
    dds_range = (
        min(r["dead_li_decay_scale"] for r in all_results),
        max(r["dead_li_decay_scale"] for r in all_results),
    )
    prs_range = (
        min(r.get("plating_rate_scale", 1.0) for r in all_results),
        max(r.get("plating_rate_scale", 1.0) for r in all_results),
    )

    arc_real  = real_targets["mean_arc_ratio"]
    arc_oracle = best["oracle_arc_ratio"]
    r1_real   = real_targets["r1_growth_pct"]
    r1_oracle = best["oracle_r1_growth_pct"]

    arc_close  = (arc_real and arc_oracle and abs(arc_oracle / arc_real - 1.0) < 0.25)
    arc_status = "validated" if arc_close else "partial — gap > 25%"
    r1_close   = (r1_real and r1_oracle and abs(r1_oracle / r1_real - 1.0) < 2.0)
    r1_status  = "validated" if r1_close else "partial — gap > 200% of target"
    dds_best   = best.get("dead_li_decay_scale", 1.0)
    prs_best   = best.get("plating_rate_scale", 1.0)

    eol_target  = _eol_target_cycles(preset, oracle_defaults)
    eol_implied = best.get("implied_eol_cycle")
    eol_close   = (eol_target and eol_implied and abs(math.log(eol_implied / eol_target)) < math.log(2.0))
    eol_status  = "validated (within 2x)" if eol_close else "partial — implied EOL off by >2x documented target"

    fade_real   = real_targets.get("soh_fade_per_cycle")
    fade_oracle = best.get("oracle_soh_fade_per_cycle")
    if fade_real:
        fade_close  = fade_oracle and fade_oracle > 0 and abs(math.log(fade_oracle / fade_real)) < math.log(1.5)
        fade_status = "validated (within 1.5x)" if fade_close else "partial — fade rate off by >1.5x real"
    else:
        fade_status = "n/a — cache carries no real_soh (EIS-driven calibration)"

    crate_ratio   = best.get("crate_sensitivity_ratio")
    crate_skipped = best.get("crate_probe_skipped", False)
    if crate_skipped:
        crate_status = "skipped (--skip-crate-probe-legacy; not the primary check for jones2022)"
    elif crate_ratio is not None and crate_ratio >= crate_sensitivity_min:
        crate_status = f"validated (ratio >= {crate_sensitivity_min:.2g})"
    else:
        crate_status = f"FAILED — ratio below required minimum {crate_sensitivity_min:.2g}"

    c2_slope        = best.get("oracle_slope_mAh_per_mA")
    c2_slope_skipped = real_crate2_slope is None
    if c2_slope_skipped:
        c2_slope_status = "skipped (--skip-crate2-slope; NOT validated against real C_rate_2 data)"
    elif c2_slope is not None and real_crate2_slope["ci_lo"] <= c2_slope <= real_crate2_slope["ci_hi"]:
        c2_slope_status = "validated (within real multi-cell 95% bootstrap CI)"
    else:
        c2_slope_status = "FAILED — outside real multi-cell 95% bootstrap CI"

    def fmt(v):
        return f"{v:.2f}" if v is not None else "n/a"

    def fmt_fade(v):
        return f"{v:.4g}" if v is not None else "n/a"

    _dr = drift_result or {}
    _drift_scale = _dr.get("drift_scale", 0.0)
    _drift_tau = _dr.get("drift_tau_s", 600.0)
    _drift_np = _dr.get("drift_n_periods", 4.0)

    lines = [
        f"# {dataset} oracle calibration (preset: {preset}).",
        "# Overrides config_oracle.yml protocol_scaling fields.",
        f"# Generated by battery_oracle.tune on {today}.",
        "# For the base defaults and field descriptions, see config_oracle.yml.",
        "# To regenerate (jones2022 adapter): python -m battery_forecast.bin.tune_oracle \\",
        f"#   --dataset {dataset} --cell-id {cell_id} --preset {preset} \\",
        f"#   --n-trials {len(all_results)} --max-cycles {n_cycles} \\",
        f"#   --ks-min {min(ks_swept):.3g} --ks-max {max(ks_swept):.3g} \\",
        f"#   --srs-min {min(srs_swept):.3g} --srs-max {max(srs_swept):.3g} \\",
        f"#   --dds-min {dds_range[0]:.3g} --dds-max {dds_range[1]:.3g}",
        "",
        "cycling:",
        f"  n_cycles: {_od_cycling.get('n_cycles', 1)}",
        f"  parameter_set: {chemistry}",
        f"  chemistry: {chemistry}",
        f"  temperature_K: {_od_cycling.get('temperature_K', 298.15)}",
        "",
        "eis:",
        f"  freq_min_hz: {_od_eis.get('freq_min_hz', 0.01)}",
        f"  freq_max_hz: {_od_eis.get('freq_max_hz', 10000.0)}",
        f"  n_freq_points: {_od_eis.get('n_freq_points', 60)}",
        f"  noise_level: {_od_eis.get('noise_level', 0.02)}",
        f"  noise_model: {_od_eis.get('noise_model', 'combined')}",
        "  # Non-stationarity drift (EIS measured while the OCP still relaxes); coupled to",
        "  # cycling.rest_s. Hallemans, Howey, Widanage et al. 2023 (arXiv:2304.08126)",
        "  # Eqs (40)/(43). 0.0 disables. See _calibration.drift below.",
        f"  drift_scale: {_drift_scale:.4g}",
        f"  drift_tau_s: {_drift_tau:.4g}",
        f"  drift_n_periods: {_drift_np:.4g}",
        "",
        "degradation:",
        f"  preset: {preset}",
        f"  eol_capacity_fraction: {_od_degradation.get('eol_capacity_fraction', 0.80)}",
        "  capacity_check: true",  # deliberate calibration-time override, not a stale default
        f"  ec_diffusivity_base_factor: {_od_degradation.get('ec_diffusivity_base_factor', 0.25)}",
        "",
        "protocol_scaling:",
        "  real_cell_capacity_mah_legacy_default: 200.0  # always auto-detect per cell",
        f"  kinetics_scale: {best['kinetics_scale']}",
        f"  sei_rate_scale: {best['sei_rate_scale']}",
        f"  dead_li_decay_scale: {dds_best:.4g}",
        f"  plating_rate_scale: {prs_best:.4g}",
        "",
        "_calibration:",
        f'  date: "{today}"',
        '  method: "battery_oracle.tune (battery-oracle package)"',
        "  cells_used:",
        f"    primary: {cell_id}",
        f'  cycle_window: "first {n_cycles} valid cycles"',
        "",
        "  kinetics_scale:",
        f'    status: "{arc_status}"',
        '    target_metric: "mean (R3+R5)/R1 across charge+discharge EIS fits"',
        f'    real_target: "~{fmt(arc_real)} ({cell_id})"',
        f'    achieved: "~{fmt(arc_oracle)} ({cell_id})"',
        f'    sweep_tried: "{ks_swept}"',
        '    note: "R3-vs-R5 individual labeling unstable (two similarly-sized arcs in',
        '           AutoEIS) — only the SUM (R3+R5)/R1 is a stable comparison target."',
        "",
        "  sei_rate_scale:",
        f'    status: "{r1_status}"',
        '    target_metric: "R1 growth % over cycle window (charge ECM)"',
        f'    real_target: "+{fmt(r1_real)}% over {n_cycles} cycles ({cell_id})"',
        f'    achieved: "+{fmt(r1_oracle)}% over {n_cycles} cycles"',
        f'    bo_range_searched: "[{min(srs_swept):.3g}, {max(srs_swept):.3g}]"',
        "",
        "  dead_li_decay_scale:",
        f'    best_value: {dds_best:.4g}',
        '    target_metric: "R1 growth % — controls dead-Li accumulation from plating"',
        f'    bo_range_searched: "[{dds_range[0]:.3g}, {dds_range[1]:.3g}]"',
        '    note: "Higher dds = faster dead-Li dissolution = lower R1 growth from plating pathway"',
        "",
        "  plating_rate_scale:",
        f'    best_value: {prs_best:.4g}',
        '    target_metric: "R1 growth % and EOL — controls how much Li is plated per cycle"',
        f'    bo_range_searched: "[{prs_range[0]:.3g}, {prs_range[1]:.3g}]"',
        '    note: "Lower prs = slower plating = less dead Li formed = lower R1 growth AND longer EOL (unlike dds which trades one for the other)"',
        "",
        "  implied_eol_cycle:",
        f'    status: "{eol_status}"',
        '    target_metric: "Cycles to SOH<0.80, extrapolated from per-cycle capacity loss over the calibration window (or actual EOL cycle if reached within the window)"',
        f'    real_target: "~{fmt(eol_target)} (config_oracle.yml preset_constants.{preset}.target_eol_cycles_at_1c midpoint)"',
        f'    achieved: "~{fmt(eol_implied)}"',
        '    note: "Keeps the search within ~2x of the documented per-preset EOL target,',
        '           so candidates cannot satisfy arc_ratio/r1_growth while reaching real',
        '           EOL in <10 cycles or not for hundreds of cycles."',
        "",
        "  capacity_fade:",
        f'    status: "{fade_status}"',
        '    target_metric: "mean per-cycle real-SOH loss over the calibration window"',
        f'    real_soh_fade_per_cycle: "{fmt_fade(fade_real)}"',
        f'    achieved_soh_fade_per_cycle: "{fmt_fade(fade_oracle)}"',
        '    note: "Sole health signal for EIS-less datasets (CALCE/Oxford/MATR): when the',
        '           cache has null ECMs but per-cycle real_soh, the fit is driven by this',
        '           term and the preset EOL anchor above is not used. n/a for EIS-driven',
        '           calibration (no real_soh in the cache)."',
        "",
        "  crate_sensitivity:",
        f'    status: "{crate_status}"',
        '    target_metric: "ratio of per-cycle SOH-loss-rate at high-C vs low-C synthetic probe"',
        f'    target_min_ratio: {crate_sensitivity_min:.3g}',
        f'    achieved_ratio: "{fmt(crate_ratio)}"',
        f'    probe_levels_mA: "low={fmt(best.get("crate_low_ma"))}  high={fmt(best.get("crate_high_ma"))}"',
        '    note: "Engineering target, not measured from real data. crate_probe_skipped=true',
        '           means this run used --skip-crate-probe-legacy and the config has NOT been',
        '           checked for this."',
        "",
        "  crate2_slope:",
        f'    status: "{c2_slope_status}"',
        '    target_metric: "d(capacity_fade_mAh)/d(C_rate_2_mA), oracle vs real multi-cell slope"',
        f'    real_slope_mAh_per_mA: "{fmt(real_crate2_slope["slope_mAh_per_mA"]) if real_crate2_slope else "n/a"}"',
        f'    real_95pct_ci: "[{fmt(real_crate2_slope["ci_lo"]) if real_crate2_slope else "n/a"}, '
        f'{fmt(real_crate2_slope["ci_hi"]) if real_crate2_slope else "n/a"}]"',
        f'    real_n_cells: {real_crate2_slope["n_cells"] if real_crate2_slope else "n/a"}',
        f'    achieved_slope_mAh_per_mA: "{fmt(c2_slope)}"',
        '    note: "Primary, real-data-calibrated replacement for the engineering-guessed',
        '           crate_sensitivity term above. Requires the oracle.py _C2_MAX_mA clip',
        '           fix (0.5C -> 2C)."',
        "",
        "  drift:",
        f'    status: "{"calibrated" if _drift_scale > 0 else "disabled (drift_scale=0)"}"',
        '    target_metric: "linKK low/high-freq residual ratio (non-stationarity signature)"',
        f'    real_target_ratio: "{fmt(_dr.get("real_lowhigh"))}"',
        f'    achieved_ratio: "{fmt(_dr.get("oracle_lowhigh"))}"',
        f'    ratio_when_well_rested: "{fmt(_dr.get("oracle_lowhigh_wellrested"))}"',
        f'    tau_relax_s: {_drift_tau:.4g}',
        f'    n_periods: {_drift_np:.4g}',
        '    basis: >',
        '      EIS measured while the OCP still relaxes -> a time-varying impedance whose',
        '      drift v0(t) leaks into the low-frequency, KK-violating residual. Model:',
        '      Hallemans, Howey, Widanage et al. 2023 (arXiv:2304.08126) Eqs (40)/(43).',
        '    limitation: >',
        '      PARKED default-OFF. Only the v0(t) drift-leakage part of Eq (43) is',
        '      implemented; integer-period single-sine extraction rejects slow drift, so',
        '      the oracle ratio cannot reach the real target at any scale/tau.',
        "",
        '  method: "Optuna TPE/GP Bayesian optimisation over (ks, srs, dds, prs) 4D log-space"',
        f'  n_trials: {len(all_results)}',
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Wrote config to %s", output_path)


def _json_default(obj):
    """json.dump default= hook: numpy scalars/bools aren't natively JSON-serializable."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_calibration_summary(
    path: Path | str,
    *,
    dataset: str,
    preset: str,
    cell_id: str,
    n_trials: int,
    n_cycles: int,
    crate_sensitivity_min: float,
    real_targets: dict,
    real_crate2_slope: dict | None,
    best: dict,
    best_score: float,
    drift_result: dict | None = None,
) -> Path:
    """Write the machine-readable calibration summary JSON sidecar.

    Clean, structured counterpart to :func:`write_oracle_config` for plotting
    (:mod:`battery_oracle.tune_plots`): the YAML config carries the same
    real-vs-achieved comparisons but as human-readable prose strings
    ("~0.85 (PJ121)"), which would need fragile regex parsing to plot. This
    carries the same underlying numbers as plain JSON instead.
    """
    path = Path(path)
    summary = {
        "dataset": dataset,
        "preset": preset,
        "cell_id": cell_id,
        "n_trials": n_trials,
        "n_cycles": n_cycles,
        "crate_sensitivity_min": crate_sensitivity_min,
        "eol_target_cycles": _eol_target_cycles(preset),
        "real_targets": real_targets,
        "real_crate2_slope": real_crate2_slope,
        "best": best,
        "best_score": best_score,
        "drift_result": drift_result,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    log.info("Wrote calibration summary to %s", path)
    return path


# ---------------------------------------------------------------------------
# Standalone CLI: calibrate from a cache JSON (dataset-free)
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate PyBaMMOracle hyperparameters from a cache JSON "
                    "(measured ECM-per-cycle + capacity + protocols).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cache", required=True, type=Path,
                   help="Path to the cache JSON (see module docstring for schema).")
    p.add_argument("--output-config", required=True, type=Path,
                   help="Where to write the calibrated config_oracle_*.yml.")
    p.add_argument("--targets", type=Path, default=None,
                   help="Optional real-targets JSON {mean_arc_ratio, r1_growth_pct}. "
                        "Computed from the cache when omitted.")
    p.add_argument("--crate2-slope", type=Path, default=None,
                   help="Optional real C_rate_2 slope JSON "
                        "{slope_mAh_per_mA, ci_lo, ci_hi, n_cells, n_pairs}.")
    p.add_argument("--dataset", default="custom", help="Label written into the config provenance.")
    p.add_argument("--chemistry", default="Chen2020",
                   choices=["Chen2020", "Xu2019", "Prada2013"],
                   help="Cell chemistry; written to the output config's parameter_set + chemistry (#14).")
    p.add_argument("--cell-id", default=None, help="Cell label; defaults to the cache's cell_id.")
    p.add_argument("--preset", default="accelerated",
                   choices=["nominal", "accelerated", "severe"])
    p.add_argument("--n-trials", type=int, default=35)
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallel trials. >1 runs worker PROCESSES (not threads) over "
                        "a shared Optuna store, since AutoEIS's numpyro/JAX inference "
                        "is not thread-safe. Non-deterministic trial order; per-worker "
                        "JAX/PyBaMM warmup. Default 1 (sequential, full per-trial logs).")
    p.add_argument("--summary-json", type=Path, default=None,
                   help="Optional path for the machine-readable calibration summary "
                        "JSON (consumed by battery-oracle-tune-plot).")
    p.add_argument("--sampler", choices=["tpe", "gp"], default="tpe")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ks-min", type=float, default=0.10)
    p.add_argument("--ks-max", type=float, default=0.50)
    p.add_argument("--srs-min", type=float, default=0.01)
    p.add_argument("--srs-max", type=float, default=1.0)
    p.add_argument("--dds-min", type=float, default=0.1)
    p.add_argument("--dds-max", type=float, default=1000.0)
    p.add_argument("--prs-min", type=float, default=0.01)
    p.add_argument("--prs-max", type=float, default=10.0)
    p.add_argument("--no-capacity-check", action="store_true", default=False)
    p.add_argument("--skip-crate2-slope", action="store_true", default=False)
    p.add_argument("--crate-sensitivity-min", type=float, default=3.0)
    p.add_argument("--plots-dir", type=Path, default=None,
                   help="Directory for auto-generated calibration plots (default: "
                        "alongside --output-config). The oracle-vs-ground-truth EIS "
                        "Nyquist plot is written here after the search.")
    p.add_argument("--no-eis-plot", action="store_true", default=False,
                   help="Skip the automatic oracle-vs-ground-truth EIS Nyquist plot "
                        "(e.g. on a capacity-only cache, where it is a no-op anyway).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    with open(args.cache) as f:
        cache = json.load(f)
    cell_id = args.cell_id or cache.get("cell_id", "cell")

    if args.targets is not None:
        with open(args.targets) as f:
            real_targets = json.load(f)
    else:
        real_targets = compute_real_targets(cache)

    real_crate2_slope = None
    if not args.skip_crate2_slope and args.crate2_slope is not None:
        with open(args.crate2_slope) as f:
            real_crate2_slope = json.load(f)

    out = calibrate_oracle(
        cache, real_targets,
        preset=args.preset,
        ks_min=args.ks_min, ks_max=args.ks_max,
        srs_min=args.srs_min, srs_max=args.srs_max,
        dds_min=args.dds_min, dds_max=args.dds_max,
        prs_min=args.prs_min, prs_max=args.prs_max,
        n_trials=args.n_trials, n_jobs=args.n_jobs,
        sampler=args.sampler, seed=args.seed,
        capacity_check=not args.no_capacity_check,
        skip_crate2_slope=(args.skip_crate2_slope or real_crate2_slope is None),
        crate_sensitivity_min=args.crate_sensitivity_min,
        real_crate2_slope=real_crate2_slope,
        chemistry=args.chemistry,
    )
    print_summary(
        out["scored"], out["best"], real_targets,
        dataset=args.dataset, preset=args.preset, cell_id=cell_id,
        sampler=args.sampler, crate_sensitivity_min=args.crate_sensitivity_min,
        real_crate2_slope=real_crate2_slope,
    )
    write_oracle_config(
        args.output_config, args.dataset, args.preset, cell_id,
        len(cache["cycles"]), out["best"], real_targets, out["results"],
        crate_sensitivity_min=args.crate_sensitivity_min,
        real_crate2_slope=real_crate2_slope,
        chemistry=args.chemistry,
    )
    print(f"Config written to: {args.output_config}")
    if args.summary_json is not None:
        write_calibration_summary(
            args.summary_json,
            dataset=args.dataset, preset=args.preset, cell_id=cell_id,
            n_trials=len(out["results"]), n_cycles=len(cache["cycles"]),
            crate_sensitivity_min=args.crate_sensitivity_min,
            real_targets=real_targets, real_crate2_slope=real_crate2_slope,
            best=out["best"], best_score=out["best_score"],
        )
        print(f"Calibration summary written to: {args.summary_json}")

    # Auto-generate the oracle-vs-ground-truth EIS Nyquist plot for the winning
    # candidate. Best-effort: never let a plotting hiccup fail a completed search.
    if not args.no_eis_plot:
        plots_dir = args.plots_dir or Path(args.output_config).parent
        try:
            eis_data = collect_eis_comparison(
                cache, out["best"], preset=args.preset,
                capacity_check=not args.no_capacity_check,
                circuit=cache.get("circuit"), chemistry=args.chemistry,
            )
            if eis_data is None:
                print("EIS comparison plot skipped (no ground-truth EIS in cache; "
                      "capacity-only calibration).")
            else:
                import matplotlib.pyplot as plt

                from battery_oracle.tune_plots import plot_eis_comparison
                plots_dir.mkdir(parents=True, exist_ok=True)
                eis_path = plots_dir / "oracle_tuning_eis.png"
                plt.close(plot_eis_comparison(eis_data, save_path=eis_path))
                print(f"Oracle-vs-ground-truth EIS plot written to: {eis_path}")
        except Exception as exc:
            log.warning("EIS comparison plot skipped: %s", exc)


if __name__ == "__main__":
    main()
