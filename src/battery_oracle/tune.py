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
import itertools
import json
import logging
import math
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from battery_oracle._circuit import (
    _param_labels_from_circuit,
    randles_pairs_from_circuit,
)
from battery_oracle.experiment import load_default_ecm_circuit
from battery_oracle.oracle import OracleFailure, PyBaMMOracle

log = logging.getLogger(__name__)


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

# Midpoint of the documented target_eol_cycles_at_1c ranges per preset. Used to
# penalize candidates whose implied EOL cycle count is wildly off the physically
# reasonable range — without this the BO can fit arc_ratio/r1_growth within the
# scoring window while reaching real EOL in <10 cycles or >500 cycles.
_EOL_TARGET_CYCLES = {
    "nominal":     300.0,   # documented range "200-400"
    "accelerated": 55.0,    # documented range "40-70"
    "severe":      35.0,    # documented range "20-50"
}


# ---------------------------------------------------------------------------
# Real calibration targets (operate on the cache dict; no dataset access)
# ---------------------------------------------------------------------------

def compute_real_targets(cache: dict, circuit: str | None = None) -> dict[str, float | None]:
    """Extract arc-ratio and R1-growth targets from the real ECM cache.

    Parameters
    ----------
    cache : dict
        Measured cache (``ecm_charge`` / ``ecm_discharge`` per cycle + protocols).
    circuit : str, optional
        ECM circuit string defining the parameter layout of the cached vectors.
        No structure is assumed: the ohmic/arc-resistor positions are derived from
        this circuit. ``None`` loads the default from ``config_oracle_defaults.yml``
        (``ecm.circuit``); pass ``cache.get("circuit")`` if the cache carries one.

    Returns
    -------
    dict
        Mapping with two keys: ``mean_arc_ratio`` — mean (sum arc R)/(ohmic R)
        across all cycles and states (or ``None``); and ``r1_growth_pct`` —
        ``(R1[last]/R1[first] - 1) * 100`` from the charge ECM (or ``None``).
        Outliers (ohmic R < 1e-6) are excluded as AutoEIS blowup artefacts.
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

    return {
        "mean_arc_ratio": float(np.mean(arc_ratios)) if arc_ratios else None,
        "r1_growth_pct":  r1_growth,
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
) -> dict[str, Any]:
    """Replay cached protocols through one oracle candidate and return metrics.

    ``circuit`` sets the oracle's ECM structure (and the ohmic/arc-resistor
    positions the arc-ratio / R1-growth metrics read). It is not assumed: ``None``
    loads the default from ``config_oracle_defaults.yml`` (``ecm.circuit``).
    """
    circuit = _resolve_circuit(circuit)
    ohmic_i, arc_is = _ecm_indices(circuit)
    oracle = PyBaMMOracle(
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

    # Implied EOL cycle: if the oracle actually failed within the window, that IS
    # the EOL cycle. Otherwise extrapolate from the mean per-cycle capacity loss
    # rate observed over the window to the eol_capacity_fraction threshold (0.80
    # SOH). Linear extrapolation is rough but enough to catch order-of-magnitude
    # errors either way.
    implied_eol_cycle = None
    if oracle_failure:
        implied_eol_cycle = float(n_completed)
    elif len(soh_history) >= 2:
        mean_loss_per_cycle = (soh_history[0] - soh_history[-1]) / (len(soh_history) - 1)
        if mean_loss_per_cycle > 1e-6:
            implied_eol_cycle = 0.20 / mean_loss_per_cycle

    return {
        "kinetics_scale":       kinetics_scale,
        "sei_rate_scale":       sei_rate_scale,
        "dead_li_decay_scale":  dead_li_decay_scale,
        "plating_rate_scale":   plating_rate_scale,
        "oracle_arc_ratio":     float(np.mean(arc_ratios)) if arc_ratios else None,
        "oracle_r1_growth_pct": r1_growth,
        "oracle_failure":       oracle_failure,
        "n_cycles_completed":   n_completed,
        "implied_eol_cycle":    implied_eol_cycle,
    }


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
    """Combined relative error (lower = better). Returns inf if data is missing.

    Terms: arc_ratio relative error, R1-growth relative error, EOL-rate
    log-ratio error, a legacy C_rate_1 sensitivity term (skipped when the probe
    wasn't run), and a C_rate_2 slope-matching term (see slope_match_error).
    """
    arc_err = r1_err = eol_err = float("inf")
    crate_err = 0.0
    if result["oracle_arc_ratio"] is not None and real_targets["mean_arc_ratio"]:
        arc_err = abs(result["oracle_arc_ratio"] / real_targets["mean_arc_ratio"] - 1.0)
    if result["oracle_r1_growth_pct"] is not None and real_targets["r1_growth_pct"]:
        real_r1 = real_targets["r1_growth_pct"]
        if abs(real_r1) > 1e-3:
            r1_err = abs(result["oracle_r1_growth_pct"] / real_r1 - 1.0)
    eol_target = _EOL_TARGET_CYCLES.get(preset)
    implied_eol = result.get("implied_eol_cycle")
    if eol_target and implied_eol and implied_eol > 0:
        eol_err = abs(math.log(implied_eol / eol_target))
    ratio = result.get("crate_sensitivity_ratio")
    if ratio is not None and ratio > 0:
        crate_err = max(0.0, math.log(crate_sensitivity_min / ratio))
    elif not result.get("crate_probe_skipped", False):
        crate_err = float("inf")
    crate2_slope_err = 0.0
    if real_crate2_slope is not None:
        crate2_slope_err = slope_match_error(
            result.get("oracle_slope_mAh_per_mA"), real_crate2_slope
        )
    return arc_err + r1_err + eol_err + crate_err + crate2_slope_err


# ---------------------------------------------------------------------------
# EIS non-stationarity drift calibration (oracle-side; real target passed in)
# ---------------------------------------------------------------------------

def _linkk_lowhigh_ratio(freq, Z) -> float:
    """linKK low/high-freq residual ratio — the non-stationarity signature.

    A spectrum that drifts during the sweep has KK residuals elevated at low
    frequency; returns mean|res|(low tercile) / mean|res|(high tercile). Uses
    impedance.py with the NumPy-2.x eval-builder fix (see _eis/kk.py).
    """
    import contextlib
    import io

    import impedance.models.circuits.elements as _ce
    from impedance.validation import linKK
    _ce.circuit_elements.setdefault("np", np)
    f = np.asarray(freq, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    m = np.isfinite(f) & np.isfinite(Z.real) & np.isfinite(Z.imag) & (f > 0)
    f, Z = f[m], Z[m]
    if len(f) < 8:
        return float("nan")
    o = np.argsort(f)
    f, Z = f[o], Z[o]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, _, res_real, res_imag = linKK(f, Z, c=0.85, max_M=50)
    except Exception:
        return float("nan")
    res = np.sqrt(np.asarray(res_real) ** 2 + np.asarray(res_imag) ** 2)
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
) -> dict[str, Any]:
    """Run the Optuna BO over (kinetics, sei_rate, dead_li_decay, plating)_scale.

    Pure engine: operates on *cache* + precomputed *real_targets* (and optional
    *real_crate2_slope*); no dataset access. Returns a dict with ``best``,
    ``best_score``, ``results`` (all trials) and ``scored`` (ranked).

    ``circuit`` is the ECM structure the candidate oracles fit and the metrics
    read; no layout is assumed. ``None`` loads the default from the YAML config
    (``config_oracle_defaults.yml`` ``ecm.circuit``) — pass ``cache.get("circuit")``
    when the cache was featurized with a specific circuit.
    """
    import optuna

    circuit = _resolve_circuit(circuit or cache.get("circuit"))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results: list[dict] = []

    def objective(trial: "optuna.Trial") -> float:
        ks  = trial.suggest_float("kinetics_scale",      ks_min,  ks_max,  log=True)
        srs = trial.suggest_float("sei_rate_scale",      srs_min, srs_max, log=True)
        dds = trial.suggest_float("dead_li_decay_scale", dds_min, dds_max, log=True)
        prs = trial.suggest_float("plating_rate_scale",  prs_min, prs_max, log=True)
        log.info(
            "── Trial %d/%d: ks=%.4f  srs=%.4f  dds=%.4f  prs=%.4f ──",
            trial.number + 1, n_trials, ks, srs, dds, prs,
        )
        t0 = time.time()
        result = run_oracle_candidate(cache, ks, srs, dds, prs, preset, capacity_check,
                                      circuit=circuit)
        if skip_crate_probe:
            result["crate_probe_skipped"] = True
            result["crate_sensitivity_ratio"] = None
        else:
            probe = run_crate_sensitivity_probe(
                cache, ks, srs, dds, prs, preset, capacity_check,
                probe_cycles=crate_probe_cycles,
                low_c_mult=crate_probe_low_c, high_c_mult=crate_probe_high_c,
                circuit=circuit,
            )
            result.update(probe)
            result["crate_probe_skipped"] = False
        if not skip_crate2_slope:
            slope_probe = run_crate2_slope_probe(
                cache, ks, srs, dds, prs, preset, capacity_check,
                probe_cycles=crate2_probe_cycles,
                c2_levels_mult=tuple(crate2_levels),
                circuit=circuit,
            )
            result.update(slope_probe)
        result["runtime_s"] = round(time.time() - t0, 1)
        results.append(result)
        score = score_candidate(
            result, real_targets, preset, crate_sensitivity_min, real_crate2_slope,
        )
        ratio = result.get("crate_sensitivity_ratio")
        oracle_slope = result.get("oracle_slope_mAh_per_mA")
        log.info(
            "  arc_ratio=%s (real=%s)  r1_growth=%s (real=%s)  eol_cycle~%s (target=%s)  "
            "crate_ratio=%s (min=%.1f)  c2_slope=%s (real_CI=%s)  score=%.3f  %.0fs",
            f"{result['oracle_arc_ratio']:.2f}" if result["oracle_arc_ratio"] is not None else "n/a",
            f"{real_targets['mean_arc_ratio']:.2f}" if real_targets["mean_arc_ratio"] else "?",
            f"{result['oracle_r1_growth_pct']:.1f}%" if result["oracle_r1_growth_pct"] is not None else "n/a",
            f"{real_targets['r1_growth_pct']:.1f}%" if real_targets["r1_growth_pct"] else "?",
            f"{result['implied_eol_cycle']:.0f}" if result["implied_eol_cycle"] is not None else "n/a",
            _EOL_TARGET_CYCLES.get(preset, "?"),
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

    opt_sampler = (
        optuna.samplers.GPSampler(seed=seed)
        if sampler == "gp"
        else optuna.samplers.TPESampler(seed=seed, n_startup_trials=8)
    )
    study = optuna.create_study(direction="minimize", sampler=opt_sampler)

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

    log.info("Starting Optuna BO [sampler=%s, n_trials=%d, preset=%s]", sampler, n_trials, preset)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

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
    eol_target = _EOL_TARGET_CYCLES.get(preset)
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
) -> None:
    """Write YAML config with calibration provenance to *output_path*."""
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

    eol_target  = _EOL_TARGET_CYCLES.get(preset)
    eol_implied = best.get("implied_eol_cycle")
    eol_close   = (eol_target and eol_implied and abs(math.log(eol_implied / eol_target)) < math.log(2.0))
    eol_status  = "validated (within 2x)" if eol_close else "partial — implied EOL off by >2x documented target"

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
        "  n_cycles: 1",
        "  parameter_set: Chen2020",
        "  temperature_K: 298.15",
        "",
        "eis:",
        "  freq_min_hz: 0.01",
        "  freq_max_hz: 10000.0",
        "  n_freq_points: 60",
        "  noise_level: 0.02",
        "  noise_model: combined",
        "  # Non-stationarity drift (EIS measured while the OCP still relaxes); coupled to",
        "  # cycling.rest_s. Hallemans, Howey, Widanage et al. 2023 (arXiv:2304.08126)",
        "  # Eqs (40)/(43). 0.0 disables. See _calibration.drift below.",
        f"  drift_scale: {_drift_scale:.4g}",
        f"  drift_tau_s: {_drift_tau:.4g}",
        f"  drift_n_periods: {_drift_np:.4g}",
        "",
        "degradation:",
        f"  preset: {preset}",
        "  eol_capacity_fraction: 0.80",
        "  capacity_check: true",
        "  ec_diffusivity_base_factor: 0.25",
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
    p.add_argument("--cell-id", default=None, help="Cell label; defaults to the cache's cell_id.")
    p.add_argument("--preset", default="accelerated",
                   choices=["nominal", "accelerated", "severe"])
    p.add_argument("--n-trials", type=int, default=35)
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
        n_trials=args.n_trials, sampler=args.sampler, seed=args.seed,
        capacity_check=not args.no_capacity_check,
        skip_crate2_slope=(args.skip_crate2_slope or real_crate2_slope is None),
        crate_sensitivity_min=args.crate_sensitivity_min,
        real_crate2_slope=real_crate2_slope,
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
    )
    print(f"Config written to: {args.output_config}")


if __name__ == "__main__":
    main()
