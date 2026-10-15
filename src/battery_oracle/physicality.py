"""Physicality report: checks that a calibration produced a physically sensible cell.

Every check returns ``{"name", "status", "value", "limit", "note"}`` with
``status`` one of ``"pass"``, ``"warn"`` (suspicious -- inspect before trusting),
``"fail"`` (unphysical -- do not use) or ``"skip"`` (could not be evaluated).
:func:`summarise` reduces a list to an overall status (the worst one); the
battery-forecast pipeline refuses to start the degradation tune (stage B) on a
stage-A result whose overall status is ``"fail"`` unless forced.

Stage A (:func:`check_bol`, on a :func:`battery_oracle.bol_fit.fit_bol` report):

* balancing: stoichiometry window inside [0, 1]; fitted parameters pinned at a
  search bound; negative series resistance (PyBOP's balancing allows it to absorb
  hysteresis);
* geometry: derived electrode area and negative thickness vs the template, N:P
  capacity ratio, fitted capacity vs the nominal one;
* fitted kinetics / transport: values at a reference state inside broad
  plausibility windows (about an order of magnitude either side of published
  LiCoO2/graphite values -- a guard against runaway fits, not a literature fit);
  EIS multipliers or contact resistance pinned at a bound;
* electrode state of health: PyBaMM's electrode-SOH solve on the fitted cell
  succeeds with x/y limits strictly inside (0, 1);
* one demanding DFN cycle on the fitted cell: completes at full fidelity, no
  electrolyte depletion, no surface-stoichiometry saturation, lithium conserved
  (PyBaMM's two independent lithium-loss measures agree).

Stage B (:func:`check_tune`, on the tune's best candidate and search ranges):
degradation scales pinned at a search bound, capacity gain between replayed
cycles, FAILED calibration targets.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_ORDER = {"pass": 0, "skip": 0, "warn": 1, "fail": 2}


def _check(name: str, status: str, value: Any = None, limit: Any = None, note: str = "") -> dict:
    return {"name": name, "status": status, "value": value, "limit": limit, "note": note}


def summarise(checks: list[dict]) -> dict:
    """Overall status (the worst) and counts per status."""
    counts = {s: sum(c["status"] == s for c in checks) for s in ("pass", "warn", "fail", "skip")}
    worst = max((c["status"] for c in checks), key=lambda s: _ORDER[s], default="pass")
    return {"status": "pass" if worst == "skip" else worst, "counts": counts}


def at_bound(value: float, lo: float, hi: float, *, log_scale: bool = False,
             rel: float = 0.01) -> str | None:
    """``"lower"``/``"upper"`` if ``value`` is within ``rel`` of the bound (in log
    space for ``log_scale``), else ``None``."""
    if log_scale:
        if value <= 0 or lo <= 0:
            return None
        span = math.log(hi) - math.log(lo)
        v, a, b = math.log(value), math.log(lo), math.log(hi)
    else:
        span, v, a, b = hi - lo, value, lo, hi
    if span <= 0:
        return None
    if v - a <= rel * span:
        return "lower"
    if b - v <= rel * span:
        return "upper"
    return None


def _range_check(name: str, value: float, lo: float, hi: float, note: str,
                 fail_factor: float = 10.0) -> dict:
    """pass inside [lo, hi]; warn within ``fail_factor`` outside; fail beyond."""
    if value is None or not np.isfinite(value):
        return _check(name, "skip", value, [lo, hi], "could not be evaluated")
    if lo <= value <= hi:
        status = "pass"
    elif lo / fail_factor <= value <= hi * fail_factor:
        status = "warn"
    else:
        status = "fail"
    return _check(name, status, float(value), [lo, hi], note)


# ---------------------------------------------------------------------------
# Plausibility windows for fitted kinetics / transport (LiCoO2 / graphite)
# ---------------------------------------------------------------------------
#: (parameter, evaluation args builder, low, high, note). Values are evaluated on
#: the fitted parameter set at a reference state (sto 0.5, c_e 1000 mol/m3, 298 K).
PLAUSIBILITY_WINDOWS: tuple[tuple[str, str, float, float, str], ...] = (
    ("Negative particle diffusivity [m2.s-1]", "D", 1e-16, 1e-12,
     "graphite solid diffusivity; published values ~1e-15-1e-13"),
    ("Positive particle diffusivity [m2.s-1]", "D", 1e-17, 1e-11,
     "LiCoO2 solid diffusivity; published values ~1e-16-1e-12"),
    ("Negative electrode exchange-current density [A.m-2]", "j0", 0.05, 50.0,
     "graphite exchange current at 50% lithiation; typical ~0.5-5 A/m2"),
    ("Positive electrode exchange-current density [A.m-2]", "j0", 0.05, 50.0,
     "LiCoO2 exchange current at 50% lithiation; typical ~0.5-5 A/m2"),
    ("Electrolyte conductivity [S.m-1]", "kappa", 0.2, 3.0,
     "1 M LiPF6 carbonate electrolyte ~0.8-1.2 S/m at 25 C"),
    ("Electrolyte diffusivity [m2.s-1]", "kappa", 5e-11, 2e-9,
     "1 M LiPF6 carbonate electrolyte ~1e-10-8e-10 m2/s at 25 C"),
    ("Negative electrode double-layer capacity [F.m-2]", "C", 0.01, 2.0,
     "typical 0.1-0.4 F/m2 of active surface"),
    ("Positive electrode double-layer capacity [F.m-2]", "C", 0.01, 2.0,
     "typical 0.1-0.4 F/m2 of active surface"),
)


def evaluate_parameter(pv, name: str, kind: str, T: float = 298.15) -> float:
    """Value of a (possibly functional) parameter at the reference state."""
    import pybamm

    f = pv[name]
    if not callable(f):
        return float(f)
    if kind == "D":
        args = (pybamm.Scalar(0.5), pybamm.Scalar(T))
    elif kind == "j0":
        d = "negative" if name.startswith("Negative") else "positive"
        c_max = float(pv[f"Maximum concentration in {d} electrode [mol.m-3]"])
        args = (pybamm.Scalar(1000.0), pybamm.Scalar(0.5 * c_max), pybamm.Scalar(c_max),
                pybamm.Scalar(T))
    elif kind == "kappa":
        args = (pybamm.Scalar(1000.0), pybamm.Scalar(T))
    else:
        args = ()
    return float(np.asarray(pv.process_symbol(f(*args)).evaluate()).ravel()[0])


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------

def check_balance(fit_report: dict, pv_template) -> list[dict]:
    """Balancing checks from ``fit_bol``'s report (all cells)."""
    out = []
    q_nom = float(pv_template["Nominal cell capacity [A.h]"])
    for cid, b in fit_report["fit"]["balance"].items():
        out.append(_check(f"balance[{cid}] stoichiometry window in [0, 1]",
                          "pass" if b.get("window_valid", True) else "fail", b.get("window"),
                          [0, 1], "window read at the voltage limits; clipped when outside"))
        for k, v in b.get("theoretical_capacity_ah", {}).items():
            hit = at_bound(v, *b["bounds"]["theoretical_capacity_ah"])
            out.append(_check(f"balance[{cid}] {k} theoretical capacity not at bound",
                              "warn" if hit else "pass", v, b["bounds"]["theoretical_capacity_ah"],
                              f"at {hit} bound: not identified by the data" if hit else ""))
        for br in b.get("branches", []):
            for key, bkey in (("min_sto_n", "min_stoichiometry_n"), ("min_sto_p", "min_stoichiometry")):
                bnd = b["bounds"].get(bkey, b["bounds"]["min_stoichiometry"])
                if key in br and at_bound(br[key], *bnd) == "upper":
                    out.append(_check(f"balance[{cid}] {br['direction']} {key} not at bound",
                                      "warn", br[key], bnd))
        r0 = b.get("series_resistance_ohm")
        out.append(_check(f"balance[{cid}] series resistance >= 0", "warn" if r0 < 0 else "pass",
                          r0, [0, None], "negative R0 absorbs hysteresis/overpotential, not a "
                          "physical resistance" if r0 < 0 else ""))
        r_bounds = b.get("bounds", {}).get("series_resistance_ohm")
        if r_bounds:
            hit = at_bound(r0, *r_bounds)
            out.append(_check(f"balance[{cid}] series resistance not at bound",
                              "warn" if hit else "pass", r0, r_bounds,
                              f"at {hit} bound: the OCV model needs more overpotential than "
                              "the bound allows" if hit else ""))
        caps = b.get("theoretical_capacity_ah", {})
        if caps:
            np_ratio = caps["negative"] / caps["positive"]
            out.append(_range_check(f"balance[{cid}] N:P theoretical-capacity ratio", np_ratio,
                                    1.0, 1.5, "<1 invites lithium plating; >1.5 unusual for "
                                    "commercial cells", fail_factor=1.5))
        q_cell = np.mean([br["Q_soc_ah"] for br in b.get("branches", [])]) if b.get("branches") else None
        if q_cell is not None:
            out.append(_range_check(f"balance[{cid}] capacity between voltage limits vs nominal",
                                    q_cell / q_nom, 0.8, 1.2, "ratio to the parameter set's "
                                    "nominal capacity", fail_factor=1.5))
    ov = fit_report["parameter_overrides"]
    t_side = math.sqrt(float(pv_template["Electrode height [m]"]) * float(pv_template["Electrode width [m]"]))
    if "Electrode height [m]" in ov:
        out.append(_range_check("geometry: electrode area / template",
                                ov["Electrode height [m]"] * ov["Electrode width [m]"] / t_side ** 2,
                                0.5, 2.0, "template area is scaled from Ai2020 area-per-capacity",
                                fail_factor=2.0))
    if "Negative electrode thickness [m]" in ov:
        out.append(_range_check("geometry: negative thickness / template",
                                ov["Negative electrode thickness [m]"]
                                / float(pv_template["Negative electrode thickness [m]"]),
                                0.5, 2.0, "commercial anodes ~50-120 um", fail_factor=2.0))
    return out


def check_fitted_values(pv_fitted, fit_report: dict) -> list[dict]:
    """Fitted kinetics/transport vs :data:`PLAUSIBILITY_WINDOWS`, and bound hits."""
    out = []
    for name, kind, lo, hi, note in PLAUSIBILITY_WINDOWS:
        try:
            value = evaluate_parameter(pv_fitted, name, kind)
        except Exception as exc:
            out.append(_check(f"value: {name}", "skip", None, [lo, hi], f"not evaluable ({exc})"))
            continue
        out.append(_range_check(f"value: {name}", value, lo, hi, note))
    eis = fit_report["fit"].get("eis", {})
    sb, cb = eis.get("scale_bounds"), eis.get("cr_bounds")
    for name, factor in (fit_report.get("parameter_scales") or {}).items():
        hit = at_bound(factor, *sb, log_scale=True) if sb else None
        out.append(_check(f"EIS multiplier not at bound: {name}", "warn" if hit else "pass",
                          factor, sb, f"at {hit} bound: not identified by the spectra" if hit else ""))
    cr = fit_report.get("contact_resistance_ohm")
    if cr is not None and cb:
        hit = at_bound(cr, *cb, log_scale=True)
        out.append(_check("EIS contact resistance not at bound", "warn" if hit else "pass",
                          cr, cb, f"at {hit} bound" if hit else ""))
    return out


def check_esoh(pv_fitted) -> list[dict]:
    """Electrode-SOH solve on the fitted cell: limits strictly inside (0, 1)."""
    import pybamm

    try:
        x0, x100, y100, y0 = (float(v) for v in
                              pybamm.lithium_ion.get_min_max_stoichiometries(pv_fitted))
    except Exception as exc:
        return [_check("electrode SOH solve succeeds", "fail", None, None, str(exc)[:200])]
    lims = {"x_0": x0, "x_100": x100, "y_100": y100, "y_0": y0}
    ok = all(0.0 < v < 1.0 for v in lims.values()) and x0 < x100 and y100 < y0
    return [_check("electrode SOH limits strictly inside (0, 1) and ordered",
                   "pass" if ok else "fail", lims, "(0, 1)",
                   "voltage limits map to valid stoichiometries on the fitted cell")]


def check_dfn_cycle(oracle_kwargs: dict, *, c1_crate: float = 3.6, c2_crate: float = 2.0,
                    d_crate: float = 3.0) -> list[dict]:
    """One demanding cycle on the fitted cell; internal-state and conservation checks.

    ``oracle_kwargs`` must build the fitted oracle (chemistry, model, BOL values,
    bounds, temperature). Currents are given at the stated C-rates of its own
    nominal capacity, so the real-cell scaling is 1.
    """
    from battery_oracle.oracle import OracleFailure, PyBaMMOracle, _randles_stub_ecm

    kw = {**oracle_kwargs, "ecm_model_fn": _randles_stub_ecm}
    try:
        oracle = PyBaMMOracle(**kw)
        nominal = oracle._nominal_mAh
        oracle._real_cell_capacity_mah = nominal
        oracle._cap_scale = 1.0
        oracle(np.array([c1_crate * nominal, c2_crate * nominal, 0.25, 0.25,
                         d_crate * nominal, 4.0]))
    except OracleFailure as exc:
        return [_check("demanding DFN cycle completes", "fail", None, None, str(exc)[:200])]
    except Exception as exc:
        return [_check("demanding DFN cycle completes", "fail", None, None,
                       f"{type(exc).__name__}: {str(exc)[:200]}")]
    h = oracle.history[-1]
    sol = oracle._prev_solution
    out = [_check("demanding DFN cycle completes at full fidelity",
                  "pass" if h.get("fidelity") == "full" else "warn", h.get("fidelity"), "full",
                  f"{c1_crate}C/{c2_crate}C charge, {d_crate}C discharge")]

    def _entries(var):
        return np.asarray(sol[var].entries, dtype=float)

    try:
        ce = _entries("Electrolyte concentration [mol.m-3]")
        ce0 = float(oracle._pv["Initial concentration in electrolyte [mol.m-3]"])
        ratio = float(np.min(ce)) / ce0
        status = "fail" if ratio <= 0 else ("warn" if ratio < 0.1 else "pass")
        out.append(_check("electrolyte not depleted (min c_e / initial)", status, ratio, ">= 0.1",
                          "local depletion makes the electrolyte the rate limit"))
    except Exception as exc:
        out.append(_check("electrolyte not depleted", "skip", None, None, str(exc)[:120]))
    for d in ("negative", "positive"):
        try:
            s_min = float(np.min(_entries(f"Minimum {d} particle surface stoichiometry")))
            s_max = float(np.max(_entries(f"Maximum {d} particle surface stoichiometry")))
            status = "fail" if (s_min <= 0.0 or s_max >= 1.0) else (
                "warn" if (s_min < 0.005 or s_max > 0.995) else "pass")
            out.append(_check(f"{d} surface stoichiometry not saturated", status,
                              [s_min, s_max], "(0.005, 0.995)"))
        except Exception as exc:
            out.append(_check(f"{d} surface stoichiometry not saturated", "skip", None, None,
                              str(exc)[:120]))
    try:
        from battery_oracle.bol_fit import ocp_data_range

        s_min = float(np.min(_entries("Minimum positive particle surface stoichiometry")))
        s_max = float(np.max(_entries("Maximum positive particle surface stoichiometry")))
        lo, hi = ocp_data_range(oracle._pv, "positive")
        out.append(_check("positive stoichiometry inside the OCP data range",
                          "pass" if (s_min >= lo - 1e-3 and s_max <= hi + 1e-3) else "warn",
                          [s_min, s_max], [lo, hi],
                          "outside it PyBaMM extrapolates the electrode curve"))
    except Exception as exc:
        out.append(_check("positive stoichiometry inside the OCP data range", "skip", None, None, str(exc)[:120]))
    try:
        lost = _entries("Total lithium lost [mol]")[-1]
        lost_side = _entries("Total lithium lost to side reactions [mol]")[-1]
        total0 = _entries("Total lithium [mol]")[0]
        rel = abs(lost - lost_side) / abs(total0)
        status = "pass" if rel < 1e-3 else ("warn" if rel < 1e-2 else "fail")
        out.append(_check("lithium conserved (lost == lost to side reactions)", status, rel,
                          "< 1e-3 of total lithium", "PyBaMM's two independent loss measures"))
    except Exception as exc:
        out.append(_check("lithium conserved", "skip", None, None, str(exc)[:120]))
    return out


def check_joint(fit_report: dict, cell_reports: list[dict], pv_template, oracle_kwargs: dict) -> dict:
    """Stage-A physicality report for a joint DFN fit (:mod:`battery_oracle.bol_joint`).

    Per cell: fit quality (formation and cycling voltage RMSE, EIS relative error,
    trajectories simulated to the end), parameters at a search bound. Population:
    geometry/N:P vs template, fitted kinetics/transport windows, electrode-SOH
    solve, one demanding DFN cycle, and the electrode-curve data range.
    """
    from battery_oracle.bol_joint import CONTACT_R, EPS_N, N_PARALLEL
    from battery_oracle.params import apply_parameter_overrides, apply_parameter_scales

    checks: list[dict] = []
    b = fit_report["fit"]["bounds"]
    for c in cell_reports:
        cid = c["cell_id"]
        for r in c.get("records", []):
            lab = r["label"]
            if "error" in r:
                checks.append(_check(f"fit[{cid}] {r['kind']} {lab} evaluated", "fail", None, None, r["error"]))
            elif r["kind"] in ("formation", "cycling"):
                done = r["n_simulated"] / max(r["n_points"], 1)
                st = "fail" if (done < 0.99 or r["rmse_v"] > 0.1) else ("warn" if r["rmse_v"] > 0.03 else "pass")
                checks.append(_check(f"fit[{cid}] {r['kind']} voltage RMSE {lab}", st, round(r["rmse_v"], 4),
                                     [0.03, 0.1], f"{done:.0%} of the record simulated"))
            else:
                st = "fail" if r["rel_rmse"] > 0.4 else ("warn" if r["rel_rmse"] > 0.15 else "pass")
                checks.append(_check(f"fit[{cid}] EIS relative RMSE {lab}", st, round(r["rel_rmse"], 3),
                                     [0.15, 0.4]))
        for name, v in c["best_inputs"].items():
            if name == EPS_N:
                hit = at_bound(v, *b["eps_n"])
            elif name == CONTACT_R:
                hit = at_bound(v, *b["contact_r"], log_scale=True)
            elif name == N_PARALLEL:
                hit = at_bound(v, *b["n_parallel"], log_scale=True)
            else:
                hit = at_bound(v, *b["scale"], log_scale=True)
            if hit:
                checks.append(_check(f"fit[{cid}] {name[:60]} not at bound", "warn", v, None,
                                     f"at {hit} bound: not identified by the data"))
    ov = fit_report["parameter_overrides"]
    t_area = float(pv_template["Electrode height [m]"]) * float(pv_template["Electrode width [m]"])
    checks.append(_range_check("geometry: electrode area / template",
                               ov["Electrode height [m]"] * ov["Electrode width [m]"] / t_area,
                               0.5, 2.0, "template area scaled from Ai2020 area-per-capacity", fail_factor=2.0))
    checks.append(_range_check("geometry: negative active fraction / template", ov[EPS_N] / float(pv_template[EPS_N]),
                               0.7, 1.3, "N:P knob; commercial anodes 0.5-0.7", fail_factor=1.5))
    if "Nominal cell capacity [A.h]" in ov:
        checks.append(_range_check("capacity between voltage limits vs nominal",
                                   ov["Nominal cell capacity [A.h]"] / float(pv_template["Nominal cell capacity [A.h]"]),
                                   0.8, 1.2, "fitted-cell capacity (electrode SOH) vs template nominal",
                                   fail_factor=1.5))
    fitted_kwargs = {**oracle_kwargs, "parameter_overrides": ov,
                     "parameter_scales": fit_report["parameter_scales"],
                     "contact_resistance_ohm": fit_report["contact_resistance_ohm"]}
    pv_fitted = apply_parameter_overrides(pv_template.copy(), ov)
    apply_parameter_scales(pv_fitted, fit_report["parameter_scales"])
    pv_fitted.update({CONTACT_R: fit_report["contact_resistance_ohm"]}, check_already_exists=False)
    shim = {"fit": {"eis": {"scale_bounds": b["scale"], "cr_bounds": b["contact_r"]}},
            "parameter_scales": fit_report["parameter_scales"],
            "contact_resistance_ohm": fit_report["contact_resistance_ohm"]}
    checks += check_fitted_values(pv_fitted, shim) + check_esoh(pv_fitted) + check_dfn_cycle(fitted_kwargs)
    report = {"stage": "A (joint)", **summarise(checks), "checks": checks}
    log.info("stage-A (joint) physicality: %s %s", report["status"], report["counts"])
    return report


def check_chain(fit_report: dict, cell_reports: list[dict], pv_template, oracle_kwargs: dict,
                *, ambient_k: float | None = None) -> dict:
    """Stage-A physicality report for a chain fit (:mod:`battery_oracle.bol_chain`).

    Per cell: fit quality per record and spectrum, whole history simulated,
    parameters at a search bound (each parameter's own bounds), OCP data-range
    excursions, cell temperature rise, electrolyte depletion. Population: geometry,
    N:P, thickness, lithium inventory, heat transfer vs plausible ranges, kinetics/
    transport windows, electrode-SOH solve and one demanding DFN cycle on the
    fitted oracle (lumped thermal).
    """
    from battery_oracle.bol_chain import H_TOTAL, LI_FRACTION, THICKNESS_SCALE
    from battery_oracle.bol_joint import EPS_N
    from battery_oracle.params import apply_parameter_overrides, apply_parameter_scales

    checks: list[dict] = []
    for c in cell_reports:
        cid = c["cell_id"]
        bounds, logs = c.get("param_bounds", {}), set(c.get("log_params", []))
        for r in c.get("records", []):
            lab = r["label"]
            if "error" in r:
                checks.append(_check(f"fit[{cid}] {r['kind']} {lab} evaluated", "fail", None, None, r["error"]))
            elif r["kind"] in ("formation", "cycling"):
                done = r["n_simulated"] / max(r["n_points"], 1)
                st = "fail" if (done < 0.99 or r["rmse_v"] > 0.1) else ("warn" if r["rmse_v"] > 0.03 else "pass")
                checks.append(_check(f"fit[{cid}] {r['kind']} voltage RMSE {lab}", st, round(r["rmse_v"], 4),
                                     [0.03, 0.1], f"{done:.0%} of the record simulated"))
            else:
                st = "fail" if r["rel_rmse"] > 0.4 else ("warn" if r["rel_rmse"] > 0.15 else "pass")
                checks.append(_check(f"fit[{cid}] EIS relative RMSE {lab}", st, round(r["rel_rmse"], 3),
                                     [0.15, 0.4]))
        for name, v in c.get("best_inputs", {}).items():
            if name in bounds:
                hit = at_bound(v, *bounds[name], log_scale=name in logs)
                if hit:
                    checks.append(_check(f"fit[{cid}] {name[:60]} not at bound", "warn", v, bounds[name],
                                         f"at {hit} bound: not identified by the data"))
        d = c.get("diagnostics") or {}
        if "error" in d:
            checks.append(_check(f"fit[{cid}] history simulated", "fail", None, None, d["error"]))
            continue
        done = d.get("t_simulated_s", 0.0) / max(d.get("t_total_s", 1.0), 1.0)
        checks.append(_check(f"fit[{cid}] whole history simulated", "pass" if done > 0.999 else "fail",
                             round(done, 4), 1.0, d.get("termination", "")))
        for dom, o in (d.get("ocp") or {}).items():
            st = "pass" if o["fraction_out"] == 0 else ("warn" if o["excursion"] <= 0.02 else "fail")
            checks.append(_check(f"fit[{cid}] {dom} surface stoichiometry inside OCP data", st,
                                 round(o["fraction_out"], 4), 0.0,
                                 f"simulated {o['min']:.3f}-{o['max']:.3f}, data {o['data_range'][0]:.3f}-"
                                 f"{o['data_range'][1]:.3f} (fraction of time outside)"))
        if "T_max_K" in d and ambient_k is not None:
            rise = d["T_max_K"] - ambient_k
            checks.append(_check(f"fit[{cid}] cell temperature rise", "pass" if rise < 10 else
                                 ("warn" if rise < 20 else "fail"), round(rise, 2), [0, 10],
                                 "lumped thermal; coin cell at <=4C should stay within ~10 K"))
        if "c_e_min_rel" in d:
            ce = d["c_e_min_rel"]
            checks.append(_check(f"fit[{cid}] electrolyte not depleted", "pass" if ce > 0.2 else
                                 ("warn" if ce > 0.05 else "fail"), round(ce, 4), 0.2,
                                 "minimum electrolyte concentration / initial over the history"))

    ov = fit_report["parameter_overrides"]
    t_area = float(pv_template["Electrode height [m]"]) * float(pv_template["Electrode width [m]"])
    checks.append(_range_check("geometry: electrode area / template",
                               ov["Electrode height [m]"] * ov["Electrode width [m]"] / t_area,
                               0.5, 2.0, "template area scaled from Ai2020 area-per-capacity", fail_factor=2.0))
    checks.append(_range_check("geometry: negative active fraction / template", ov[EPS_N] / float(pv_template[EPS_N]),
                               0.7, 1.3, "N:P knob; commercial anodes 0.5-0.7", fail_factor=1.5))
    for dom in ("Negative", "Positive"):
        act = float(ov.get(f"{dom} electrode active material volume fraction",
                           pv_template[f"{dom} electrode active material volume fraction"]))
        por = float(ov.get(f"{dom} electrode porosity", pv_template[f"{dom} electrode porosity"]))
        checks.append(_check(f"geometry: {dom.lower()} volume fractions (active + pores) <= 1",
                             "pass" if act + por <= 1.0 + 1e-9 and por >= 0.15 else "fail",
                             round(act + por, 4), 1.0, f"active {act:.3f}, porosity {por:.3f}"))
    med = {k: v["median"] for k, v in fit_report["fit"]["distributions"].items()}
    if THICKNESS_SCALE in med:
        checks.append(_range_check("geometry: electrode thickness / template", med[THICKNESS_SCALE], 0.52, 1.31,
                                   "40-100 um per electrode (Chen 2020; Ecker 2015); Ai2020 68-77 um",
                                   fail_factor=1.2))
    if LI_FRACTION in med:
        checks.append(_range_check("cyclable lithium / positive capacity", med[LI_FRACTION], 0.85, 1.0,
                                   "LiCoO2 is made fully lithiated (1.0); 5-15% lost at formation",
                                   fail_factor=1.1))
    if H_TOTAL in med:
        checks.append(_range_check("heat-transfer coefficient [W/(m2 K)]", med[H_TOTAL], 5.0, 25.0,
                                   "natural convection 5-25 W/(m2 K), plus conduction into the holder",
                                   fail_factor=2.0))
    esoh = fit_report["fit"].get("esoh") or {}
    meas = fit_report["fit"].get("measured_capacity_mah")
    if "capacity_ah" in esoh and meas:
        checks.append(_range_check("capacity between OCV limits / measured 1C capacity",
                                   1000.0 * esoh["capacity_ah"] / meas, 1.0, 1.3,
                                   "open-circuit capacity must hold the measured ~1C capacity, "
                                   "with a modest rate loss", fail_factor=1.3))
    elif "error" in esoh:
        checks.append(_check("electrode SOH on the fitted cell", "fail", None, None, esoh["error"]))
    for key, nom in (("ocv100", 4.2), ("ocv0", 3.0)):
        if key in esoh:
            clamped = esoh.get(f"{key}_clamped")
            checks.append(_check(f"SOC window {key} inside OCP data", "warn" if clamped else "pass",
                                 round(esoh[key], 4), nom,
                                 "clamped to the electrode OCP data edge" if clamped else "protocol value"))
    fitted_kwargs = {**oracle_kwargs, "parameter_overrides": ov,
                     "parameter_scales": fit_report["parameter_scales"],
                     "contact_resistance_ohm": fit_report["contact_resistance_ohm"]}
    if fit_report.get("thermal"):
        fitted_kwargs.update(thermal=fit_report["thermal"], h_total_W_per_m2K=fit_report["h_total_W_per_m2K"])
        if ambient_k is not None:
            fitted_kwargs["T_ambient_K"] = ambient_k
    pv_fitted = apply_parameter_overrides(pv_template.copy(), ov)
    apply_parameter_scales(pv_fitted, fit_report["parameter_scales"])
    pv_fitted.update({"Contact resistance [Ohm]": fit_report["contact_resistance_ohm"]}, check_already_exists=False)
    # bound hits are checked per cell above (each parameter has its own centred bounds)
    shim = {"fit": {"eis": {}},
            "parameter_scales": fit_report["parameter_scales"],
            "contact_resistance_ohm": fit_report["contact_resistance_ohm"]}
    checks += _microstructure_checks(pv_fitted, fit_report["fit"].get("measured_capacity_mah"))
    checks += check_fitted_values(pv_fitted, shim) + check_esoh(pv_fitted) + check_dfn_cycle(fitted_kwargs)
    report = {"stage": "A (chain)", **summarise(checks), "checks": checks}
    log.info("stage-A (chain) physicality: %s %s", report["status"], report["counts"])
    return report


def _microstructure_checks(pv, measured_capacity_mah: float | None) -> list[dict]:
    """Tortuosities, positive utilisation and N:P of the fitted cell vs measured
    ranges (graphite tau 3-10: Landesfeind et al. 2016, Ebner et al. 2014; separator
    MacMullin 2.5-5: Landesfeind 2016, Thorat et al. 2009; LiCoO2 cycles ~40-65% of
    its theoretical capacity; practical N:P ~1.05-1.2 (reversible negative/positive))."""
    from battery_oracle.bol_chain import electrode_capacities_ah

    out = []
    for dom, lo, hi, ref in (("Negative electrode", 3.0, 10.0, "graphite tau 3-10"),
                             ("Positive electrode", 1.5, 6.0, "LiCoO2 electrodes tau ~2-5"),
                             ("Separator", None, None, "")):
        eps = float(pv[f"{dom} porosity"])
        b = float(pv[f"{dom} Bruggeman coefficient (electrolyte)"])
        tau = eps ** (1.0 - b)
        if dom == "Separator":
            n_m = eps ** (-b)
            out.append(_range_check("separator MacMullin number", n_m, 2.5, 5.0,
                                    "Celgard-type separators ~2.8-4.5", fail_factor=1.5))
        else:
            out.append(_range_check(f"{dom.lower()} tortuosity", tau, lo, hi, ref, fail_factor=1.5))
    try:
        q_n, q_p = electrode_capacities_ah(pv)
        if measured_capacity_mah:
            q = measured_capacity_mah / 1000.0
            out.append(_range_check("positive utilisation (measured / theoretical)", q / q_p, 0.40, 0.65,
                                    "LiCoO2 cycles ~0.45-1.0 of its lithium", fail_factor=1.3))
            out.append(_range_check("practical N:P (0.93 Q_n / measured)", 0.93 * q_n / q, 1.0, 1.4,
                                    "commercial 1.05-1.2", fail_factor=1.3))
    except Exception as exc:                    # noqa: BLE001
        out.append(_check("electrode capacities", "skip", None, None, str(exc)[:120]))
    return out


def check_bol(fit_report: dict, pv_template, oracle_kwargs: dict) -> dict:
    """Stage-A physicality report for a :func:`battery_oracle.bol_fit.fit_bol` result.

    ``pv_template`` is the base parameter set before the fit (what geometry ratios
    are taken against); ``oracle_kwargs`` build the base oracle (chemistry, model,
    bounds, temperature) -- the fitted BOL values are added here.
    """
    from battery_oracle.params import apply_parameter_overrides, apply_parameter_scales

    fitted_kwargs = {**oracle_kwargs,
                     "parameter_overrides": fit_report["parameter_overrides"],
                     "parameter_scales": fit_report["parameter_scales"],
                     "contact_resistance_ohm": fit_report["contact_resistance_ohm"]}
    pv_fitted = apply_parameter_overrides(pv_template.copy(), fit_report["parameter_overrides"])
    apply_parameter_scales(pv_fitted, fit_report["parameter_scales"])
    checks = (check_balance(fit_report, pv_template) + check_fitted_values(pv_fitted, fit_report)
              + check_esoh(pv_fitted) + check_dfn_cycle(fitted_kwargs))
    report = {"stage": "A", **summarise(checks), "checks": checks}
    log.info("stage-A physicality: %s %s", report["status"], report["counts"])
    return report


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------

def check_tune(best: dict, ranges: dict, calibration_statuses: dict | None = None,
               soh_history: list[float] | None = None) -> dict:
    """Stage-B physicality report: degradation scales at bounds, capacity gains,
    FAILED calibration targets."""
    checks = []
    for key, lo_k, hi_k in (("sei_rate_scale", "srs_min", "srs_max"),
                            ("dead_li_decay_scale", "dds_min", "dds_max"),
                            ("plating_rate_scale", "prs_min", "prs_max"),
                            ("dod_lam_scale", "dl_min", "dl_max"),
                            ("kinetics_scale", "ks_min", "ks_max")):
        lo, hi = ranges.get(lo_k), ranges.get(hi_k)
        v = best.get(key)
        if v is None or lo is None or hi is None or lo == hi:
            continue
        hit = at_bound(float(v), float(lo), float(hi), log_scale=True)
        checks.append(_check(f"{key} not at search bound", "warn" if hit else "pass", v, [lo, hi],
                             f"at {hit} bound: widen the range or the target does not "
                             f"constrain it" if hit else ""))
    if soh_history and len(soh_history) > 1:
        gains = np.diff(np.asarray(soh_history, dtype=float))
        worst = float(np.max(gains))
        checks.append(_check("no capacity gain between cycles (max SOH increase)",
                             "pass" if worst <= 1e-4 else "warn", worst, "<= 1e-4",
                             "SOH should not rise between cycles"))
    for target, status in (calibration_statuses or {}).items():
        checks.append(_check(f"calibration target: {target}",
                             "fail" if str(status).startswith("FAILED") else
                             ("warn" if str(status).startswith("partial") else "pass"), status))
    report = {"stage": "B", **summarise(checks), "checks": checks}
    log.info("stage-B physicality: %s %s", report["status"], report["counts"])
    return report
