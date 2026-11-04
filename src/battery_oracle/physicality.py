"""Physicality checks: is a parameterised oracle a physically sensible cell?

Every check returns ``{"name", "status", "value", "limit", "note"}`` with
``status`` one of ``"pass"``, ``"warn"`` (suspicious -- inspect before trusting),
``"fail"`` (unphysical -- do not use) or ``"skip"`` (could not be evaluated).
:func:`summarise` reduces a list to an overall status (the worst one).

Validity before accuracy: a calibration need not reproduce the cell exactly, but it
must be physically valid; these checks judge the internal solution and the
parameters, never the fit error.

* fitted kinetics/transport at a reference state inside broad plausibility windows
  (about an order of magnitude either side of published LiCoO2/graphite values -- a
  guard against runaway fits, not a literature fit), :func:`check_fitted_values`;
* electrode state of health: PyBaMM's electrode-SOH solve succeeds with x/y limits
  strictly inside (0, 1), :func:`check_esoh`;
* one demanding DFN cycle on the deployed oracle (:func:`check_dfn_cycle`): completes
  with the primary solver, no electrolyte depletion (:data:`CE_FAIL_REL`), no surface-
  stoichiometry saturation, lithium conserved, and the two-mesh validity gate of
  :mod:`battery_oracle.validity`.

The stage-A checks of a chain fit build on these (:mod:`battery_oracle.fit.checks`).
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_ORDER = {"pass": 0, "skip": 0, "warn": 1, "fail": 2}

#: Electrolyte depletion (minimum c_e / initial), one pair for every check and the fits'
#: floor (:data:`battery_oracle.fit.config.CE_FLOOR_DEFAULT`): below CE_FAIL_REL the
#: electrolyte, not the electrodes, is the rate limit, and the DFN is least reliable;
#: below CE_WARN_REL the margin to it is small.
CE_FAIL_REL = 0.1
CE_WARN_REL = 0.2


def electrolyte_status(ce_min_rel: float) -> str:
    """pass / warn / fail of a minimum electrolyte concentration relative to initial."""
    return "fail" if ce_min_rel < CE_FAIL_REL else ("warn" if ce_min_rel < CE_WARN_REL else "pass")


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


def run_demanding_cycle(oracle_kwargs: dict, *, c1_crate: float = 3.8, c2_crate: float = 2.0,
                        d_crate: float = 3.8, **overrides):
    """The fitted oracle (``oracle_kwargs`` plus ``overrides``, Randles-stub ECM) after one
    demanding cycle at the stated C-rates of its own nominal capacity (real-cell scaling 1):
    two 0.25 h charge stages, then a discharge of up to 4 h. The oracle's current bounds
    and DFN C-rate ceiling are raised to the stated rates where they are lower, so the
    cycle runs at those rates (never silently clipped). Raises what the oracle raises."""
    from battery_oracle.ecm import randles_stub_ecm
    from battery_oracle.oracle import PyBaMMOracle

    oracle = PyBaMMOracle(**{**oracle_kwargs, "ecm_model_fn": randles_stub_ecm, **overrides})
    nominal = oracle._nominal_mAh
    oracle._real_cell_capacity_mah = nominal
    oracle._cap_scale = 1.0
    top = max(c1_crate, d_crate)
    oracle._C_MAX_mA = max(oracle._C_MAX_mA, top * nominal)
    oracle._C2_MAX_mA = max(oracle._C2_MAX_mA, c2_crate * nominal)
    oracle(np.array([c1_crate * nominal, c2_crate * nominal, 0.25, 0.25, d_crate * nominal, 4.0]))
    return oracle


def check_dfn_cycle(oracle_kwargs: dict, *, c1_crate: float = 3.8, c2_crate: float = 2.0,
                    d_crate: float = 3.8) -> list[dict]:
    """One demanding cycle on the fitted cell (default: the jones2022 protocol envelope,
    ~3.8C charge and discharge); internal-state and conservation checks.

    ``oracle_kwargs`` must build the fitted oracle (chemistry, model, BOL values,
    bounds, temperature). Currents are given at the stated C-rates of its own
    nominal capacity, so the real-cell scaling is 1.
    """
    from battery_oracle.failures import OracleFailure

    kw = dict(oracle_kwargs)
    rates = {"c1_crate": c1_crate, "c2_crate": c2_crate, "d_crate": d_crate}
    try:
        oracle = run_demanding_cycle(kw, **rates)
        nominal = oracle._nominal_mAh
    except OracleFailure as exc:
        return [_check("demanding DFN cycle completes", "fail", None, None, str(exc)[:200])]
    except Exception as exc:
        return [_check("demanding DFN cycle completes", "fail", None, None,
                       f"{type(exc).__name__}: {str(exc)[:200]}")]
    h = oracle.history[-1]
    sol = oracle._prev_solution
    fk = h.get("failure_kind")
    out = [_check("demanding DFN cycle completes with the primary solver",
                  "pass" if fk is None else "warn", None if fk is None else str(fk), None,
                  f"{c1_crate}C/{c2_crate}C charge, {d_crate}C discharge")]

    def _entries(var):
        return np.asarray(sol[var].entries, dtype=float)

    try:
        ce = _entries("Electrolyte concentration [mol.m-3]")
        ce0 = float(oracle._pv["Initial concentration in electrolyte [mol.m-3]"])
        ratio = float(np.min(ce)) / ce0
        out.append(_check("electrolyte not depleted (min c_e / initial)", electrolyte_status(ratio), ratio,
                          [CE_FAIL_REL, CE_WARN_REL],
                          "the fits' electrolyte floor; below it the electrolyte is the rate limit"))
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
        from battery_oracle.parameter_sets import ocp_data_range

        s_min = float(np.min(_entries("Minimum positive particle surface stoichiometry")))
        s_max = float(np.max(_entries("Maximum positive particle surface stoichiometry")))
        lo, hi = ocp_data_range(oracle._pv, "positive")
        out.append(_check("positive stoichiometry inside the OCP data range",
                          "pass" if (s_min >= lo - 1e-3 and s_max <= hi + 1e-3) else "warn",
                          [s_min, s_max], [lo, hi],
                          "outside it PyBaMM extrapolates the electrode curve"))
    except Exception as exc:
        out.append(_check("positive stoichiometry inside the OCP data range", "skip", None, None, str(exc)[:120]))
    # the two-mesh validity gate (battery_oracle.validity.judge) on the oracle's own
    # demanding cycle -- the model exp2 uses: the same cycle on the oracle's mesh and on
    # it with the electrode/separator mesh refined x2
    try:
        from battery_oracle import validity
        tag = "oracle demanding cycle"
        mesh = dict(kw.get("radial_mesh") or {})
        xf = int(mesh.get("x_factor") or 1)
        fine_kw = {**kw, "radial_mesh": {**mesh, "x_factor": 2 * xf}}
        fine = run_demanding_cycle(fine_kw, **rates)
        fsol = fine.last_solution
        names = (f"x{xf}", f"x{2 * xf}")
        m_dep = validity.measure(sol, oracle._pv)
        ladder = None
        if any((m_dep.get(k) or 0.0) > 10.0 * oracle._dfn_solver_rtol for k in ("charge", "charge_integral", "lithium")):
            # an exact-law residual above 10 rtol: is it a tolerance error? (10x and 100x tighter solves)
            ladder = []
            for f in validity.TOLERANCE_LADDER:
                try:
                    tight = run_demanding_cycle(kw, **rates, dfn_solver_rtol=oracle._dfn_solver_rtol * f,
                                                dfn_solver_atol=oracle._dfn_solver_atol * f)
                    tsol = tight.last_solution
                    complete = (tight.history[-1].get("failure_kind") is None
                                and float(tsol.t[-1]) >= float(sol.t[-1]) - 1.0)
                except OracleFailure:
                    complete = False
                # a failed or truncated tighter solve is no evidence either way
                ladder.append(validity.measure(tsol, tight._pv) if complete else {})
        out += validity.judge(m_dep, validity.measure(fsol, fine._pv), factor=2, cell=tag, mesh=names,
                              rtol=oracle._dfn_solver_rtol, ladder=ladder)
        def _tvi(s):
            return (s.t, s["Voltage [V]"].entries, s["Current [A]"].entries)

        _, dv = validity.voltage_difference(*_tvi(sol), *_tvi(fsol), i_1c=nominal / 1000.0)
        p99 = float(np.percentile(dv, 99))
        out.append(_check(f"validity[{tag}] {names[0]} mesh resolves the voltage",
                          "pass" if p99 < validity.VOLTAGE_NOISE_V else "fail", 1e3 * p99,
                          1e3 * validity.VOLTAGE_NOISE_V, f"p99 |dV| mV {names[0]} vs {names[1]} outside "
                          f"{validity.STEP_WINDOW_S:g} s of current steps; limit the instrument noise "
                          "(a fail means the oracle must run the refined mesh)"))
    except Exception as exc:                    # noqa: BLE001 - unevaluable validity fails
        out.append(_check("validity[oracle demanding cycle] two-mesh gate evaluated", "fail", None, None,
                          f"{type(exc).__name__}: {str(exc)[:160]}"))
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


def fitted_oracle_kwargs(fit_report: dict, oracle_kwargs: dict, ambient_k: float | None = None) -> dict:
    """The deployed oracle of a chain fit report: ``oracle_kwargs`` plus the fitted values,
    thermal submodel and particle/x mesh -- what the replays and exp2 build from it."""
    kw = {**oracle_kwargs, "parameter_overrides": fit_report["parameter_overrides"],
          "parameter_scales": fit_report["parameter_scales"],
          "contact_resistance_ohm": fit_report["contact_resistance_ohm"]}
    if fit_report.get("thermal"):
        kw.update(thermal=fit_report["thermal"], h_total_W_per_m2K=fit_report["h_total_W_per_m2K"])
        if ambient_k is not None:
            kw["T_ambient_K"] = ambient_k
    if fit_report.get("radial_mesh"):
        # the demanding cycle and its two-mesh gate judge the deployed mesh, not PyBaMM's default
        kw["radial_mesh"] = fit_report["radial_mesh"]
    return kw
