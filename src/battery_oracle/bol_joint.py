"""Stage A, joint DFN fit per cell, with the measurement modelled as performed.

Replaces the earlier two-step stage A (equilibrium OCV balancing on reshaped
formation branches, then an equilibrium EIS fit). One PyBOP ``MetaProblem`` per
cell, fitted with CMA-ES, holds every dataset the cell has, each simulated with
its own real history:

* the **formation record** -- current-driven DFN, rests included. The ~240 mV
  instant and ~250 mV slow voltage recovery after the 0.9C discharge constrain
  the series resistance and solid diffusion directly (no equilibrium assumption);
* **cycling records** (charge and discharge of a few cycles, 1-4C): charge
  acceptance and rate behaviour;
* the cell's **EIS spectra**, each computed by
  :func:`battery_oracle.eis_nonstationary.nonstationary_impedance` (through
  :mod:`battery_oracle.patches`): the DFN integrated through that record's real
  current history, linearised at the state where each frequency was measured.

Parameters (all PyBOP ``Parameter`` objects, log-uniform where positive scales):
effective electrode area (as a continuous parallel-layer count, exactly equivalent
in the 1-D DFN and free of mesh rebuilds), N:P ratio (negative active-material
fraction), exchange currents, solid diffusivities, electrolyte conductivity,
double-layer capacitances, contact resistance. The open-circuit-voltage limits
that define 0% / 100% state of charge are fixed (3.0 V = the discharge cut-off,
4.2 V = the formation CV level) so PyBaMM's electrode-SOH solve and the fit agree.

The warm start is the equilibrium balancing of :mod:`battery_oracle.bol_fit`
(its weakness -- state-of-charge dependent overpotential -- is exactly what this
fit removes; it is only a starting guess here).

Inputs are plain dicts built by battery-forecast's ``oracle_recal.joint_fit_data``
(model-frame currents [A, positive = discharge], voltages, spectra). Output is
JSON-serialisable and carries measured vs fitted curves for plotting.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from battery_oracle.bol_fit import EIS_SCALES
from battery_oracle.eis_nonstationary import History
from battery_oracle.params import add_scale_parameters, apply_parameter_overrides, scale_parameter_name

log = logging.getLogger(__name__)

N_PARALLEL = "Number of electrodes connected in parallel to make a cell"
EPS_N = "Negative electrode active material volume fraction"
CONTACT_R = "Contact resistance [Ohm]"
OCV0, OCV100 = "Open-circuit voltage at 0% SOC [V]", "Open-circuit voltage at 100% SOC [V]"

#: Typical misfit that counts as 1 in the combined cost (voltage 10 mV; impedance
#: 5% of |Z|): keeps the time-domain and EIS parts comparable.
SIGMA_V = 0.010
SIGMA_Z_REL = 0.05
WIDE_WINDOW = (2.5, 4.6)


def widened(pv):
    pv = pv.copy()
    pv.update({"Lower voltage cut-off [V]": WIDE_WINDOW[0], "Upper voltage cut-off [V]": WIDE_WINDOW[1]})
    return pv


def time_model():
    """DFN with contact resistance (the oracle's cycling model without degradation)."""
    import pybamm
    return pybamm.lithium_ion.DFN(options={"contact resistance": "true"})


def eis_model_options() -> dict:
    return {"surface form": "differential", "contact resistance": "true"}


def _centred(bounds, value, min_factor=1.5):
    """Log-bounds whose geometric mean is ``value`` (and inside ``bounds``).

    PyBOP's population optimisers (CMAES, DE) start from the *mean of the prior
    distribution* and ignore ``initial_value``; for a LogUniform prior that is the
    geometric midpoint of the bounds. Centring the bounds on the warm start makes the
    warm start effective and sets the initial step (the prior's std) to a sensible size.
    """
    lo, hi = bounds
    v = float(np.clip(value, lo, hi))
    f = max(min(v / lo, hi / v), min_factor)
    return (v / f, v * f)


def fit_parameters(pv, warm: dict | None = None, *, scale_bounds=(0.01, 100.0),
                   cr_bounds=(1e-6, 0.5), n_bounds=(0.2, 5.0), eps_n_bounds=(0.3, 0.8),
                   free: list[str] | None = None) -> dict:
    """The PyBOP parameters of the joint fit, with warm-start initial values.

    ``free`` (names as in the dict returned): fit only these; the rest keep the
    values in ``pv`` (multipliers 1, contact resistance / N / eps_n as in ``pv``).
    """
    import pybop

    warm = warm or {}
    fit = {}
    for n in EIS_SCALES:
        v0 = warm.get(n, 1.0)
        fit[scale_parameter_name(n)] = pybop.Parameter(
            distribution=pybop.LogUniform(*_centred(scale_bounds, v0)), initial_value=v0,
            transformation=pybop.LogTransformation())
    fit[CONTACT_R] = pybop.Parameter(
        distribution=pybop.LogUniform(*_centred(cr_bounds, warm.get(CONTACT_R, 0.05))),
        initial_value=float(np.clip(warm.get(CONTACT_R, 0.05), *cr_bounds)),
        transformation=pybop.LogTransformation())
    fit[N_PARALLEL] = pybop.Parameter(
        distribution=pybop.LogUniform(*_centred(n_bounds, warm.get(N_PARALLEL, 1.0))),
        initial_value=float(np.clip(warm.get(N_PARALLEL, 1.0), *n_bounds)),
        transformation=pybop.LogTransformation())
    fit[EPS_N] = pybop.Parameter(
        bounds=tuple(eps_n_bounds),
        initial_value=float(np.clip(warm.get(EPS_N, float(pv[EPS_N])), *eps_n_bounds)))
    if free is not None:
        missing = [k for k in free if k not in fit]
        if missing:
            raise KeyError(f"unknown free parameters {missing}; choose from {sorted(fit)}")
        fit = {k: v for k, v in fit.items() if k in free}
    return fit


def _with_fit(pv, fit: dict):
    pv = widened(pv)
    add_scale_parameters(pv, EIS_SCALES)
    pv.update(fit, check_already_exists=False)
    return pv


def _time_problem(rec: dict, pv, fit: dict):
    import pybop

    dataset = pybop.Dataset({"Time [s]": rec["t"], "Current [A]": rec["I"], "Voltage [V]": rec["V"]})
    from battery_oracle.patches import single_thread_solver

    sim = pybop.pybamm.Simulator(time_model(), parameter_values=_with_fit(pv, fit), protocol=dataset,
                                 solver=single_thread_solver(),
                                 initial_state={"Initial open-circuit voltage [V]": float(rec["V"][0])})
    return pybop.Problem(sim, pybop.RootMeanSquaredError(dataset))


def _eis_problem(rec: dict, spec: dict, pv, fit: dict, amplitude_a: float):
    import pybamm
    import pybop

    from battery_oracle import patches

    patches.apply(["eis_nonstationary", "graded_failure"], strict=True)
    f = np.asarray(spec["f"], float)
    dataset = pybop.Dataset({"Frequency [Hz]": f, "Current [A]": np.zeros_like(f),
                             "Impedance": np.asarray(spec["Z"], complex)}, domain="Frequency [Hz]")
    sim = pybop.pybamm.EISSimulator(pybamm.lithium_ion.DFN(options=eis_model_options()), f_eval=f,
                                    parameter_values=_with_fit(pv, fit))
    sim.set_measurement(History(t=np.asarray(rec["t"], float), I=np.asarray(rec["I"], float),
                                initial_voltage=float(rec["V"][0])),
                        sweep_times=spec.get("sweep_t"), n_periods=spec.get("n_periods", 1.0),
                        amplitude_a=amplitude_a)
    return pybop.Problem(sim, pybop.RootMeanSquaredError(dataset, target="Impedance"))


def build_problems(data: dict, pv, fit: dict, *, amplitude_a: float = 5e-3):
    """[(kind, label, problem, weight, target_dataset_arrays)] for one cell's data."""
    out = []
    form = data["formation"]
    out.append(("formation", form["label"], _time_problem(form, pv, fit), 1.0 / SIGMA_V, form))
    for rec in data["cycles"]:
        out.append(("cycling", rec["label"], _time_problem(rec, pv, fit), 1.0 / SIGMA_V, rec))
        spec = rec.get("spectrum")
        if spec is not None:
            sigma = SIGMA_Z_REL * float(np.median(np.abs(spec["Z"])))
            out.append(("eis", rec["label"], _eis_problem(rec, spec, pv, fit, amplitude_a),
                        1.0 / sigma, spec))
    return out


def fit_cell_joint(cell_id: str, data: dict, pv, *, warm: dict | None = None,
                   max_iterations: int = 150, max_unchanged: int = 30,
                   amplitude_a: float = 5e-3, seed: int | None = 0,
                   kinds: tuple[str, ...] = ("formation", "cycling", "eis"),
                   free: list[str] | None = None) -> dict[str, Any]:
    """Joint DFN fit of one cell (see module docstring); returns a JSON-able report."""
    import pybop

    import os

    from battery_oracle import patches

    patches.apply()
    fit = fit_parameters(pv, warm, free=free)
    problems = [p for p in build_problems(data, pv, fit, amplitude_a=amplitude_a) if p[0] in kinds]
    meta = pybop.MetaProblem(*[p[2] for p in problems], weights=[p[3] for p in problems])
    meta.set_workers(int(os.environ.get("ORACLE_FIT_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", 1))))
    if seed is not None:
        np.random.seed(seed)
    options = pybop.PintsOptions(max_iterations=max_iterations, max_unchanged_iterations=max_unchanged,
                                 verbose=True)
    t0 = time.time()
    result = pybop.CMAES(meta, options=options).run()
    runtime = time.time() - t0
    best = {k: float(np.asarray(v).ravel()[0]) for k, v in result.best_inputs.items()}
    log.info("%s: joint fit cost %s in %.0f s: %s", cell_id, result.best_cost, runtime, best)

    records = []
    for kind, label, prob, weight, target in problems:
        entry = {"kind": kind, "label": label, "weight": weight}
        try:
            sol = prob.simulator.solve(inputs=best)
            if kind == "eis":
                z = np.asarray(sol["Impedance"].data, dtype=complex)
                zt = np.asarray(target["Z"], complex)
                entry.update(f=np.asarray(target["f"]).tolist(), z_real=zt.real.tolist(),
                             z_imag=zt.imag.tolist(), zfit_real=z.real.tolist(), zfit_imag=z.imag.tolist(),
                             rmse_ohm=float(np.sqrt(np.mean(np.abs(z - zt) ** 2))),
                             rel_rmse=float(np.sqrt(np.mean(np.abs(z - zt) ** 2)) / np.median(np.abs(zt))))
            else:
                v = np.asarray(sol["Voltage [V]"].data)
                n = min(v.size, len(target["V"]))
                step = max(1, len(target["t"]) // 600)
                entry.update(t=np.asarray(target["t"])[::step].tolist(), v=np.asarray(target["V"])[::step].tolist(),
                             vfit=np.pad(v, (0, len(target["V"]) - n), constant_values=np.nan)[::step].tolist(),
                             n_simulated=int(n), n_points=int(len(target["V"])),
                             rmse_v=float(np.sqrt(np.mean((v[:n] - np.asarray(target["V"])[:n]) ** 2))))
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        records.append(entry)
    return {
        "cell_id": cell_id, "best_inputs": best, "cost": float(np.asarray(result.best_cost).ravel()[0]),
        "bounds": {"scale": [0.01, 100.0], "contact_r": [1e-6, 0.5],
                   "n_parallel": [0.2, 5.0], "eps_n": [0.3, 0.8]},
        "records": records, "runtime_s": round(runtime, 1), "n_iterations": int(result.n_iterations)
        if hasattr(result, "n_iterations") else None,
    }


# ---------------------------------------------------------------------------
# From a fit to the oracle's bol section
# ---------------------------------------------------------------------------

def overrides_from_inputs(pv, best: dict) -> tuple[dict, dict, float]:
    """(parameter_overrides, parameter_scales, contact_resistance) for the oracle.

    The fitted parallel-layer count becomes one layer of the scaled total area
    (the oracle's own EIS corrections read height x width); N:P enters through the
    negative active-material fraction; the 0% / 100% SOC voltages are the fixed
    fit limits; nominal capacity is the capacity PyBaMM's electrode-SOH solve
    gives between them on the fitted cell.
    """
    import pybamm

    area = float(pv["Electrode height [m]"]) * float(pv["Electrode width [m]"]) * best[N_PARALLEL]
    side = float(np.sqrt(area))
    ov = {"Electrode height [m]": side, "Electrode width [m]": side, N_PARALLEL: 1.0,
          EPS_N: best[EPS_N], OCV0: float(pv[OCV0]) if OCV0 in pv.keys() else 3.0,
          OCV100: float(pv[OCV100]) if OCV100 in pv.keys() else 4.2}
    pvf = apply_parameter_overrides(pv.copy(), ov)
    try:
        x0, x100, y100, y0 = (float(v) for v in pybamm.lithium_ion.get_min_max_stoichiometries(pvf))
        q_n = (float(pvf[EPS_N]) * float(pvf["Negative electrode thickness [m]"]) * area
               * float(pvf["Maximum concentration in negative electrode [mol.m-3]"]) * 96485.33212 / 3600.0)
        ov["Nominal cell capacity [A.h]"] = float((x100 - x0) * q_n)
    except Exception as exc:
        log.warning("nominal capacity from electrode SOH failed (%s); keeping template value", exc)
    scales = {n: best[scale_parameter_name(n)] for n in EIS_SCALES}
    return ov, scales, float(best[CONTACT_R])


def aggregate_joint(cell_reports: list[dict], pv) -> dict:
    """Median over cells (geometric for multiplicative parameters) -> bol section +
    per-parameter distributions across cells."""
    ok = [c for c in cell_reports if c.get("best_inputs")]
    failed = [{"cell_id": c["cell_id"], "reason": c.get("error", "no fit")} for c in cell_reports
              if not c.get("best_inputs")]
    if not ok:
        raise ValueError(f"no cell has a joint fit: {failed}")
    keys = ok[0]["best_inputs"].keys()
    geo = [k for k in keys if k != EPS_N]
    med = {k: float(np.exp(np.median([np.log(c["best_inputs"][k]) for c in ok]))) for k in geo}
    med[EPS_N] = float(np.median([c["best_inputs"][EPS_N] for c in ok]))
    ov, scales, cr = overrides_from_inputs(pv, med)
    dist = {}
    for k in keys:
        x = np.asarray([c["best_inputs"][k] for c in ok], float)
        dist[k] = {"values": x.tolist(), "median": float(np.median(x)),
                   "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}
    return {"parameter_overrides": ov, "parameter_scales": scales, "contact_resistance_ohm": cr,
            "fit": {"method": "joint DFN fit per cell (PyBOP CMAES MetaProblem: formation + cycling "
                              "records + non-stationary EIS); oracle values = cross-cell median",
                    "cells": [c["cell_id"] for c in ok], "failed_cells": failed,
                    "distributions": dist,
                    "bounds": ok[0]["bounds"],
                    "per_cell_cost": {c["cell_id"]: c["cost"] for c in ok}}}
