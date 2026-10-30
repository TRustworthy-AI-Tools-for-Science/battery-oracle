"""Beginning-of-life (BOL) fit: stage A of the oracle calibration, using PyBOP's methods.

``battery_oracle.tune`` calibrates the *degradation* scales; it assumes the fresh
cell already behaves like the real one. This module makes that true with two of
PyBOP's published parameterisation workflows (Planden et al. 2025, JOSS 10:7874;
github.com/pybop-team/PyBOP, ``examples/scripts``):

1. **Electrode balancing** -- PyBOP's ``dfn_parameterisation/3_cell_balancing.py``.
   A closed-form full-cell OCV model (:func:`ocv_balance_simulator`, PyBOP's
   ``OCVBalance`` simulator) is fitted to the charge and discharge branches of a
   pseudo-OCV measurement, plotted against a naive state of charge. Both branches
   share the electrode theoretical capacities and a series resistance; each has
   its own minimum stoichiometries; a Coulombic-efficiency factor closes each
   branch on its relaxed end voltage; ``pybop.SciPyDifferentialEvolution``
   optimises an RMSE in mV. The fitted OCV is then inverted (``pybop.InverseOCV``)
   at the oracle's voltage limits to give the stoichiometry window.
2. **Kinetics / transport from impedance** -- PyBOP's
   ``battery_parameterisation/simple_eis.py`` (frequency-domain fitting with
   ``pybop.pybamm.EISSimulator`` on the DFN; Hallemans et al. 2025, J.
   Electrochem. Soc.).
   Multipliers on exchange currents, solid diffusivities, electrolyte
   conductivity and double-layer capacitances, plus a contact resistance, are
   fitted to measured spectra, each simulated at its own relaxed voltage on the
   balanced cell, with ``pybop.CMAES`` on an impedance RMSE.

Whether the result *cycles* like the real cell is checked separately, not fitted
(battery-forecast's ``bin/oracle_step_baseline.py``).

Deviation from PyBOP's reference data: its pseudo-OCV uses C/30 branches with 2 h
rests and electrode OCPs measured on half cells. jones2022 offers only ~1C
formation branches with ~5 min rests and no half-cell data, so the electrode OCPs
are the parameter set's literature curves (LiR2032: Ai2020 LiCoO2/graphite; see
:func:`compare_balancing` to test alternatives) and the series-resistance term and
the Coulombic-efficiency closure absorb more overpotential than in PyBOP's example.

Inputs are dataset-agnostic (built by e.g. ``battery_forecast.oracle_recal``):

* a balancing *branch*: ``t`` [s], ``I`` [A, positive = discharge, scaled to the
  simulated cell's nominal capacity at the real C-rate -- for the LiR2032 set this
  is the real cell itself], ``V`` [V], ``direction``
  ("charge"/"discharge"), ``v_relax`` (voltage at the end of the following rest);
* a *spectrum*: ``f`` [Hz], ``Z`` [complex Ohm, scaled likewise], ``v_rest``
  (relaxed voltage at which it was measured), ``label``.

:func:`fit_bol` returns the oracle config's ``bol`` section
(``parameter_overrides`` -- the four concentrations -- ``parameter_scales``,
``contact_resistance_ohm``). PyBOP is an optional dependency, imported lazily.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from battery_oracle.parameter_sets import ocp_data_range  # noqa: F401  (re-exported for callers)

from battery_oracle.params import (
    add_scale_parameters,
    apply_parameter_overrides,
    scale_parameter_name,
)

log = logging.getLogger(__name__)

#: Multipliers fitted to impedance (stage 2).
EIS_SCALES: tuple[str, ...] = (
    "Negative electrode exchange-current density [A.m-2]",
    "Positive electrode exchange-current density [A.m-2]",
    "Negative particle diffusivity [m2.s-1]",
    "Positive particle diffusivity [m2.s-1]",
    "Electrolyte conductivity [S.m-1]",
    "Negative electrode double-layer capacity [F.m-2]",
    "Positive electrode double-layer capacity [F.m-2]",
)
#: Parameters written back to the oracle by the balancing step (see
#: :func:`balance_overrides`).
BALANCE_PARAMS: tuple[str, ...] = (
    "Open-circuit voltage at 0% SOC [V]",
    "Open-circuit voltage at 100% SOC [V]",
    "Electrode height [m]",
    "Electrode width [m]",
    "Negative electrode thickness [m]",
    "Nominal cell capacity [A.h]",
    "Initial concentration in negative electrode [mol.m-3]",
    "Initial concentration in positive electrode [mol.m-3]",
)
CONTACT_R = "Contact resistance [Ohm]"


# ---------------------------------------------------------------------------
# Parameter values
# ---------------------------------------------------------------------------

def oracle_parameter_values(**oracle_kwargs):
    """The parameter values a :class:`PyBaMMOracle` built with ``oracle_kwargs`` uses.

    Includes the degradation preset's parameter edits and ``kinetics_scale``, with
    the oracle's own voltage limits as the cut-offs, so fitted values are relative
    to exactly what the oracle simulates.
    """
    from battery_oracle.oracle import PyBaMMOracle, _randles_stub_ecm

    oracle = PyBaMMOracle(ecm_model_fn=_randles_stub_ecm, **oracle_kwargs)
    pv = oracle._pv.copy()
    pv.update({"Upper voltage cut-off [V]": oracle._V_CHARGE_MAX,
               "Lower voltage cut-off [V]": oracle._V_DISCHARGE_MIN})
    return pv


def _areal_capacity_ah_m2(pv, domain: str) -> float:
    """Theoretical areal capacity [Ah/m2] of one electrode (stoichiometry 0 -> 1)."""
    d = domain.capitalize()
    eps = float(pv[f"{d} electrode active material volume fraction"])
    L = float(pv[f"{d} electrode thickness [m]"])
    c_max = float(pv[f"Maximum concentration in {domain} electrode [mol.m-3]"])
    return eps * L * c_max * 96485.33212 / 3600.0


def balance_overrides(pv, q_th_n: float, q_th_p: float, window: dict,
                      q_cell_ah: float, ocv_limits: tuple[float, float]) -> dict[str, float]:
    """Map a balancing result onto the parameter set (template materials kept fixed).

    * electrode area from the positive theoretical capacity at the template's
      positive loading (square electrode, one layer);
    * negative thickness from the negative theoretical capacity at that area, so
      the N:P ratio is the fitted one while volume fractions stay physical;
    * nominal capacity = the capacity between the voltage limits;
    * initial concentrations at 100% state of charge (``x_100``, ``y_100``), which
      fix the cyclable lithium that PyBaMM's electrode-SOH solver works from;
    * the open-circuit voltages that define 0% / 100% state of charge
      (``ocv_limits``) -- the same voltages the window was read at, so PyBaMM's
      electrode-SOH solve reproduces the window.
    """
    area = q_th_p / _areal_capacity_ah_m2(pv, "positive")
    side = float(np.sqrt(area))
    neg_per_m = _areal_capacity_ah_m2(pv, "negative") / float(pv["Negative electrode thickness [m]"])
    return {
        "Open-circuit voltage at 0% SOC [V]": float(ocv_limits[0]),
        "Open-circuit voltage at 100% SOC [V]": float(ocv_limits[1]),
        "Electrode height [m]": side,
        "Electrode width [m]": side,
        "Negative electrode thickness [m]": q_th_n / (neg_per_m * area),
        "Nominal cell capacity [A.h]": float(q_cell_ah),
        "Initial concentration in negative electrode [mol.m-3]":
            window["x_100"] * float(pv["Maximum concentration in negative electrode [mol.m-3]"]),
        "Initial concentration in positive electrode [mol.m-3]":
            window["y_100"] * float(pv["Maximum concentration in positive electrode [mol.m-3]"]),
    }


def _electrode_capacity_ah(pv, domain: str) -> float:
    """Theoretical capacity [Ah] of one electrode (stoichiometry 0 -> 1)."""
    d = domain.capitalize()
    eps = float(pv[f"{d} electrode active material volume fraction"])
    L = float(pv[f"{d} electrode thickness [m]"])
    area = float(pv["Electrode height [m]"]) * float(pv["Electrode width [m]"])
    c_max = float(pv[f"Maximum concentration in {domain} electrode [mol.m-3]"])
    return eps * L * area * c_max * 96485.33212 / 3600.0


def ocp_interpolants(pv, n: int = 1001):
    """(positive, negative) electrode OCPs of ``pv`` as ``pybop.Interpolant``.

    PyBOP's balancing reads half-cell pOCP interpolants
    (``pybop.applications.utils.get_ocp_functions``); with no half-cell data, the
    parameter set's own OCPs -- functions or interpolants -- are sampled instead.
    """
    import pybamm
    import pybop

    sto = np.linspace(0.0, 1.0, n)
    out = []
    for d in ("Positive", "Negative"):
        f = pv[f"{d} electrode OCP [V]"]
        u = f(pybamm.Vector(sto)) if callable(f) else f
        y = np.asarray(pv.process_symbol(u).evaluate(), dtype=float).ravel()
        out.append(pybop.Interpolant(sto, y, name=f"{d} OCP"))
    return tuple(out)


def ocv_soc_limits(pv) -> tuple[float, float]:
    """Open-circuit voltages that define 0% / 100% SOC (PyBaMM's electrode-SOH
    limits); fall back to the voltage cut-offs when the set does not define them."""
    try:
        return (float(pv["Open-circuit voltage at 0% SOC [V]"]),
                float(pv["Open-circuit voltage at 100% SOC [V]"]))
    except KeyError:
        return float(pv["Lower voltage cut-off [V]"]), float(pv["Upper voltage cut-off [V]"])


def initial_stoichiometries(pv, voltage: float) -> tuple[float, float]:
    """(x_n, y_p) at a relaxed open-circuit ``voltage`` (PyBaMM electrode-SOH
    inversion; voltages outside the 0-100% SOC open-circuit limits are clipped)."""
    import pybamm

    v_lo, v_hi = ocv_soc_limits(pv)
    v = float(np.clip(voltage, v_lo + 1e-3, v_hi - 1e-3))
    x, y = pybamm.lithium_ion.get_initial_stoichiometries(f"{v} V", pv)
    return float(x), float(y)


# ---------------------------------------------------------------------------
# Stage 1: electrode balancing (PyBOP 3_cell_balancing)
# ---------------------------------------------------------------------------

def ocv_balance_simulator(parameter_values: dict, dataset, direction: str, ocps):
    """PyBOP's ``OCVBalance`` simulator (``3_cell_balancing.py``), same form.

    ``V = U_p(y(soc)) - U_n(x(soc)) - R0 * I`` over the dataset's "Naive SoC",
    with x/y spanning ``Cell capacity / theoretical capacity`` from each branch's
    minimum stoichiometries.
    """
    import pybop

    positive_ocp, negative_ocp = ocps

    class OCVBalance(pybop.BaseSimulator):
        def __init__(self):
            self._parameter_values = dict(parameter_values)
            super().__init__(parameters=self._parameter_values)
            self.domain_data = dataset["Naive SoC"]
            self.input_data = dataset["Current [A]"]
            self.direction = direction

        def get_model_parameters(self, inputs):
            param = dict(self._parameter_values)
            param.update(inputs)
            Q = param["Cell capacity [A.h]"]
            sto_n_0 = param["Minimum negative stoichiometry " + self.direction]
            sto_n_100 = sto_n_0 + Q / param["Negative electrode theoretical capacity [A.h]"]
            sto_p_100 = param["Minimum positive stoichiometry " + self.direction]
            sto_p_0 = sto_p_100 + Q / param["Positive electrode theoretical capacity [A.h]"]
            R0 = param["Series resistance [Ohm]"]
            return Q, sto_n_0, sto_n_100, sto_p_100, sto_p_0, R0

        def solve_batch(self, inputs, calculate_sensitivities: bool = False):
            solutions = []
            for x in inputs:
                Q, sto_n_0, sto_n_100, sto_p_100, sto_p_0, R0 = self.get_model_parameters(x)
                soc = self.domain_data
                pos = positive_ocp(sto_p_0 + (sto_p_100 - sto_p_0) * soc)
                neg = negative_ocp(sto_n_0 + (sto_n_100 - sto_n_0) * soc)
                sol = pybop.Solution()
                sol.set_solution_variable("Voltage [V]", data=pos - neg - R0 * self.input_data)
                solutions.append(sol)
            return solutions

    return OCVBalance()


def _monotone(v: np.ndarray, frac: np.ndarray, increasing: bool):
    """Strictly monotone (voltage, fraction) pairs for an inverse-pOCV interpolant
    (PyBOP smooths with ``monotonic=True``; a CV hold repeats the same voltage)."""
    v_m = np.maximum.accumulate(v) if increasing else np.minimum.accumulate(v)
    keep = np.r_[True, np.diff(v_m) != 0]
    v_m, f_m = v_m[keep], frac[keep]
    order = np.argsort(v_m)
    return v_m[order], f_m[order]


def _branch_dataset(branch: dict, dv_keep: float = 1e-3):
    """PyBOP dataset of one branch, thinned to ~``dv_keep`` volt steps (PyBOP's
    ``smoothing.downsample(sampling_interval=0.001)``), with its throughput."""
    import pybop

    t, I, V = (np.asarray(branch[k], float) for k in ("t", "I", "V"))
    dc = np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t))]) / 3600.0
    keep = np.zeros(t.size, bool)
    keep[[0, -1]] = True
    last = V[0]
    for k in range(1, t.size):
        if abs(V[k] - last) >= dv_keep or abs(I[k] - I[k - 1]) > 1e-3 * max(abs(I[k]), 1e-9):
            keep[k] = True
            last = V[k]
    return pybop.Dataset({"Time [s]": t[keep], "Current [A]": I[keep], "Voltage [V]": V[keep],
                          "Discharge capacity [A.h]": dc[keep]})


def fit_balance(branches: list[dict], pv, *, maxiter: int = 100, seed: int | None = 0,
                q_th_ratio: tuple[float, float] = (1.0, 2.5),
                ocv_limits: tuple[float, float] | None = None) -> dict[str, Any]:
    """PyBOP ``3_cell_balancing`` on one cell's charge + discharge branches.

    Returns the stoichiometry window at ``pv``'s voltage limits per branch, the
    shared theoretical capacities / series resistance, and the parameter
    overrides of :func:`balance_overrides` (window averaged over the two
    branches), plus each branch's measured and fitted curves (``curve``).

    Theoretical-capacity bounds are ``q_th_ratio`` x the nominal capacity
    (``pv``'s, in the same frame as the branch currents). PyBOP's example uses
    1-1.5x; the default here is 1-2.5x because the Ai2020 template's own positive
    electrode holds 2.0x its nominal capacity, so 1.5x would exclude the template.

    The window is read where the fitted OCV equals ``ocv_limits`` (0% and 100%
    SOC). Default: from the branches -- the discharge branch's relaxed end
    voltage (0%) and the charge branch's highest voltage, its CV level (100%).
    PyBOP's example reads it at the voltage cut-offs; on these cells the rested
    voltage never reaches the 3.0 V discharge cut-off (~3.58 V after a full
    discharge), so the cut-off gives no valid window.
    """
    import pybop
    from pybop.applications.utils import OpenCircuitVoltage

    q_nom_ah = float(pv["Nominal cell capacity [A.h]"])
    ocps = ocp_interpolants(pv)
    q_lo, q_hi = q_th_ratio[0] * q_nom_ah, q_th_ratio[1] * q_nom_ah
    # Minimum-stoichiometry bounds: PyBOP's example uses [0, 0.4] for both
    # electrodes (NMC-like). Here each starts at the OCP data's own lower limit
    # and spans 0.4 above it, so the fit stays where the curve is measured --
    # Ai2020 LiCoO2 data cover y = 0.4-0.999; [0, 0.4] would force it to the edge.
    n_lo = ocp_data_range(pv, "negative")[0]
    p_lo = ocp_data_range(pv, "positive")[0]
    sto_bounds = {"n": [n_lo, n_lo + 0.4], "p": [p_lo, min(p_lo + 0.4, 0.999)]}
    Q_th_n = pybop.Parameter(initial_value=1.2 * q_nom_ah, bounds=[q_lo, q_hi])
    Q_th_p = pybop.Parameter(initial_value=1.2 * q_nom_ah, bounds=[q_lo, q_hi])
    R0 = pybop.Parameter(initial_value=0, bounds=[-1, 1])

    problems, info = [], []
    for branch in sorted(branches, key=lambda b: b["direction"] != "charge"):
        direction = branch["direction"]
        dataset = _branch_dataset(branch)
        throughput = dataset["Discharge capacity [A.h]"] - dataset["Discharge capacity [A.h]"][0]
        Q_meas = abs(throughput[-1])
        # Coulombic efficiency: scale so the relaxed end voltage is full dis/charge.
        v_inv, f_inv = _monotone(np.asarray(dataset["Voltage [V]"]),
                                 np.asarray(throughput / throughput[-1]),
                                 increasing=(direction == "charge"))
        inverse_pocv = pybop.Interpolant(v_inv, f_inv, name="Inverse pOCV")
        CE = float(np.minimum(inverse_pocv(branch["v_relax"]), 1.0))
        Q_cell = CE * Q_meas
        dataset["Naive SoC"] = (0.0 if direction == "charge" else 1.0) - throughput / Q_cell
        params = {
            "Negative electrode theoretical capacity [A.h]": Q_th_n,
            "Positive electrode theoretical capacity [A.h]": Q_th_p,
            "Cell capacity [A.h]": Q_cell,
            # Start inside the bounds, not on them: SciPy's differential evolution
            # rejects an x0 that rounds outside after rescaling to [0, 1].
            "Minimum negative stoichiometry " + direction: pybop.Parameter(
                initial_value=float(np.interp(0.25, [0, 1], sto_bounds["n"])),
                bounds=sto_bounds["n"]),
            "Minimum positive stoichiometry " + direction: pybop.Parameter(
                initial_value=float(np.interp(0.25, [0, 1], sto_bounds["p"])),
                bounds=sto_bounds["p"]),
            "Series resistance [Ohm]": R0,
        }
        cost = pybop.RootMeanSquaredError(dataset, target="Voltage [V]", weighting="domain")
        sim = ocv_balance_simulator(params, dataset, direction, ocps)
        problems.append(pybop.Problem(simulator=sim, cost=pybop.WeightedCost(cost, weights=[1e3])))
        info.append({"direction": direction, "CE": CE, "Q_cell_ah": Q_cell,
                     "_soc": np.asarray(dataset["Naive SoC"]),
                     "_v": np.asarray(dataset["Voltage [V]"])})

    t0 = time.time()
    result = pybop.SciPyDifferentialEvolution(
        pybop.MetaProblem(*problems),
        options=pybop.SciPyDifferentialEvolutionOptions(maxiter=maxiter, seed=seed)).run()
    runtime = time.time() - t0
    best = {k: float(np.asarray(v).ravel()[0]) for k, v in result.best_inputs.items()}

    positive_ocp, negative_ocp = ocps
    if ocv_limits is None:
        dis = [b for b in branches if b["direction"] == "discharge"]
        chg = [b for b in branches if b["direction"] == "charge"]
        ocv_limits = (float(dis[0]["v_relax"]) if dis else ocv_soc_limits(pv)[0],
                      float(np.max(chg[0]["V"])) if chg else ocv_soc_limits(pv)[1])
    v_lo, v_hi = ocv_limits
    per_branch = []
    for problem, inf in zip(problems, info):
        inputs = problem.get_model_inputs(best)
        Q, n0, n100, p100, p0, _ = problem.simulator.get_model_parameters(inputs)
        ocv = OpenCircuitVoltage(positive_ocp, p0, p100, negative_ocp, n0, n100)
        inverse = pybop.InverseOCV(ocv)
        inverse.parameters["Root"] = pybop.Parameter(initial_value=0.5)
        lower_soc, upper_soc = float(inverse(v_lo)), float(inverse(v_hi))
        idx = np.unique(np.linspace(0, inf["_v"].size - 1, min(400, inf["_v"].size)).astype(int))
        v_model = np.asarray(problem.simulator.solve(inputs=inputs)["Voltage [V]"].data)
        curve = {"naive_soc": inf["_soc"][idx].tolist(), "v_data": inf["_v"][idx].tolist(),
                 "v_model": v_model[idx].tolist()}
        per_branch.append({
            **{k: v for k, v in inf.items() if not k.startswith("_")}, "curve": curve,
            "x_0": n0 + (n100 - n0) * lower_soc, "x_100": n0 + (n100 - n0) * upper_soc,
            "y_100": p0 + (p100 - p0) * upper_soc, "y_0": p0 + (p100 - p0) * lower_soc,
            "min_sto_n": n0, "min_sto_p": p100,
            "Q_soc_ah": Q * (upper_soc - lower_soc),
            "rmse_mV": float(problem.evaluate(inputs).values.item()),
        })

    window = {k: float(np.mean([b[k] for b in per_branch])) for k in ("x_0", "x_100", "y_100", "y_0")}
    # The window is read at pv's voltage limits, which can lie outside the measured
    # range (oracle 3.0/4.3 V vs jones2022 formation ~3.09/4.2 V); a poor fit then
    # extrapolates past physical stoichiometry. Flag it and clip into (0, 1) so
    # PyBaMM never receives a concentration above its maximum.
    window_valid = all(0.0 <= v <= 1.0 for v in window.values())
    if not window_valid:
        log.warning("balance: stoichiometry window %s outside [0, 1]; clipped", window)
        window = {k: float(np.clip(v, 1e-3, 1 - 1e-3)) for k, v in window.items()}
    q_n = best["Negative electrode theoretical capacity [A.h]"]
    q_p = best["Positive electrode theoretical capacity [A.h]"]
    q_cell = float(np.mean([b["Q_soc_ah"] for b in per_branch]))
    log.info("balance: Q_th n/p %.3f/%.3f Ah, R0 %.4f Ohm, window %s, RMSE %s mV (%.0f s)",
             q_n, q_p, best["Series resistance [Ohm]"], window,
             [round(b["rmse_mV"], 1) for b in per_branch], runtime)
    return {
        "overrides": balance_overrides(pv, q_n, q_p, window, q_cell, (v_lo, v_hi)),
        "ocv_limits_v": [v_lo, v_hi],
        "theoretical_capacity_ah": {"negative": q_n, "positive": q_p},
        "series_resistance_ohm": best["Series resistance [Ohm]"],
        "window": window, "window_valid": window_valid, "branches": per_branch,
        "bounds": {"theoretical_capacity_ah": [q_lo, q_hi],
                   "min_stoichiometry": sto_bounds["p"], "min_stoichiometry_n": sto_bounds["n"],
                   "series_resistance_ohm": [-1.0, 1.0]},
        "runtime_s": round(runtime, 1),
    }


# ---------------------------------------------------------------------------
# Stage 2: kinetics / transport from impedance (PyBOP simple_eis)
# ---------------------------------------------------------------------------

def eis_model():
    """DFN with the differential surface form (double layer) + contact resistance --
    the oracle's EIS model form, on the full DFN rather than a reduced model."""
    import pybamm
    return pybamm.lithium_ion.DFN(
        options={"surface form": "differential", "contact resistance": "true"})


def spectrum_initial_state(spec: dict, pv) -> tuple[float, float]:
    """(x, y) of a spectrum's relaxed voltage on ``pv``; ValueError if not in (0, 1).

    Done here (PyBaMM electrode-SOH inversion) rather than inside PyBOP's
    simulator, so an inconsistent balanced cell is reported per spectrum instead
    of failing the model build.
    """
    x, y = initial_stoichiometries(pv, spec["v_rest"])
    if not (0.0 < x < 1.0 and 0.0 < y < 1.0):
        raise ValueError(f"initial stoichiometries x={x:.4f}, y={y:.4f} outside (0, 1) "
                         f"at {spec['v_rest']:.3f} V")
    return x, y


def _eis_problem(spec: dict, pv, fit_params: dict):
    """One PyBOP EIS problem: the spectrum simulated at its own relaxed voltage."""
    import pybop

    x, y = spectrum_initial_state(spec, pv)
    pv = pv.copy()
    pv.update({
        "Initial concentration in negative electrode [mol.m-3]":
            x * float(pv["Maximum concentration in negative electrode [mol.m-3]"]),
        "Initial concentration in positive electrode [mol.m-3]":
            y * float(pv["Maximum concentration in positive electrode [mol.m-3]"]),
    })
    add_scale_parameters(pv, EIS_SCALES)
    pv.update(fit_params, check_already_exists=False)
    sim = pybop.pybamm.EISSimulator(eis_model(), f_eval=spec["f"], parameter_values=pv)
    dataset = pybop.Dataset({"Frequency [Hz]": spec["f"], "Current [A]": np.zeros_like(spec["f"]),
                             "Impedance": spec["Z"]}, domain="Frequency [Hz]")
    return pybop.Problem(sim, pybop.RootMeanSquaredError(dataset, target="Impedance"))


def _eis_fit_params(scale_bounds, cr_bounds, contact_r0: float):
    import pybop

    fit = {scale_parameter_name(n): pybop.Parameter(
        distribution=pybop.LogUniform(*scale_bounds), initial_value=1.0,
        transformation=pybop.LogTransformation()) for n in EIS_SCALES}
    fit[CONTACT_R] = pybop.Parameter(
        distribution=pybop.LogUniform(*cr_bounds),
        initial_value=float(np.clip(contact_r0, *cr_bounds)),
        transformation=pybop.LogTransformation())
    return fit


def spectrum_model(spec: dict, pv, inputs: dict) -> np.ndarray | None:
    """Simulated impedance [complex Ohm] of one spectrum at ``inputs`` (scale-parameter
    names + contact resistance); ``None`` if the simulation fails."""
    try:
        problem = _eis_problem(spec, pv, _eis_fit_params((1e-3, 1e3), (1e-7, 10.0), 1e-3))
        return np.asarray(problem.simulator.solve(inputs=inputs)["Impedance"].data, dtype=complex)
    except Exception as exc:
        log.warning("%s: EIS simulation failed (%s)", spec.get("label"), exc)
        return None


def spectrum_rmse(spec: dict, pv, inputs: dict) -> float:
    """|Z| RMSE [Ohm] of one spectrum at ``inputs``; ``nan`` if the simulation fails."""
    z = spectrum_model(spec, pv, inputs)
    return float("nan") if z is None else float(np.sqrt(np.mean(np.abs(z - spec["Z"]) ** 2)))


def fit_eis(spectra: list[dict], pv, *, scale_bounds: tuple[float, float] = (0.01, 100.0),
            cr_bounds: tuple[float, float] = (1e-6, 0.1), contact_r0: float = 1e-3,
            max_iterations: int = 200, max_unchanged: int = 40,
            seed: int | None = 0) -> dict[str, Any]:
    """PyBOP ``simple_eis``-style fit of :data:`EIS_SCALES` + contact resistance."""
    import pybop

    fit = _eis_fit_params(scale_bounds, cr_bounds, contact_r0)
    problems, skipped = [], []
    for sp in spectra:
        try:
            problems.append(_eis_problem(sp, pv, fit))
        except Exception as exc:
            skipped.append({"label": sp.get("label"), "reason": str(exc)[:300]})
    if not problems:
        raise ValueError(f"no spectrum could be set up on the balanced cell: {skipped}")
    if seed is not None:
        np.random.seed(seed)
    options = pybop.PintsOptions(max_iterations=max_iterations,
                                 max_unchanged_iterations=max_unchanged, verbose=True)
    t0 = time.time()
    result = pybop.CMAES(pybop.MetaProblem(*problems), options=options).run()
    runtime = time.time() - t0
    best = {k: float(np.asarray(v).ravel()[0]) for k, v in result.best_inputs.items()}
    log.info("eis: summed RMSE %s over %d spectra in %.0f s: %s",
             result.best_cost, len(spectra), runtime, best)
    return {"scales": {n: best[scale_parameter_name(n)] for n in EIS_SCALES},
            "contact_resistance_ohm": best[CONTACT_R], "inputs": best,
            "cost": float(np.asarray(result.best_cost).ravel()[0]),
            "scale_bounds": list(scale_bounds), "cr_bounds": list(cr_bounds),
            "n_spectra_fitted": len(problems), "skipped_spectra": skipped,
            "runtime_s": round(runtime, 1)}


def _rmse_summary(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    ok = x[np.isfinite(x)]
    return {"n": int(x.size), "n_failed": int(x.size - ok.size),
            "median_ohm": float(np.median(ok)) if ok.size else None,
            "max_ohm": float(np.max(ok)) if ok.size else None}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def unit_eis_inputs(contact_r: float = 1e-3) -> dict:
    """EIS inputs with every multiplier 1 (the unfitted parameter set)."""
    return {**{scale_parameter_name(n): 1.0 for n in EIS_SCALES}, CONTACT_R: contact_r}


def fit_cell(cell_id: str, branches: list[dict], spectra: list[dict], pv0, *,
             compare_pvs: dict | None = None, de_maxiter: int = 100,
             max_iterations: int = 200, max_unchanged: int = 40,
             scale_bounds: tuple[float, float] = (0.01, 100.0),
             cr_bounds: tuple[float, float] = (1e-6, 0.1),
             q_th_ratio: tuple[float, float] = (1.0, 2.5), seed: int | None = 0) -> dict:
    """Stage A for ONE cell: balancing on its formation branches, then the EIS fit
    on its own spectra, on its own balanced cell.

    The per-cell results feed the parameter distributions; :func:`aggregate_cells`
    turns them into the oracle's single ``bol`` section. Measured and fitted
    curves/spectra are kept for plotting. ``compare_pvs`` (name -> parameter
    values) re-runs the balancing with other OCP templates for comparison.
    """
    bal = fit_balance(branches, pv0, maxiter=de_maxiter, seed=seed, q_th_ratio=q_th_ratio)
    comparison = {}
    for name, pv_c in (compare_pvs or {}).items():
        try:
            c = fit_balance(branches, pv_c, maxiter=de_maxiter, seed=seed, q_th_ratio=q_th_ratio)
            comparison[name] = {"rmse_mV": [b["rmse_mV"] for b in c["branches"]],
                                "window_valid": c["window_valid"]}
        except Exception as exc:
            comparison[name] = {"error": str(exc)[:200]}
    pv_bal = apply_parameter_overrides(pv0.copy(), bal["overrides"])
    try:
        eis = fit_eis(spectra, pv_bal, scale_bounds=scale_bounds, cr_bounds=cr_bounds,
                      max_iterations=max_iterations, max_unchanged=max_unchanged, seed=seed)
    except Exception as exc:
        log.warning("%s: EIS fit failed (%s)", cell_id, exc)
        return {"cell_id": cell_id, "balance": bal, "template_comparison": comparison,
                "eis": {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}, "spectra": []}
    unit = unit_eis_inputs()
    spec_out = []
    for sp in spectra:
        z0 = spectrum_model(sp, pv_bal, unit)
        z1 = spectrum_model(sp, pv_bal, eis["inputs"])
        spec_out.append({
            "label": sp.get("label"), "v_rest": sp["v_rest"], "f": np.asarray(sp["f"]).tolist(),
            "z_real": np.real(sp["Z"]).tolist(), "z_imag": np.imag(sp["Z"]).tolist(),
            "zfit_real": None if z1 is None else np.real(z1).tolist(),
            "zfit_imag": None if z1 is None else np.imag(z1).tolist(),
            "rmse_before_ohm": None if z0 is None else float(np.sqrt(np.mean(np.abs(z0 - sp["Z"]) ** 2))),
            "rmse_after_ohm": None if z1 is None else float(np.sqrt(np.mean(np.abs(z1 - sp["Z"]) ** 2))),
        })
    return {"cell_id": cell_id, "balance": bal, "template_comparison": comparison,
            "eis": {k: v for k, v in eis.items() if k != "inputs"}, "spectra": spec_out}


def aggregate_cells(cell_reports: list[dict]) -> dict:
    """The oracle's ``bol`` section from per-cell :func:`fit_cell` results.

    Balancing overrides: per-key median across cells. EIS multipliers and contact
    resistance: geometric median (median in log space; they are log-searched).
    Returns a report shaped like :func:`fit_bol`'s, so the physicality checks
    apply unchanged, plus per-parameter distributions across cells.
    """
    failed = [{"cell_id": c["cell_id"], "reason": c["eis"]["error"]}
              for c in cell_reports if "error" in c.get("eis", {})]
    cell_reports = [c for c in cell_reports if "scales" in c.get("eis", {})]
    if not cell_reports:
        raise ValueError(f"no cell has a complete stage-A fit; failures: {failed}")
    keys = cell_reports[0]["balance"]["overrides"].keys()
    overrides = {k: float(np.median([c["balance"]["overrides"][k] for c in cell_reports]))
                 for k in keys}
    scale_names = cell_reports[0]["eis"]["scales"].keys()
    scales = {n: float(np.exp(np.median([np.log(c["eis"]["scales"][n]) for c in cell_reports])))
              for n in scale_names}
    contact_r = float(np.exp(np.median([np.log(c["eis"]["contact_resistance_ohm"])
                                        for c in cell_reports])))

    def _dist(values):
        x = np.asarray(values, float)
        return {"values": x.tolist(), "median": float(np.median(x)),
                "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}

    distributions = {f"override: {k}": _dist([c["balance"]["overrides"][k] for c in cell_reports])
                     for k in keys}
    distributions.update({f"scale: {n}": _dist([c["eis"]["scales"][n] for c in cell_reports])
                          for n in scale_names})
    distributions["contact_resistance_ohm"] = _dist(
        [c["eis"]["contact_resistance_ohm"] for c in cell_reports])
    for key in ("x_0", "x_100", "y_100", "y_0"):
        distributions[f"window: {key}"] = _dist([c["balance"]["window"][key] for c in cell_reports])
    for key in ("negative", "positive"):
        distributions[f"theoretical capacity: {key}"] = _dist(
            [c["balance"]["theoretical_capacity_ah"][key] for c in cell_reports])
    first_eis = cell_reports[0]["eis"]
    return {
        "parameter_overrides": overrides,
        "parameter_scales": scales,
        "contact_resistance_ohm": contact_r,
        "fit": {
            "method": "per-cell PyBOP 3_cell_balancing + simple_eis (DFN); oracle values = "
                      "median (balancing) / geometric median (EIS) across cells",
            "cells": [c["cell_id"] for c in cell_reports],
            "failed_cells": failed,
            "balance": {c["cell_id"]: {k: v for k, v in c["balance"].items()
                                       if k not in ("overrides",)} for c in cell_reports},
            "eis": {"scale_bounds": first_eis.get("scale_bounds"),
                    "cr_bounds": first_eis.get("cr_bounds")},
            "distributions": distributions,
        },
    }


def compare_balancing(balance_cells: dict[str, list[dict]], chemistries: list[str], *,
                      oracle_kwargs: dict | None = None, de_maxiter: int = 100,
                      seed: int | None = 0) -> dict[str, Any]:
    """Run the balancing step with each chemistry's electrode OCPs and summarise.

    For choosing the OCP template with data: per chemistry, the per-branch RMSE
    [mV] across cells and whether every cell's window stayed inside [0, 1].
    """
    out = {}
    for chem in chemistries:
        pv = oracle_parameter_values(**{**(oracle_kwargs or {}), "chemistry": chem})
        fits = {cid: fit_balance(br, pv, maxiter=de_maxiter, seed=seed)
                for cid, br in balance_cells.items()}
        rmse = [b["rmse_mV"] for f in fits.values() for b in f["branches"]]
        out[chem] = {"median_rmse_mV": float(np.median(rmse)), "max_rmse_mV": float(np.max(rmse)),
                     "all_windows_valid": all(f["window_valid"] for f in fits.values()),
                     "per_cell": {cid: {"rmse_mV": [round(b["rmse_mV"], 2) for b in f["branches"]],
                                        "window": f["window"], "window_valid": f["window_valid"],
                                        "theoretical_capacity_ah": f["theoretical_capacity_ah"]}
                                  for cid, f in fits.items()}}
        log.info("balancing with %s: median %.1f mV, max %.1f mV, windows valid: %s", chem,
                 out[chem]["median_rmse_mV"], out[chem]["max_rmse_mV"], out[chem]["all_windows_valid"])
    return out


def fit_bol(balance_cells: dict[str, list[dict]], spectra: list[dict],
            holdout_spectra: list[dict] | None = None, *, oracle_kwargs: dict | None = None,
            scale_bounds: tuple[float, float] = (0.01, 100.0), de_maxiter: int = 100,
            max_iterations: int = 200, max_unchanged: int = 40,
            seed: int | None = 0) -> dict[str, Any]:
    """Balancing per cell (median across cells), then the EIS fit on the balanced cell.

    ``balance_cells``: ``{cell_id: [charge branch, discharge branch]}``.
    ``spectra`` / ``holdout_spectra``: impedance spectra (see module docstring).
    Returns a JSON-serialisable report whose ``parameter_overrides`` /
    ``parameter_scales`` / ``contact_resistance_ohm`` are the oracle's ``bol``
    config section.
    """
    import pybop

    pv0 = oracle_parameter_values(**(oracle_kwargs or {}))
    balances = {cid: fit_balance(br, pv0, maxiter=de_maxiter, seed=seed)
                for cid, br in balance_cells.items()}
    if not balances:
        raise ValueError("no balancing data")
    overrides = {k: float(np.median([b["overrides"][k] for b in balances.values()]))
                 for k in BALANCE_PARAMS}
    pv_bal = apply_parameter_overrides(pv0.copy(), overrides)

    r0 = 1e-3
    unit = {**{scale_parameter_name(n): 1.0 for n in EIS_SCALES}, CONTACT_R: r0}
    holdout_spectra = holdout_spectra or []
    before = {"fit": [spectrum_rmse(s, pv_bal, unit) for s in spectra],
              "holdout": [spectrum_rmse(s, pv_bal, unit) for s in holdout_spectra]}
    eis = fit_eis(spectra, pv_bal, scale_bounds=scale_bounds, contact_r0=r0,
                  max_iterations=max_iterations, max_unchanged=max_unchanged, seed=seed)
    after = {"fit": [spectrum_rmse(s, pv_bal, eis["inputs"]) for s in spectra],
             "holdout": [spectrum_rmse(s, pv_bal, eis["inputs"]) for s in holdout_spectra]}

    return {
        "parameter_overrides": overrides,
        "parameter_scales": eis["scales"],
        "contact_resistance_ohm": eis["contact_resistance_ohm"],
        "fit": {
            "method": "PyBOP 3_cell_balancing (OCVBalance + SciPyDifferentialEvolution) on "
                      "formation branches, then PyBOP simple_eis (EISSimulator on the DFN + "
                      "CMAES) on measured spectra",
            "pybop_version": pybop.__version__,
            "oracle_kwargs": {k: v for k, v in (oracle_kwargs or {}).items()
                              if isinstance(v, (int, float, str, bool))},
            "balance": {cid: {k: v for k, v in b.items() if k != "overrides"}
                        for cid, b in balances.items()},
            "balance_overrides_per_cell": {cid: b["overrides"] for cid, b in balances.items()},
            "eis": {k: v for k, v in eis.items() if k not in ("scales", "inputs")},
            "n_spectra": len(spectra), "n_holdout_spectra": len(holdout_spectra),
            "eis_rmse_before": {k: _rmse_summary(v) for k, v in before.items()},
            "eis_rmse_after": {k: _rmse_summary(v) for k, v in after.items()},
        },
    }
