"""Impedance spectra from the DFN: equilibrium and non-stationary (measured as performed).

Equilibrium EIS (PyBaMM ``EISSimulation``) linearises the model at one fully relaxed
state and assumes the cell does not change during the sweep. The jones2022 spectra
are measured after a 20 min rest at open circuit following a 1-4C step, with a
~300 s sweep (10 kHz -> 0.02 Hz): the cell is still relaxing. The non-stationary
spectrum is computed the way the measurement happened:

1. Integrate the DFN with the EIS voltage/current algebraic variables
   (:func:`eis_model`) through the current history, the rest and the sweep itself.
2. Frequency ``f_k`` is measured in a window ``[lo_k, hi_k]``. Linearise the model at
   the *state at the window centre*: ``Z_k = -x[V] / x[I]`` with
   ``(j w_k M - J(y(t_k))) x = b`` (:func:`linearised_impedance`).
3. Add the lock-in drift leakage (:func:`drift_leakage`). Over whole periods of a
   sinusoidal current ``A sin(wt)`` a linear voltage drift ``dV/dt`` adds
   ``-2 (dV/dt) / (w A)`` to the measured impedance (the lock-in of ``t`` over whole
   periods is ``j T / w``; the current's is ``A / 2j``). Hallemans et al. 2023
   (arXiv:2304.08126) give the general treatment; this is its leading term.

The oracle (:class:`battery_oracle.oracle.PyBaMMOracle`) and the stage-A history
fit (:mod:`battery_oracle.bol_chain`) both use these functions.
"""
from __future__ import annotations

import numpy as np


def eis_model(model_cls, options: dict, initial_state=None):
    """The DFN (or other model) plus the EIS voltage/current algebraic variables --
    exactly the model PyBOP's ``EISSimulator`` builds. ``initial_state``: a solution
    whose last state the model starts from (mapped by variable name in physical
    units, before the EIS variables are added)."""
    from pybop.pybamm.eis_simulator import EISSimulator

    model = model_cls(options=dict(options))
    if initial_state is not None:
        model.set_initial_conditions_from(initial_state, inplace=True)
    return EISSimulator.set_up_for_eis(None, model)


def impedance_scale(model, pv) -> float:
    """[Ohm] per unit of the EIS model's ``-x[V] / x[I]`` (its voltage/current scales)."""
    v_scale = getattr(model.variables["Voltage [V]"], "scale", 1)
    i_scale = getattr(model.variables["Current [A]"], "scale", 1)
    return float(np.asarray(pv.evaluate(v_scale / i_scale)).ravel()[0])


def linearised_impedance(sim, t_lin, y_states, frequencies, z_scale, *, per_frequency: bool = True,
                         inputs: dict | None = None):
    """Impedance [Ohm] of a built EIS model (``eis_model``) linearised at given states.

    ``y_states[:, k]`` is the state at ``t_lin[k]`` where frequency ``k`` is
    measured: ``Z_k = -x[V] / x[I] * z_scale`` with ``(j w_k M - J(y_k)) x = b``.
    ``per_frequency=False`` linearises every frequency at the first state.
    ``inputs``: values of the model's input parameters (e.g. the chain fit's cycler
    cut-off, 0 at rest), in the model's input order.
    """
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spsolve

    f = np.asarray(frequencies, float)
    built = sim.built_model
    if not hasattr(built, "jac_rhs_algebraic_eval") or built.jac_rhs_algebraic_eval is None:
        sim.solver.set_up(built, inputs=dict(inputs or {}))
    p = np.asarray(list(inputs.values()), float) if inputs else []
    M = csc_matrix(built.mass_matrix.entries)
    b = np.zeros(M.shape[0])
    b[-1] = -1.0
    z = np.empty(f.size, dtype=complex)
    cache: dict[int, object] = {}
    for k, fk in enumerate(f):
        key = k if per_frequency else 0
        if key not in cache:
            cache[key] = csc_matrix(built.jac_rhs_algebraic_eval(float(t_lin[k]), y_states[:, k], p).sparse())
        x = spsolve(1.0j * 2 * np.pi * fk * M - cache[key], b)
        z[k] = -x[-2] / x[-1]
    return z * z_scale


def drift_leakage(t, v, lo, hi, frequencies, amplitude_a: float) -> np.ndarray:
    """Lock-in leakage [Ohm, real] of the voltage drift in each measurement window.

    ``t``, ``v``: simulated time [s] and voltage [V]; window ``k`` is
    ``[lo[k], hi[k]]`` on the same clock. The drift is the mean slope over the
    window, ``(v(hi) - v(lo)) / (hi - lo)``; the leakage is ``-2 (dV/dt) / (w A)``.
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    dvdt = (np.interp(hi, t, v) - np.interp(lo, t, v)) / np.maximum(hi - lo, 1e-9)
    return -2.0 * dvdt / (2 * np.pi * np.asarray(frequencies, float) * amplitude_a)


def sweep_windows(frequencies, periods: float, overhead_s: float):
    """(start, end) [s after the sweep starts] of each frequency's window, in the given
    order, for a sweep run from high to low frequency: each window is
    ``periods / f + overhead_s`` long (jones2022: 57 points, 20 kHz -> 0.02 Hz in
    ~297 s, reproduced with 1.15 s/point of instrument overhead)."""
    f = np.asarray(frequencies, float)
    order = np.argsort(-f)
    length = float(periods) / f[order] + float(overhead_s)
    end = np.cumsum(length)
    lo, hi = np.empty_like(f), np.empty_like(f)
    lo[order] = end - length
    hi[order] = end
    return lo, hi


def windows_from_stamps(stamps):
    """(start, end) [s] of each frequency's measurement window from the EIS file's
    ``time/s`` stamps, on the same clock as the cycling data.

    A stamp marks the end of its frequency's window; the window starts at the
    previous stamp (so it holds the real number of periods plus the instrument's
    settling overhead, instead of an assumed one period). The first window is given
    the length of the second (both are high frequency and sub-second).
    """
    end = np.asarray(stamps, float)
    start = np.empty_like(end)
    start[1:] = end[:-1]
    start[0] = end[0] - (end[1] - end[0] if end.size > 1 else 1e-3)
    return start, end


def equilibrium_impedance(model_cls, pv, frequencies, soc: float) -> np.ndarray:
    """Impedance [Ohm] of a relaxed cell at ``soc`` (PyBaMM ``EISSimulation``, double
    layer on, isothermal)."""
    import pybamm

    sim = pybamm.EISSimulation(model_cls(options={"surface form": "differential"}), parameter_values=pv)
    return np.array(sim.solve(frequencies, initial_soc=soc)["Impedance [Ohm]"])
