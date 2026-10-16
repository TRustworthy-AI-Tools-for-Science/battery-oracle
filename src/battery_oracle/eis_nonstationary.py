"""Impedance of a cell that is still relaxing (non-stationary EIS), from the DFN.

Standard EIS simulation (pybamm-eis, PyBOP ``EISSimulator``) linearises the model at
one fully relaxed state and assumes the cell does not change during the sweep.
The jones2022 spectra are measured after a 20 min rest at open circuit following
a 1-4C step, with a ~300 s sweep (10 kHz -> 0.02 Hz): the cell is still relaxing.

This module computes the impedance the way the measurement happened:

1. Integrate the DFN (with the EIS algebraic variables) through the cycle's real
   current history, rest and the sweep itself.
2. Frequency ``f_k`` is measured in a window centred at ``t_k``. Linearise the
   model at the *state at t_k*: ``Z_k = -x[V] / x[I]`` with
   ``(j w_k M - J(y(t_k))) x = b``.
3. Add the lock-in drift leakage. Over ``n`` whole periods of a sinusoidal current
   ``A sin(wt)`` a linear voltage drift ``dV/dt`` adds ``-2 (dV/dt) / (w A)`` to the
   measured impedance (derivation: the lock-in of ``t`` over whole periods is
   ``j T / w``; the current's is ``A / 2j``). Hallemans et al. 2023 (arXiv:2304.08126)
   give the general treatment; this is its leading term.

With no history it reduces to the stock linearisation at the initial state (tested
against PyBOP's ``EISSimulator``). Implemented standalone, using only PyBaMM and
PyBOP's ``EISSimulator.set_up_for_eis`` (to build the same EIS model), so both the
PyBOP fits (via :mod:`battery_oracle.patches`) and the oracle can use it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class History:
    """Current history leading to the EIS measurement (simulated cell frame).

    ``t`` [s] from 0, ``I`` [A] with PyBaMM's sign (positive = discharge), ending at
    the start of the EIS sweep (end of the rest). Either ``initial_voltage`` [V]
    (relaxed voltage at ``t = 0``) or the parameter set's initial concentrations
    define the state at ``t = 0``.
    """
    t: np.ndarray
    I: np.ndarray
    initial_voltage: float | None = None


@dataclass
class SweepInfo:
    """Diagnostics of one non-stationary impedance computation."""
    window_centers_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    voltage_v: np.ndarray = field(default_factory=lambda: np.empty(0))
    drift_v_per_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    leakage_ohm: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=complex))
    z_stationary: np.ndarray | None = None


def eis_model(model_cls, options: dict):
    """The DFN (or other model) plus the EIS voltage/current algebraic variables --
    exactly the model PyBOP's ``EISSimulator`` builds."""
    from pybop.pybamm.eis_simulator import EISSimulator

    return EISSimulator.set_up_for_eis(None, model_cls(options=dict(options)))


#: Delay between the start of the EIS technique and its first measured point
#: [s] (jones2022 files: the first frequency is stamped ~2.3 s).
FIRST_POINT_DELAY_S = 2.0
#: A sweep longer than this is not a sweep: the stamps are absolute (see below).
MAX_SWEEP_S = 3600.0


def relative_sweep_times(stamps, first_delay_s: float = FIRST_POINT_DELAY_S):
    """Seconds after the sweep started, from the EIS file's ``time/s`` column.

    That column is time since the start of the test step sequence, not since the
    sweep began (e.g. a first stamp of 5140 s); using it as is places every
    frequency thousands of seconds into extra rest. Rebase to the first stamp and
    add the first point's start-up delay. Pass the stamps of ALL frequencies, then
    select the subset used.
    """
    s = np.asarray(stamps, float)
    return s - s[0] + first_delay_s


def window_times(frequencies, n_periods: float = 1.0, sweep_times=None):
    """Centre of each frequency's measurement window, seconds after the sweep starts.

    ``sweep_times`` [s]: the measured time stamp of each frequency (end of its
    window; jones2022 EIS files' ``time/s``). Without it, windows run back to
    back in the given order, each ``n_periods / f`` long.
    """
    f = np.asarray(frequencies, float)
    length = n_periods / f
    if sweep_times is None:
        end = np.cumsum(length)
    else:
        end = np.asarray(sweep_times, float)
        if end.max() > MAX_SWEEP_S:
            raise ValueError(f"sweep_times reach {end.max():.0f} s: they look absolute; "
                             "use relative_sweep_times() on the EIS file's time column")
    return end - length / 2.0, length


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


def linearised_impedance(sim, t_lin, y_states, frequencies, z_scale, *, per_frequency: bool = True,
                         inputs: dict | None = None):
    """Impedance [Ohm] of a built EIS model (``eis_model``) linearised at given states.

    ``y_states[:, k]`` is the state at ``t_lin[k]`` where frequency ``k`` is
    measured: ``Z_k = -x[V] / x[I] * z_scale`` with ``(j w_k M - J(y_k)) x = b``.
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


def nonstationary_impedance(model_cls, options: dict, pv, frequencies, *,
                            history: History | None = None, sweep_times=None,
                            n_periods: float = 1.0, amplitude_a: float = 5e-3,
                            drift_leakage: bool = True, per_frequency: bool = True,
                            solver=None, return_info: bool = False):
    """Complex impedance [Ohm] at ``frequencies`` [Hz] (see module docstring).

    ``pv`` must hold plain numbers (inputs already applied). ``history=None``
    linearises at the initial state, as stock EIS does. ``per_frequency=False``
    linearises everything at the end of the history (state at the sweep start).
    """
    import pybamm

    f = np.asarray(frequencies, float)
    model = eis_model(model_cls, options)
    pv = pv.copy()
    v_scale = getattr(model.variables["Voltage [V]"], "scale", 1)
    i_scale = getattr(model.variables["Current [A]"], "scale", 1)
    z_scale = pv.evaluate(v_scale / i_scale)

    centers, length = window_times(f, n_periods, sweep_times)
    info = SweepInfo()
    if history is None:
        pv["Current function [A]"] = 0
        sim = pybamm.Simulation(model, parameter_values=pv,
                                solver=solver or pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6))
        sim.build()
        y_t = None
        t_lin = np.zeros_like(f)
    else:
        t_h, i_h = np.asarray(history.t, float), np.asarray(history.I, float)
        t_end_h = float(t_h[-1])
        t_sweep_end = t_end_h + float(np.max(centers + length / 2)) + 1.0
        # Rest during the sweep: the 5 mA perturbation is the *probe*, not the baseline.
        t_all = np.concatenate([t_h, [t_end_h + 1e-3, t_sweep_end]])
        i_all = np.concatenate([i_h, [0.0, 0.0]])
        pv["Current function [A]"] = pybamm.Interpolant(t_all, i_all, pybamm.t)
        sim = pybamm.Simulation(model, parameter_values=pv,
                                solver=solver or pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6))
        t_lin = t_end_h + (centers if per_frequency else np.zeros_like(f))
        t_eval = np.unique(np.concatenate([t_h, t_lin, [t_end_h]]))
        init = ({"initial_soc": f"{float(history.initial_voltage)} V"}
                if history.initial_voltage is not None else {})
        sol = sim.solve(t_eval=t_eval, **init)
        if sol.termination not in ("final time", None) and sol.t[-1] < t_eval[-1] - 1e-6:
            raise RuntimeError(f"history simulation stopped early: {sol.termination} "
                               f"at t={sol.t[-1]:.1f} s of {t_eval[-1]:.1f} s")
        idx = np.searchsorted(sol.t, t_lin)
        idx = np.clip(idx, 0, sol.t.size - 1)
        y_t = np.asarray(sol.y)[:, idx]
        v_t = np.asarray(sol["Voltage [V]"].entries)[idx]
        # voltage drift at each window centre: central difference over +-half a window
        dv = np.zeros_like(f)
        for k, (tc, half) in enumerate(zip(t_lin, length / 2)):
            lo, hi = np.searchsorted(sol.t, [tc - half, tc + half])
            lo, hi = max(lo, 0), min(hi, sol.t.size - 1)
            if hi > lo:
                dv[k] = (np.asarray(sol["Voltage [V]"].entries)[hi]
                         - np.asarray(sol["Voltage [V]"].entries)[lo]) / (sol.t[hi] - sol.t[lo])
        info.window_centers_s, info.voltage_v, info.drift_v_per_s = t_lin, v_t, dv

    built = sim.built_model
    if y_t is None:
        if not hasattr(built, "jac_rhs_algebraic_eval") or built.jac_rhs_algebraic_eval is None:
            sim.solver.set_up(built, inputs={})
        y0 = np.asarray(built.concatenated_initial_conditions.evaluate(0, inputs={}), float).ravel()
        y_t = np.tile(y0[:, None], (1, f.size))
    z = linearised_impedance(sim, t_lin, y_t, f, z_scale, per_frequency=per_frequency)

    if drift_leakage and history is not None:
        w = 2 * np.pi * f
        leak = -2.0 * info.drift_v_per_s / (w * amplitude_a)
        info.leakage_ohm = leak.astype(complex)
        z = z + leak
    if return_info:
        return z, info
    return z
