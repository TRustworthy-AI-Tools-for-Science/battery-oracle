"""Monkey patches that put non-equilibrium physics into the PyBOP/PyBaMM workflow.

Why patches. PyBOP's fitting workflow (simulators, costs, optimisers, samplers) is
verified and reused as is; but two of its assumptions do not hold for cells measured
out of equilibrium (jones2022: ~1C formation, 290 s / 20 min rests, ~300 s EIS
sweeps on a relaxing cell):

``eis_nonstationary``
    ``pybop.pybamm.EISSimulator._solve`` linearises the model once at the fully
    relaxed initial state. Patched: when a measurement history is set on the
    simulator (``simulator.set_measurement(...)``), the impedance comes from
    :func:`battery_oracle.eis_nonstationary.nonstationary_impedance` -- the DFN
    integrated through the real current history, linearised per frequency at the
    state where that frequency was measured, plus lock-in drift leakage. Without a
    measurement set the stock path runs unchanged.

``graded_failure``
    ``pybop.ErrorMeasure.evaluate_batch`` gives any simulation that stops early
    (voltage/stoichiometry event) one flat failure cost, so a population of
    infeasible candidates is a flat landscape the optimiser cannot descend. Patched:
    a truncated trajectory is scored on the part simulated plus a per-point
    penalty on the missing part (graded in the fraction missing).

``parallel_batches``
    ``pybop.MetaProblem.evaluate_batch`` evaluates a population one candidate
    after another. A joint DFN fit costs ~20 s per candidate, so a CMA-ES iteration
    (10 candidates) takes minutes. Patched: after ``problem.set_workers(n)`` the
    candidates are split over ``n`` forked worker processes (the built PyBaMM models
    are inherited, not pickled) and the costs recombined in order. With one worker,
    or with sensitivities requested, the stock path runs. Fork-safety: forking after
    PyBaMM's multi-threaded (OpenMP) IDAKLU has run deadlocks the workers, so solvers
    used with this patch must be single-threaded (:func:`single_thread_solver`);
    each candidate runs in its own process under a hard time limit
    (``ORACLE_FIT_CANDIDATE_TIMEOUT`` s, default 600): a candidate that crawls is
    killed and scored ``inf`` like any failed simulation, so one pathological
    parameter set cannot freeze a whole population.

``harmonic_flux_coefficients``
    PyBaMM averages a coefficient onto cell edges harmonically only when it
    multiplies a bare gradient; the DFN electrolyte current multiplies its
    effective conductivity by a *combination* of gradients, so at the
    electrode|separator interfaces (where eps^b jumps several-fold) it is averaged
    arithmetically. Patched to the harmonic (series) mean for any flux expression.

Safety. Each patch is installed only if the installed pybamm/PyBOP match the
tested versions and the patched targets have the expected signatures; otherwise
:func:`apply` raises (``strict``) or skips it with a warning. Every patch can be
reverted (:func:`revert`) and is tested to leave the stock behaviour intact when
its non-equilibrium feature is not used.

Usage::

    from battery_oracle import patches
    patches.apply()                      # all patches
    sim = pybop.pybamm.EISSimulator(model, f_eval=f, parameter_values=pv)
    sim.set_measurement(history, sweep_times=t_sweep, amplitude_a=5e-3)
"""
from __future__ import annotations

import inspect
import logging
import warnings
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

#: Versions the patches were written and tested against (PyBaMM 26.10 is the only supported
#: version: pyproject pins >= 26.10). A string is accepted (verification scripts set the
#: installed version to force-install every patch).
TESTED_PYBAMM: tuple[str, ...] | str = ("26.10.1.0",)
TESTED_PYBOP_PREFIX = "26.3"   # develop branch at commit 25d0b5b reports 26.3

_ORIGINALS: dict[str, Callable] = {}
_APPLIED: set[str] = set()


class PatchError(RuntimeError):
    """A patch target is missing or has an unexpected signature/version."""


def _check_versions() -> None:
    import pybamm
    import pybop

    tested = (TESTED_PYBAMM,) if isinstance(TESTED_PYBAMM, str) else tuple(TESTED_PYBAMM)
    if pybamm.__version__ not in tested:
        raise PatchError(f"pybamm {pybamm.__version__} not in tested {tested}")
    if not str(pybop.__version__).startswith(TESTED_PYBOP_PREFIX):
        raise PatchError(f"pybop {pybop.__version__} not {TESTED_PYBOP_PREFIX}.x")


def _require_signature(fn, names: list[str], where: str) -> None:
    params = list(inspect.signature(fn).parameters)
    if params[: len(names)] != names:
        raise PatchError(f"{where} signature {params} != expected prefix {names}")


# ---------------------------------------------------------------------------
# Patch 1: non-stationary EIS
# ---------------------------------------------------------------------------

def _install_eis_nonstationary() -> None:
    from pybop.pybamm.eis_simulator import EISSimulator

    _require_signature(EISSimulator._solve, ["self", "inputs"], "EISSimulator._solve")
    for attr in ("_f_eval", "_simulator", "parameter_values"):
        # _f_eval and _simulator are instance attributes; parameter_values a property
        if attr == "parameter_values" and not hasattr(EISSimulator, attr):
            raise PatchError("EISSimulator.parameter_values missing")
    original = EISSimulator._solve
    _ORIGINALS["EISSimulator._solve"] = original

    def set_measurement(self, history, *, sweep_times=None, n_periods: float = 1.0,
                        amplitude_a: float = 5e-3, drift_leakage: bool = True,
                        per_frequency: bool = True) -> None:
        """Make this simulator compute non-stationary impedance (see module doc)."""
        self._measurement = dict(history=history, sweep_times=sweep_times, n_periods=n_periods,
                                 amplitude_a=amplitude_a, drift_leakage=drift_leakage,
                                 per_frequency=per_frequency)

    def clear_measurement(self) -> None:
        self._measurement = None

    def _solve(self, inputs):
        meas = getattr(self, "_measurement", None)
        if meas is None:
            return original(self, inputs)
        from pybop.simulators.base_simulator import Solution

        from battery_oracle.eis_nonstationary import nonstationary_impedance

        pv = self.parameter_values.copy()
        pv.update(inputs or {})
        model = self._simulator._model
        try:
            z = nonstationary_impedance(type(model), dict(model.options), pv,
                                        np.asarray(self._f_eval, float), **meas)
        except Exception as exc:
            # PyBOP turns RuntimeError/ValueError/ZeroDivisionError into a failed
            # candidate; PyBaMM's SolverError/ModelError (an extreme parameter set
            # the DFN cannot integrate) would otherwise abort the whole fit.
            raise RuntimeError(f"non-stationary EIS failed: {type(exc).__name__}: "
                               f"{str(exc)[:200]}") from exc
        solution = Solution()
        solution.set_solution_variable("Impedance", data=np.asarray(z, dtype=complex))
        return solution

    EISSimulator._solve = _solve
    EISSimulator.set_measurement = set_measurement
    EISSimulator.clear_measurement = clear_measurement


def _remove_eis_nonstationary() -> None:
    from pybop.pybamm.eis_simulator import EISSimulator

    EISSimulator._solve = _ORIGINALS.pop("EISSimulator._solve")
    for name in ("set_measurement", "clear_measurement"):
        if hasattr(EISSimulator, name):
            delattr(EISSimulator, name)


# ---------------------------------------------------------------------------
# Patch 2: graded cost for truncated trajectories
# ---------------------------------------------------------------------------

#: Per-point penalty (in the target's units) added for each unsimulated point.
DEFAULT_TRUNCATION_PENALTY = 0.5


def _install_graded_failure() -> None:
    from pybop.costs.error_measures import ErrorMeasure

    _require_signature(ErrorMeasure.evaluate_batch, ["self", "solution", "inputs"],
                       "ErrorMeasure.evaluate_batch")
    original = ErrorMeasure.evaluate_batch
    _ORIGINALS["ErrorMeasure.evaluate_batch"] = original

    def evaluate_batch(self, solution, inputs, calculate_sensitivities: bool = False):
        penalty = getattr(self, "truncation_penalty", DEFAULT_TRUNCATION_PENALTY)
        if calculate_sensitivities or penalty is None:
            return original(self, solution, inputs, calculate_sensitivities)
        from pybop.simulators.base_simulator import Solution
        from pybop.simulators.failed_solution import FailedSolution

        fixed = []
        for sol in solution:
            if isinstance(sol, FailedSolution):
                fixed.append(sol)
                continue
            try:
                short = any(len(sol[k].data) < len(self._target_data[k]) for k in self.target)
            except Exception:
                short = False
            if not short:
                fixed.append(sol)
                continue
            padded = Solution()
            for k in self.target:
                d = np.asarray(sol[k].data)
                tgt = np.asarray(self._target_data[k])
                n = d.size
                if n == 0:                     # nothing simulated: leave to the flat failure cost
                    padded = None
                    break
                padded.set_solution_variable(k, data=np.concatenate([d, tgt[n:] + penalty]))
            fixed.append(sol if padded is None else padded)
        return original(self, fixed, inputs, calculate_sensitivities)

    ErrorMeasure.evaluate_batch = evaluate_batch


def _remove_graded_failure() -> None:
    from pybop.costs.error_measures import ErrorMeasure

    ErrorMeasure.evaluate_batch = _ORIGINALS.pop("ErrorMeasure.evaluate_batch")


# ---------------------------------------------------------------------------
# Patch 3: process-parallel population evaluation
# ---------------------------------------------------------------------------

_ACTIVE_META = None


def single_thread_solver(rtol: float = 1e-6, atol: float = 1e-6, output_variables: list[str] | None = None):
    """PyBOP-style IDAKLU solver (failures ignored, algebraic errors suppressed) with
    ``num_threads = 1``: process parallelism replaces OpenMP threads, which also
    avoids PyBOP's default of one thread per core of the whole node. With
    ``output_variables`` the solver stores only those variables (not the state
    history; the terminal state is kept for continuing)."""
    import pybamm

    return pybamm.IDAKLUSolver(rtol=rtol, atol=atol, on_failure="ignore", output_variables=output_variables,
                               options={"suppress_algebraic_error": True, "num_threads": 1})


def _candidate_worker(conn, x):
    """Forked worker: stock evaluation of ONE candidate on the inherited problem."""
    try:
        original = _ORIGINALS["MetaProblem.evaluate_batch"]
        conn.send(("ok", float(np.asarray(original(_ACTIVE_META, [x], False).values).ravel()[0])))
    except BaseException as exc:                      # noqa: BLE001 - report, never hang the parent
        conn.send(("err", f"{type(exc).__name__}: {str(exc)[:200]}"))
    finally:
        conn.close()


def run_candidates(meta, inputs: list[dict], n_workers: int, timeout_s: float) -> np.ndarray:
    """Costs of ``inputs``, each in its own forked process, at most ``n_workers`` at once.

    A candidate still running after ``timeout_s`` is killed and scored ``inf`` (like
    any other failed simulation): an extreme parameter set can make the DFN
    integration crawl, and a pool would wait for it forever. An exception inside a
    candidate is also scored ``inf`` (logged); if every candidate fails, the first
    error is raised so a real bug is not hidden.

    A worker KILLED from outside (SIGKILL without a result: the kernel's out-of-memory
    killer) says nothing about the candidate, so it is not scored ``inf``: the
    candidate is run again with one worker fewer (kept for later batches too). If it
    is killed again with a single worker, the job does not have the memory for one
    simulation and this raises. (Before 2026-10-18 such candidates were scored
    ``inf``: the v20 pilots lost 251 / 706 of their candidates to OOM kills.)
    """
    import multiprocessing as mp
    import time
    from multiprocessing.connection import wait

    global _ACTIVE_META
    ctx = mp.get_context("fork")
    _ACTIVE_META = meta
    costs = np.full(len(inputs), np.inf)
    errors: list[str] = []
    pending = list(range(len(inputs)))
    running: dict = {}                       # conn -> (index, process, start time)
    limit = max(1, int(n_workers))
    try:
        while pending or running:
            while pending and len(running) < limit:
                k = pending.pop(0)
                recv, send = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=_candidate_worker, args=(send, inputs[k]), daemon=True)
                proc.start()
                send.close()
                running[recv] = (k, proc, time.time())
            for conn in wait(list(running), timeout=1.0):
                k, proc, _ = running.pop(conn)
                try:
                    kind, val = conn.recv()
                except EOFError:             # died without reporting
                    kind, val = "err", "worker died without a result"
                conn.close()
                proc.join(timeout=5)
                if kind == "err" and val == "worker died without a result" and proc.exitcode == -9:
                    # killed from outside (out of memory): not the candidate's fault
                    POOL_STATS["killed_workers"] += 1
                    if limit == 1 and len(running) == 0:
                        raise RuntimeError(
                            "a candidate worker was killed (SIGKILL, out of memory?) while running "
                            "alone: the job has too little memory for one simulation")
                    limit = max(1, min(limit, len(running) + 1) - 1)
                    setattr(meta, "_n_workers", limit)
                    POOL_STATS["workers"] = limit
                    pending.insert(0, k)
                    log.warning("candidate %d: worker killed (out of memory?); rerun with %d workers",
                                k, limit)
                    continue
                if kind == "ok":
                    costs[k] = val
                else:
                    errors.append(val)
            now = time.time()
            for conn, (k, proc, start) in list(running.items()):
                if now - start > timeout_s:
                    proc.kill()
                    proc.join(timeout=5)
                    conn.close()
                    del running[conn]
                    errors.append(f"candidate {k} exceeded {timeout_s:.0f} s (killed, scored inf)")
                    log.warning("candidate %d killed after %.0f s: %s", k, timeout_s,
                                {key[:24]: round(v, 4) for key, v in inputs[k].items()})
    finally:
        for conn, (_, proc, _) in running.items():
            proc.kill()
        _ACTIVE_META = None
    if errors and not np.isfinite(costs).any():
        raise RuntimeError(f"every candidate failed; first error: {errors[0]}")
    return costs


_BEST: dict = {"cost": np.inf}
#: worker-pool health of this process: workers killed from outside (rerun, not
#: scored) and the current number of concurrent workers
POOL_STATS: dict = {"killed_workers": 0, "workers": None}


def _checkpoint_best(inputs: list, costs) -> None:
    """Keep the best candidate seen so far; when ``ORACLE_FIT_CHECKPOINT`` names a
    file, rewrite it on every improvement, so a fit stopped by the job's time limit
    still leaves its best parameters (PyBOP writes results only at the end)."""
    import json
    import os
    import time

    k = int(np.argmin(costs)) if len(costs) else -1
    if k < 0 or not np.isfinite(costs[k]) or costs[k] >= _BEST["cost"]:
        return
    best = {key: float(np.asarray(v, dtype=float).ravel()[0]) for key, v in dict(inputs[k]).items()}
    _BEST.update(cost=float(costs[k]), inputs=best, time=time.time())
    path = os.environ.get("ORACLE_FIT_CHECKPOINT")
    if path:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump({"best_cost": _BEST["cost"], "best_inputs": best,
                       "written": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=1)
        os.replace(tmp, path)


def _install_parallel_batches() -> None:
    from pybop.costs.evaluation import Evaluation
    from pybop.problems.meta_problem import MetaProblem

    _require_signature(MetaProblem.evaluate_batch, ["self", "inputs", "calculate_sensitivities"],
                       "MetaProblem.evaluate_batch")
    original = MetaProblem.evaluate_batch
    _ORIGINALS["MetaProblem.evaluate_batch"] = original

    def set_workers(self, n: int) -> None:
        self._n_workers = max(1, int(n))

    def evaluate_batch(self, inputs, calculate_sensitivities: bool = False):
        import os

        n = getattr(self, "_n_workers", 1)
        if n <= 1 or len(inputs) <= 1 or calculate_sensitivities:
            return original(self, inputs, calculate_sensitivities)
        timeout = float(os.environ.get("ORACLE_FIT_CANDIDATE_TIMEOUT", 600))
        inputs = list(inputs)
        costs = run_candidates(self, inputs, n, timeout)
        _checkpoint_best(inputs, costs)
        return Evaluation(values=costs)

    MetaProblem.evaluate_batch = evaluate_batch
    MetaProblem.set_workers = set_workers


def _remove_parallel_batches() -> None:
    from pybop.problems.meta_problem import MetaProblem

    MetaProblem.evaluate_batch = _ORIGINALS.pop("MetaProblem.evaluate_batch")
    if hasattr(MetaProblem, "set_workers"):
        del MetaProblem.set_workers


# ---------------------------------------------------------------------------
# Patch 4: harmonic averaging of coefficients that multiply a flux expression
# ---------------------------------------------------------------------------

def _is_flux_term(sym) -> bool:
    """A gradient, possibly negated or multiplied by a node coefficient."""
    import pybamm

    if isinstance(sym, pybamm.Gradient):
        return True
    if isinstance(sym, pybamm.Negate):
        return _is_flux_term(sym.children[0])
    if isinstance(sym, pybamm.Multiplication):
        a, b = sym.children
        return ((isinstance(b, pybamm.Gradient) and not a.evaluates_on_edges("primary"))
                or (isinstance(a, pybamm.Gradient) and not b.evaluates_on_edges("primary")))
    return False


def _is_flux(sym) -> bool:
    """True for a COMBINATION of gradient terms (a flux driving force such as
    ``chi grad(c) - grad(phi)``): an Addition/Subtraction whose terms are all
    gradients (optionally scaled). A bare gradient is left to stock PyBaMM (already
    harmonic); nested products are left alone (a coefficient that is zero on both
    sides of an edge has no harmonic mean)."""
    import pybamm

    if isinstance(sym, (pybamm.Addition, pybamm.Subtraction)):
        return all(_is_flux_term(c) or _is_flux(c) for c in sym.children)
    return False


def _install_harmonic_flux_coefficients() -> None:
    """PyBaMM's finite volumes map a node coefficient onto cell edges with the
    harmonic mean only when it multiplies a bare ``Gradient``; for any other flux
    expression they use the arithmetic mean. The DFN electrolyte current is written
    ``(kappa tor) * (chi grad(c_e) - grad(phi_e))``, so its effective conductivity
    is ARITHMETICALLY averaged at the electrode|separator interfaces, where the
    transport efficiency eps^b jumps (Ai2020: 0.040 | 0.354 | 0.124): the interface
    conductance is then overestimated by (a+b)^2/(4ab) (~2.8x for a 9x jump)
    against the series (harmonic) value of the control-volume method (Patankar 1980;
    the Recktenwald note PyBaMM cites). Patched: a coefficient multiplying any flux
    expression (see ``_is_flux``) gets the harmonic mean, so every DFN flux --
    electrolyte salt and current, solid current, heat -- treats discontinuous
    coefficients at interfaces the same, consistent way."""
    import pybamm

    fv = pybamm.FiniteVolume
    _require_signature(fv.process_binary_operators,
                       ["self", "bin_op", "left", "right", "disc_left", "disc_right"],
                       "FiniteVolume.process_binary_operators")
    original = fv.process_binary_operators
    _ORIGINALS["FiniteVolume.process_binary_operators"] = original

    def process_binary_operators(self, bin_op, left, right, disc_left, disc_right):
        le = left.evaluates_on_edges("primary")
        re_ = right.evaluates_on_edges("primary")
        # only a coefficient MULTIPLYING a flux gets the harmonic mean (as upstream PR #5856):
        # an addend (e.g. a node term added to a flux in the EIS-wrapped model) has no series
        # meaning, and is zero at rest, where its harmonic mean 1/(b/0 + (1-b)/0) is undefined
        # (non-finite Jacobian at t = 0; found with a stoichiometry-dependent conductivity,
        # 2026-10-21); division stays arithmetic (PyBaMM#5863)
        if isinstance(bin_op, pybamm.Multiplication) and le != re_:
            if le and not isinstance(left, pybamm.Gradient) and _is_flux(left):
                disc_right = self.node_to_edge(disc_right, method="harmonic")
                return pybamm.simplify_if_constant(bin_op.create_copy([disc_left, disc_right]))
            if re_ and not isinstance(right, pybamm.Gradient) and _is_flux(right):
                disc_left = self.node_to_edge(disc_left, method="harmonic")
                return pybamm.simplify_if_constant(bin_op.create_copy([disc_left, disc_right]))
        return original(self, bin_op, left, right, disc_left, disc_right)

    fv.process_binary_operators = process_binary_operators


def _remove_harmonic_flux_coefficients() -> None:
    import pybamm

    pybamm.FiniteVolume.process_binary_operators = _ORIGINALS.pop("FiniteVolume.process_binary_operators")


# ---------------------------------------------------------------------------

_PATCHES = {
    "eis_nonstationary": (_install_eis_nonstationary, _remove_eis_nonstationary),
    "graded_failure": (_install_graded_failure, _remove_graded_failure),
    "parallel_batches": (_install_parallel_batches, _remove_parallel_batches),
    "harmonic_flux_coefficients": (_install_harmonic_flux_coefficients, _remove_harmonic_flux_coefficients),
}


def apply(names: list[str] | None = None, *, strict: bool = True) -> list[str]:
    """Install the named patches (default all); returns the names now applied.

    ``strict``: raise :class:`PatchError` if versions/signatures do not match;
    otherwise skip that patch with a warning.
    """
    applied = []
    for name in names or list(_PATCHES):
        if name in _APPLIED:
            applied.append(name)
            continue
        try:
            _check_versions()
            if _PATCHES[name][0]() is False:            # the patch found an upstream fix and stood down
                continue
        except Exception as exc:
            if strict:
                raise PatchError(f"patch {name!r} not applied: {exc}") from exc
            warnings.warn(f"patch {name!r} skipped: {exc}", stacklevel=2)
            continue
        _APPLIED.add(name)
        applied.append(name)
        log.info("applied patch %s", name)
    return applied


def revert(names: list[str] | None = None) -> None:
    """Remove the named patches (default all applied), restoring stock behaviour."""
    for name in names or list(_APPLIED):
        if name in _APPLIED:
            _PATCHES[name][1]()
            _APPLIED.discard(name)


def applied() -> set[str]:
    return set(_APPLIED)
