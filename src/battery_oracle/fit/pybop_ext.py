"""PyBOP extensions of the stage-A fit: patches for fits of cells measured out of
equilibrium, the fork-safe solver they need, and the CMA-ES warm restart.

PyBOP's fitting workflow (simulators, costs, optimisers) is reused as is; two of its
assumptions do not hold for the stage-A fit (jones2022: ~1C formation, 290 s / 20 min
rests, ~300 s EIS sweeps on a relaxing cell):

``graded_failure``
    ``pybop.ErrorMeasure.evaluate_batch`` gives any simulation that stops early
    (voltage/stoichiometry event) one flat failure cost, so a population of
    infeasible candidates is a flat landscape the optimiser cannot descend. Patched:
    a truncated trajectory is scored on the part simulated plus a per-point
    penalty on the missing part (graded in the fraction missing).

``parallel_batches``
    ``pybop.MetaProblem.evaluate_batch`` evaluates a population one candidate
    after another. A joint DFN fit costs ~20 s per candidate, so a CMA-ES iteration
    (10 candidates) takes minutes. Patched: after ``problem.set_run(workers=n)`` the
    candidates are split over ``n`` forked worker processes (the built PyBaMM models
    are inherited, not pickled) and the costs recombined in order. With one worker,
    or with sensitivities requested, the stock path runs. Fork-safety: forking after
    PyBaMM's multi-threaded (OpenMP) IDAKLU has run deadlocks the workers, so solvers
    used with this patch must be single-threaded (:func:`single_thread_solver`);
    each candidate runs in its own process under a hard time limit
    (``set_run(timeout_s=...)``, default 600 s): a candidate that crawls is
    killed and scored ``inf`` like any failed simulation, so one pathological
    parameter set cannot freeze a whole population.

Warm restart (:func:`restart_population`): PyBOP starts population optimisers at the
prior mean and has no option for another start; the restart replaces the PINTS
optimiser it built, which needs PyBOP's private attributes.

Safety. Everything that touches PyBOP internals runs only if the installed
PyBaMM/PyBOP match the tested versions and the targets have the expected
signatures; otherwise :func:`apply` raises (``strict``) or skips the patch with a
warning, and :func:`restart_population` raises. Every patch can be reverted
(:func:`revert`) and is tested to leave the stock behaviour intact when its feature
is not used.

Usage::

    from battery_oracle.fit import pybop_ext
    pybop_ext.apply()                    # all patches
"""
from __future__ import annotations

import logging
import warnings
from typing import Callable

import numpy as np

from battery_oracle.patches import PatchError, check_pybamm_version, require_signature

log = logging.getLogger(__name__)

#: PyBOP version the extensions were written and tested against (the PyBaMM version is
#: battery_oracle.patches.TESTED_PYBAMM)
TESTED_PYBOP_PREFIX = "26.3"   # develop branch at commit 25d0b5b reports 26.3

_ORIGINALS: dict[str, Callable] = {}
_APPLIED: set[str] = set()


def _check_versions() -> None:
    import pybop

    check_pybamm_version()
    if not str(pybop.__version__).startswith(TESTED_PYBOP_PREFIX):
        raise PatchError(f"pybop {pybop.__version__} not {TESTED_PYBOP_PREFIX}.x")


_require_signature = require_signature


# ---------------------------------------------------------------------------
# Patch: graded cost for truncated trajectories
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
# Patch: process-parallel population evaluation
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
    simulation and this raises.
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


#: worker-pool health of this process: workers killed from outside (rerun, not
#: scored) and the current number of concurrent workers
POOL_STATS: dict = {"killed_workers": 0, "workers": None}


def _checkpoint_best(best: dict, path: str | None, inputs: list, costs) -> None:
    """Keep the best candidate seen so far in ``best`` (one per fit problem); when
    ``path`` names a file, rewrite it on every improvement, so a fit stopped by the
    job's time limit still leaves its best parameters (PyBOP writes results only at the
    end)."""
    import json
    import os
    import time

    costs = np.asarray(costs, dtype=float).ravel()
    k = int(np.argmin(costs)) if costs.size else -1
    if k < 0 or not np.isfinite(costs[k]) or costs[k] >= best.get("cost", np.inf):
        return
    inputs_k = {key: float(np.asarray(v, dtype=float).ravel()[0]) for key, v in dict(inputs[k]).items()}
    best.update(cost=float(costs[k]), inputs=inputs_k, time=time.time())
    if path:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump({"best_cost": best["cost"], "best_inputs": inputs_k,
                       "written": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=1)
        os.replace(tmp, path)


def _install_parallel_batches() -> None:
    from pybop.costs.evaluation import Evaluation
    from pybop.problems.meta_problem import MetaProblem

    _require_signature(MetaProblem.evaluate_batch, ["self", "inputs", "calculate_sensitivities"],
                       "MetaProblem.evaluate_batch")
    original = MetaProblem.evaluate_batch
    _ORIGINALS["MetaProblem.evaluate_batch"] = original

    def set_run(self, workers: int = 1, timeout_s: float = 600.0, checkpoint: str | None = None) -> None:
        """Concurrent candidate workers, the per-candidate time limit and the
        best-so-far checkpoint file of this problem's fit (its best so far resets)."""
        self._n_workers = max(1, int(workers))
        self._timeout_s = float(timeout_s)
        self._checkpoint = checkpoint
        self._best = {"cost": np.inf}

    def evaluate_batch(self, inputs, calculate_sensitivities: bool = False):
        if calculate_sensitivities:
            return original(self, inputs, calculate_sensitivities)
        inputs = list(inputs)
        n = getattr(self, "_n_workers", 1)
        if n <= 1 or len(inputs) <= 1:
            result = original(self, inputs, calculate_sensitivities)
            costs = result.values
        else:
            costs = run_candidates(self, inputs, n, self._timeout_s)
            result = Evaluation(values=costs)
        # both paths keep the problem's best so far (and its checkpoint file)
        _checkpoint_best(self.__dict__.setdefault("_best", {"cost": np.inf}), getattr(self, "_checkpoint", None),
                         inputs, costs)
        return result

    MetaProblem.evaluate_batch = evaluate_batch
    MetaProblem.set_run = set_run


def _remove_parallel_batches() -> None:
    from pybop.problems.meta_problem import MetaProblem

    MetaProblem.evaluate_batch = _ORIGINALS.pop("MetaProblem.evaluate_batch")
    if hasattr(MetaProblem, "set_run"):
        del MetaProblem.set_run


# ---------------------------------------------------------------------------

_PATCHES = {
    "graded_failure": (_install_graded_failure, _remove_graded_failure),
    "parallel_batches": (_install_parallel_batches, _remove_parallel_batches),
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


# ---------------------------------------------------------------------------
# CMA-ES warm restart
# ---------------------------------------------------------------------------

def restart_population(optim, sigma_factor: float) -> None:
    """Restart PyBOP's population optimiser ``optim`` at its parameters' initial
    values (PyBOP starts it at the prior mean), with the prior's step sizes times
    ``sigma_factor``. Raises :class:`PatchError` on an untested PyBOP."""
    _check_versions()
    for name in ("_optimiser", "_pints_optimiser", "_std0", "_boundaries"):
        if not hasattr(optim, name):
            raise PatchError(f"{type(optim).__name__} has no {name}: PyBOP internals changed")
    optim._optimiser = optim._pints_optimiser(
        x0=optim.problem.parameters.get_initial_values(transformed=True),
        sigma0=np.asarray(optim._std0, float) * float(sigma_factor), boundaries=optim._boundaries)
