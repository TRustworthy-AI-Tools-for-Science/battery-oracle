"""Local polish of a stage-A fit after CMA-ES (2026-10-20).

The synthetic-recovery test (battery-forecast scripts/oracle_recal/verify/synthetic_recovery.py)
showed CMA-ES stopping ("no significant change for 30 iterations") at a median voltage
misfit of ~13 mV on data whose truth misfits by ~1 mV (the noise floor), in both seeds.
Whether that is a premature stop or a genuine local minimum is decided by a local method
started at the CMA-ES best: if it keeps descending, the stop was premature.

A candidate evaluation is a full chain solve (~minutes), so a serial local optimiser
(Powell, Nelder-Mead) is too slow. This is a projected L-BFGS in bound-normalised
coordinates u in [0, 1]^n (log-scaled where the bounds span > 10x) whose evaluations are
batched: per iteration one batch of n forward-difference points (the gradient) and one
batch of trial step lengths (the line search), both run in parallel by the caller's
``evaluate_batch``.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np

log = logging.getLogger(__name__)


class BoxMap:
    """Parameter values <-> u in [0, 1]^n; log scale where 0 < lo and hi / lo > 10."""

    def __init__(self, names: Sequence[str], bounds: Sequence[tuple[float, float]]):
        self.names = list(names)
        lo, hi = np.asarray([b[0] for b in bounds], float), np.asarray([b[1] for b in bounds], float)
        if np.any(hi <= lo):
            raise ValueError("every upper bound must exceed its lower bound")
        self.log = (lo > 0) & (hi / np.where(lo > 0, lo, 1.0) > 10.0)
        self._lo = np.where(self.log, np.log(np.where(self.log, lo, 1.0)), lo)
        self._hi = np.where(self.log, np.log(np.where(self.log, hi, 1.0)), hi)

    def to_u(self, values: Sequence[float]) -> np.ndarray:
        v = np.asarray(values, float)
        g = np.where(self.log, np.log(np.where(self.log, np.maximum(v, 1e-300), 1.0)), v)
        return np.clip((g - self._lo) / (self._hi - self._lo), 0.0, 1.0)

    def from_u(self, u: np.ndarray) -> dict[str, float]:
        g = self._lo + np.clip(u, 0.0, 1.0) * (self._hi - self._lo)
        v = np.where(self.log, np.exp(g), g)
        return {k: float(x) for k, x in zip(self.names, v)}


def _two_loop(g: np.ndarray, s_hist: list, y_hist: list) -> np.ndarray:
    """L-BFGS direction -H g from the (s, y) history (steepest descent if empty)."""
    q = g.copy()
    alphas = []
    for s, y in reversed(list(zip(s_hist, y_hist))):
        rho = 1.0 / float(y @ s)
        a = rho * float(s @ q)
        alphas.append((rho, a))
        q -= a * y
    if s_hist:
        s, y = s_hist[-1], y_hist[-1]
        q *= float(s @ y) / float(y @ y)
    for (s, y), (rho, a) in zip(zip(s_hist, y_hist), reversed(alphas)):
        b = rho * float(y @ q)
        q += (a - b) * s
    return -q


def polish(evaluate_batch: Callable[[list[dict]], Sequence[float]], names: Sequence[str],
           bounds: Sequence[tuple[float, float]], x0: dict[str, float], *, max_iter: int = 15,
           h: float = 5e-3, n_trials: int = 8, max_step: float = 0.2, rel_tol: float = 1e-4,
           memory: int = 6, f0: float | None = None, central: bool = True) -> dict:
    """Minimise from ``x0`` within ``bounds``. ``evaluate_batch(list of {name: value})``
    returns one cost per candidate (non-finite = failed). ``central``: central differences
    (2n points, O(h^2) error) instead of forward (n points, O(h) error); the forward bias
    of stiff directions corrupts the direction in the weak ones. Returns the best point, its
    cost and a per-iteration history (cost, projected-gradient norm, accepted step)."""
    box = BoxMap(names, bounds)
    n = len(box.names)
    ev = lambda us: np.asarray(evaluate_batch([box.from_u(u) for u in us]), float)   # noqa: E731
    u = box.to_u([x0[k] for k in box.names])
    f = float(ev([u])[0]) if f0 is None else float(f0)
    if not np.isfinite(f):
        raise RuntimeError("the polish start point does not evaluate")
    s_hist: list = []
    y_hist: list = []
    g_prev = u_prev = None
    history = [{"iter": 0, "cost": f}]
    reason = "max_iter"
    for it in range(1, max_iter + 1):
        # gradient: central differences where both points are inside the box, else one-sided inward
        eye = np.eye(n)
        both = central & (u - h >= 0.0) & (u + h <= 1.0)
        steps = np.where(u + h <= 1.0, h, -h)
        pts = [u + steps[i] * eye[i] for i in range(n)] + [u - h * eye[i] for i in range(n) if both[i]]
        fv = ev(pts)
        fs, fb = fv[:n], np.full(n, np.nan)
        fb[both] = fv[n:]
        g = np.where(both, (fs - fb) / (2 * h), (fs - f) / steps)
        g = np.where(np.isfinite(g), g, 0.0)
        # projected gradient: drop components pushing out of the box at an active bound
        free = ~(((u <= 0.0) & (g > 0)) | ((u >= 1.0) & (g < 0)))
        pg = np.where(free, g, 0.0)
        gnorm = float(np.linalg.norm(pg))
        if g_prev is not None:
            s, y = u - u_prev, g - g_prev
            if float(s @ y) > 1e-12 * float(np.linalg.norm(s) * np.linalg.norm(y)):
                s_hist.append(s)
                y_hist.append(y)
                del s_hist[:-memory], y_hist[:-memory]
        if gnorm == 0.0:
            reason = "zero projected gradient"
            history.append({"iter": it, "cost": f, "pgrad_norm": 0.0})
            break
        d = np.where(free, _two_loop(pg, s_hist, y_hist), 0.0)
        if float(d @ pg) >= 0:                                  # not a descent direction: reset
            s_hist.clear()
            y_hist.clear()
            d = -pg
        scale = max_step / max(float(np.max(np.abs(d))), 1e-300)
        alphas = min(1.0, scale) * 0.5 ** np.arange(n_trials)
        trials = [np.clip(u + a * d, 0.0, 1.0) for a in alphas]
        ft = ev(trials)
        k = int(np.nanargmin(np.where(np.isfinite(ft), ft, np.inf)))
        if not np.isfinite(ft[k]) or ft[k] >= f:
            if s_hist:                                          # retry once along steepest descent
                s_hist.clear()
                y_hist.clear()
                d = -pg
                scale = max_step / max(float(np.max(np.abs(d))), 1e-300)
                alphas = min(1.0, scale) * 0.5 ** np.arange(n_trials)
                trials = [np.clip(u + a * d, 0.0, 1.0) for a in alphas]
                ft = ev(trials)
                k = int(np.nanargmin(np.where(np.isfinite(ft), ft, np.inf)))
            if not np.isfinite(ft[k]) or ft[k] >= f:
                reason = "no descent along the gradient (local minimum at this resolution)"
                history.append({"iter": it, "cost": f, "pgrad_norm": gnorm, "accepted_step": 0.0})
                break
        improvement = (f - ft[k]) / max(abs(f), 1e-300)
        u_prev, g_prev = u, g
        u, f = trials[k], float(ft[k])
        history.append({"iter": it, "cost": f, "pgrad_norm": gnorm, "accepted_step": float(alphas[k]),
                        "rel_improvement": float(improvement)})
        log.info("polish iter %d: cost %.6g (rel -%.2e), |pg| %.3g, step %.3g", it, f, improvement, gnorm, alphas[k])
        if improvement < rel_tol:
            reason = f"relative improvement below {rel_tol:g}"
            break
    return {"best_inputs": box.from_u(u), "best_cost": f, "start_cost": history[0]["cost"],
            "iterations": len(history) - 1, "stop_reason": reason, "history": history,
            "log_scaled": [k for k, lg in zip(box.names, box.log) if lg]}
