"""polish: batched projected L-BFGS in bound-normalised coordinates (synthetic functions only)."""
import numpy as np
import pytest

from battery_oracle.polish import BoxMap, polish

NAMES = [f"p{i}" for i in range(6)]
BOUNDS = [(0.1, 10.0), (1e-3, 1.0), (-2.0, 2.0), (0.5, 1.5), (1.0, 3.0), (1e-6, 1e-4)]


def _evaluator(fun, seen):
    def ev(cands):
        out = []
        for c in cands:
            x = np.array([c[k] for k in NAMES])
            seen.append(x)
            out.append(fun(x))
        return out
    return ev


def _in_bounds(seen):
    lo, hi = np.array([b[0] for b in BOUNDS]), np.array([b[1] for b in BOUNDS])
    tol_lo, tol_hi = 1e-12 * np.abs(lo) + 1e-15, 1e-12 * np.abs(hi) + 1e-15
    return all(np.all(x >= lo - tol_lo) and np.all(x <= hi + tol_hi) for x in seen)


def test_boxmap_round_trip_and_log_choice():
    box = BoxMap(NAMES, BOUNDS)
    assert list(box.log) == [True, True, False, False, False, True]
    x = {k: v for k, v in zip(NAMES, [3.0, 0.02, 0.5, 1.1, 2.5, 3e-5])}
    back = box.from_u(box.to_u([x[k] for k in NAMES]))
    assert all(back[k] == pytest.approx(x[k], rel=1e-12) for k in NAMES)
    assert np.allclose(box.to_u([b[0] for b in BOUNDS]), 0) and np.allclose(box.to_u([b[1] for b in BOUNDS]), 1)


def _quadratic(target_u, weights):
    box = BoxMap(NAMES, BOUNDS)
    return lambda x: float(np.sum(weights * (box.to_u(x) - target_u) ** 2)), box


def test_converges_to_an_interior_minimum_of_an_ill_conditioned_quadratic():
    target = np.array([0.3, 0.7, 0.45, 0.6, 0.2, 0.55])
    fun, box = _quadratic(target, np.array([1.0, 30.0, 3.0, 100.0, 10.0, 0.3]))
    seen = []
    x0 = box.from_u(np.full(6, 0.5))
    res = polish(_evaluator(fun, seen), NAMES, BOUNDS, x0, max_iter=60, rel_tol=1e-10)
    assert res["best_cost"] < 1e-6 * res["start_cost"]
    assert np.allclose(box.to_u([res["best_inputs"][k] for k in NAMES]), target, atol=2e-3)
    assert _in_bounds(seen)


def test_forward_differences_stall_where_central_ones_converge():
    """Why central is the default: on an ill-conditioned problem the O(h) forward bias stops
    the weak direction short."""
    target = np.array([0.3, 0.7, 0.45, 0.6, 0.2, 0.55])
    fun, box = _quadratic(target, np.array([1.0, 30.0, 3.0, 100.0, 10.0, 0.3]))
    x0 = box.from_u(np.full(6, 0.5))
    fwd = polish(_evaluator(fun, []), NAMES, BOUNDS, x0, max_iter=60, rel_tol=1e-10, central=False)
    cen = polish(_evaluator(fun, []), NAMES, BOUNDS, x0, max_iter=60, rel_tol=1e-10, central=True)
    assert cen["best_cost"] < 1e-2 * fwd["best_cost"]


def test_minimum_outside_the_box_ends_on_the_bound():
    target = np.array([1.3, 0.5, -0.2, 0.5, 0.5, 0.5])         # first coordinate beyond u = 1, third below 0
    fun, box = _quadratic(target, np.ones(6))
    seen = []
    res = polish(_evaluator(fun, seen), NAMES, BOUNDS, box.from_u(np.full(6, 0.4)), max_iter=60, rel_tol=1e-12)
    u = box.to_u([res["best_inputs"][k] for k in NAMES])
    assert u[0] == pytest.approx(1.0, abs=1e-6) and u[2] == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(u[[1, 3, 4, 5]], 0.5, atol=2e-2)
    assert _in_bounds(seen)


def test_start_at_the_minimum_does_not_move():
    target = np.full(6, 0.5)
    fun, box = _quadratic(target, np.ones(6))
    x0 = box.from_u(target)
    res = polish(_evaluator(fun, []), NAMES, BOUNDS, x0, max_iter=10)
    assert res["best_cost"] == pytest.approx(0.0, abs=1e-12)
    assert all(res["best_inputs"][k] == pytest.approx(x0[k], rel=1e-9) for k in NAMES)


def test_failed_candidates_are_skipped_not_accepted():
    """A region that fails to evaluate (inf) between start and minimum must never be accepted."""
    target = np.full(6, 0.8)
    base, box = _quadratic(target, np.ones(6))

    def fun(x):
        u = box.to_u(x)
        return np.inf if 0.55 < u[0] < 0.6 else base(x)

    seen = []
    res = polish(_evaluator(fun, seen), NAMES, BOUNDS, box.from_u(np.full(6, 0.3)), max_iter=40, rel_tol=1e-12)
    assert np.isfinite(res["best_cost"]) and res["best_cost"] < res["start_cost"]
    assert all(np.isfinite(h["cost"]) for h in res["history"])


def test_batches_are_used():
    """Each iteration evaluates its gradient points and its line-search trials as batches."""
    sizes = []
    fun, box = _quadratic(np.full(6, 0.6), np.ones(6))

    def ev(cands):
        sizes.append(len(cands))
        return [fun(np.array([c[k] for k in NAMES])) for c in cands]

    polish(ev, NAMES, BOUNDS, box.from_u(np.full(6, 0.4)), max_iter=3, n_trials=5)
    assert sizes[0] == 1 and 12 in sizes and 5 in sizes          # 2n central points, n_trials steps
