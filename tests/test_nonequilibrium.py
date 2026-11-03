"""The PyBOP/PyBaMM patches."""
import numpy as np
import pytest


# --------------------------------------------------------------------------- patches

@pytest.fixture(autouse=True)
def clean_patch_state():
    """Every test here starts with no patches installed (other test modules install them
    globally, and apply() reports an already-installed patch without re-checking it) and
    leaves the global state as it found it."""
    from battery_oracle import patches

    before = patches.applied()
    patches.revert()
    yield
    patches.revert()
    if before:
        patches.apply(sorted(before), strict=False)


def test_patches_apply_and_revert_cleanly():
    pybop = pytest.importorskip("pybop")
    from pybop.costs.error_measures import ErrorMeasure

    from battery_oracle import patches

    stock_eval = ErrorMeasure.evaluate_batch
    try:
        got = set(patches.apply())
        assert {"graded_failure", "parallel_batches", "harmonic_flux_coefficients"} <= got
        assert got <= set(patches._PATCHES)
        assert ErrorMeasure.evaluate_batch is not stock_eval
    finally:
        patches.revert()
    assert ErrorMeasure.evaluate_batch is stock_eval
    assert patches.applied() == set()
    _ = pybop


def test_apply_refuses_on_version_mismatch(monkeypatch):
    pytest.importorskip("pybop")
    import pybamm

    from battery_oracle import patches

    monkeypatch.setattr(pybamm, "__version__", "99.0")
    with pytest.raises(patches.PatchError):
        patches.apply()
    with pytest.warns(UserWarning):
        assert patches.apply(strict=False) == []
    assert patches.applied() == set()


def test_graded_failure_penalises_truncation_in_proportion():
    pybop = pytest.importorskip("pybop")
    from pybop.simulators.base_simulator import Solution

    from battery_oracle import patches

    t = np.linspace(0.0, 100.0, 101)
    v = 3.7 + 0.1 * np.sin(t / 20)
    dataset = pybop.Dataset({"Time [s]": t, "Current [A]": np.zeros_like(t), "Voltage [V]": v})

    def cost_for(n_sim: int) -> float:
        sol = Solution()
        sol.set_solution_variable("Voltage [V]", data=v[:n_sim])
        cost = pybop.RootMeanSquaredError(dataset)
        out = cost.evaluate_batch([sol], [{}])
        return float(np.asarray(out.values).ravel()[0])

    stock = [cost_for(n) for n in (101, 80, 40)]          # stock: flat failure cost when short
    patches.apply(["graded_failure"])
    try:
        graded = [cost_for(n) for n in (101, 80, 40)]
    finally:
        patches.revert(["graded_failure"])
    assert graded[0] == pytest.approx(stock[0], abs=1e-12)            # complete: unchanged
    assert stock[1] == stock[2]                                       # stock: flat
    assert graded[0] < graded[1] < graded[2]                          # patched: graded
    assert graded[1] == pytest.approx(0.5 * np.sqrt(21 / 101), rel=1e-6)
    assert graded[2] == pytest.approx(0.5 * np.sqrt(61 / 101), rel=1e-6)


def test_parallel_batches_match_sequential_and_preserve_order():
    pybop = pytest.importorskip("pybop")
    import pybamm

    from battery_oracle import patches

    model = pybamm.lithium_ion.DFN()
    pv = pybamm.ParameterValues("Chen2020")
    t = np.linspace(0, 300, 61)
    truth = pybamm.Simulation(model, parameter_values=pv).solve(t_eval=t)
    dataset = pybop.Dataset({"Time [s]": t, "Current [A]": truth["Current [A]"](t),
                             "Voltage [V]": truth["Voltage [V]"](t)})
    pvf = pv.copy()
    pvf.update({"Negative electrode active material volume fraction": pybop.Parameter(
        pybop.Gaussian(0.68, 0.05), initial_value=0.68)})
    # Single-threaded solver: forking after a multi-threaded (OpenMP) solve deadlocks.
    sim = pybop.pybamm.Simulator(model, parameter_values=pvf, protocol=dataset,
                                 solver=patches.single_thread_solver())
    meta = pybop.MetaProblem(pybop.Problem(sim, pybop.RootMeanSquaredError(dataset)))
    xs = [{"Negative electrode active material volume fraction": v} for v in (0.60, 0.68, 0.72, 0.64, 0.70)]
    stock = np.asarray(meta.evaluate_batch(xs, False).values)
    patches.apply(["parallel_batches"])
    try:
        meta.set_run(workers=1)
        one = np.asarray(meta.evaluate_batch(xs, False).values)
        meta.set_run(workers=3)
        par = np.asarray(meta.evaluate_batch(xs, False).values)
    finally:
        patches.revert()
    assert np.allclose(one, stock) and np.allclose(par, stock, rtol=1e-9)
    # The data come from Chen2020's own value (0.75): cost must rank candidates by their
    # distance from it -- a shuffled recombination of worker results would not.
    values = np.array([0.60, 0.68, 0.72, 0.64, 0.70])
    assert np.array_equal(np.argsort(par), np.argsort(np.abs(values - 0.75)))


def test_parallel_batches_kill_a_runaway_candidate_and_keep_the_rest(monkeypatch):
    """One candidate that never returns must cost inf, not freeze the population."""
    pytest.importorskip("pybop")
    import time

    from pybop.costs.evaluation import Evaluation
    from pybop.problems.meta_problem import MetaProblem

    from battery_oracle import patches

    def fake(self, inputs, calculate_sensitivities=False):
        if inputs[0]["a"] == 2.0:
            time.sleep(120)                                  # the runaway
        if inputs[0]["a"] == 4.0:
            raise ValueError("bad candidate")                # an exception: inf, not a crash
        return Evaluation(values=np.array([inputs[0]["a"] * 10.0]))

    patches.apply(["parallel_batches"])
    real = patches._ORIGINALS["MetaProblem.evaluate_batch"]     # revert() must get THIS back
    try:
        patches._ORIGINALS["MetaProblem.evaluate_batch"] = fake
        meta = MetaProblem.__new__(MetaProblem)              # workers only call the fake
        meta.set_run(workers=2, timeout_s=3)
        t0 = time.time()
        out = np.asarray(meta.evaluate_batch([{"a": float(k)} for k in (1, 2, 3, 4, 5)], False).values)
        elapsed = time.time() - t0
        assert np.allclose(out[[0, 2, 4]], [10.0, 30.0, 50.0])   # order preserved around the failures
        assert np.isinf(out[1]) and np.isinf(out[3])
        assert elapsed < 30                                      # killed at ~3 s, not 120 s
        patches._ORIGINALS["MetaProblem.evaluate_batch"] = (
            lambda self, inputs, calc=False: (_ for _ in ()).throw(ValueError("boom")))
        with pytest.raises(RuntimeError, match="every candidate failed"):
            meta.evaluate_batch([{"a": 1.0}, {"a": 2.0}], False)
    finally:
        patches._ORIGINALS["MetaProblem.evaluate_batch"] = real   # no leaked fake after revert
        patches.revert()
    from pybop.problems.meta_problem import MetaProblem as _MP

    assert list(__import__("inspect").signature(_MP.evaluate_batch).parameters)[:3] == [
        "self", "inputs", "calculate_sensitivities"]
