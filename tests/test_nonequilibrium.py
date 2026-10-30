"""Non-equilibrium physics: non-stationary EIS and the PyBOP/PyBaMM patches."""
import numpy as np
import pytest

from battery_oracle.eis_nonstationary import History, window_times


# --------------------------------------------------------------------------- maths

def test_window_times_back_to_back_and_given_stamps():
    f = np.array([100.0, 1.0, 0.1])
    c, length = window_times(f, n_periods=2.0)
    assert np.allclose(length, [0.02, 2.0, 20.0])
    assert np.allclose(c, np.cumsum(length) - length / 2)
    c2, _ = window_times(f, n_periods=1.0, sweep_times=[1.0, 5.0, 20.0])   # stamps = window ends
    assert np.allclose(c2, np.array([1.0, 5.0, 20.0]) - np.array([0.01, 1.0, 10.0]) / 2)


@pytest.mark.parametrize("f_hz,n", [(0.02, 1), (0.1, 3), (5.0, 2)])
def test_drift_leakage_formula_matches_numerical_lock_in(f_hz, n):
    """Lock-in of A sin(wt) over n whole periods with a linear voltage drift:
    the impedance error is -2 (dV/dt) / (w A) (module docstring derivation)."""
    A, vdot = 5e-3, -5e-6                       # 5 mA probe, -0.3 mV/min drift
    w = 2 * np.pi * f_hz
    T = n / f_hz
    t = np.linspace(0.0, T, 200001)
    i = A * np.sin(w * t)
    v = vdot * (t - T / 2) + 1e-3 * 0.0         # pure drift (+ zero-mean response removed)
    lock = lambda x: np.trapezoid(x * np.exp(-1j * w * t), t) / T   # noqa: E731
    z_leak = lock(v) / lock(i)
    assert z_leak.real == pytest.approx(-2 * vdot / (w * A), rel=1e-4)
    assert abs(z_leak.imag) < 1e-6 * abs(z_leak.real)


def test_leakage_is_about_one_percent_at_the_lowest_frequency():
    """The size quoted in the report: -0.3 mV/min at 0.02 Hz vs |Z| ~ 1.5 Ohm."""
    vdot, w, A = -0.3e-3 / 60, 2 * np.pi * 0.02, 5e-3
    assert abs(-2 * vdot / (w * A)) / 1.5 == pytest.approx(0.01, abs=0.005)


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


# --------------------------------------------------------------------------- DFN (slow)

def _setup():
    import pybamm

    from battery_oracle.parameter_sets import lir2032

    pv = lir2032()
    pv.update({"Contact resistance [Ohm]": 0.05}, check_already_exists=False)
    options = {"surface form": "differential", "contact resistance": "true"}
    return pybamm.lithium_ion.DFN, options, pv


@pytest.mark.slow
def test_no_history_reproduces_stock_pybop_eis():
    import pybop

    from battery_oracle.eis_nonstationary import nonstationary_impedance

    cls, options, pv = _setup()
    f = np.logspace(-1, 3, 12)
    stock = pybop.pybamm.EISSimulator(cls(options=dict(options)), f_eval=f,
                                      parameter_values=pv.copy()).solve()["Impedance"].data
    mine = nonstationary_impedance(cls, options, pv, f, history=None)
    assert np.allclose(mine, np.asarray(stock), rtol=1e-5, atol=1e-9)


@pytest.mark.slow
def test_relaxing_state_changes_low_frequency_impedance_only():
    from battery_oracle.eis_nonstationary import nonstationary_impedance

    cls, options, pv = _setup()
    f = np.array([1e4, 1e2, 1.0, 0.1, 0.02])
    # 20 min at ~1C discharge (42 mA), then 20 min rest, from 4.1 V relaxed.
    t = np.concatenate([np.linspace(0, 1200, 41), [1201.0, 2400.0]])
    i = np.concatenate([np.full(41, 0.042), [0.0, 0.0]])
    hist = History(t=t, I=i, initial_voltage=4.1)
    sweep = np.cumsum(1.0 / f)
    z_ns, info = nonstationary_impedance(cls, options, pv, f, history=hist, sweep_times=sweep,
                                         n_periods=1.0, drift_leakage=False, return_info=True)
    z_end = nonstationary_impedance(cls, options, pv, f, history=hist, sweep_times=sweep,
                                    n_periods=1.0, drift_leakage=False, per_frequency=False)
    assert np.all(np.isfinite(z_ns)) and np.all(z_ns.real > 0)
    assert np.all(info.voltage_v > 3.0) and np.all(info.voltage_v < 4.3)
    rel = np.abs(z_ns - z_end) / np.abs(z_end)
    assert rel[0] < 1e-6                         # 10 kHz: measured at the start of the sweep
    assert rel[-1] >= rel[0]                     # the slowest frequency sees the most relaxation
    # leakage: sign follows -2 vdot/(w A); magnitude ~1% or less of |Z|
    z_leak, info2 = nonstationary_impedance(cls, options, pv, f, history=hist, sweep_times=sweep,
                                            n_periods=1.0, drift_leakage=True, return_info=True)
    assert np.allclose((z_leak - z_ns).real, info2.leakage_ohm.real, rtol=1e-6, atol=1e-12)
    assert np.all(np.abs(info2.leakage_ohm) < 0.05 * np.abs(z_ns))


def test_relative_sweep_times_rebase_absolute_stamps_and_guard():
    from battery_oracle.eis_nonstationary import relative_sweep_times

    stamps = np.array([5140.25, 5141.38, 5143.58, 5420.60])        # real PJ248 c9 stamps
    rel = relative_sweep_times(stamps)
    assert rel[0] == pytest.approx(2.0) and rel[-1] == pytest.approx(282.35)
    c, length = window_times(np.array([1e4, 5e3, 1e3, 0.02]), 1.0, rel)    # fine: relative
    assert c[-1] < rel[-1]
    with pytest.raises(ValueError, match="absolute"):
        window_times(np.array([1e4, 5e3, 1e3, 0.02]), 1.0, stamps)


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
        meta.set_workers(1)
        one = np.asarray(meta.evaluate_batch(xs, False).values)
        meta.set_workers(3)
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
        monkeypatch.setenv("ORACLE_FIT_CANDIDATE_TIMEOUT", "3")
        meta = MetaProblem.__new__(MetaProblem)              # workers only call the fake
        meta.set_workers(2)
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
