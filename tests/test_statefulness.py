"""The oracle's core claim: a stateful cell. Each call is one cycle that continues from
the cell the previous calls left; reset() returns to the fresh cell; a seed fixes the
whole trajectory, measurement noise included.

All of these solve the DFN (Randles-stub ECM, no AutoEIS), so all are slow.
"""
import numpy as np
import pytest

from battery_oracle import PyBaMMOracle, randles_stub_ecm
from conftest import PROTOCOL

pytestmark = pytest.mark.slow

N_CYCLES = 3


def _oracle(seed=0):
    return PyBaMMOracle(ecm_model_fn=randles_stub_ecm, seed=seed)


def _run(oracle, n):
    return [np.asarray(oracle(PROTOCOL)) for _ in range(n)]


@pytest.fixture(scope="module")
def three_cycles():
    """A fresh seeded oracle after N_CYCLES identical calls: (oracle, states, history)."""
    o = _oracle()
    states = _run(o, N_CYCLES)
    return o, states, [dict(h) for h in o.history]


def test_degradation_accumulates_across_calls(three_cycles):
    """SOH never rises; lithium loss and charge throughput only grow, call after call."""
    o, _, hist = three_cycles
    assert [h["call_idx"] for h in hist] == list(range(N_CYCLES))
    soh = np.array([h["end_soh"] for h in hist])
    lli = np.array([h["cumulative_lli_total_ah"] for h in hist])
    assert np.all(np.diff(soh) <= 0.0) and soh[-1] < 1.0
    assert np.all(np.diff(lli) > 0.0)
    thr = np.asarray(o.last_solution["Throughput capacity [A.h]"].entries, float)
    assert np.all(np.diff(thr) >= -1e-12) and thr[-1] > 0.0


def test_one_call_per_cycle_equals_one_experiment(three_cycles):
    """N calls of one cycle each hand the cell on exactly as one PyBaMM experiment of N
    cycles over the same steps, solved by the same model and solver."""
    import pybamm

    o, _, _ = three_cycles
    exp, _ = o._protocol_to_experiment(PROTOCOL)
    sim = pybamm.Simulation(o._cycling_model, experiment=pybamm.Experiment(list(exp.cycles) * N_CYCLES),
                            var_pts=o._model_var_pts(o._cycling_model),
                            submesh_types=o._model_submesh(o._cycling_model),
                            parameter_values=o._pv, solver=o._solver)
    whole = sim.solve(initial_soc=o._initial_soc)
    calls = o.last_solution
    for var in ("Voltage [V]", "Loss of capacity to negative SEI [A.h]", "Throughput capacity [A.h]",
                "Total lithium in particles [mol]"):
        a = float(np.asarray(calls[var].entries).ravel()[-1])
        b = float(np.asarray(whole[var].entries).ravel()[-1])
        assert a == pytest.approx(b, rel=1e-4, abs=1e-9), var
    assert float(calls.t[-1]) == pytest.approx(float(whole.t[-1]), rel=1e-6)


def test_capacity_is_each_cycles_own_discharge(three_cycles):
    """Without the capacity check, capacity_ah is the charge the cycle's discharge step
    passed (the same charge its step statistics integrate), in every cycle, not a running
    maximum over the history."""
    _, _, hist = three_cycles
    for h in hist:
        assert h["capacity_ah"] == pytest.approx(h["step_stats"][0]["throughput_ah"], rel=1e-3)


def test_earlier_calls_are_not_kept(three_cycles):
    """Each call continues from the previous call's final state only: the last solution
    holds that state and the one cycle run, not the whole history (memory stays bounded
    over a long replay)."""
    o, _, _ = three_cycles
    assert len(o.last_solution.cycles) <= 2
    assert o._prev_solution.t.size == 1


def test_reset_and_seed_reproduce_the_trajectory(three_cycles):
    """A second oracle with the same seed reproduces the trajectory, noise included; after
    reset() it starts from the fresh cell again and reproduces it once more."""
    _, ref, ref_hist = three_cycles
    o = _oracle()
    before = _run(o, 2)
    o.reset()
    assert o.history == [] and o.last_solution is None and o.last_parameter_values is None
    after = _run(o, 2)
    for got in (before, after):
        for k in range(2):
            np.testing.assert_allclose(got[k], ref[k], rtol=1e-10, equal_nan=True)
    assert [h["end_soh"] for h in o.history] == pytest.approx([h["end_soh"] for h in ref_hist[:2]], rel=1e-12)


def test_age_advances_the_cell_as_run_cycle_does(three_cycles):
    """age() runs the same cycles without the measurement: the same degradation, state
    of health and step statistics (rest voltages included) as run_cycle, and history
    rows with no spectra, ECM, linKK or DRT entries."""
    _, _, ref_hist = three_cycles
    o = _oracle()
    rows = [dict(o.age(PROTOCOL)) for _ in range(N_CYCLES)]
    assert [h["call_idx"] for h in o.history] == list(range(N_CYCLES))
    for got, ref in zip(rows, ref_hist):
        for key in ("end_soh", "cumulative_lli_total_ah", "capacity_ah"):
            assert got[key] == pytest.approx(ref[key], rel=1e-12), key
        for a, b in zip(got["step_stats"], ref["step_stats"]):
            assert a["kind"] == b["kind"]
            assert a["throughput_ah"] == pytest.approx(b["throughput_ah"], rel=1e-12, abs=1e-15)
            assert a["v_end"] == pytest.approx(b["v_end"], rel=1e-12, nan_ok=True)
        measured = [k for k in got if k.startswith(("Z_", "ecm_")) or k in ("drt_peaks", "linkk_rmse")]
        assert measured == [], measured
    assert o.last_solution is not None


def test_another_seed_changes_only_the_noise(three_cycles):
    """The seed drives the measurement noise only: another seed gives other spectra on the
    same cell (identical SOH)."""
    _, _, ref_hist = three_cycles
    o = _oracle(seed=1)
    o(PROTOCOL)
    h = o.history[0]
    assert h["end_soh"] == pytest.approx(ref_hist[0]["end_soh"], rel=1e-12)
    assert not np.allclose(h["Z_charge_real"], ref_hist[0]["Z_charge_real"])
