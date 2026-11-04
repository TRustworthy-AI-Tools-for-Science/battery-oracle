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


def test_another_seed_changes_only_the_noise(three_cycles):
    """The seed drives the measurement noise only: another seed gives other spectra on the
    same cell (identical SOH)."""
    _, _, ref_hist = three_cycles
    o = _oracle(seed=1)
    o(PROTOCOL)
    h = o.history[0]
    assert h["end_soh"] == pytest.approx(ref_hist[0]["end_soh"], rel=1e-12)
    assert not np.allclose(h["Z_charge_real"], ref_hist[0]["Z_charge_real"])
