"""EIS maths: measurement windows, lock-in drift leakage, linearised impedance."""
import numpy as np
import pytest

from battery_oracle.eis import drift_leakage, sweep_windows


def test_sweep_windows_run_high_to_low_back_to_back_in_given_order():
    f = np.array([1.0, 100.0, 0.1])                 # not sorted: windows follow sweep order
    lo, hi = sweep_windows(f, periods=2.0, overhead_s=0.5)
    length = 2.0 / f + 0.5
    assert np.allclose(hi - lo, length)
    order = np.argsort(-f)
    assert lo[order[0]] == 0.0                      # highest frequency first
    assert np.allclose(lo[order[1:]], hi[order[:-1]])
    assert hi.max() == pytest.approx(length.sum())


@pytest.mark.parametrize("f_hz,n", [(0.02, 1), (0.1, 3), (5.0, 2)])
def test_drift_leakage_matches_numerical_lock_in(f_hz, n):
    """Lock-in of A sin(wt) over n whole periods with a linear voltage drift gives the
    leakage drift_leakage computes from the window's mean slope."""
    A, vdot = 5e-3, -5e-6
    w = 2 * np.pi * f_hz
    T = n / f_hz
    t = np.linspace(0.0, T, 200001)
    i = A * np.sin(w * t)
    v = vdot * (t - T / 2)
    lock = lambda x: np.trapezoid(x * np.exp(-1j * w * t), t) / T   # noqa: E731
    z_leak = lock(v) / lock(i)
    got = drift_leakage(t, v, [0.0], [T], [f_hz], A)
    assert got[0] == pytest.approx(z_leak.real, rel=1e-4)
    assert abs(z_leak.imag) < 1e-6 * abs(z_leak.real)


def test_drift_leakage_vanishes_without_drift_and_is_linear_in_it():
    t = np.linspace(0.0, 100.0, 101)
    f, lo, hi = np.array([1.0, 0.1]), np.array([10.0, 20.0]), np.array([11.0, 30.0])
    assert np.allclose(drift_leakage(t, np.full_like(t, 3.7), lo, hi, f, 5e-3), 0.0)
    one = drift_leakage(t, 3.7 + 1e-6 * t, lo, hi, f, 5e-3)
    two = drift_leakage(t, 3.7 + 2e-6 * t, lo, hi, f, 5e-3)
    assert np.allclose(two, 2 * one) and np.all(one < 0)


@pytest.mark.slow
def test_linearised_impedance_at_the_initial_state_reproduces_pybop_eis():
    import pybamm
    import pybop

    from battery_oracle.eis import eis_model, impedance_scale, linearised_impedance
    from battery_oracle.parameter_sets import lir2032

    pv = lir2032()
    pv.update({"Contact resistance [Ohm]": 0.05}, check_already_exists=False)
    options = {"surface form": "differential", "contact resistance": "true"}
    cls = pybamm.lithium_ion.DFN
    f = np.logspace(-1, 3, 12)
    stock = pybop.pybamm.EISSimulator(cls(options=dict(options)), f_eval=f,
                                      parameter_values=pv.copy()).solve()["Impedance"].data
    model = eis_model(cls, options)
    pv0 = pv.copy()
    pv0["Current function [A]"] = 0
    sim = pybamm.Simulation(model, parameter_values=pv0, solver=pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6))
    sim.build()
    sim.solver.set_up(sim.built_model, inputs={})
    y0 = np.asarray(sim.built_model.concatenated_initial_conditions.evaluate(0, inputs={}), float).ravel()
    mine = linearised_impedance(sim, np.zeros(f.size), np.tile(y0[:, None], (1, f.size)), f,
                                impedance_scale(model, pv0))
    assert np.allclose(mine, np.asarray(stock), rtol=1e-5, atol=1e-9)


@pytest.mark.slow
def test_equilibrium_impedance_carries_the_contact_resistance():
    """With the contact-resistance option the whole spectrum shifts right by R_c (a series
    resistor); without it the parameter is ignored."""
    import pybamm

    from battery_oracle.eis import equilibrium_impedance

    pv = pybamm.ParameterValues("Chen2020")
    r_c = 0.05
    pv.update({"Contact resistance [Ohm]": r_c}, check_already_exists=False)
    f = np.logspace(3, -1, 6)
    plain = equilibrium_impedance(pybamm.lithium_ion.DFN(options={"surface form": "differential"}), pv, f, 0.5)
    with_rc = equilibrium_impedance(pybamm.lithium_ion.DFN(options={"surface form": "differential",
                                                                    "contact resistance": "true"}), pv, f, 0.5)
    np.testing.assert_allclose(with_rc.real - plain.real, r_c, rtol=1e-6)
    np.testing.assert_allclose(with_rc.imag, plain.imag, rtol=1e-6, atol=1e-12)
