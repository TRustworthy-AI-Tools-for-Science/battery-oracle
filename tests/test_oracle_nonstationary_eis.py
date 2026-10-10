"""Oracle EIS measured as performed (PyBaMMOracle eis_mode="nonstationary")."""
import numpy as np
import pytest


_BASE_OPTS = {"surface form": "differential", "contact resistance": "true"}
#: The oracle's configuration for a chain-fitted LiR2032 cell (accelerated preset).
_ORACLE_OPTS = {**_BASE_OPTS, "thermal": "lumped", "use lumped thermal capacity": "true",
                "SEI": "ec reaction limited", "lithium plating": "partially reversible"}


def _oracle_pv():
    from battery_oracle.bol_fit import oracle_parameter_values

    pv = oracle_parameter_values(degradation_preset="accelerated", kinetics_scale=1.0,
                                 chemistry="LiR2032", temperature_K=296.15)
    pv.update({"Contact resistance [Ohm]": 0.05}, check_already_exists=False)
    return pv


def _lir_pv():
    from battery_oracle.parameter_sets import lir2032

    pv = lir2032()
    pv.update({"Contact resistance [Ohm]": 0.05}, check_already_exists=False)
    return pv


@pytest.mark.slow
@pytest.mark.parametrize("opts,pv_fn", [(_BASE_OPTS, _lir_pv), (_ORACLE_OPTS, _oracle_pv)],
                         ids=["dfn", "oracle_full"])
def test_cycling_states_map_onto_the_eis_model(opts, pv_fn):
    """The cycling model (double-layer DFN) state + [V, I] is the EIS model's state:
    linearising at a relaxed cycling state reproduces PyBaMM's own EIS at that state.
    Checked for the plain DFN and for the oracle's full configuration (lumped thermal,
    SEI growth, partially reversible plating)."""
    import pybamm

    from battery_oracle.eis_nonstationary import eis_model, linearised_impedance

    pv = pv_fn()
    f = np.logspace(3, -1, 9)

    cyc = pybamm.Simulation(pybamm.lithium_ion.DFN(options=opts), parameter_values=pv,
                            experiment=pybamm.Experiment(["Rest for 10 seconds (1 second period)"]))
    sol = cyc.solve(initial_soc=0.5)
    y = np.asarray(sol.y)[:, :1]
    v = np.asarray(sol["Voltage [V]"].entries)[:1]

    # same initial values as the cycling solve (PyBaMM's state references depend on them)
    pv0 = cyc.parameter_values.copy()
    pv0["Current function [A]"] = 0.0
    m = eis_model(pybamm.lithium_ion.DFN, opts)
    sim = pybamm.Simulation(m, parameter_values=pv0)
    sim.build()
    z_scale = float(np.asarray(pv0.evaluate(getattr(m.variables["Voltage [V]"], "scale", 1)
                                             / getattr(m.variables["Current [A]"], "scale", 1))).ravel()[0])
    z_map = linearised_impedance(sim, np.zeros(f.size), np.tile(np.vstack([y, v[None, :], [[0.0]]]), (1, f.size)),
                                 f, z_scale)

    stock = pybamm.EISSimulation(pybamm.lithium_ion.DFN(options=opts), parameter_values=pv)
    z_ref = np.asarray(stock.solve(f, initial_soc=0.5)["Impedance [Ohm]"])
    assert np.asarray(sol.y).shape[0] + 2 == np.asarray(
        sim.built_model.concatenated_initial_conditions.evaluate(0, inputs={})).size
    assert np.allclose(z_map, z_ref, rtol=2e-3, atol=1e-5), np.abs(z_map - z_ref) / np.abs(z_ref)


@pytest.mark.slow
def test_oracle_spectrum_of_a_relaxed_cell_equals_pybamm_eis():
    """The oracle's measurement model (a double-layer DFN started from the cycling
    state by variable name, integrated at zero current, linearised per frequency)
    reproduces PyBaMM's own EIS for a cell that is already relaxed -- with the oracle's
    full options (lumped thermal, SEI growth, plating, contact resistance)."""
    import pybamm

    from battery_oracle.oracle import PyBaMMOracle, _randles_stub_ecm

    f = np.logspace(3, -1, 9)
    # side reactions off (no plating; SEI growth ~0): a resting cell with SEI growth
    # carries a small steady interfacial current, so it is never fully relaxed and its
    # low-frequency impedance differs from the equilibrium one -- physically right,
    # but not a clean reference
    o = PyBaMMOracle(ecm_model_fn=_randles_stub_ecm, chemistry="LiR2032", model="DFN",
                     thermal="lumped", T_ambient_K=298.15, contact_resistance_ohm=0.05,
                     degradation_preset="nominal", sei_rate_scale=1e-9,
                     eis_mode="nonstationary", eis_noise_level=0.0, frequencies=f, rest_s=600.0)
    pv = o._pv.copy()
    rest = pybamm.Simulation(o._model_cls(options=o._deg_opts), parameter_values=pv,
                             experiment=pybamm.Experiment(["Rest for 4 hours"]))
    sol = rest.solve(initial_soc=0.5)
    z = o._nonstationary_spectrum(sol.cycles[-1].steps[-1], pv)
    assert o._eis_lin_ok
    stock = pybamm.EISSimulation(o._model_cls(options={**o._deg_opts, "surface form": "differential"}),
                                 parameter_values=pv)
    z_ref = np.asarray(stock.solve(f, initial_soc=0.5)["Impedance [Ohm]"])
    assert np.allclose(z, z_ref, rtol=0.02, atol=1e-4), np.abs(z - z_ref) / np.abs(z_ref)


@pytest.mark.slow
def test_oracle_cycle_measures_spectra_inside_the_sweep():
    from battery_oracle.oracle import PyBaMMOracle, _randles_stub_ecm

    f = np.logspace(3, -1, 12)
    o = PyBaMMOracle(ecm_model_fn=_randles_stub_ecm, chemistry="LiR2032", model="DFN",
                     thermal="lumped", T_ambient_K=296.15, contact_resistance_ohm=0.1,
                     eis_mode="nonstationary", eis_noise_level=0.0, frequencies=f,
                     real_cell_capacity_mah=42.0, c_max_mA=150.0, c2_max_mA=150.0, dfn_max_crate=4.0)
    o(np.array([80.0, 40.0, 0.2, 0.2, 60.0, 0.5]))
    h = o.history[-1]
    assert h["fidelity"] == "full", h["fidelity"]                  # cycling model unchanged
    assert h["eis_mode_used"] == "nonstationary", h["eis_mode_used"]
    for side in ("charge", "discharge"):
        zr, zi = np.asarray(h[f"Z_{side}_real"]), np.asarray(h[f"Z_{side}_neg_imag"])
        assert np.all(np.isfinite(zr)) and np.all(np.isfinite(zi)) and np.all(zr > 0), side
    # contact resistance is now in the spectrum: high-frequency real part above it
    assert h["Z_charge_real"][int(np.argmax(f))] > 0.1
