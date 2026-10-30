"""battery_oracle.parameter_sets: the real-scale LiR2032 set, its LiCoO2 diffusivity law, and
its use in the oracle."""
import numpy as np
import pytest

pybamm = pytest.importorskip("pybamm")

from battery_oracle import parameter_sets as ps  # noqa: E402


def _area(pv):
    return (float(pv["Electrode height [m]"]) * float(pv["Electrode width [m]"])
            * float(pv["Number of electrodes connected in parallel to make a cell"]))


def test_lir2032_is_one_electrode_pair_scaled_from_the_template_by_capacity():
    """One layer whose area is the template's area per capacity times the nominal capacity;
    the voltage window and the materials' geometry are the set's constants and the template's."""
    ai = pybamm.ParameterValues("Ai2020")
    per_ah = _area(ai) / float(ai["Nominal cell capacity [A.h]"])
    for q in (0.02, ps.LIR2032_NOMINAL_AH, 0.08):
        pv = ps.lir2032(nominal_ah=q)
        assert float(pv["Number of electrodes connected in parallel to make a cell"]) == 1.0
        assert float(pv["Nominal cell capacity [A.h]"]) == q
        assert _area(pv) == pytest.approx(per_ah * q, rel=1e-12)
    pv = ps.lir2032()
    assert (float(pv["Lower voltage cut-off [V]"]), float(pv["Upper voltage cut-off [V]"])) == (
        ps.LIR2032_V_MIN, ps.LIR2032_V_MAX)
    for key in ("Positive electrode thickness [m]", "Negative electrode porosity",
                "Maximum concentration in positive electrode [mol.m-3]"):
        assert pv[key] == ai[key]


def test_lir2032_uses_the_template_electrolyte_and_the_marquis_lico2_ocp():
    pv, ai = ps.lir2032(), pybamm.ParameterValues("Ai2020")
    assert pv["Electrolyte diffusivity [m2.s-1]"] is ai["Electrolyte diffusivity [m2.s-1]"]
    assert pv["Positive electrode OCP [V]"] is pybamm.ParameterValues("Marquis2019")["Positive electrode OCP [V]"]


def test_set_names_and_options_resolve():
    """The plain name and the name the stage-A fits ran under are the same set; an unknown
    graphite option is rejected; PyBaMM's own sets pass through."""
    a = ps.resolve_parameter_values("LiR2032")
    b = ps.resolve_parameter_values("LiR2032_MohtatGraphite_MarquisLCO_ThermoD")
    assert a["Negative electrode OCP [V]"] is b["Negative electrode OCP [V]"]
    assert ps.nominal_capacity_mah("LiR2032") == pytest.approx(1000.0 * ps.LIR2032_NOMINAL_AH)
    assert ps.nominal_capacity_mah("Chen2020") == pytest.approx(
        1000.0 * float(pybamm.ParameterValues("Chen2020")["Nominal cell capacity [A.h]"]))
    with pytest.raises(ValueError):
        ps.lir2032(graphite_ocp="not-a-graphite")


def test_ocp_thermodynamic_factor_is_the_lattice_gas_law():
    """D = D0 Theta with Theta = -(F / R T) y (1 - y) dU/dy from the OCP itself, checked against a
    finite difference of an arbitrary smooth monotone OCP, at two temperatures."""
    d0, cmax = 1.0e-14, 5.0e4
    # an OCP that rescales its argument first (as Marquis2019 does with 1.062 * sto): PyBaMM folds
    # the constants, so differentiating w.r.t. the passed-in expression would give 0
    ocp = lambda s: 4.0 - 0.3 * (1.05 * s) - 0.2 * pybamm.tanh((1.05 * s - 0.9) / 0.02)      # noqa: E731
    f = ps._with_ocp_thermodynamic_factor(lambda sto, T: d0, ocp)
    F, R = 96485.33212, 8.314462618
    c = pybamm.StateVector(slice(0, 1))                       # stoichiometry as in the model: c / c_max
    for T in (283.15, 318.15):
        d = f(c / cmax, pybamm.Scalar(T))
        for y in (0.3, 0.6, 0.85, 0.9):
            h = 1e-6
            du = (float(np.ravel(ocp(pybamm.Scalar(y + h)).evaluate())[0])
                  - float(np.ravel(ocp(pybamm.Scalar(y - h)).evaluate())[0])) / (2 * h)
            expect = d0 * max(-(F / (R * T)) * y * (1 - y) * du, ps.LICO2_THETA_FLOOR)
            got = float(np.ravel(d.evaluate(y=np.array([y * cmax])))[0])
            assert got == pytest.approx(expect, rel=1e-6)


def test_lico2_diffusivity_follows_the_installed_ocp():
    """The set's LiCoO2 D is the thermodynamic-factor law of its own OCP: small on the flat
    plateau, large where the OCP falls steeply near full lithiation."""
    pv = ps.lir2032()
    cmax = float(pv["Maximum concentration in positive electrode [mol.m-3]"])
    c = pybamm.StateVector(slice(0, 1))
    d = pv.process_symbol(pv["Positive particle diffusivity [m2.s-1]"](c / cmax, pybamm.Scalar(298.15)))
    u = pv.process_symbol(pv["Positive electrode OCP [V]"](c / cmax))
    val = lambda e, y: float(np.ravel(e.evaluate(y=np.array([y * cmax])))[0])       # noqa: E731
    slope = lambda y, h=1e-4: (val(u, y + h) - val(u, y - h)) / (2 * h)            # noqa: E731
    F, R, T = 96485.33212, 8.314462618, 298.15
    theta = lambda y: max(-(F / (R * T)) * y * (1 - y) * slope(y), ps.LICO2_THETA_FLOOR)   # noqa: E731
    lo, hi = 0.85, 0.96
    assert abs(slope(hi)) > abs(slope(lo))                    # the wall is steeper than the plateau
    assert val(d, hi) / val(d, lo) == pytest.approx(theta(hi) / theta(lo), rel=1e-2)
    assert val(d, hi) > val(d, lo)


def test_oracle_scales_currents_by_the_simulated_cell_capacity():
    """The protocol's real-cell currents are scaled by simulated / real capacity: 1 when the
    real cell is the simulated one, nominal / real otherwise."""
    from battery_oracle.oracle import PyBaMMOracle, _randles_stub_ecm

    q = ps.nominal_capacity_mah("LiR2032")
    o = PyBaMMOracle(chemistry="LiR2032", ecm_model_fn=_randles_stub_ecm, real_cell_capacity_mah=q)
    assert o._cap_scale == pytest.approx(1.0)
    chen = PyBaMMOracle(ecm_model_fn=_randles_stub_ecm, real_cell_capacity_mah=q)
    assert chen._cap_scale == pytest.approx(ps.nominal_capacity_mah("Chen2020") / q)


@pytest.mark.slow
def test_dfn_lir2032_oracle_applies_real_cell_currents_unscaled():
    """A real-scale DFN cycle: a stage current equals the requested real-cell current."""
    from battery_oracle.oracle import PyBaMMOracle, _randles_stub_ecm

    q = ps.nominal_capacity_mah("LiR2032")
    o = PyBaMMOracle(chemistry="LiR2032", ecm_model_fn=_randles_stub_ecm,
                     real_cell_capacity_mah=q, discharge_mode="protocol",
                     c_min_mA=0.01 * q, c2_min_mA=0.01 * q, c_max_mA=4.5 * q, c2_max_mA=4.5 * q,
                     dfn_max_crate=4.5, dur_min_s=1.0, ecm_rescale_target_r0=None)
    o(np.array([3.5 * q, 1.5 * q, 0.25, 0.25, 1.3 * q, 0.1]))
    h = o.history[-1]
    assert h["fidelity"] == "full"
    assert h["step_stats"][2]["current_A"] == pytest.approx(3.5 * q / 1000.0, rel=1e-3)
    assert h["step_stats"][2]["frac"] > 0.0
