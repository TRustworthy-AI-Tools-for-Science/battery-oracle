"""battery_oracle.parameter_sets: the real-scale LiR2032 set and its use in the oracle."""
import math

import numpy as np
import pytest

pybamm = pytest.importorskip("pybamm")

from battery_oracle.parameter_sets import (  # noqa: E402
    LIR2032_NOMINAL_AH,
    lir2032,
    nominal_capacity_mah,
    resolve_parameter_values,
)


def test_lir2032_is_a_single_layer_coin_cell_from_ai2020():
    pv = lir2032()
    ai = pybamm.ParameterValues("Ai2020")
    assert float(pv["Nominal cell capacity [A.h]"]) == LIR2032_NOMINAL_AH
    assert float(pv["Number of electrodes connected in parallel to make a cell"]) == 1.0
    assert (float(pv["Lower voltage cut-off [V]"]), float(pv["Upper voltage cut-off [V]"])) == (3.0, 4.3)
    # Area scaled from Ai2020's area-per-capacity: ~15 cm2 for 42 mAh.
    area = float(pv["Electrode height [m]"]) * float(pv["Electrode width [m]"])
    ai_area = (float(ai["Electrode height [m]"]) * float(ai["Electrode width [m]"])
               * float(ai["Number of electrodes connected in parallel to make a cell"]))
    assert area == pytest.approx(ai_area * LIR2032_NOMINAL_AH / float(ai["Nominal cell capacity [A.h]"]))
    assert 10e-4 < area < 20e-4
    # Materials untouched.
    for key in ("Positive electrode thickness [m]", "Negative electrode porosity",
                "Maximum concentration in positive electrode [mol.m-3]"):
        assert pv[key] == ai[key]


def test_resolver_and_ramadass_template():
    assert math.isclose(nominal_capacity_mah("LiR2032"), 42.0)
    assert math.isclose(nominal_capacity_mah("Chen2020"), 5000.0)
    rv = resolve_parameter_values("LiR2032_Ramadass2004")
    assert rv["Positive electrode OCP [V]"] is not lir2032()["Positive electrode OCP [V]"]
    with pytest.raises(ValueError):
        lir2032(template="Chen2020")


def test_oracle_scales_currents_by_the_simulated_cell_capacity():
    from battery_oracle.oracle import PyBaMMOracle, _randles_stub_ecm

    o = PyBaMMOracle(chemistry="LiR2032", ecm_model_fn=_randles_stub_ecm,
                     real_cell_capacity_mah=42.0)
    assert o._cap_scale == pytest.approx(1.0)          # real cell simulated at real size
    chen = PyBaMMOracle(ecm_model_fn=_randles_stub_ecm, real_cell_capacity_mah=42.0)
    assert chen._cap_scale == pytest.approx(5000.0 / 42.0)


@pytest.mark.slow
def test_dfn_lir2032_oracle_runs_the_real_jones2022_currents():
    """One real-scale DFN cycle at the jones2022 extremes (~3.6C CC1, ~1.3C discharge)."""
    from battery_oracle.oracle import PyBaMMOracle, _randles_stub_ecm

    o = PyBaMMOracle(chemistry="LiR2032", model="DFN", ecm_model_fn=_randles_stub_ecm,
                     real_cell_capacity_mah=42.0, discharge_mode="protocol",
                     c_min_mA=0.42, c2_min_mA=0.42, c_max_mA=189.0, c2_max_mA=189.0,
                     dfn_max_crate=4.5, dur_min_s=1.0, ecm_rescale_target_r0=None)
    protocol = np.array([150.0, 60.0, 0.25, 0.25, 55.0, 0.1])
    o(protocol)
    h = o.history[-1]
    assert h["fidelity"] == "full"
    steps = h["step_stats"]
    assert steps[2]["current_A"] == pytest.approx(0.150, rel=1e-3)   # real mA, unscaled
    assert steps[2]["frac"] > 0.0


def test_lir2032_corrects_the_ai2020_electrolyte_diffusivity_units():
    pv, ai = lir2032(), pybamm.ParameterValues("Ai2020")
    c, T = pybamm.Scalar(1000.0), pybamm.Scalar(298.15)

    def d(p):
        return float(p.process_symbol(p["Electrolyte diffusivity [m2.s-1]"](c, T)).evaluate())

    assert d(ai) > 1e-6                     # PyBaMM's Ai2020: cm2/s returned as m2/s
    assert 1e-10 < d(pv) < 1e-9             # corrected: ~3.2e-10 m2/s
    assert d(pv) == pytest.approx(1e-4 * d(ai))
