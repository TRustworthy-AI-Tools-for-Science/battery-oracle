"""Step-shape pieces: BOL parameter scales and the oracle's discharge_mode / step_stats.

The pure-function tests are fast; the oracle tests run PyBaMM and are marked slow.
"""
import math

import numpy as np
import pytest

from battery_oracle.params import (
    add_scale_parameters,
    apply_parameter_overrides,
    apply_parameter_scales,
    scale_parameter_name,
)


# --------------------------------------------------------------------------- params

def test_scales_numeric_and_functional_agree_with_scale_parameters():
    pybamm = pytest.importorskip("pybamm")
    j0 = "Negative electrode exchange-current density [A.m-2]"
    D = "Negative particle diffusivity [m2.s-1]"
    direct = pybamm.ParameterValues("Chen2020")
    base_D = float(direct[D])
    apply_parameter_scales(direct, {j0: 2.0, D: 0.5})
    assert math.isclose(float(direct[D]), 0.5 * base_D)

    wrapped = pybamm.ParameterValues("Chen2020")
    names = add_scale_parameters(wrapped, [j0, D])
    assert names == {j0: scale_parameter_name(j0), D: scale_parameter_name(D)}
    wrapped.update({scale_parameter_name(j0): 2.0, scale_parameter_name(D): 0.5})

    # Same expression either way: evaluate the j0 function at one state.
    args = (pybamm.Scalar(1000.0), pybamm.Scalar(15000.0), pybamm.Scalar(33133.0),
            pybamm.Scalar(298.15))
    a = direct.process_symbol(direct[j0](*args)).evaluate()
    b = wrapped.process_symbol(wrapped[j0](*args)).evaluate()
    assert np.allclose(a, b)
    assert np.isclose(wrapped.process_symbol(wrapped[D]).evaluate(), 0.5 * base_D)


def test_scale_of_one_is_noop_and_overrides_add_names():
    pybamm = pytest.importorskip("pybamm")
    pv = pybamm.ParameterValues("Chen2020")
    before = pv["Negative particle diffusivity [m2.s-1]"]
    apply_parameter_scales(pv, {"Negative particle diffusivity [m2.s-1]": 1.0})
    assert pv["Negative particle diffusivity [m2.s-1]"] == before
    apply_parameter_overrides(pv, {"Contact resistance [Ohm]": 0.02})
    assert pv["Contact resistance [Ohm]"] == 0.02


# --------------------------------------------------------------------------- oracle (PyBaMM)

@pytest.mark.slow
def test_oracle_protocol_discharge_mode_honours_duration_and_records_steps():
    from battery_oracle.ecm import randles_stub_ecm
    from battery_oracle.oracle import PyBaMMOracle

    proto = np.array([150.0, 60.0, 0.25, 0.25, 60.0, 0.05])   # 3 min discharge request
    kw = dict(ecm_model_fn=randles_stub_ecm, real_cell_capacity_mah=42.0,
              c_max_mA=22_500.0, c2_max_mA=22_500.0, dur_min_s=1.0)
    timed = PyBaMMOracle(discharge_mode="protocol", **kw)
    timed(proto)
    steps = timed.history[-1]["step_stats"]
    assert [s["kind"] for s in steps] == ["dis", "rest", "chg", "chg", "rest"]
    assert steps[0]["duration_s"] == pytest.approx(180.0, abs=2.0)
    assert not steps[0]["hit_cutoff"]

    legacy = PyBaMMOracle(**kw)
    legacy(proto)
    assert legacy.history[-1]["step_stats"][0]["duration_s"] > 180.0   # runs to 3.0 V


@pytest.mark.slow
def test_oracle_contact_resistance_lowers_charge_acceptance():
    from battery_oracle.ecm import randles_stub_ecm
    from battery_oracle.oracle import PyBaMMOracle

    proto = np.array([150.0, 60.0, 0.25, 0.25, 60.0, 0.05])
    kw = dict(ecm_model_fn=randles_stub_ecm, real_cell_capacity_mah=42.0,
              c_max_mA=22_500.0, c2_max_mA=22_500.0, dur_min_s=1.0, discharge_mode="protocol")
    ref = PyBaMMOracle(**kw)
    ref(proto)
    resist = PyBaMMOracle(contact_resistance_ohm=0.05, **kw)
    resist(proto)
    q = lambda o: o.history[-1]["step_stats"][2]["frac"]  # noqa: E731
    assert q(resist) < q(ref)
