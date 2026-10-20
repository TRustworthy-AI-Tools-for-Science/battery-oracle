"""The Ai2020 electrolyte-diffusivity unit correction is applied only when the installed
PyBaMM still returns cm2/s (algorithmic check on synthetic parameter values)."""
import pybamm

from battery_oracle.parameter_sets import _ai2020_electrolyte_diffusivity_in_cm2


def _pv(scale):
    pv = pybamm.ParameterValues({"Electrolyte diffusivity [m2.s-1]":
                                 lambda c_e, T: scale * 10 ** (-4.43 - 54 / (T - 229 - 5e-3 * c_e) - 0.22e-3 * c_e)})
    return pv


def test_cm2_values_need_the_correction():
    assert _ai2020_electrolyte_diffusivity_in_cm2(_pv(1.0))


def test_si_values_are_left_alone():
    assert not _ai2020_electrolyte_diffusivity_in_cm2(_pv(1e-4))


def test_constant_values():
    assert _ai2020_electrolyte_diffusivity_in_cm2(pybamm.ParameterValues({"Electrolyte diffusivity [m2.s-1]": 3e-6}))
    assert not _ai2020_electrolyte_diffusivity_in_cm2(pybamm.ParameterValues({"Electrolyte diffusivity [m2.s-1]": 3e-10}))
