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


import pytest

import battery_oracle.parameter_sets as ps


def _d_e(pv):
    f = pv["Electrolyte diffusivity [m2.s-1]"]
    return float(pv.process_symbol(f(pybamm.Scalar(1000.0), pybamm.Scalar(298.15))).evaluate())


@pytest.mark.parametrize("lico2_ds", ["knee", "thermo", "vacancy"])
def test_model_forms_apply_whatever_the_unit_check_says(monkeypatch, lico2_ds):
    """The unit check decides only the electrolyte-diffusivity scale; the OCP continuation,
    activation energies, D_s forms and j0 shape must be installed in both cases (they were
    dropped on PyBaMM >= 26.10 until 2026-10-20)."""
    built = {}
    for needs_fix in (True, False):
        monkeypatch.setattr(ps, "_ai2020_electrolyte_diffusivity_in_cm2", lambda pv, v=needs_fix: v)
        pv = ps.lir2032(lico2_ds=lico2_ds)
        assert pv["Positive electrode OCP [V]"] is ps.lico2_ocp_monotone
        assert ps.J0_SHAPE_P in pv.keys() and ps.GRAPHITE_DS_KNEE in pv.keys()
        assert (ps.LICO2_DS_KNEE in pv.keys()) == (lico2_ds == "knee")
        built[needs_fix] = _d_e(pv)
    assert built[True] == pytest.approx(built[False] * ps.AI2020_ELECTROLYTE_DIFFUSIVITY_UNIT_FIX, rel=1e-9)


def test_vacancy_limited_diffusivity_is_d0_times_vacancy_fraction():
    """D(y) = D0 max(1 - y, floor): exact at interior y, floored at full lithiation."""
    d0 = 2.5e-15
    f = ps._with_vacancy_limit(lambda sto, T: d0)
    for y in (0.0, 0.3, 0.9, 0.999):
        val = float(f(pybamm.Scalar(y), pybamm.Scalar(298.15)).evaluate())
        assert val == pytest.approx(d0 * max(1.0 - y, ps.LICO2_VACANCY_FLOOR), rel=1e-12)
    assert float(f(pybamm.Scalar(1.0), pybamm.Scalar(298.15)).evaluate()) == pytest.approx(d0 * ps.LICO2_VACANCY_FLOOR)
