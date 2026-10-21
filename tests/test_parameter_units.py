"""The Ai2020 electrolyte-diffusivity unit correction is applied only when the installed
PyBaMM still returns cm2/s (algorithmic check on synthetic parameter values)."""
import numpy as np
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


@pytest.mark.parametrize("lico2_ds", ["knee", "vacancy"])
def test_marquis_lico2_ocp_replaces_the_ai2020_table(lico2_ds):
    """lico2_ocp="Marquis2019" installs PyBaMM's Dualfoil LiCoO2 OCP (not the Ai2020 table),
    with the D_s form as requested and no Ai2020 entropic term."""
    pv = ps.lir2032(graphite_ocp="Mohtat2020", lico2_ocp="Marquis2019", lico2_ds=lico2_ds)
    stock = pybamm.ParameterValues("Marquis2019")["Positive electrode OCP [V]"]
    assert pv["Positive electrode OCP [V]"] is stock
    assert float(pv["Positive electrode OCP entropic change [V.K-1]"]) == 0.0
    assert (ps.LICO2_DS_KNEE in pv.keys()) == (lico2_ds == "knee")


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


def test_marquis_thermo_chemistry_uses_the_installed_ocp():
    pv = ps.lir2032(graphite_ocp="Mohtat2020", lico2_ocp="Marquis2019", lico2_ds="thermo")
    assert pv["Positive electrode OCP [V]"] is pybamm.ParameterValues("Marquis2019")["Positive electrode OCP [V]"]
    cmax = float(pv["Maximum concentration in positive electrode [mol.m-3]"])
    c = pybamm.StateVector(slice(0, 1))
    d = pv.process_symbol(pv["Positive particle diffusivity [m2.s-1]"](c / cmax, pybamm.Scalar(298.15)))
    lo, hi = (float(np.ravel(d.evaluate(y=np.array([y * cmax])))[0]) for y in (0.85, 0.96))
    assert hi > 50 * lo            # Theta: small on the plateau, large at the wall


def test_mit_conductivity_follows_the_lever_rule():
    """sigma = sigma_0 below the two-phase region, sigma_0 x ratio above it, log-linear in the
    insulating-phase fraction between; a ratio of 1 is the constant conductivity."""
    lo, hi = ps.LICO2_MIT_RANGE
    f = ps._lico2_mit_conductivity(10.0)
    for ratio in (1.0, 0.1, 1e-3):
        pv = pybamm.ParameterValues({ps.LICO2_SIGMA_RATIO: ratio})
        ev = lambda y: float(np.ravel(pv.process_symbol(f(pybamm.Scalar(y), pybamm.Scalar(298.15))).evaluate())[0])  # noqa: E731
        assert ev(lo - 0.1) == pytest.approx(10.0) and ev(hi + 0.03) == pytest.approx(10.0 * ratio)
        mid = 0.5 * (lo + hi)
        assert ev(mid) == pytest.approx(10.0 * ratio ** 0.5, rel=1e-9)
    assert float(np.ravel(f(pybamm.Scalar(298.15)).evaluate())[0]) == pytest.approx(10.0)
