"""Fitted porosities with a closed solid fraction (bol_chain.geometry_overrides): the active
fraction is SOLID_FRACTION - porosity, the fixed N:P and positive capacity are kept by deriving
the negative thickness and the area (synthetic template values only)."""
import pytest

from battery_oracle import bol_chain as bc

F_AH = bc.F_AH
TEMPLATE = {
    "Negative electrode thickness [m]": 80e-6, "Positive electrode thickness [m]": 70e-6,
    bc.EPS_N: 0.60, bc.POROSITY_N: 0.32, bc.EPS_P: 0.62, bc.POROSITY_P: 0.32,
    "Maximum concentration in negative electrode [mol.m-3]": 30000.0,
    "Maximum concentration in positive electrode [mol.m-3]": 50000.0,
    "Separator porosity": 0.5,
}


def _capacities(ov):
    t = {**TEMPLATE, **ov}
    area = ov["Electrode height [m]"] * ov["Electrode width [m]"]
    q_n = t[bc.EPS_N] * t["Negative electrode thickness [m]"] * area * t["Maximum concentration in negative electrode [mol.m-3]"] * F_AH
    q_p = t[bc.EPS_P] * t["Positive electrode thickness [m]"] * area * t["Maximum concentration in positive electrode [mol.m-3]"] * F_AH
    return q_n, q_p


@pytest.mark.parametrize("eps_n, eps_p, scale", [(0.25, 0.35, 1.0), (0.30, 0.30, 0.8), (0.35, 0.25, 1.2)])
def test_closed_volume_fraction_and_fixed_balance(eps_n, eps_p, scale):
    np_ratio, qp_ratio, q_meas = 0.6, 2.2, 0.040
    inputs = {bc.POROSITY_N: eps_n, bc.POROSITY_P: eps_p, bc.THICKNESS_SCALE: scale,
              bc.NP_RATIO: np_ratio, bc.Q_P_RATIO: qp_ratio}
    ov = bc.geometry_overrides(TEMPLATE, inputs, q_meas)
    assert ov[bc.EPS_N] == pytest.approx(bc.SOLID_FRACTION[bc.POROSITY_N] - eps_n)
    assert ov[bc.EPS_P] == pytest.approx(bc.SOLID_FRACTION[bc.POROSITY_P] - eps_p)
    q_n, q_p = _capacities(ov)
    assert q_n / q_p == pytest.approx(np_ratio, rel=1e-12)        # N:P of the OCV balance kept
    assert q_p == pytest.approx(qp_ratio * q_meas, rel=1e-12)     # positive capacity kept
    assert ov["Positive electrode thickness [m]"] == pytest.approx(scale * 70e-6)


def test_without_porosity_inputs_the_old_parametrisation_is_unchanged():
    inputs = {bc.NEG_THICKNESS_RATIO: 1.1, bc.THICKNESS_SCALE: 1.0, bc.NP_RATIO: 0.6, bc.Q_P_RATIO: 2.2}
    ov = bc.geometry_overrides(TEMPLATE, inputs, 0.040)
    assert ov["Negative electrode thickness [m]"] == pytest.approx(1.1 * 70e-6)
    q_n, q_p = _capacities(ov)
    assert q_n / q_p == pytest.approx(0.6, rel=1e-12)
    assert bc.POROSITY_P not in ov                               # positive porosity stays the template's


def test_porosity_mode_swaps_the_thickness_ratio_for_the_porosities(monkeypatch):
    monkeypatch.setenv("ORACLE_FIT_POROSITY", "1")
    b = dict(bc.LITERATURE_BOUNDS)
    b.update(bc.POROSITY_BOUNDS)
    assert set(bc.POROSITY_BOUNDS) == {bc.POROSITY_N, bc.POROSITY_P}
    assert all(lo == 0.25 and hi == 0.35 for lo, hi in bc.POROSITY_BOUNDS.values())
