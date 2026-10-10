"""battery_oracle.bol_fit: PyBOP-method balancing and EIS fitting.

Synthetic round trips: data generated from known values with the same model forms
must be recovered. The full jones2022 fit runs on a compute node (battery-forecast
scripts/oracle_recal/02_bol_fit.sh).
"""
import numpy as np
import pytest

pybamm = pytest.importorskip("pybamm")

from battery_oracle.bol_fit import (  # noqa: E402
    BALANCE_PARAMS,
    _electrode_capacity_ah,
    initial_stoichiometries,
    ocp_interpolants,
)


@pytest.fixture(scope="module")
def pv():
    p = pybamm.ParameterValues("Chen2020")
    p.update({"Lower voltage cut-off [V]": 3.0, "Upper voltage cut-off [V]": 4.3})
    return p


def test_initial_stoichiometries_monotone_in_voltage(pv):
    x_lo, y_lo = initial_stoichiometries(pv, 3.6)
    x_hi, y_hi = initial_stoichiometries(pv, 4.0)
    assert 0 < x_lo < x_hi < 1          # more lithium in the negative when fuller
    assert 0 < y_hi < y_lo < 1
    x_top, _ = initial_stoichiometries(pv, 4.5)   # clipped inside the limits
    assert x_top >= x_hi


def test_electrode_capacity_matches_nominal_scale(pv):
    # Chen2020 is a 5 Ah cell; each electrode holds more than that over its full
    # 0->1 stoichiometry range (the cell uses only part of it).
    for d in ("negative", "positive"):
        assert 5.0 < _electrode_capacity_ah(pv, d) < 10.0


def _synthetic_branches(pv, q_th_n=6.5, q_th_p=7.0, sto_n_0=0.03, sto_p_100=0.25, q=5.0):
    """Charge then discharge at 1 A over the full window, no overpotential."""
    pos, neg = ocp_interpolants(pv)
    soc = np.linspace(0.0, 1.0, 400)
    sto_n = sto_n_0 + soc * q / q_th_n
    sto_p = sto_p_100 + (1 - soc) * q / q_th_p
    ocv = np.asarray(pos(sto_p)) - np.asarray(neg(sto_n))
    t = soc * q * 3600.0                           # 1 A for q Ah
    charge = {"t": t, "I": -np.ones_like(t), "V": ocv, "direction": "charge",
              "v_relax": float(ocv[-1]), "label": "syn/charge"}
    discharge = {"t": t, "I": np.ones_like(t), "V": ocv[::-1], "direction": "discharge",
                 "v_relax": float(ocv[0]), "label": "syn/discharge"}
    return [charge, discharge]


def test_fit_balance_recovers_synthetic_window(pv):
    pytest.importorskip("pybop")
    from battery_oracle.bol_fit import fit_balance

    out = fit_balance(_synthetic_branches(pv), pv, maxiter=60, seed=1)
    assert all(b["rmse_mV"] < 5.0 for b in out["branches"])
    assert all(b["CE"] == pytest.approx(1.0, abs=1e-6) for b in out["branches"])
    assert out["series_resistance_ohm"] == pytest.approx(0.0, abs=5e-3)
    assert out["theoretical_capacity_ah"]["negative"] == pytest.approx(6.5, rel=0.1)
    assert out["theoretical_capacity_ah"]["positive"] == pytest.approx(7.0, rel=0.1)
    assert set(out["overrides"]) == set(BALANCE_PARAMS)
    w = out["window"]
    assert 0 <= w["x_0"] < w["x_100"] <= 1 and 0 <= w["y_100"] < w["y_0"] <= 1


@pytest.mark.slow
def test_eis_fit_pieces_round_trip(pv):
    """Synthetic spectrum at known multipliers: zero RMSE there, larger elsewhere."""
    pytest.importorskip("pybop")
    from battery_oracle.bol_fit import (
        CONTACT_R,
        EIS_SCALES,
        _eis_fit_params,
        _eis_problem,
        spectrum_rmse,
    )
    from battery_oracle.params import scale_parameter_name

    f = np.logspace(-1, 3, 15)
    truth = {**{scale_parameter_name(n): 1.0 for n in EIS_SCALES}, CONTACT_R: 2e-3}
    truth[scale_parameter_name(EIS_SCALES[0])] = 3.0
    probe = {"f": f, "Z": np.zeros_like(f, dtype=complex), "v_rest": 3.9, "label": "syn"}
    # "Measured" spectrum generated through the same simulator path.
    problem = _eis_problem(probe, pv, _eis_fit_params((1e-3, 1e3), (1e-7, 10.0), 1e-3))
    z_true = np.asarray(problem.simulator.solve(inputs=truth)["Impedance"].data)
    spec = {**probe, "Z": z_true}
    assert spectrum_rmse(spec, pv, truth) == pytest.approx(0.0, abs=1e-9)
    off = dict(truth)
    off[scale_parameter_name(EIS_SCALES[0])] = 1.0
    assert spectrum_rmse(spec, pv, off) > 1e-5


def test_balanced_window_is_reproduced_by_pybamm_esoh(pv):
    """The handoff: PyBaMM's electrode-SOH solve on the balanced parameters must
    give back the window PyBOP's balancing read (same 0% / 100% OCV definition)."""
    pytest.importorskip("pybop")
    from battery_oracle.bol_fit import fit_balance
    from battery_oracle.params import apply_parameter_overrides

    out = fit_balance(_synthetic_branches(pv), pv, maxiter=60, seed=1)
    pvb = apply_parameter_overrides(pv.copy(), out["overrides"])
    x0, x100, y100, y0 = (float(v) for v in pybamm.lithium_ion.get_min_max_stoichiometries(pvb))
    w = out["window"]
    assert x100 == pytest.approx(w["x_100"], abs=0.02)
    assert y100 == pytest.approx(w["y_100"], abs=0.02)
    assert x0 == pytest.approx(w["x_0"], abs=0.02)
    assert y0 == pytest.approx(w["y_0"], abs=0.02)
