"""Stage A as one continuous history (battery_oracle.bol_chain)."""
import numpy as np
import pytest

from battery_oracle.eis_nonstationary import windows_from_stamps


def _pv():
    from battery_oracle.bol_chain import base_values
    from battery_oracle.parameter_sets import lir2032

    pv = lir2032()
    pv.update({"Contact resistance [Ohm]": 0.05}, check_already_exists=False)
    return base_values(pv, 296.15)


def test_windows_from_stamps_follow_measured_times():
    lo, hi = windows_from_stamps([2.0, 2.1, 2.5, 10.0, 60.0])
    assert np.allclose(hi, [2.0, 2.1, 2.5, 10.0, 60.0])
    assert np.allclose(lo[1:], [2.0, 2.1, 2.5, 10.0])
    assert np.isclose(lo[0], 1.9)


def test_initial_stoichiometries_hold_voltage_and_lithium():
    from battery_oracle.bol_chain import _ocp_evaluators, electrode_capacities_ah, initial_stoichiometries

    pv = _pv()
    q_n, q_p = electrode_capacities_ah(pv)
    for v0, f_li in [(3.96, 0.85), (3.70, 0.95), (4.10, 0.80)]:
        x, y = initial_stoichiometries(pv, v0, f_li, 296.15)
        un, up = _ocp_evaluators(pv, 296.15)
        assert abs(up(y) - un(x) - v0) < 1e-6
        assert abs(x * q_n + y * q_p - f_li * q_p) < 1e-9
        assert 0 < x < 1 and 0 < y < 1


def test_lithium_inventory_is_independent_of_np_ratio():
    """Changing the negative active fraction (N:P) at fixed lithium fraction keeps the
    cyclable lithium; only the stoichiometries move (the old coupling is gone)."""
    from battery_oracle.bol_chain import EPS_N, LI_FRACTION, candidate_values, electrode_capacities_ah

    pv = _pv()
    tot = []
    for eps in (0.5, 0.65):
        p = candidate_values(pv, {EPS_N: eps, LI_FRACTION: 0.9}, 3.96)
        q_n, q_p = electrode_capacities_ah(p)
        x = float(p["Initial concentration in negative electrode [mol.m-3]"]) / float(
            p["Maximum concentration in negative electrode [mol.m-3]"])
        y = float(p["Initial concentration in positive electrode [mol.m-3]"]) / float(
            p["Maximum concentration in positive electrode [mol.m-3]"])
        tot.append(x * q_n + y * q_p)
    assert np.isclose(tot[0], tot[1], rtol=1e-9)


def test_impossible_lithium_inventory_is_a_failed_candidate():
    """More cyclable lithium than both electrodes can hold has no balance (a little
    lithium is still a balance: the OCPs' PyBaMM end terms span every voltage)."""
    from battery_oracle.bol_chain import electrode_capacities_ah, initial_stoichiometries

    pv = _pv()
    q_n, q_p = electrode_capacities_ah(pv)
    with pytest.raises(ValueError):
        initial_stoichiometries(pv, 3.96, 1.01 * (q_n + q_p) / q_p, 296.15)


def test_oracle_soc_window_stays_inside_ocp_data_and_nominal_is_measured():
    """The oracle's 0/100% SOC limits must map to stoichiometries inside the OCP data
    (the pilot's 4.2 V limit sent LiCoO2 below its y = 0.4 data edge), and the nominal
    capacity is the measured one so the oracle replays real currents."""
    from battery_oracle.bol_chain import (EPS_N, LI_FRACTION, N_PARALLEL, THICKNESS_SCALE,
                                          overrides_from_chain)
    from battery_oracle.parameter_sets import ocp_data_range

    pv = _pv()
    best = {N_PARALLEL: 0.91, EPS_N: 0.78, THICKNESS_SCALE: 2.0, LI_FRACTION: 0.74}   # pilot PJ248
    ov, _, _, _, esoh = overrides_from_chain(pv, best, 41.9)
    assert "error" not in esoh, esoh
    y_lo, y_hi = ocp_data_range(pv, "positive")
    assert y_lo <= esoh["y_100"] < esoh["y_0"] <= y_hi
    assert 0 <= esoh["x_0"] < esoh["x_100"] <= 1
    assert np.isclose(ov["Nominal cell capacity [A.h]"], 0.0419)
    assert 3.0 <= ov["Open-circuit voltage at 0% SOC [V]"] < ov["Open-circuit voltage at 100% SOC [V]"] <= 4.2


def _synthetic_data(discharge_s: float = 600.0):
    """One group: rest, 1C discharge (``discharge_s``), rest, 1C charge, 20 min rest, EIS
    sweep, a second group after an unrecorded gap with a short discharge."""
    d = float(discharge_s)
    t1 = np.concatenate([np.arange(0, 60, 20.0), np.arange(60, 60 + d, 20.0), np.arange(60 + d, 360 + d, 30.0),
                         np.arange(360 + d, 960 + d, 20.0), np.arange(960 + d, 2160 + d, 60.0)])
    i1 = np.zeros_like(t1)
    i1[(t1 >= 60) & (t1 < 60 + d)] = 0.042
    i1[(t1 >= 360 + d) & (t1 < 960 + d)] = -0.042
    f = np.logspace(2, -1, 8)
    stamps = 2160.0 + d + np.cumsum(2.0 / f) + 1.0
    lo, hi = windows_from_stamps(stamps)
    t2 = np.concatenate([np.arange(0, 60, 20.0), np.arange(60, 360, 10.0), np.arange(360, 960, 60.0)])
    i2 = np.zeros_like(t2)
    i2[(t2 >= 60) & (t2 < 360)] = 0.084
    groups = [{"name": "000", "duration_s": float(hi[-1]) + 1.0, "gap_before": None, "formation": True},
              {"name": "002", "duration_s": float(t2[-1]) + 1.0, "gap_before": "fit", "formation": False}]
    return {"cell_id": "syn", "cell_capacity_mah": 42.0, "groups": groups, "initial_voltage": 3.9,
            "amplitude_a": 5e-3,
            "current": [{"group": 0, "t": t1, "I": i1}, {"group": 1, "t": t2, "I": i2}],
            "records": [{"group": 0, "label": "syn/000_01", "kind": "formation", "t": t1, "V": np.zeros_like(t1)},
                        {"group": 1, "label": "syn/002_02", "kind": "cycling", "t": t2, "V": np.zeros_like(t2)}],
            "spectra": [{"group": 0, "label": "syn/000_02", "f": f, "Z": np.ones(f.size, complex),
                         "t_start": lo, "t_end": hi}]}


def test_offsets_insert_fitted_gap():
    from battery_oracle.bol_chain import ChainModel, gap_name

    data = _synthetic_data()
    chain = ChainModel(data, None)
    off = chain.offsets({gap_name("002"): 500.0})
    assert off[0] == 0.0 and np.isclose(off[1], data["groups"][0]["duration_s"] + 500.0)
    t, i = chain.current_knots(off)
    assert np.all(np.diff(t) > 0)
    assert np.isclose(np.interp(off[1] + 100.0, t, i), 0.084)


@pytest.mark.slow
def test_chain_simulates_history_with_heating_and_spectrum():
    from battery_oracle.bol_chain import ChainModel, gap_name

    data = _synthetic_data()
    res = ChainModel(data, _pv()).solve({gap_name("002"): 3600.0})
    assert "error" not in res, res.get("error")
    assert [len(r) for r in res["records"]] == [len(r["t"]) for r in data["records"]]
    z = res["spectra"][0]
    assert np.all(np.isfinite(z)) and np.all(z.real > 0)
    d = res["diag"]
    assert d["T_max_K"] > 296.15 + 1e-3              # the 1C steps heat the cell
    assert d["ocp"]["positive"]["fraction_out"] == 0.0


@pytest.mark.slow
def test_chain_fit_recovers_known_parameters_from_synthetic_data():
    """Round trip: data simulated by the chain model at known parameters; the fit
    (5 free parameters incl. lithium inventory and the unrecorded gap, perturbed
    start) recovers them. The discharge reaches the LiCoO2 wall, where the thermodynamic-
    factor diffusivity is felt: doubling D0 moves the voltage by ~50 mV there, against
    ~4 mV for a 600 s discharge on the plateau (probe 2026-10-30)."""
    from battery_oracle.bol_chain import (CONTACT_R, LI_FRACTION, ChainModel, fit_cell_chain, gap_name,
                                          lithium_fraction)
    from battery_oracle.params import scale_parameter_name

    pv = _pv()
    data = _synthetic_data(discharge_s=1800.0)
    f_li0 = lithium_fraction(pv)
    free = [scale_parameter_name("Negative electrode exchange-current density [A.m-2]"),
            scale_parameter_name("Positive particle diffusivity [m2.s-1]"), CONTACT_R, LI_FRACTION,
            gap_name("002")]
    truth = {free[0]: 2.0, free[1]: 0.5, free[2]: 0.15, free[3]: f_li0 + 0.03, free[4]: 1800.0}
    res = ChainModel(data, pv).solve(truth)
    assert "error" not in res, res.get("error")
    for r, v in zip(data["records"], res["records"]):
        r["V"] = np.asarray(v)
    data["spectra"][0]["Z"] = res["spectra"][0]

    out = fit_cell_chain("syn", data, pv, warm={CONTACT_R: 0.05, gap_name("002"): 3600.0},
                         max_iterations=60, max_unchanged=20, seed=1, free=free, ambient_k=296.15)
    best = out["best_inputs"]
    assert out["cost"] < 1.0, out["cost"]
    for name in free[:3]:
        assert 0.6 * truth[name] < best[name] < 1.6 * truth[name], (name, best[name], truth[name])
    assert abs(best[LI_FRACTION] - truth[LI_FRACTION]) < 0.02, (best[LI_FRACTION], truth[LI_FRACTION])


@pytest.mark.slow
def test_oracle_lumped_thermal_uses_coin_cell_heat_capacity():
    from battery_oracle.oracle import PyBaMMOracle, _randles_stub_ecm

    o = PyBaMMOracle(ecm_model_fn=_randles_stub_ecm, chemistry="LiR2032", model="DFN", thermal="lumped",
                     T_ambient_K=296.15, h_total_W_per_m2K=12.0)
    assert o._deg_opts.get("use lumped thermal capacity") == "true"
    assert float(o._pv["Total heat transfer coefficient [W.m-2.K-1]"]) == 12.0
    assert float(o._pv["Cell volume [m3]"]) < 2e-6            # the coin can, not the pouch


def test_lico2_ocp_is_monotone_and_ocp_data_range_reports_the_data():
    """The set's LiCoO2 OCP falls monotonically over the stoichiometry; ocp_data_range is
    (0, 1) for an analytic OCP and the x range of an interpolated one (synthetic table)."""
    import pybamm

    from battery_oracle.bol_chain import _ocp_evaluators
    from battery_oracle.parameter_sets import ocp_data_range

    pv = _pv()
    _, up = _ocp_evaluators(pv, 298.15)
    ys = np.linspace(0.01, 0.99, 197)
    u = np.array([up(y) for y in ys])
    assert np.all(np.diff(u) < 0), ys[1:][np.diff(u) >= 0]
    assert ocp_data_range(pv, "positive") == (0.0, 1.0)
    x = np.linspace(0.2, 0.9, 15)
    table = pybamm.ParameterValues({"Positive electrode OCP [V]":
                                    lambda sto: pybamm.Interpolant(x, 4.2 - x, sto, name="synthetic_ocp")})
    assert ocp_data_range(table, "positive") == pytest.approx((0.2, 0.9))


def test_negative_volume_fractions_sum_to_one():
    """Fitting the negative active fraction moves the porosity with it (pilot 2 fitted
    eps_n = 0.797 with the template porosity 0.33: 1.13 of the electrode volume)."""
    from battery_oracle.bol_chain import EPS_N, POROSITY_N, candidate_values, fit_parameters_chain

    pv = _pv()
    filler = 1 - float(pv[EPS_N]) - float(pv[POROSITY_N])
    for eps in (0.45, 0.6, 0.74):
        p = candidate_values(pv, {EPS_N: eps}, 3.96)
        assert np.isclose(float(p[EPS_N]) + float(p[POROSITY_N]) + filler, 1.0)
        assert float(p[POROSITY_N]) >= 0.19
    fit = fit_parameters_chain(pv, {"groups": []})
    assert fit[EPS_N].bounds[1] <= 0.74 + 1e-9


def test_literature_bounds_and_derived_microstructure():
    """Every fitted parameter's prior is a published range; the derived Bruggeman
    exponents reproduce the fitted tortuosity / MacMullin number; the area follows the
    positive capacity ratio."""
    from battery_oracle.bol_chain import (BRUGG_N, BRUGG_S, EPS_N, LITERATURE_BOUNDS, MACMULLIN_S, POROSITY_N,
                                          Q_P_RATIO, THICKNESS_SCALE, TORTUOSITY_N, electrode_capacities_ah,
                                          candidate_values, fit_parameters_chain)

    pv = _pv()
    fit = fit_parameters_chain(pv, {"groups": []})
    for k, (lo, hi) in LITERATURE_BOUNDS.items():
        assert np.allclose(fit[k].bounds, (lo, hi)), k
    x = {EPS_N: 0.65, TORTUOSITY_N: 6.0, MACMULLIN_S: 4.0, Q_P_RATIO: 1.8, THICKNESS_SCALE: 0.8}
    p = candidate_values(pv, x, 3.96, 0.042)
    eps_n, b_n = float(p[POROSITY_N]), float(p[BRUGG_N])
    assert np.isclose(eps_n ** (1 - b_n), 6.0)
    eps_s, b_s = float(p["Separator porosity"]), float(p[BRUGG_S])
    assert np.isclose(eps_s ** (-b_s), 4.0)
    _, q_p = electrode_capacities_ah(p)
    assert np.isclose(q_p, 1.8 * 0.042)
    assert np.isclose(float(p["Positive electrode thickness [m]"]), 0.8 * float(pv["Positive electrode thickness [m]"]))


def test_ocv_balance_recovers_a_known_balance():
    """Rest voltages generated from a known balance with the set's own OCPs are mapped
    back to that balance (two-step stage A)."""
    from battery_oracle.bol_chain import _ocp_evaluators, ocv_balance

    pv = _pv()
    un, up = _ocp_evaluators(pv, 298.15)
    x0, y0, qn, qp = 0.70, 0.55, 0.050, 0.100
    qs = np.array([0.0, 0.01, 0.02, 0.03, 0.035, 0.015, 0.0, 0.025, 0.005])
    t = np.arange(qs.size) * 2000.0
    # a single group whose current knots move exactly the charges qs between rests
    tk, ik = [0.0], [0.0]
    for k in range(1, qs.size):
        dq = qs[k] - qs[k - 1]
        t0 = t[k - 1] + 100.0
        tk += [t0, t0 + 0.05, t0 + 600.05, t0 + 600.1]
        ik += [0.0, dq * 3600.0 / 600.0, dq * 3600.0 / 600.0, 0.0]
    tk.append(t[-1] + 1000.0); ik.append(0.0)
    rec_t = np.concatenate([np.linspace(t[k - 1] + 700.2, t[k] + 99.0, 5) for k in range(1, qs.size)])
    v = []
    for tt in rec_t:
        k = int(np.searchsorted(t, tt - 99.0)) - 0
        q = qs[min(k, qs.size - 1)] if tt > t[0] + 700 else 0.0
        v.append(up(y0 + q / qp) - un(x0 - q / qn))
    data = {"groups": [{"name": "000", "duration_s": t[-1] + 1000.0, "gap_before": None}],
            "current": [{"group": 0, "t": np.array(tk), "I": np.array(ik)}],
            "records": [{"group": 0, "t": rec_t, "V": np.array(v), "kind": "cycling", "label": "syn"}],
            "initial_voltage": up(y0) - un(x0)}
    b = ocv_balance(data, pv)
    assert b["rmse_mV"] < 1.0, b
    # rest points over a partial window need not pin the balance: the truth must lie
    # inside the reported near-optimal range (identifiability is reported, not assumed)
    lo, hi = b["np_ratio_near_range"]
    assert lo - 0.02 <= qn / qp <= hi + 0.02, b


def test_ocv_balance_constrained_stays_in_literature():
    """With the measured capacity, the balance stays inside the literature limits even
    when the rest voltages alone prefer a graphite smaller than the cell (PJ248 with
    Ai2020); the unconstrained fit is reported beside it."""
    from battery_oracle.bol_chain import (GRAPHITE_USABLE, LI_FRACTION, LITERATURE_BOUNDS, NP_PRACTICAL_BOUNDS,
                                          Q_P_RATIO, _ocp_evaluators, ocv_balance)

    pv = _pv()
    un, up = _ocp_evaluators(pv, 298.15)
    q_meas = 0.042
    x0, y0, qn, qp = 0.95, 0.50, 0.040, 0.110          # unphysical truth: practical N:P 0.89, Q_p ratio 2.6
    qs = np.array([0.0, 0.01, 0.02, 0.03, 0.035, 0.015, 0.0])
    t = np.arange(qs.size) * 2000.0
    tk, ik = [0.0], [0.0]
    for k in range(1, qs.size):
        dq = qs[k] - qs[k - 1]
        t0 = t[k - 1] + 100.0
        tk += [t0, t0 + 0.05, t0 + 600.05, t0 + 600.1]
        ik += [0.0, dq * 3600.0 / 600.0, dq * 3600.0 / 600.0, 0.0]
    tk.append(t[-1] + 1000.0); ik.append(0.0)
    rec_t = np.concatenate([np.linspace(t[k - 1] + 700.2, t[k] + 99.0, 5) for k in range(1, qs.size)])
    v = [up(y0 + (qs[min(int(np.searchsorted(t, tt - 99.0)), qs.size - 1)] if tt > t[0] + 700 else 0.0) / qp)
         - un(x0 - (qs[min(int(np.searchsorted(t, tt - 99.0)), qs.size - 1)] if tt > t[0] + 700 else 0.0) / qn)
         for tt in rec_t]
    data = {"groups": [{"name": "000", "duration_s": t[-1] + 1000.0, "gap_before": None}],
            "current": [{"group": 0, "t": np.array(tk), "I": np.array(ik)}],
            "records": [{"group": 0, "t": rec_t, "V": np.array(v), "kind": "cycling", "label": "syn"}],
            "initial_voltage": up(y0) - un(x0), "cell_capacity_mah": 1000 * q_meas}
    b = ocv_balance(data, pv)
    assert b["constrained"] and "unconstrained" in b
    tol = 1e-6
    lo, hi = LITERATURE_BOUNDS[Q_P_RATIO]
    assert lo - tol <= b["Q_p_Ah"] / q_meas <= hi + tol, b
    lo, hi = NP_PRACTICAL_BOUNDS
    assert lo - tol <= GRAPHITE_USABLE * b["Q_n_Ah"] / q_meas <= hi + tol, b
    lo, hi = LITERATURE_BOUNDS[LI_FRACTION]
    assert lo - 0.01 <= b["li_fraction"] <= hi + 0.01, b
    assert b["rmse_mV"] >= b["unconstrained"]["rmse_mV"] - 1e-6


def test_np_ratio_sets_the_negative_active_fraction():
    from battery_oracle.bol_chain import (EPS_N, NEG_THICKNESS_RATIO, NP_RATIO, POROSITY_N, THICKNESS_SCALE,
                                          candidate_values, electrode_capacities_ah)

    pv = _pv()
    p = candidate_values(pv, {NP_RATIO: 0.6, NEG_THICKNESS_RATIO: 0.9, THICKNESS_SCALE: 1.1}, 3.96, 0.042)
    q_n, q_p = electrode_capacities_ah(p)
    assert np.isclose(q_n / q_p, 0.6)
    assert np.isclose(float(p["Negative electrode thickness [m]"]), 0.9 * float(p["Positive electrode thickness [m]"]))
    assert np.isclose(float(p[EPS_N]) + float(p[POROSITY_N]), 1 - (1 - float(pv[EPS_N]) - float(pv[POROSITY_N])))
