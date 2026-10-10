"""Step-shape calibration pieces: BOL parameter scales, GCPL segmentation,
step summaries/match error, and the oracle's discharge_mode / step_stats.

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
from battery_oracle.tune import (
    STEP_FRAC_FLOOR,
    score_candidate,
    segment_gcpl_steps,
    step_match_error,
    summarise_steps,
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


# --------------------------------------------------------------------------- segmentation

def _record():
    """rest 10 s | CC1 150 mA 100 s (last sample 142 mA) | rest 10 s | CC2 40 mA 60 s | rest."""
    t = np.arange(0, 400, 2.0)
    i = np.zeros_like(t)
    i[(t >= 10) & (t < 110)] = 150.0
    i[t == 108] = 142.0                      # potentiostat's averaged cutoff sample
    i[(t >= 120) & (t < 180)] = 40.0
    v = np.linspace(3.9, 4.2, t.size)
    return t, i, v


def test_segment_merges_cutoff_sample_into_its_stage():
    t, i, v = _record()
    rows = segment_gcpl_steps(t, i, v, capacity_mah=42.0)
    kinds = [r["kind"] for r in rows]
    assert kinds == ["rest", "chg", "rest", "chg", "rest"]
    cc1, cc2 = rows[1], rows[3]
    assert math.isclose(cc1["crate"], 150.0 / 42.0, rel_tol=0.01)
    # ~150 mA x 100 s = 4.17 mAh -> ~0.099 of 42 mAh
    assert 0.09 < cc1["frac"] < 0.105
    assert math.isclose(cc2["current_mA"], 40.0)


def test_segment_splits_direct_current_change():
    t = np.arange(0, 100, 1.0)
    i = np.where(t < 50, 150.0, 80.0)
    rows = segment_gcpl_steps(t, i, np.full(t.size, 4.0), 40.0)
    assert [r["kind"] for r in rows] == ["chg", "chg"]


# --------------------------------------------------------------------------- summary / error

def _rows(frac_chg, frac_dis, rest_v, crate=3.0):
    rows = [{"kind": "chg", "crate": crate, "frac": f, "duration_s": 100.0} for f in frac_chg]
    rows += [{"kind": "dis", "crate": 1.2, "frac": f, "duration_s": 100.0} for f in frac_dis]
    rows += [{"kind": "rest", "crate": 0.0, "frac": 0.0, "duration_s": 1200.0, "v_end": v}
             for v in rest_v]
    return rows


def test_summary_ignores_short_rests():
    rows = _rows([0.2] * 5, [0.1] * 5, [4.0] * 5)
    rows.append({"kind": "rest", "crate": 0.0, "frac": 0.0, "duration_s": 10.0, "v_end": 3.0})
    s = summarise_steps(rows)
    assert s["rest_v"]["n"] == 5 and s["rest_v"]["p5"] == pytest.approx(4.0)


def test_step_match_error_zero_for_identical_and_grows_with_gap():
    real = summarise_steps(_rows([0.25] * 30, [0.5] * 30, np.linspace(3.8, 4.2, 30)))
    assert step_match_error(real, real) == pytest.approx(0.0)
    starved = summarise_steps(_rows([0.01] * 30, [0.5] * 30, np.linspace(3.8, 4.2, 30)))
    low_soc = summarise_steps(_rows([0.25] * 30, [0.5] * 30, np.linspace(3.0, 3.4, 30)))
    e_starved = step_match_error(starved, real)
    assert e_starved == pytest.approx(
        np.mean([math.log((0.25 + STEP_FRAC_FLOOR) / (0.01 + STEP_FRAC_FLOOR))] * 2 + [0, 0]))
    assert step_match_error(low_soc, real) == pytest.approx(1.0)   # rest band fully outside


def test_step_match_error_none_without_target_inf_without_overlap():
    real = summarise_steps(_rows([0.25] * 30, [0.5] * 30, [4.0] * 30))
    assert step_match_error(real, None) is None
    other_bin = summarise_steps(_rows([0.25] * 30, [], [4.0] * 30, crate=5.5))
    no_dis = {**other_bin, "dis": [{**b, "n": 0} for b in other_bin["dis"]]}
    assert step_match_error(no_dis, real) == math.inf


def test_score_candidate_adds_weighted_step_and_fade_terms():
    real_targets = {"mean_arc_ratio": 10.0, "r1_growth_pct": 5.0, "soh_fade_per_cycle": 0.004}
    result = {"oracle_arc_ratio": 10.0, "oracle_r1_growth_pct": 5.0, "implied_eol_cycle": 55.0,
              "oracle_soh_fade_per_cycle": 0.002, "crate_probe_skipped": True,
              "step_match_error": 0.5}
    base = score_candidate(result, real_targets)
    assert score_candidate(result, real_targets, step_weight=2.0) == pytest.approx(base + 1.0)
    assert score_candidate(result, real_targets, fit_fade=True) == pytest.approx(
        base + abs(math.log(0.5)))


# --------------------------------------------------------------------------- oracle (PyBaMM)

@pytest.mark.slow
def test_oracle_protocol_discharge_mode_honours_duration_and_records_steps():
    from battery_oracle.oracle import PyBaMMOracle, _randles_stub_ecm

    proto = np.array([150.0, 60.0, 0.25, 0.25, 60.0, 0.05])   # 3 min discharge request
    kw = dict(ecm_model_fn=_randles_stub_ecm, real_cell_capacity_mah=42.0,
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
    from battery_oracle.oracle import PyBaMMOracle, _randles_stub_ecm

    proto = np.array([150.0, 60.0, 0.25, 0.25, 60.0, 0.05])
    kw = dict(ecm_model_fn=_randles_stub_ecm, real_cell_capacity_mah=42.0,
              c_max_mA=22_500.0, c2_max_mA=22_500.0, dur_min_s=1.0, discharge_mode="protocol")
    ref = PyBaMMOracle(**kw)
    ref(proto)
    resist = PyBaMMOracle(contact_resistance_ohm=0.05, **kw)
    resist(proto)
    q = lambda o: o.history[-1]["step_stats"][2]["frac"]  # noqa: E731
    assert q(resist) < q(ref)
