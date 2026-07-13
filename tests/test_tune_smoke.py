"""Smoke tests for the tune-oracle engine's pure (non-PyBaMM) functions.

The full calibrate_oracle loop drives PyBaMM and is exercised end-to-end via the
battery_forecast jones2022 adapter; here we test the dataset-free scoring,
target-extraction, and config-writing helpers.
"""
import numpy as np
import pytest

from battery_oracle.tune import (
    CALIBRATION_MODEL,
    _eol_target_cycles,
    _eol_target_cycles_from_range,
    collect_eis_comparison,
    compute_real_targets,
    score_candidate,
    slope_match_error,
    write_calibration_summary,
    write_oracle_config,
)


def test_calibration_model_is_spme():
    """#5: calibration always runs on SPMe, independent of experiment-time model."""
    assert CALIBRATION_MODEL == "SPMe"


def _synthetic_cache():
    #                 R1    P2w  P2n   R3    P4w  P4n   R5    P6w  P6n
    ecm0 = [0.10, 1.0, 0.9, 0.03, 1.0, 0.9, 0.02, 1.0, 0.9]
    ecm1 = [0.11, 1.0, 0.9, 0.03, 1.0, 0.9, 0.02, 1.0, 0.9]  # R1 grew 10%
    return {
        "cell_id": "C01",
        "real_cell_capacity_mah": 200.0,
        # ECM vectors above use the legacy 9-param layout; the engine derives
        # its ohmic/arc positions from this declared circuit.
        "circuit": "R1-P2-[R3,P4]-[R5,P6]",
        "cycles": ["0", "1"],
        "first_real_capacity_mah": 200.0,
        "data": {
            "0": {"protocol": [200, 150, 0.16, 0.08, 200, 0.38],
                  "real_capacity_mah": 200.0, "real_soh": 1.0,
                  "ecm_charge": ecm0, "ecm_discharge": ecm0},
            "1": {"protocol": [200, 150, 0.16, 0.08, 200, 0.38],
                  "real_capacity_mah": 198.0, "real_soh": 0.99,
                  "ecm_charge": ecm1, "ecm_discharge": ecm1},
        },
    }


def test_compute_real_targets():
    targets = compute_real_targets(_synthetic_cache())
    # arc = (R3+R5)/R1 = 0.05/0.10 = 0.5 for cycle 0, 0.05/0.11 ≈ 0.4545 for cycle 1
    assert targets["mean_arc_ratio"] == \
        pytest.approx(np.mean([0.5, 0.5, 0.05 / 0.11, 0.05 / 0.11]), rel=1e-6)
    assert targets["r1_growth_pct"] == pytest.approx(10.0, rel=1e-6)
    # real_soh 1.0 -> 0.99 over two cycles => 0.01 SOH loss per cycle
    assert targets["soh_fade_per_cycle"] == pytest.approx(0.01, rel=1e-6)


def test_compute_real_targets_capacity_only():
    """EIS-less cache (null ECMs, per-cycle real_soh): fade target, no ECM targets."""
    cache = _synthetic_cache()
    for cyc in cache["cycles"]:
        cache["data"][cyc]["ecm_charge"] = None
        cache["data"][cyc]["ecm_discharge"] = None
    targets = compute_real_targets(cache)
    assert targets["mean_arc_ratio"] is None
    assert targets["r1_growth_pct"] is None
    assert targets["soh_fade_per_cycle"] == pytest.approx(0.01, rel=1e-6)


def test_compute_real_targets_fade_from_capacity_mah():
    """Fade falls back to real_capacity_mah / reference when real_soh is absent."""
    cache = _synthetic_cache()
    for cyc in cache["cycles"]:
        del cache["data"][cyc]["real_soh"]
    # 200 -> 198 mAh vs first_real_capacity_mah=200 => 0.01 SOH loss per cycle
    assert compute_real_targets(cache)["soh_fade_per_cycle"] == pytest.approx(0.01, rel=1e-6)


def test_score_candidate_perfect_vs_off():
    real = {"mean_arc_ratio": 0.5, "r1_growth_pct": 10.0}
    good = {"oracle_arc_ratio": 0.5, "oracle_r1_growth_pct": 10.0,
            "implied_eol_cycle": 55.0, "crate_probe_skipped": True}
    bad = {"oracle_arc_ratio": 2.0, "oracle_r1_growth_pct": 100.0,
           "implied_eol_cycle": 5.0, "crate_probe_skipped": True}
    s_good = score_candidate(good, real, preset="accelerated")
    s_bad = score_candidate(bad, real, preset="accelerated")
    assert np.isfinite(s_good) and np.isfinite(s_bad)
    assert s_good < s_bad


def test_score_candidate_capacity_only():
    """EIS-less cache: scored on the capacity-fade term alone — finite & ordered."""
    real = {"mean_arc_ratio": None, "r1_growth_pct": None, "soh_fade_per_cycle": 0.005}
    good = {"oracle_arc_ratio": None, "oracle_r1_growth_pct": None,
            "oracle_soh_fade_per_cycle": 0.0051, "crate_probe_skipped": True}
    bad = {"oracle_arc_ratio": None, "oracle_r1_growth_pct": None,
           "oracle_soh_fade_per_cycle": 0.02, "crate_probe_skipped": True}
    s_good = score_candidate(good, real, preset="accelerated")
    s_bad = score_candidate(bad, real, preset="accelerated")
    assert np.isfinite(s_good) and np.isfinite(s_bad)
    assert s_good < s_bad


def test_score_candidate_no_signal_is_inf():
    """Neither ECM targets nor a measured fade rate: nothing to fit -> inf."""
    real = {"mean_arc_ratio": None, "r1_growth_pct": None, "soh_fade_per_cycle": None}
    cand = {"oracle_arc_ratio": None, "oracle_r1_growth_pct": None,
            "oracle_soh_fade_per_cycle": None, "crate_probe_skipped": True}
    assert score_candidate(cand, real, preset="accelerated") == float("inf")


def test_score_candidate_present_target_missing_oracle_is_inf():
    """A real target present but the oracle failed to produce it -> inf (bad cand)."""
    real = {"mean_arc_ratio": 0.5, "r1_growth_pct": 10.0}
    cand = {"oracle_arc_ratio": None, "oracle_r1_growth_pct": 10.0,
            "implied_eol_cycle": 55.0, "crate_probe_skipped": True}
    assert score_candidate(cand, real, preset="accelerated") == float("inf")


def test_collect_eis_comparison_none_when_no_ecm():
    """Capacity-only cache (null ECMs) -> None, short-circuiting before any oracle
    is built (so the auto EIS plot is a graceful no-op for EIS-less datasets)."""
    cache = _synthetic_cache()
    for cyc in cache["cycles"]:
        cache["data"][cyc]["ecm_charge"] = None
        cache["data"][cyc]["ecm_discharge"] = None
    best = {"kinetics_scale": 0.3, "sei_rate_scale": 0.1,
            "dead_li_decay_scale": 10.0, "plating_rate_scale": 1.0}
    assert collect_eis_comparison(cache, best, preset="accelerated") is None


def test_slope_match_error():
    ci = {"ci_lo": 0.05, "ci_hi": 0.15}
    assert slope_match_error(0.10, ci) == 0.0        # inside CI
    assert slope_match_error(0.25, ci) > 0.0         # outside CI
    assert slope_match_error(None, ci) == float("inf")


def test_eol_target_cycles_from_range():
    assert _eol_target_cycles_from_range("40-70") == pytest.approx(55.0)
    assert _eol_target_cycles_from_range("200-400") == pytest.approx(300.0)
    assert _eol_target_cycles_from_range(None) is None
    assert _eol_target_cycles_from_range("") is None


def test_eol_target_cycles_sourced_from_oracle_yaml():
    # Matches config_oracle_defaults.yml's documented preset_constants ranges.
    assert _eol_target_cycles("accelerated") == pytest.approx(55.0)
    assert _eol_target_cycles("severe") == pytest.approx(35.0)
    assert _eol_target_cycles("nominal") == pytest.approx(300.0)


def test_write_oracle_config(tmp_path):
    real = {"mean_arc_ratio": 0.5, "r1_growth_pct": 10.0}
    best = {
        "kinetics_scale": 0.3, "sei_rate_scale": 0.03,
        "dead_li_decay_scale": 10.0, "plating_rate_scale": 0.1,
        "oracle_arc_ratio": 0.48, "oracle_r1_growth_pct": 9.5,
        "implied_eol_cycle": 55.0, "crate_probe_skipped": True,
    }
    out = tmp_path / "config_oracle_test.yml"
    write_oracle_config(out, "mydata", "accelerated", "C01", 2, best, real, [best],
                        chemistry="Prada2013")
    assert out.exists()
    text = out.read_text()
    assert "protocol_scaling:" in text
    assert "kinetics_scale: 0.3" in text
    assert "_calibration:" in text
    # #14: chemistry is emitted for both parameter_set and chemistry.
    assert "parameter_set: Prada2013" in text
    assert "chemistry: Prada2013" in text


def test_write_calibration_summary(tmp_path):
    import json

    real = {"mean_arc_ratio": 0.5, "r1_growth_pct": np.float64(10.0)}
    best = {
        "kinetics_scale": np.float64(0.3), "sei_rate_scale": 0.03,
        "oracle_arc_ratio": 0.48, "oracle_r1_growth_pct": 9.5,
        "implied_eol_cycle": 55.0, "score": np.float64(0.42),
        "trial_number": np.int64(7), "oracle_failure": np.False_,
    }
    out = tmp_path / "calibration_summary.json"
    path = write_calibration_summary(
        out,
        dataset="mydata", preset="accelerated", cell_id="C01",
        n_trials=2, n_cycles=10, crate_sensitivity_min=3.0,
        real_targets=real, real_crate2_slope=None,
        best=best, best_score=0.42,
    )
    assert path == out and out.exists()
    # numpy scalars must round-trip through the _json_default hook
    summary = json.loads(out.read_text())
    assert summary["best"]["trial_number"] == 7
    assert summary["best"]["kinetics_scale"] == pytest.approx(0.3)
    assert summary["best"]["oracle_failure"] is False
    # engine sources the EOL target from the preset YAML, not the caller
    assert summary["eol_target_cycles"] == pytest.approx(55.0)
