"""Smoke tests for the tune-oracle engine's pure (non-PyBaMM) functions.

The full calibrate_oracle loop drives PyBaMM and is exercised end-to-end via the
battery_forecast jones2022 adapter; here we test the dataset-free scoring,
target-extraction, and config-writing helpers.
"""
import numpy as np
import pytest

from battery_oracle.tune import (
    _eol_target_cycles,
    _eol_target_cycles_from_range,
    compute_real_targets,
    score_candidate,
    slope_match_error,
    write_oracle_config,
)


def _synthetic_cache():
    #                 R1    P2w  P2n   R3    P4w  P4n   R5    P6w  P6n
    ecm0 = [0.10, 1.0, 0.9, 0.03, 1.0, 0.9, 0.02, 1.0, 0.9]
    ecm1 = [0.11, 1.0, 0.9, 0.03, 1.0, 0.9, 0.02, 1.0, 0.9]  # R1 grew 10%
    return {
        "cell_id": "C01",
        "real_cell_capacity_mah": 200.0,
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
    write_oracle_config(out, "mydata", "accelerated", "C01", 2, best, real, [best])
    assert out.exists()
    text = out.read_text()
    assert "protocol_scaling:" in text
    assert "kinetics_scale: 0.3" in text
    assert "_calibration:" in text
