"""battery_oracle.physicality: check classification and the stage-A / stage-B reports."""
import pytest

from battery_oracle.physicality import (
    _range_check,
    at_bound,
    check_tune,
    summarise,
)


def test_at_bound_linear_and_log():
    assert at_bound(1.0, 1.0, 2.0) == "lower"
    assert at_bound(1.995, 1.0, 2.0) == "upper"
    assert at_bound(1.5, 1.0, 2.0) is None
    assert at_bound(99.0, 0.01, 100.0, log_scale=True) == "upper"
    assert at_bound(1.0, 0.01, 100.0, log_scale=True) is None


def test_range_check_pass_warn_fail():
    assert _range_check("x", 1.0, 0.5, 2.0, "")["status"] == "pass"
    assert _range_check("x", 5.0, 0.5, 2.0, "")["status"] == "warn"
    assert _range_check("x", 50.0, 0.5, 2.0, "")["status"] == "fail"
    assert _range_check("x", float("nan"), 0.5, 2.0, "")["status"] == "skip"


def test_summarise_takes_the_worst_and_ignores_skips():
    checks = [{"status": "pass"}, {"status": "skip"}, {"status": "warn"}]
    assert summarise(checks)["status"] == "warn"
    assert summarise(checks + [{"status": "fail"}])["status"] == "fail"
    assert summarise([{"status": "skip"}])["status"] == "pass"


def test_check_tune_flags_bounds_gains_and_failed_targets():
    best = {"sei_rate_scale": 0.01005, "plating_rate_scale": 0.5, "kinetics_scale": 1.0}
    ranges = {"srs_min": 0.01, "srs_max": 1.0, "prs_min": 0.01, "prs_max": 10.0,
              "ks_min": 1.0, "ks_max": 1.0}
    rep = check_tune(best, ranges, {"crate2_slope": "FAILED — outside CI",
                                    "capacity_fade": "partial — off"},
                     soh_history=[1.0, 0.999, 0.9995])
    by = {c["name"]: c["status"] for c in rep["checks"]}
    assert by["sei_rate_scale not at search bound"] == "warn"
    assert by["plating_rate_scale not at search bound"] == "pass"
    assert "kinetics_scale not at search bound" not in by          # pinned range skipped
    assert by["no capacity gain between cycles (max SOH increase)"] == "warn"
    assert by["calibration target: crate2_slope"] == "fail"
    assert by["calibration target: capacity_fade"] == "warn"
    assert rep["status"] == "fail"


def test_check_balance_flags_invalid_window_negative_r0_and_bounds():
    pybamm = pytest.importorskip("pybamm")
    from battery_oracle.parameter_sets import lir2032
    from battery_oracle.physicality import check_balance

    pv = lir2032()
    side = float(pv["Electrode height [m]"])
    report = {
        "fit": {"balance": {"PJ1": {
            "window_valid": False, "window": {"x_100": 1.2},
            "theoretical_capacity_ah": {"negative": 0.063, "positive": 0.050},
            "bounds": {"theoretical_capacity_ah": [0.042, 0.063], "min_stoichiometry": [0, 0.4],
                       "series_resistance_ohm": [-1, 1]},
            "series_resistance_ohm": -0.2,
            "branches": [{"direction": "charge", "Q_soc_ah": 0.041, "min_sto_n": 0.1,
                          "min_sto_p": 0.399}],
        }}},
        "parameter_overrides": {"Electrode height [m]": side, "Electrode width [m]": side,
                                "Negative electrode thickness [m]":
                                    3.0 * float(pv["Negative electrode thickness [m]"])},
    }
    by = {c["name"]: c["status"] for c in check_balance(report, pv)}
    assert by["balance[PJ1] stoichiometry window in [0, 1]"] == "fail"
    assert by["balance[PJ1] negative theoretical capacity not at bound"] == "warn"
    assert by["balance[PJ1] positive theoretical capacity not at bound"] == "pass"
    assert by["balance[PJ1] series resistance >= 0"] == "warn"
    assert by["balance[PJ1] series resistance not at bound"] == "pass"
    assert by["balance[PJ1] charge min_sto_p not at bound"] == "warn"
    assert by["geometry: electrode area / template"] == "pass"
    assert by["geometry: negative thickness / template"] == "warn"


def test_fitted_values_and_esoh_on_the_template_pass():
    pytest.importorskip("pybamm")
    from battery_oracle.parameter_sets import lir2032
    from battery_oracle.physicality import check_esoh, evaluate_parameter, PLAUSIBILITY_WINDOWS

    pv = lir2032()
    for name, kind, lo, hi, _ in PLAUSIBILITY_WINDOWS:
        v = evaluate_parameter(pv, name, kind)
        assert v > 0, name
    (esoh,) = check_esoh(pv)
    assert esoh["status"] == "pass", esoh


@pytest.mark.slow
def test_demanding_dfn_cycle_on_the_template():
    pytest.importorskip("pybamm")
    from battery_oracle.physicality import check_dfn_cycle

    kw = dict(chemistry="LiR2032", model="DFN", c_min_mA=0.42, c2_min_mA=0.42,
              c_max_mA=189.0, c2_max_mA=189.0, dfn_max_crate=4.5, dur_min_s=1.0,
              ecm_rescale_target_r0=None, temperature_K=296.15)
    checks = check_dfn_cycle(kw)
    names = [c["name"] for c in checks]
    assert names[0].startswith("demanding DFN cycle completes")
    assert checks[0]["status"] != "fail", checks[0]
    assert any(c["name"].startswith("lithium conserved") and c["status"] != "skip"
               for c in checks), checks
