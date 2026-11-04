"""battery_oracle.physicality and battery_oracle.fit.checks: check classification and the
stage-A physical-validity checks."""
import pytest

from battery_oracle.physicality import (
    _range_check,
    CE_FAIL_REL,
    CE_WARN_REL,
    at_bound,
    electrolyte_status,
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


def test_one_electrolyte_threshold_pair_for_checks_and_fit_floor():
    from battery_oracle.fit.config import CE_FLOOR_DEFAULT

    assert CE_FLOOR_DEFAULT == CE_FAIL_REL < CE_WARN_REL
    assert electrolyte_status(0.5 * CE_FAIL_REL) == "fail"
    assert electrolyte_status(0.5 * (CE_FAIL_REL + CE_WARN_REL)) == "warn"
    assert electrolyte_status(2.0 * CE_WARN_REL) == "pass"


def test_cell_checks_judge_validity_not_accuracy():
    """A cell's physical checks come from its diagnostics; its fit errors are not judged."""
    from battery_oracle.fit.checks import cell_checks

    valid = {"name": "validity[c] lithium conserved", "status": "pass", "value": 0.0}
    rep = {"cell_id": "c", "records": [{"kind": "cycling", "label": "c/1", "rmse_v": 9.9,
                                        "n_simulated": 10, "n_points": 10}],
           "diagnostics": {"validity": [valid], "t_simulated_s": 100.0, "t_total_s": 100.0,
                           "c_e_min_rel": 0.5 * CE_FAIL_REL, "T_max_K": 301.0}}
    by = {c["name"]: c["status"] for c in cell_checks(rep, ambient_k=300.0)}
    assert by == {valid["name"]: "pass", "fit[c] whole history simulated": "pass",
                  "fit[c] cell temperature rise": "pass", "fit[c] electrolyte not depleted": "fail"}
    failed = cell_checks({"cell_id": "c", "diagnostics": {"error": "solver"}})
    assert [c["status"] for c in failed] == ["fail"]


@pytest.mark.slow
def test_demanding_dfn_cycle_on_the_template():
    pytest.importorskip("pybamm")
    from battery_oracle.physicality import check_dfn_cycle

    kw = dict(chemistry="LiR2032", c_min_mA=0.42, c2_min_mA=0.42,
              c_max_mA=189.0, c2_max_mA=189.0, dfn_max_crate=4.5, dur_min_s=1.0,
              ecm_rescale_target_r0=None, temperature_K=296.15)
    checks = check_dfn_cycle(kw)
    names = [c["name"] for c in checks]
    assert names[0].startswith("demanding DFN cycle completes")
    assert checks[0]["status"] != "fail", checks[0]
    assert any(c["name"].startswith("lithium conserved") and c["status"] != "skip"
               for c in checks), checks


def test_fitted_oracle_kwargs_carry_the_fit_mesh_and_thermal():
    from battery_oracle.physicality import fitted_oracle_kwargs

    rep = {"parameter_overrides": {"a": 1.0}, "parameter_scales": {"b": 2.0}, "contact_resistance_ohm": 0.1,
           "thermal": "lumped", "h_total_W_per_m2K": 7.0,
           "radial_mesh": {"points": 10, "kind": "surface", "stretch": 2.0, "x_factor": 3}}
    kw = fitted_oracle_kwargs(rep, {"chemistry": "X", "radial_mesh": {"x_factor": 1}}, ambient_k=300.0)
    assert kw["radial_mesh"] == rep["radial_mesh"]          # the fit's mesh overrides the base
    assert kw["T_ambient_K"] == 300.0 and kw["thermal"] == "lumped" and kw["chemistry"] == "X"
    bare = fitted_oracle_kwargs({k: rep[k] for k in ("parameter_overrides", "parameter_scales",
                                                      "contact_resistance_ohm")}, {"chemistry": "X"})
    assert "radial_mesh" not in bare and "thermal" not in bare
