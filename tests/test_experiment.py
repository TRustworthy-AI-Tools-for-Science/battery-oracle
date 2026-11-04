"""Tests for the experiment-protocol loader/runner.

The parse + kwargs-mapping layer is PyBaMM-free, so everything here runs without
a simulation; the actual run_experiment execution is exercised as a `slow` test.
"""
import copy

import numpy as np
import pytest

from battery_oracle.experiment import (
    _PROTOCOL_FIELDS,
    _validate_config,
    build_oracle_from_config,
    load_experiment_config,
    load_oracle_config,
    oracle_kwargs_from_config,
    oracle_kwargs_from_oracle_config,
    protocols_from_config,
)


def test_load_packaged_default():
    cfg = load_experiment_config(None)
    for section in ("cycling", "degradation", "eis", "protocols"):
        assert section in cfg
    assert len(cfg["protocols"]) >= 1


def test_oracle_kwargs_mapping():
    cfg = load_experiment_config(None)
    kw = oracle_kwargs_from_config(cfg)
    # core fields
    assert kw["degradation_preset"] == cfg["degradation"]["preset"]
    assert kw["n_cycles"] == int(cfg["cycling"]["n_cycles"])
    assert kw["temperature_K"] == float(cfg["cycling"]["temperature_K"])
    # parameter set carried as a plain string (PyBaMM-free layer)
    assert kw["parameter_set"] == cfg["cycling"]["parameter_set"]
    assert isinstance(kw["parameter_set"], str)
    # frequencies converted to a logspace array with the right endpoints
    freqs = kw["frequencies"]
    assert isinstance(freqs, np.ndarray)
    assert len(freqs) == int(cfg["eis"]["n_freq_points"])
    assert freqs.min() == pytest.approx(float(cfg["eis"]["freq_min_hz"]))
    assert freqs.max() == pytest.approx(float(cfg["eis"]["freq_max_hz"]))


def test_scale_defaults_when_omitted():
    cfg = load_experiment_config(None)
    for f in ("kinetics_scale", "sei_rate_scale", "dead_li_decay_scale", "plating_rate_scale"):
        cfg["degradation"].pop(f, None)
    kw = oracle_kwargs_from_config(cfg)
    for f in ("kinetics_scale", "sei_rate_scale", "dead_li_decay_scale", "plating_rate_scale"):
        assert kw[f] == 1.0


def test_ecm_fitter_default_and_mapping():
    cfg = load_experiment_config(None)
    # packaged default declares 'randles' (works on the core install)
    assert oracle_kwargs_from_config(cfg)["ecm_fitter"] == "randles"
    # explicit autoeis is carried through
    cfg["ecm"] = {"fitter": "autoeis"}
    assert oracle_kwargs_from_config(cfg)["ecm_fitter"] == "autoeis"
    # omitted 'ecm' section defaults to randles
    cfg.pop("ecm", None)
    assert oracle_kwargs_from_config(cfg)["ecm_fitter"] == "randles"


def test_bad_ecm_fitter_rejected():
    cfg = load_experiment_config(None)
    cfg["ecm"] = {"fitter": "nope"}
    with pytest.raises(ValueError):
        _validate_config(cfg)


def test_protocols_shape_and_order():
    cfg = load_experiment_config(None)
    protos = protocols_from_config(cfg)
    assert len(protos) == len(cfg["protocols"])
    for vec, proto in zip(protos, cfg["protocols"]):
        assert vec.shape == (6,)
        assert vec.dtype == np.float64
        expected = [float(proto[f]) for f in _PROTOCOL_FIELDS]
        assert list(vec) == expected


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["degradation"].__setitem__("preset", "boom"),  # unknown preset
        lambda c: c.__setitem__("protocols", []),               # empty protocols
        lambda c: c["protocols"][0].pop("dur_1_h"),             # missing field
        lambda c: c.pop("eis"),                                  # missing section
    ],
)
def test_validation_errors(mutate):
    cfg = copy.deepcopy(load_experiment_config(None))
    mutate(cfg)
    with pytest.raises((ValueError, KeyError)):
        _validate_config(cfg)


def test_load_oracle_config_packaged_default():
    oc = load_oracle_config(None)
    for section in ("cycling", "solver", "protocol_bounds", "eis", "ecm", "degradation",
                    "protocol_scaling"):
        assert section in oc
    assert "dfn" in oc["solver"] and "emergency" in oc["solver"]
    assert set(oc["degradation"]["preset_constants"]) == {"nominal", "accelerated", "severe"}


def test_build_oracle_rejects_chemistry_parameter_set_mismatch():
    """#14: a calibration YAML declaring one chemistry must not pair with another set."""
    from battery_oracle.experiment import build_oracle_from_config
    from battery_oracle.parameter_sets import SUPPORTED_PARAMETER_SETS
    a, b = SUPPORTED_PARAMETER_SETS[:2]
    cfg = {
        "cycling": {"parameter_set": a, "chemistry": b},
        "degradation": {"preset": "accelerated"},
        "eis": {}, "ecm": {},
        "protocols": [{"C_rate_1": 100, "C_rate_2": 50, "duration_1": 1,
                       "duration_2": 0.5, "D_rate": 100, "duration_d": 1}],
    }
    with pytest.raises(ValueError, match="does not match"):
        build_oracle_from_config(cfg)


def test_oracle_kwargs_from_oracle_config_preset_resolution():
    oc = load_oracle_config()
    severe = oracle_kwargs_from_oracle_config(oc, preset="severe")
    assert severe["degradation_preset"] == "severe"
    assert severe["preset_constants"]["plating_kinetic_rate_constant_m_s"] == pytest.approx(1e-7)
    nominal = oracle_kwargs_from_oracle_config(oc, preset="nominal")
    assert "plating_kinetic_rate_constant_m_s" not in nominal["preset_constants"]


def test_oracle_config_field_omission_falls_back(monkeypatch, tmp_path):
    """A partial oracle YAML (missing whole sections) must not raise."""
    partial = tmp_path / "config_oracle_partial.yml"
    partial.write_text("cycling:\n  n_cycles: 3\n")
    kw = oracle_kwargs_from_oracle_config(load_oracle_config(partial))
    assert kw["n_cycles"] == 3
    assert kw["temperature_K"] == pytest.approx(298.15)  # fell back, no KeyError


def test_built_oracle_matches_resolved_kwargs():
    """The oracle actually constructed from the packaged YAMLs carries the
    same values oracle_kwargs_from_config resolved -- catches drift between
    the mapping layer and PyBaMMOracle.__init__'s attribute wiring.
    """
    cfg = load_experiment_config(None)
    oc = load_oracle_config()
    kw = oracle_kwargs_from_config(cfg, oc)
    o = build_oracle_from_config(cfg)
    attr_to_kwarg = {
        "_dfn_solver_rtol": "dfn_solver_rtol", "_dfn_solver_atol": "dfn_solver_atol",
        "_dfn_solver_dt_max_s": "dfn_solver_dt_max_s",
        "_emergency_solver_rtol": "emergency_solver_rtol",
        "_DUR_MIN_s": "dur_min_s", "_DUR_MAX_s": "dur_max_s",
        "_V_CHARGE_MAX": "v_charge_max", "_V_DISCHARGE_MIN": "v_discharge_min",
        "_CHARGE_STAGE_MAX_s": "charge_stage_max_s",
        "_initial_soc": "initial_soc", "_soc_clip_min": "soc_clip_min",
        "_soc_clip_max": "soc_clip_max", "_lam_ceiling": "lam_ceiling",
        "_noise_combined_flicker_frac": "noise_combined_flicker_frac",
        "_noise_combined_white_frac": "noise_combined_white_frac",
        "_linkk_c": "linkk_c", "_linkk_max_M": "linkk_max_M",
        "_cpe_w_default": "cpe_w_default", "_cpe_n_default": "cpe_n_default",
        "_ecm_rescale_target_r0": "ecm_rescale_target_r0",
        "_autoeis_num_warmup": "autoeis_num_warmup",
        "_autoeis_num_samples": "autoeis_num_samples",
        "_rest_s": "rest_s", "_real_cell_capacity_mah": "real_cell_capacity_mah",
        "_temperature_K": "temperature_K", "eol_capacity_fraction": "eol_capacity_fraction",
        "capacity_check": "capacity_check", "n_cycles": "n_cycles",
        "_circuit": "circuit",
    }
    for attr, key in attr_to_kwarg.items():
        assert getattr(o, attr) == kw[key], f"{attr} != kwargs[{key!r}]"
    for attr, key in (("_C_MIN_mA", "c_min_crate"), ("_C_MAX_mA", "c_max_crate"),
                      ("_C2_MIN_mA", "c2_min_crate"), ("_C2_MAX_mA", "c2_max_crate")):
        assert getattr(o, attr) == pytest.approx(kw[key] * o._nominal_mAh), attr
    assert o._cpe_w_seed == kw["cpe_w_seed"]
    assert o._cpe_n_seed == kw["cpe_n_seed"]


def test_oracle_config_pointer_overrides_defaults(tmp_path):
    """An experiment YAML's oracle_config field points at a custom oracle YAML
    whose fields (e.g. a tune-oracle-skill calibration) actually take effect.
    """
    custom_oracle = tmp_path / "config_oracle_custom.yml"
    custom_oracle.write_text(
        "protocol_scaling:\n"
        "  kinetics_scale: 0.42\n"
        "solver:\n"
        "  dfn:\n"
        "    rtol: 1.0e-4\n"
    )
    cfg = load_experiment_config(None)
    cfg["oracle_config"] = str(custom_oracle)
    o = build_oracle_from_config(cfg)
    assert o._dfn_solver_rtol == pytest.approx(1e-4)
    # Fields the custom oracle YAML omits (e.g. protocol_bounds entirely) fall
    # back to PyBaMMOracle's own Python-literal defaults, not an error.
    assert o._C_MAX_mA == pytest.approx(1.5 * o._nominal_mAh)


@pytest.mark.slow
def test_run_experiment_default():
    """run_experiment on the packaged default returns a non-empty history."""
    import importlib.resources as r

    from battery_oracle import run_experiment

    path = r.files("battery_oracle").joinpath("config_experiment_defaults.yml")
    history = run_experiment(path)
    assert isinstance(history, list)
    assert len(history) >= 1


def test_oracle_config_builder_takes_the_ecm_fitter():
    """The ECM fitter is not in the oracle config: the Randles stub by default, or the
    fitter passed."""
    from battery_oracle import PyBaMMOracle, randles_stub_ecm

    def fitter(*a, **k):
        return randles_stub_ecm(*a, **k)

    assert PyBaMMOracle.from_config().ecm_model_fn is randles_stub_ecm
    assert PyBaMMOracle.from_config(ecm_model_fn=fitter).ecm_model_fn is fitter
