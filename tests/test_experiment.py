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
    for section in ("model", "cycling", "degradation", "eis", "protocols"):
        assert section in cfg
    assert cfg["model"]["type"] in ("SPMe", "SPM", "DFN")
    assert len(cfg["protocols"]) >= 1


def test_oracle_kwargs_mapping():
    cfg = load_experiment_config(None)
    kw = oracle_kwargs_from_config(cfg)
    # model passthrough + core fields
    assert kw["model"] == cfg["model"]["type"]
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


def test_model_passthrough_spm():
    cfg = load_experiment_config(None)
    cfg["model"]["type"] = "SPM"
    _validate_config(cfg)  # SPM is valid
    assert oracle_kwargs_from_config(cfg)["model"] == "SPM"


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
        lambda c: c["model"].__setitem__("type", "XYZ"),        # unknown model
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
    assert "primary" in oc["solver"] and "emergency" in oc["solver"]
    assert set(oc["degradation"]["preset_constants"]) == {"nominal", "accelerated", "severe"}


def test_oracle_kwargs_from_oracle_config_defaults_match_pybamm_literals():
    """Every field the packaged oracle YAML documents must resolve to the same
    literal PyBaMMOracle's own Python defaults use -- this is the parity
    guarantee the whole refactor depends on (YAML present == default unchanged).
    """
    kw = oracle_kwargs_from_oracle_config(load_oracle_config())
    expected = {
        "n_cycles": 1, "temperature_K": 298.15, "parameter_set": "Chen2020",
        "real_cell_capacity_mah": 200.0, "rest_s": 1200.0, "initial_soc": 0.8,
        "degradation_preset": "accelerated", "eol_capacity_fraction": 0.80,
        "capacity_check": False, "ec_diffusivity_base_factor": 0.25, "lam_ceiling": 0.95,
        "dod_lam_scale": 0.0, "c2_stress_scale": 0.0, "c2_stress_slope_mah_per_ma": 0.0794,
        "c2_stress_ref_ma": 75.27, "kinetics_scale": 1.0, "sei_rate_scale": 1.0,
        "dead_li_decay_scale": 1.0, "plating_rate_scale": 1.0,
        "eis_noise_level": 0.02, "eis_noise_model": "combined", "eis_drift_scale": 0.0,
        "eis_drift_tau_s": 600.0, "eis_drift_n_periods": 4.0,
        "noise_combined_flicker_frac": 0.75, "noise_combined_white_frac": 0.25,
        "soc_clip_min": 0.05, "soc_clip_max": 0.99, "linkk_c": 0.85, "linkk_max_M": 50,
        "cpe_w_default": 0.1, "cpe_n_default": 0.80, "ecm_rescale_target_r0": 0.1334,
        "autoeis_num_warmup": 500, "autoeis_num_samples": 200,
        "solver_rtol": 1e-3, "solver_atol": 1e-6, "solver_dt_max_s": 60.0,
        "emergency_solver_rtol": 1e-2, "emergency_solver_atol": 1e-5,
        "emergency_solver_dt_max_s": 10.0,
        "c_min_mA": 50.0, "c_max_mA": 10_000.0, "c2_min_mA": 20.0, "c2_max_mA": 10_000.0,
        "dur_min_s": 60.0, "dur_max_s": 28_800.0, "v_charge_max": 4.3,
        "v_discharge_min": 3.0, "charge_stage_max_s": 900.0,
    }
    for key, val in expected.items():
        if isinstance(val, float):
            assert kw[key] == pytest.approx(val), key
        else:
            assert kw[key] == val, key
    assert kw["cpe_w_seed"] == {"P3w": 0.071, "P5w": 0.043,
                                "P2w": 7.32, "P4w": 0.071, "P6w": 0.043}
    assert kw["cpe_n_seed"] == {"P3n": 0.80, "P5n": 0.75,
                                "P2n": 0.85, "P4n": 0.80, "P6n": 0.75}
    assert kw["preset_constants"]["plating_kinetic_rate_constant_m_s"] == pytest.approx(1e-8)


def test_base_oracle_yaml_emits_model():
    """#1: model is now in the base oracle YAML layer (was only in the experiment YAML)."""
    kw = oracle_kwargs_from_oracle_config(load_oracle_config())
    assert kw["model"] == "SPMe"


def test_base_oracle_yaml_emits_dfn_solver_settings():
    """#2: DFN solver tolerances + current ceiling map from the oracle YAML."""
    kw = oracle_kwargs_from_oracle_config(load_oracle_config())
    assert kw["dfn_solver_rtol"] == pytest.approx(1e-6)
    assert kw["dfn_solver_atol"] == pytest.approx(1e-8)
    assert kw["dfn_solver_dt_max_s"] == pytest.approx(1.0)
    assert kw["dfn_max_crate"] == pytest.approx(1.5)


def test_base_oracle_yaml_emits_thermal_and_arrhenius():
    """#9/#10/#11: thermal, temperature-protocol, and Arrhenius fields map from YAML."""
    kw = oracle_kwargs_from_oracle_config(load_oracle_config())
    assert kw["thermal"] == "isothermal"
    assert kw["T_ambient_K"] == pytest.approx(298.15)
    assert kw["h_total_W_per_m2K"] == pytest.approx(10.0)
    assert kw["use_temperature_protocol"] is False
    assert kw["E_a_J_per_mol"] == pytest.approx(30e3)
    assert kw["E_a_electrolyte_J_per_mol"] == pytest.approx(15e3)


def test_base_oracle_yaml_emits_detrend():
    """#8: detrend config maps from the oracle YAML."""
    kw = oracle_kwargs_from_oracle_config(load_oracle_config())
    assert kw["detrend_alpha"] == pytest.approx(0.1)
    assert kw["n_protocol_groups"] == 8
    assert kw["detrend_warmup"] == 20


def test_base_oracle_yaml_emits_chemistry():
    """#14: chemistry maps from the oracle YAML (defaults to Chen2020)."""
    kw = oracle_kwargs_from_oracle_config(load_oracle_config())
    assert kw["chemistry"] == "Chen2020"


def test_build_oracle_rejects_chemistry_parameter_set_mismatch():
    """#14: an LFP-declared calibration YAML must not pair with an NMC parameter set."""
    from battery_oracle.experiment import build_oracle_from_config
    cfg = {
        "model": {"type": "SPMe"},
        "cycling": {"parameter_set": "Chen2020", "chemistry": "Prada2013"},
        "degradation": {"preset": "accelerated"},
        "eis": {}, "ecm": {},
        "protocols": [{"C_rate_1": 100, "C_rate_2": 50, "duration_1": 1,
                       "duration_2": 0.5, "D_rate": 100, "duration_d": 1}],
    }
    with pytest.raises(ValueError, match="does not match"):
        build_oracle_from_config(cfg)


@pytest.mark.parametrize("dataset", ["calce", "oxford", "matr"])
def test_packaged_dataset_configs_load_and_map(dataset):
    """#12: each packaged config_oracle_{dataset}.yml loads, maps, and declares a
    chemistry that matches its parameter_set."""
    from battery_oracle.experiment import (
        _resolve_dataset_config,
        oracle_kwargs_from_oracle_config,
    )
    cfg = _resolve_dataset_config(dataset)
    kw = oracle_kwargs_from_oracle_config(cfg)
    assert kw["chemistry"] in ("Chen2020", "Xu2019", "Prada2013")
    assert kw["parameter_set"] == kw["chemistry"]  # must agree (validated at build)


def test_config_dataset_and_oracle_config_are_mutually_exclusive():
    """#12: both select the base layer, so setting both must raise."""
    from battery_oracle.experiment import build_oracle_from_config
    cfg = {
        "model": {"type": "SPMe"}, "cycling": {}, "degradation": {"preset": "nominal"},
        "eis": {}, "ecm": {}, "oracle_config": "somewhere.yml",
        "protocols": [{"C_rate_1": 100, "C_rate_2": 50, "duration_1": 1,
                       "duration_2": 0.5, "D_rate": 100, "duration_d": 1}],
    }
    with pytest.raises(ValueError, match="only one"):
        build_oracle_from_config(cfg, config_dataset="matr")


def test_resolve_dataset_config_rejects_unknown():
    from battery_oracle.experiment import _resolve_dataset_config
    with pytest.raises(ValueError):
        _resolve_dataset_config("nope")


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
        "_solver_rtol": "solver_rtol", "_solver_atol": "solver_atol",
        "_solver_dt_max_s": "solver_dt_max_s",
        "_emergency_solver_rtol": "emergency_solver_rtol",
        "_C_MIN_mA": "c_min_mA", "_C_MAX_mA": "c_max_mA",
        "_C2_MIN_mA": "c2_min_mA", "_C2_MAX_mA": "c2_max_mA",
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
        "  primary:\n"
        "    rtol: 1.0e-4\n"
    )
    cfg = load_experiment_config(None)
    cfg["oracle_config"] = str(custom_oracle)
    o = build_oracle_from_config(cfg)
    assert o._solver_rtol == pytest.approx(1e-4)
    # Fields the custom oracle YAML omits (e.g. protocol_bounds entirely) fall
    # back to PyBaMMOracle's own Python-literal defaults, not an error.
    assert o._C_MAX_mA == pytest.approx(10_000.0)


@pytest.mark.slow
def test_run_experiment_default():
    """run_experiment on the packaged default returns a non-empty history."""
    import importlib.resources as r

    from battery_oracle import run_experiment

    path = r.files("battery_oracle").joinpath("config_experiment_defaults.yml")
    history = run_experiment(path)
    assert isinstance(history, list)
    assert len(history) >= 1
    assert history[0].get("model") == "SPMe"
