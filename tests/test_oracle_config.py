"""OracleConfig: one default per setting, YAML locations, unknown-key rejection, and the
experiment-over-oracle YAML precedence (synthetic configs only)."""

import numpy as np
import pytest

from battery_oracle.experiment import load_oracle_config, oracle_kwargs_from_config, oracle_kwargs_from_oracle_config
from battery_oracle.oracle_config import (FIELD_NAMES, YAML_KEYS, OracleConfig, settings_from_yaml,
                                          unknown_yaml_keys)


def _set(d, path, value):
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


def test_replace_changes_one_field_and_rejects_unknown_names():
    base = OracleConfig()
    c = base.replace(n_cycles=base.n_cycles + 3)
    assert c.n_cycles == base.n_cycles + 3
    assert all(getattr(c, f) is getattr(base, f) for f in FIELD_NAMES if f != "n_cycles")
    with pytest.raises(TypeError):
        base.replace(not_a_setting=1)


def test_an_empty_yaml_gives_the_defaults():
    kw = oracle_kwargs_from_oracle_config({})
    defaults = OracleConfig().to_dict()
    assert {k: kw[k] for k in defaults} == defaults


@pytest.mark.parametrize("name", sorted(YAML_KEYS))
def test_every_yaml_location_reaches_its_field(name):
    """A value written at a field's YAML location (any nesting) is the setting."""
    default = getattr(OracleConfig(), name)
    if isinstance(default, bool):
        value = not default
    elif isinstance(default, (int, float)):
        value = type(default)(default * 2 + 1)
    elif isinstance(default, str):
        value = default                     # free-form strings are validated by the oracle
    else:                                   # None / mapping defaults
        value = {"k": 1.0} if name in ("parameter_scales", "parameter_overrides", "radial_mesh",
                                       "cpe_w_seed", "cpe_n_seed") else 1.5
    cfg = {}
    _set(cfg, YAML_KEYS[name], value)
    assert settings_from_yaml(cfg)[name] == value


def test_unknown_keys_under_oracle_sections_are_rejected():
    cfg = {"eis": {"noise_level": 0.01, "noise_levle": 0.02}, "_notes": {"anything": 1}}
    assert unknown_yaml_keys(cfg) == ["eis.noise_levle"]
    with pytest.raises(ValueError, match="noise_levle"):
        settings_from_yaml(cfg)


def test_packaged_config_has_no_unknown_keys():
    assert unknown_yaml_keys(load_oracle_config()) == []


def test_frequency_grid_explicit_list_wins_over_range():
    grid = [1e3, 1e1, 1e-1]
    s = settings_from_yaml({"eis": {"frequencies_hz": grid, "freq_min_hz": 1.0, "n_freq_points": 5}})
    assert np.array_equal(s["frequencies"], grid)
    s = settings_from_yaml({"eis": {"freq_min_hz": 1.0, "freq_max_hz": 100.0, "n_freq_points": 3}})
    assert np.allclose(s["frequencies"], [1.0, 10.0, 100.0])
    assert "frequencies" not in settings_from_yaml({})


def test_preset_constants_follow_the_resolved_preset():
    cfg = {"degradation": {"preset": "a", "preset_constants": {"a": {"x": 1.0}, "b": {"x": 2.0}}}}
    assert settings_from_yaml(cfg)["preset_constants"] == {"x": 1.0}
    assert settings_from_yaml(cfg, preset="b")["preset_constants"] == {"x": 2.0}


def test_experiment_yaml_overrides_the_oracle_yaml():
    oracle_cfg = {"cycling": {"n_cycles": 2, "rest_s": 10.0},
                  "protocol_scaling": {"kinetics_scale": 0.5, "real_cell_capacity_mah": 11.0}}
    exp = {"cycling": {"n_cycles": 7, "real_cell_capacity_mah": 33.0},
           "degradation": {"preset": "nominal", "kinetics_scale": 0.25}, "eis": {}}
    kw = oracle_kwargs_from_config(exp, oracle_cfg)
    assert kw["n_cycles"] == 7 and kw["rest_s"] == 10.0              # experiment wins; the rest stays
    assert kw["real_cell_capacity_mah"] == 33.0 and kw["kinetics_scale"] == 0.25
    assert kw["degradation_preset"] == "nominal"
    assert oracle_cfg["cycling"]["n_cycles"] == 2                     # inputs not mutated


def test_oracle_keeps_its_config():
    from battery_oracle.oracle import PyBaMMOracle

    o = PyBaMMOracle(OracleConfig(n_cycles=3), rest_s=7.0)
    assert (o.config.n_cycles, o.config.rest_s) == (3, 7.0)
    with pytest.raises(TypeError):
        PyBaMMOracle(not_a_setting=1)


def test_packaged_yaml_documents_the_defaults():
    """The packaged oracle YAML states OracleConfig's defaults: every field it sets
    equals the default, except fields whose default is None (resolved at build time:
    the YAML supplies them)."""
    import math

    default = OracleConfig()
    stated = settings_from_yaml(load_oracle_config())
    assert stated, "the packaged YAML sets no field"
    for name, value in stated.items():
        want = getattr(default, name)
        if want is None:
            continue
        if isinstance(want, float):
            assert math.isclose(float(value), want, rel_tol=1e-12), name
        else:
            assert value == want, name
    assert set(stated) <= set(YAML_KEYS) | {"frequencies", "chemistry", "preset_constants"}

