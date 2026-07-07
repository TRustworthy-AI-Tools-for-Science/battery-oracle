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
    load_experiment_config,
    oracle_kwargs_from_config,
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
