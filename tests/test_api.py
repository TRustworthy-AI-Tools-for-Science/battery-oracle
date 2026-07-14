"""Tests for the novice-facing API surface added by the packaging refactor:
``Protocol``, ``CycleResult``/``run_cycle``, seedable randomness (A4), and
import/version hygiene.

Fast (construction-only / no-solve) tests dominate, matching
tests/test_oracle_smoke.py's convention that any test invoking
``oracle(...)``/``oracle.run_cycle(...)`` (a real PyBaMM solve) is
``@pytest.mark.slow``.
"""
import logging

import numpy as np
import pytest

import battery_oracle
from battery_oracle import (
    CycleResult,
    OracleFailure,
    Protocol,
    PyBaMMOracle,
    make_pybamm_candidates,
    randles_stub_ecm,
)
from battery_oracle._circuit import ACTION_FEATURE_NAMES
from battery_oracle._eis.noise import (
    add_flicker_noise,
    add_relaxation_drift,
    add_white_noise,
)
from battery_oracle.protocol import PROTOCOL_FIELD_NAMES

# ── Protocol round-trip ─────────────────────────────────────────────────────

def _make_protocol(**overrides) -> Protocol:
    defaults = dict(
        charge_current_1_mA=1000.0,
        charge_current_2_mA=500.0,
        charge_duration_1_h=0.25,
        charge_duration_2_h=0.3,
        discharge_current_mA=900.0,
        discharge_duration_h=1.1,
    )
    defaults.update(overrides)
    return Protocol(**defaults)


def test_protocol_round_trip_6d():
    p = _make_protocol()
    arr = p.to_array()
    assert arr.shape == (6,)
    assert Protocol.from_array(arr) == p
    # to_array's slot order matches the canonical field tuple (derived from
    # ACTION_FEATURE_NAMES, not a separately hardcoded order).
    assert list(arr) == [getattr(p, name) for name in PROTOCOL_FIELD_NAMES]
    assert len(PROTOCOL_FIELD_NAMES) == len(ACTION_FEATURE_NAMES)


def test_protocol_round_trip_7d():
    p = _make_protocol(T_ambient_K=305.0)
    arr = p.to_array()
    assert arr.shape == (7,)
    assert Protocol.from_array(arr) == p
    assert arr[6] == pytest.approx(305.0)


def test_protocol_from_array_wrong_length_raises():
    with pytest.raises(ValueError):
        Protocol.from_array(np.zeros(5))
    with pytest.raises(ValueError):
        Protocol.from_array(np.zeros(8))


# ── __call__ golden path + Protocol/ndarray equivalence (needs a solve) ─────

@pytest.mark.slow
def test_call_returns_bare_ndarray_and_protocol_array_equivalence():
    """__call__ still returns a bare ndarray of state_vector_len, and calling
    with a Protocol produces the same state as calling with the equivalent
    ndarray, given the same seed (two fresh oracles so neither call's
    degradation/RNG state leaks into the other)."""
    protocol_arr = make_pybamm_candidates(n_candidates=1)[0]

    o1 = PyBaMMOracle(ecm_model_fn=randles_stub_ecm, capacity_check=False, seed=0)
    o1.reset()
    try:
        state_arr = o1(protocol_arr)
    except OracleFailure as exc:
        pytest.skip(f"oracle reported EOL/solver failure on the first cycle: {exc}")
    assert isinstance(state_arr, np.ndarray)
    assert state_arr.shape == (o1.state_vector_len,)

    o2 = PyBaMMOracle(ecm_model_fn=randles_stub_ecm, capacity_check=False, seed=0)
    o2.reset()
    protocol_obj = Protocol.from_array(protocol_arr)
    try:
        state_from_protocol = o2(protocol_obj)
    except OracleFailure as exc:
        pytest.skip(f"oracle reported EOL/solver failure on the first cycle (Protocol path): {exc}")

    assert np.allclose(state_arr, state_from_protocol, equal_nan=True)


# ── run_cycle / CycleResult (needs a solve) ─────────────────────────────────

@pytest.mark.slow
def test_run_cycle_returns_cycle_result():
    oracle = PyBaMMOracle(ecm_model_fn=randles_stub_ecm, capacity_check=False, seed=0)
    oracle.reset()
    protocol = make_pybamm_candidates(n_candidates=1)[0]
    try:
        result = oracle.run_cycle(protocol)
    except OracleFailure as exc:
        pytest.skip(f"oracle reported EOL/solver failure on the first cycle: {exc}")

    assert isinstance(result, CycleResult)
    assert np.allclose(np.asarray(result), result.state, equal_nan=True)

    # .block() slices by schema key, matching the block's own recorded width.
    lo, hi = result.schema["means_charge"]
    block = result.block("means_charge")
    assert block.shape == (hi - lo,)
    assert np.allclose(block, result.state[lo:hi], equal_nan=True)

    # ECM dict key sets equal the oracle's own (circuit-derived) param labels.
    expected_labels = set(oracle._ecm_param_names)
    assert set(result.ecm_charge) == expected_labels
    assert set(result.ecm_discharge) == expected_labels
    assert set(result.ecm_std_charge) == expected_labels
    assert set(result.ecm_std_discharge) == expected_labels


# ── Seedable randomness: noise functions (A4) ───────────────────────────────

def _sample_spectrum():
    freq = np.logspace(-2, 4, 30)
    Z = 0.05 + 0.02 / (1 + 1j * freq / 10.0)
    return freq, Z


def test_add_white_noise_seed_determinism():
    freq, Z = _sample_spectrum()
    out1 = add_white_noise(Z, 0.05, rng=np.random.default_rng(42))
    out2 = add_white_noise(Z, 0.05, rng=np.random.default_rng(42))
    out3 = add_white_noise(Z, 0.05, rng=np.random.default_rng(43))
    assert np.array_equal(out1, out2)
    assert not np.array_equal(out1, out3)
    # rng=None (default) still works, unseeded.
    out_none = add_white_noise(Z, 0.05, rng=None)
    assert out_none.shape == Z.shape
    assert np.all(np.isfinite(out_none))


def test_add_flicker_noise_seed_determinism():
    freq, Z = _sample_spectrum()
    out1 = add_flicker_noise(freq, Z, 0.05, rng=np.random.default_rng(42))
    out2 = add_flicker_noise(freq, Z, 0.05, rng=np.random.default_rng(42))
    out3 = add_flicker_noise(freq, Z, 0.05, rng=np.random.default_rng(43))
    assert np.array_equal(out1, out2)
    assert not np.array_equal(out1, out3)
    out_none = add_flicker_noise(freq, Z, 0.05, rng=None)
    assert out_none.shape == Z.shape
    assert np.all(np.isfinite(out_none))


def test_add_relaxation_drift_seed_determinism():
    freq, Z = _sample_spectrum()
    kwargs = dict(rest_s=10.0, drift_scale=0.01)
    out1 = add_relaxation_drift(freq, Z, rng=np.random.default_rng(42), **kwargs)
    out2 = add_relaxation_drift(freq, Z, rng=np.random.default_rng(42), **kwargs)
    out3 = add_relaxation_drift(freq, Z, rng=np.random.default_rng(43), **kwargs)
    assert np.array_equal(out1, out2)
    assert not np.array_equal(out1, out3)
    out_none = add_relaxation_drift(freq, Z, rng=None, **kwargs)
    assert out_none.shape == Z.shape
    assert np.all(np.isfinite(out_none))


def test_oracle_seed_reset_reproducible():
    """PyBaMMOracle(seed=...) then reset() re-derives the same RNG stream, so
    a reset-then-run reproduces a fresh instance's noise realisation."""
    o = PyBaMMOracle(seed=7)
    draws1 = o._rng.random(5)
    o.reset()
    draws2 = o._rng.random(5)
    assert np.array_equal(draws1, draws2)

    o_unseeded_a = PyBaMMOracle(seed=None)
    o_unseeded_b = PyBaMMOracle(seed=None)
    # Two unseeded oracles should (with overwhelming probability) diverge.
    assert not np.array_equal(o_unseeded_a._rng.random(5), o_unseeded_b._rng.random(5))


# ── Sanitisation: loud warning + wrong-length ValueError ────────────────────

def test_sanitisation_warns_once_then_dedupes(caplog):
    """An out-of-range protocol logs exactly one WARNING (mentioning the
    clamped slot); an identical second call logs no new WARNING (only a
    DEBUG-level repeat) -- see PyBaMMOracle._warn_clamps."""
    o = PyBaMMOracle()
    bad = np.array([999_999.0, 250.0, 0.25, 0.25, 500.0, 1.0])  # C1 wildly out of bounds

    with caplog.at_level(logging.DEBUG, logger="battery_oracle.oracle"):
        o._protocol_to_experiment(bad)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "C1" in warnings[0].getMessage()

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="battery_oracle.oracle"):
        o._protocol_to_experiment(bad)
    warnings2 = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings2) == 0
    debugs2 = [r for r in caplog.records if r.levelno == logging.DEBUG
              and "sanitised (repeat)" in r.getMessage()]
    assert len(debugs2) == 1


def test_protocol_to_experiment_wrong_length_raises():
    o = PyBaMMOracle()
    with pytest.raises(ValueError):
        o._protocol_to_experiment(np.zeros(5))
    with pytest.raises(ValueError):
        o._protocol_to_experiment(np.zeros(7))  # 7-D requires use_temperature_protocol=True


# ── Import hygiene ───────────────────────────────────────────────────────────

def test_all_hygiene_no_underscore_names():
    assert all(not name.startswith("_") for name in battery_oracle.__all__)


def test_deprecated_underscore_aliases_warn_and_match():
    with pytest.warns(DeprecationWarning):
        randles_alias = battery_oracle._randles_stub_ecm
    assert randles_alias is battery_oracle.randles_stub_ecm

    with pytest.warns(DeprecationWarning):
        autoeis_alias = battery_oracle._autoeis_ecm
    assert autoeis_alias is battery_oracle.autoeis_ecm


def test_bare_install_defaults_to_randles_stub(monkeypatch, caplog):
    # Regression: without the [autoeis] extra the constructor must default the
    # ECM fitter to the Randles stub instead of raising ImportError on the
    # first cycle (pre-existing bug exposed by the packaged-wheel smoke test).
    from battery_oracle import oracle as oracle_mod

    monkeypatch.setattr(oracle_mod, "_AUTOEIS_AVAILABLE", False)
    with caplog.at_level(logging.WARNING, logger="battery_oracle.oracle"):
        oracle = PyBaMMOracle()
    assert oracle.ecm_model_fn is battery_oracle.randles_stub_ecm
    assert any("Randles stub" in r.message for r in caplog.records)


def test_version_is_nonempty_string():
    assert isinstance(battery_oracle.__version__, str)
    assert len(battery_oracle.__version__) > 0


# ── run_experiment() no-arg ──────────────────────────────────────────────────

@pytest.mark.slow
def test_run_experiment_no_arg():
    """run_experiment() with no path loads the packaged default config (A1.5)."""
    from battery_oracle import run_experiment

    history = run_experiment()
    assert isinstance(history, list)
    assert len(history) >= 1
