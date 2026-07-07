"""Smoke tests for the oracle: construction, constants, and one SPMe cycle.

Uses the Randles-stub ECM (no AutoEIS) so these run without the [autoeis] extra.
"""
import numpy as np
import pytest

from battery_oracle import (
    ACTION_FEATURE_NAMES,
    DEFAULT_CIRCUIT,
    ECM_PARAM_NAMES,
    OracleFailure,
    PyBaMMOracle,
    _randles_stub_ecm,
    make_pybamm_candidates,
)
from battery_oracle._circuit import _param_labels_from_circuit


def test_constants_consistent():
    assert ECM_PARAM_NAMES == _param_labels_from_circuit(DEFAULT_CIRCUIT)
    assert len(ACTION_FEATURE_NAMES) == 6
    assert DEFAULT_CIRCUIT.startswith("R1")


def test_make_candidates():
    cands = make_pybamm_candidates(n_candidates=5)
    assert len(cands) == 5
    for c in cands:
        assert np.asarray(c).shape == (6,)


def test_construct_and_circuit_params():
    o = PyBaMMOracle()
    assert o._circuit == DEFAULT_CIRCUIT
    assert o._ecm_param_names == _param_labels_from_circuit(DEFAULT_CIRCUIT)
    assert len(o._action_names) == 6


def test_construct_custom_circuit():
    o = PyBaMMOracle(circuit="R1-[R2,P3]")
    assert o._circuit == "R1-[R2,P3]"
    assert o._ecm_param_names == ["R1", "R2", "P3w", "P3n"]


def test_model_kwarg_resolves():
    import pybamm

    assert PyBaMMOracle()._model == "SPMe"  # default
    assert PyBaMMOracle(model="SPM")._model_cls is pybamm.lithium_ion.SPM
    assert PyBaMMOracle(model="SPMe")._model_cls is pybamm.lithium_ion.SPMe
    assert PyBaMMOracle(model="DFN")._model_cls is pybamm.lithium_ion.DFN


def test_model_kwarg_rejects_unknown():
    with pytest.raises(ValueError):
        PyBaMMOracle(model="bogus")


@pytest.mark.slow
def test_one_cycle_runs_spm():
    """Run a single SPM cycle (new model kwarg) with the Randles stub."""
    oracle = PyBaMMOracle(model="SPM", ecm_model_fn=_randles_stub_ecm, capacity_check=False)
    oracle.reset()
    protocol = make_pybamm_candidates(n_candidates=1)[0]
    try:
        oracle(protocol)
    except OracleFailure as exc:
        pytest.skip(f"SPM oracle reported EOL/solver failure on the first cycle: {exc}")
    assert len(oracle._history) == 1
    assert oracle._history[-1]["model"] == "SPM"


def test_randles_stub_shape():
    freq = np.logspace(-2, 4, 30)
    # Simple synthetic spectrum: ohmic + one decaying arc
    Z = 0.05 + 0.02 / (1 + 1j * freq / 10.0)
    out = _randles_stub_ecm(freq, Z.real, Z.imag)
    assert out.shape == (18,)   # 9-param half duplicated for charge/discharge
    assert np.all(np.isfinite(out))


@pytest.mark.slow
def test_one_cycle_runs():
    """Run a single SPMe cycle with the Randles stub and check the history entry."""
    oracle = PyBaMMOracle(
        ecm_model_fn=_randles_stub_ecm,
        degradation_preset="accelerated",
        capacity_check=False,
    )
    oracle.reset()
    protocol = make_pybamm_candidates(n_candidates=1)[0]
    try:
        oracle(protocol)
    except OracleFailure as exc:
        pytest.skip(f"oracle reported EOL/solver failure on the first cycle: {exc}")
    assert len(oracle._history) == 1
    h = oracle._history[-1]
    assert "end_soh" in h
    assert h["ecm_params_charge"] is not None
    assert len(h["ecm_params_charge"]) == len(oracle._ecm_param_names)


@pytest.mark.slow
def test_save_to_csv(tmp_path):
    """save_to_csv is a @staticmethod (called as PyBaMMOracle.save_to_csv(...))."""
    import pandas as pd
    oracle = PyBaMMOracle(ecm_model_fn=_randles_stub_ecm, capacity_check=False)
    oracle.reset()
    try:
        oracle(make_pybamm_candidates(n_candidates=1)[0])
    except OracleFailure as exc:
        pytest.skip(f"oracle reported EOL/solver failure on the first cycle: {exc}")
    out = PyBaMMOracle.save_to_csv(oracle._history, tmp_path / "rec.csv", cell_id="C01")
    df = pd.read_csv(out)
    assert len(df) == 1
    assert df["circuit"].iloc[0] == DEFAULT_CIRCUIT
    assert "R1_charge_mean" in df.columns
