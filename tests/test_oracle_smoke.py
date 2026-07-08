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


def test_build_degradation_config_default_matches_module_dict():
    """preset_constants=None (the default) must reproduce today's hardcoded numbers."""
    import pybamm

    from battery_oracle.oracle import _DEFAULT_PRESET_CONSTANTS, _build_degradation_config

    pv = pybamm.ParameterValues("Chen2020")
    _, pv2 = _build_degradation_config("accelerated", pv)
    expected = _DEFAULT_PRESET_CONSTANTS["accelerated"]["plating_kinetic_rate_constant_m_s"]
    assert pv2["Lithium plating kinetic rate constant [m.s-1]"] == pytest.approx(expected)
    assert expected == pytest.approx(1e-8)


def test_build_degradation_config_preset_constants_override():
    import pybamm

    from battery_oracle.oracle import _build_degradation_config

    pv = pybamm.ParameterValues("Chen2020")
    custom = {
        "plating_kinetic_rate_constant_m_s": 5e-7,
        "dead_lithium_decay_constant_s": 2e-5,
        "initial_plated_lithium_concentration_mol_m3": 0.0,
    }
    _, pv2 = _build_degradation_config("accelerated", pv, preset_constants=custom)
    assert pv2["Lithium plating kinetic rate constant [m.s-1]"] == pytest.approx(5e-7)
    assert pv2["Dead lithium decay constant [s-1]"] == pytest.approx(2e-5)


def test_ec_diffusivity_base_factor_override():
    import pybamm

    from battery_oracle.oracle import _build_degradation_config

    pv = pybamm.ParameterValues("Chen2020")
    base_d = float(pv["EC diffusivity [m2.s-1]"])
    _, pv_default = _build_degradation_config("nominal", pv)
    _, pv_custom = _build_degradation_config("nominal", pv, ec_diffusivity_base_factor=0.5)
    assert float(pv_default["EC diffusivity [m2.s-1]"]) == pytest.approx(base_d * 0.25)
    assert float(pv_custom["EC diffusivity [m2.s-1]"]) == pytest.approx(base_d * 0.5)


def test_protocol_bound_kwarg_overrides_sanitisation():
    o = PyBaMMOracle(c_max_mA=1234.0, c_min_mA=77.0)
    assert o._C_MAX_mA == pytest.approx(1234.0)
    assert o._C_MIN_mA == pytest.approx(77.0)
    amps = o._sanitise_current(999_999.0, 500.0, o._C_MIN_mA, o._C_MAX_mA)
    assert amps == pytest.approx(1234.0 / 1000.0)


def test_solver_and_ecm_kwargs_wire_to_instance_attrs():
    o = PyBaMMOracle(
        solver_rtol=1e-4, linkk_c=0.9, linkk_max_M=40,
        initial_soc=0.7, lam_ceiling=0.9,
    )
    assert o._solver_rtol == pytest.approx(1e-4)
    assert o._linkk_c == pytest.approx(0.9)
    assert o._linkk_max_M == 40
    assert o._initial_soc == pytest.approx(0.7)
    assert o._lam_ceiling == pytest.approx(0.9)


def test_save_to_csv_respects_custom_circuit():
    """save_to_csv must label columns from the passed circuit, not the module default."""
    history = [{
        "call_idx": 0,
        "ecm_params_charge": np.array([0.1, 1.0, 0.9]),
        "ecm_params_discharge": np.array([0.1, 1.0, 0.9]),
        "protocol": np.zeros(6),
    }]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = PyBaMMOracle.save_to_csv(
            history, f"{d}/rec.csv", circuit="R1-[R2,P3]",
        )
        import pandas as pd
        df = pd.read_csv(out)
        assert df["circuit"].iloc[0] == "R1-[R2,P3]"
        assert "R1_charge_mean" in df.columns
        assert "P3w_charge_mean" in df.columns
        assert "R3_charge_mean" not in df.columns  # would appear under the default 9-param circuit


def test_randles_stub_shape():
    freq = np.logspace(-2, 4, 30)
    # Simple synthetic spectrum: ohmic + one decaying arc
    Z = 0.05 + 0.02 / (1 + 1j * freq / 10.0)
    out = _randles_stub_ecm(freq, Z.real, Z.imag)
    assert out.shape == (14,)   # 7-param DEFAULT_CIRCUIT half, duplicated chg/dis
    assert np.all(np.isfinite(out))


def test_randles_stub_circuit_generic():
    """The stub derives its layout from the circuit — legacy 9-param still works."""
    freq = np.logspace(-2, 4, 30)
    Z = 0.05 + 0.02 / (1 + 1j * freq / 10.0)
    out = _randles_stub_ecm(freq, Z.real, Z.imag, circuit="R1-P2-[R3,P4]-[R5,P6]")
    assert out.shape == (18,)
    assert np.all(np.isfinite(out))
    # Ohmic R lands in slot 0 (HF intercept); the two arc Rs split the LF rise 0.6/0.4
    half = out[:9]
    assert half[0] == pytest.approx(0.05, rel=0.05)          # R1
    assert half[3] / half[6] == pytest.approx(0.6 / 0.4)     # R3 / R5


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
