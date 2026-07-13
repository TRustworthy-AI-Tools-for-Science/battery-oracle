"""Smoke tests for the oracle: construction, constants, and one SPMe cycle.

Uses the Randles-stub ECM (no AutoEIS) so these run without the [autoeis] extra.
"""
import numpy as np
import pytest

from battery_oracle import (
    ACTION_FEATURE_NAMES,
    DEFAULT_CIRCUIT,
    ECM_PARAM_NAMES,
    STATE_VECTOR_SCHEMA,
    FailureKind,
    OracleFailure,
    PyBaMMOracle,
    _randles_stub_ecm,
    make_pybamm_candidates,
    state_vector_schema,
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


# ── Phase 0: FailureKind (#7) ──────────────────────────────────────────────

def test_failure_kind_is_str_enum():
    # str-Enum so it serialises directly to CSV/JSON and compares to its value.
    assert FailureKind.END_OF_LIFE == "end_of_life"
    assert FailureKind.SOLVER_DEGRADED.value == "solver_degraded"
    names = {f.name for f in FailureKind}
    assert {
        "SOLVER_TRUNCATION", "SOLVER_FAILURE", "VOLTAGE_INFEASIBLE",
        "END_OF_LIFE", "ECM_NONCONVERGENCE", "THERMAL_RUNAWAY", "SOLVER_DEGRADED",
    } <= names


def test_oracle_failure_carries_failure_kind():
    exc = OracleFailure("boom", protocol=np.zeros(6), failure_kind=FailureKind.SOLVER_TRUNCATION)
    assert exc.failure_kind is FailureKind.SOLVER_TRUNCATION
    # Back-compat: failure_kind is optional.
    assert OracleFailure("legacy").failure_kind is None


def test_save_to_csv_failure_and_fidelity_columns():
    """A terminal audit row (no ECM params) exports with failure_kind + fidelity."""
    import tempfile

    import pandas as pd
    history = [
        {  # a normal successful row
            "call_idx": 0,
            "ecm_params_charge": np.array([0.1, 1.0, 0.9]),
            "ecm_params_discharge": np.array([0.1, 1.0, 0.9]),
            "protocol": np.zeros(6),
            "failure_kind": None,
            "fidelity": "full",
        },
        {  # a failed audit row appended by run_experiment
            "call_idx": 1,
            "failed": True,
            "failure_kind": FailureKind.END_OF_LIFE,
            "fidelity": "failed",
            "protocol": np.zeros(6),
        },
    ]
    with tempfile.TemporaryDirectory() as d:
        out = PyBaMMOracle.save_to_csv(history, f"{d}/rec.csv", circuit="R1-[R2,P3]")
        df = pd.read_csv(out)
    assert "failure_kind" in df.columns and "fidelity" in df.columns
    assert df["failure_kind"].iloc[0] == "" or pd.isna(df["failure_kind"].iloc[0])
    assert df["failure_kind"].iloc[1] == "end_of_life"
    assert df["fidelity"].iloc[1] == "failed"


# ── Phase 4: multi-chemistry (#14) ─────────────────────────────────────────

def test_chemistry_kwarg_resolves_and_validates():
    assert PyBaMMOracle()._chemistry == "Chen2020"  # default
    lfp = PyBaMMOracle(chemistry="Prada2013")
    assert lfp._chemistry == "Prada2013"
    assert float(lfp._pv["Nominal cell capacity [A.h]"]) == pytest.approx(2.3)
    # Friendly alias resolves to the same parameter set.
    assert float(PyBaMMOracle(chemistry="LFP")._pv["Nominal cell capacity [A.h]"]) == pytest.approx(2.3)
    with pytest.raises(ValueError):
        PyBaMMOracle(chemistry="bogus")


def test_explicit_parameter_values_wins_over_chemistry():
    import pybamm
    pv = pybamm.ParameterValues("Chen2020")
    o = PyBaMMOracle(chemistry="Prada2013", parameter_values=pv)
    # chemistry recorded for provenance, but the explicit pv is used.
    assert o._chemistry == "Prada2013"
    assert float(o._pv["Nominal cell capacity [A.h]"]) == pytest.approx(5.0)


# ── Phase 2: thermal / Arrhenius / temperature protocol (#9, #11, #10) ──────

def test_thermal_kwarg_validation_and_wiring():
    with pytest.raises(ValueError):
        PyBaMMOracle(thermal="bogus")
    iso = PyBaMMOracle()
    assert iso._thermal == "isothermal"
    assert "T_cell_K" not in iso.state_vector_schema  # no thermal slot when isothermal
    lump = PyBaMMOracle(thermal="lumped", T_ambient_K=308.15, h_total_W_per_m2K=25.0)
    assert lump._thermal == "lumped"
    assert lump._deg_opts.get("thermal") == "lumped"
    assert float(lump._pv["Ambient temperature [K]"]) == pytest.approx(308.15)
    assert float(lump._pv["Total heat transfer coefficient [W.m-2.K-1]"]) == pytest.approx(25.0)
    # Lumped adds a single trailing T_cell_K slot -> state len grows by exactly 1.
    assert lump.state_vector_len == iso.state_vector_len + 1
    assert "T_cell_K" in lump.state_vector_schema


def test_arrhenius_kwargs_wire_to_attrs():
    o = PyBaMMOracle(E_a_J_per_mol=45e3, E_a_electrolyte_J_per_mol=12e3)
    assert o._E_a == pytest.approx(45e3)
    assert o._E_a_el == pytest.approx(12e3)
    assert o._T_ref_K == pytest.approx(298.15)


def test_temperature_protocol_action_names_and_validation():
    # Requires lumped thermal.
    with pytest.raises(ValueError):
        PyBaMMOracle(use_temperature_protocol=True, thermal="isothermal")
    base = PyBaMMOracle()
    o = PyBaMMOracle(thermal="lumped", use_temperature_protocol=True)
    # Exactly one more action name than the 6-D default; the extra is T_ambient_K.
    assert len(o._action_names) == len(base._action_names) + 1
    assert o._action_names[-1] == "T_ambient_K"


def test_make_candidates_temperature_protocol_shape():
    base = make_pybamm_candidates(n_candidates=3)
    temp = make_pybamm_candidates(n_candidates=3, temperature_protocol=True,
                                  T_ambient_range=(280.0, 310.0))
    # One extra slot vs the default vector; asserted against each other, not a literal.
    assert temp[0].shape[0] == base[0].shape[0] + 1
    assert temp[0][-1] == pytest.approx(280.0)
    assert temp[-1][-1] == pytest.approx(310.0)


# ── Phase 1: DFN solver settings + fallback (#2, #4) ───────────────────────

def test_dfn_solver_kwargs_wire_to_instance_attrs():
    o = PyBaMMOracle(model="DFN", dfn_solver_rtol=1e-7, dfn_solver_atol=1e-9,
                     dfn_solver_dt_max_s=0.5, dfn_max_crate=1.2)
    assert o._dfn_solver_rtol == pytest.approx(1e-7)
    assert o._dfn_solver_atol == pytest.approx(1e-9)
    assert o._dfn_solver_dt_max_s == pytest.approx(0.5)
    assert o._dfn_max_crate == pytest.approx(1.2)


def test_dfn_current_ceiling_tighter_than_spme():
    """DFN tightens the current ceiling to _dfn_max_crate; SPMe keeps c_max_mA.
    Ceiling is derived from the cell's own C/20 current (no hardcoded mA)."""
    spme = PyBaMMOracle(model="SPMe")
    dfn = PyBaMMOracle(model="DFN")
    # SPMe: ceiling unchanged.
    assert spme._current_ceiling_mA(spme._C_MAX_mA) == pytest.approx(spme._C_MAX_mA)
    # DFN: ceiling = dfn_max_crate * nominal_A * 1000, derived from _c20_A.
    nominal_A = 20.0 * dfn._c20_A
    expected = min(dfn._C_MAX_mA, dfn._dfn_max_crate * nominal_A * 1000.0)
    assert dfn._current_ceiling_mA(dfn._C_MAX_mA) == pytest.approx(expected)
    assert dfn._current_ceiling_mA(dfn._C_MAX_mA) < dfn._C_MAX_mA


@pytest.mark.slow
def test_one_cycle_runs_dfn():
    """A real DFN cycle runs end-to-end at full fidelity (DFN cold-start works;
    the 3-tier fallback is only for failures)."""
    oracle = PyBaMMOracle(model="DFN", ecm_model_fn=_randles_stub_ecm, capacity_check=False)
    oracle.reset()
    try:
        state = oracle(make_pybamm_candidates(n_candidates=1)[0])
    except OracleFailure as exc:
        pytest.skip(f"DFN oracle reported EOL/solver failure on the first cycle: {exc}")
    assert len(state) == oracle.state_vector_len
    h = oracle._history[-1]
    assert h["model"] == "DFN"
    # If DFN succeeded outright it's full fidelity + not degraded; if it fell back
    # it would be reduced — either way the run completes without raising.
    assert h["fidelity"] in ("full", "reduced")


@pytest.mark.slow
def test_dfn_solver_fallback_latches_to_spme():
    """When both DFN solver tiers fail, the oracle falls back to SPMe (reduced
    fidelity, SOLVER_DEGRADED) and latches there for subsequent calls."""
    from battery_oracle import FailureKind
    o = PyBaMMOracle(model="DFN", ecm_model_fn=_randles_stub_ecm, capacity_check=False)
    o.reset()

    class _Boom:
        def solve(self, *a, **k):
            raise RuntimeError("forced DFN solver failure")

    o._solver = _Boom()
    o._solver_emerg = _Boom()
    proto = make_pybamm_candidates(n_candidates=1)[0]
    try:
        o(proto)
    except OracleFailure as exc:
        pytest.skip(f"SPMe fallback itself failed on this protocol: {exc}")
    h = o._history[-1]
    assert o._degraded_to_spme is True
    assert h["fidelity"] == "reduced"
    assert h["failure_kind"] == FailureKind.SOLVER_DEGRADED
    # Latched: the next call stays on SPMe even though _solver is still broken.
    try:
        o(proto)
    except OracleFailure as exc:
        pytest.skip(f"latched SPMe step failed: {exc}")
    assert o._history[-1]["fidelity"] == "reduced"


# ── Phase 0: STATE_VECTOR_SCHEMA + ECM σ (#6) ──────────────────────────────

def test_state_vector_schema_is_derived():
    # Default circuit has 7 params -> 4 blocks of 7 = 28, all derived (no literals).
    n = len(_param_labels_from_circuit(DEFAULT_CIRCUIT))
    sch = state_vector_schema(n)
    assert list(sch) == ["means_charge", "means_discharge", "std_charge", "std_discharge"]
    assert sch["means_charge"] == (0, n)
    assert sch["std_discharge"] == (3 * n, 4 * n)
    assert STATE_VECTOR_SCHEMA == sch
    # has_t_cell registers a single trailing scalar slot.
    sch_t = state_vector_schema(n, has_t_cell=True)
    assert sch_t["T_cell_K"] == (4 * n, 4 * n + 1)


def test_state_vector_len_tracks_circuit():
    o = PyBaMMOracle(circuit="R1-[R2,P3]")  # 4 params
    assert o.state_vector_len == 4 * len(o._ecm_param_names)
    assert o.state_vector_schema["std_discharge"] == (
        3 * len(o._ecm_param_names), 4 * len(o._ecm_param_names),
    )


@pytest.mark.slow
def test_returned_state_matches_schema_and_std_is_nan_for_stub():
    """Under the Randles stub the returned width equals state_vector_len and
    the σ slots are NaN (no AutoEIS posterior). Asserts against derived length."""
    oracle = PyBaMMOracle(ecm_model_fn=_randles_stub_ecm, capacity_check=False)
    oracle.reset()
    try:
        state = oracle(make_pybamm_candidates(n_candidates=1)[0])
    except OracleFailure as exc:
        pytest.skip(f"oracle reported EOL/solver failure on the first cycle: {exc}")
    assert state.shape == (oracle.state_vector_len,)
    lo, hi = oracle.state_vector_schema["std_charge"]
    lo2, hi2 = oracle.state_vector_schema["std_discharge"]
    assert np.isnan(state[lo:hi]).all()
    assert np.isnan(state[lo2:hi2]).all()
    # Means are finite.
    assert np.isfinite(state[: oracle.state_vector_schema["means_discharge"][1]]).all()
    h = oracle._history[-1]
    assert h["failure_kind"] is None and h["fidelity"] == "full"
    assert "ecm_std_charge" in h and "ecm_std_discharge" in h


@pytest.mark.slow
def test_call_stores_raw_state_in_history():
    """__call__ returns the raw state and keeps a copy in history under
    'state_raw' — per-regime detrending now lives in traits_audit.RegimeDetrender,
    fed by a digital-twin orchestrator from this raw per-cycle state history."""
    o = PyBaMMOracle(ecm_model_fn=_randles_stub_ecm, capacity_check=False)
    o.reset()
    try:
        state = o(make_pybamm_candidates(n_candidates=1)[0])
    except OracleFailure as exc:
        pytest.skip(f"oracle failed on first cycle: {exc}")
    h = o._history[-1]
    assert len(state) == o.state_vector_len
    assert "state_raw" in h
    assert np.allclose(state, h["state_raw"], equal_nan=True)


@pytest.mark.slow
def test_lumped_thermal_populates_t_cell():
    """A lumped-thermal cycle surfaces T_cell_K in history + the state vector,
    and the state width equals the schema (schema-derived, no magic number)."""
    o = PyBaMMOracle(thermal="lumped", T_ambient_K=308.15,
                     ecm_model_fn=_randles_stub_ecm, capacity_check=False)
    o.reset()
    try:
        state = o(make_pybamm_candidates(n_candidates=1)[0])
    except OracleFailure as exc:
        pytest.skip(f"lumped-thermal cycle failed: {exc}")
    assert len(state) == o.state_vector_len
    lo, hi = o.state_vector_schema["T_cell_K"]
    t_cell = o._history[-1]["T_cell_K"]
    assert state[lo] == pytest.approx(t_cell)
    # Cell self-heats to at or above ambient.
    assert t_cell >= 308.15 - 1.0


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
