"""Smoke tests for the oracle: construction, constants, and one DFN cycle.

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
    randles_stub_ecm,
    state_vector_schema,
)
from battery_oracle.circuit import param_labels_from_circuit
from battery_oracle.oracle_config import OracleConfig
from conftest import PROTOCOL


def test_constants_consistent():
    assert ECM_PARAM_NAMES == param_labels_from_circuit(DEFAULT_CIRCUIT)
    assert len(ACTION_FEATURE_NAMES) == 6
    assert DEFAULT_CIRCUIT.startswith("R1")


def test_construct_and_circuit_params():
    o = PyBaMMOracle()
    assert o._circuit == DEFAULT_CIRCUIT
    assert o._ecm_param_names == param_labels_from_circuit(DEFAULT_CIRCUIT)
    assert len(o._action_names) == 6


def test_construct_custom_circuit():
    o = PyBaMMOracle(circuit="R1-[R2,P3]")
    assert o._circuit == "R1-[R2,P3]"
    assert o._ecm_param_names == ["R1", "R2", "P3w", "P3n"]


def test_build_degradation_config_default_matches_module_dict():
    """preset_constants=None (the default) takes the module's constants."""
    import pybamm

    from battery_oracle.degradation import _build_degradation_config, _DEFAULT_PRESET_CONSTANTS

    pv = pybamm.ParameterValues("Chen2020")
    _, pv2 = _build_degradation_config("accelerated", pv)
    expected = _DEFAULT_PRESET_CONSTANTS["accelerated"]["plating_kinetic_rate_constant_m_s"]
    assert pv2["Lithium plating kinetic rate constant [m.s-1]"] == pytest.approx(expected)
    expected = _DEFAULT_PRESET_CONSTANTS["accelerated"]["dead_lithium_decay_constant_s"]
    assert pv2["Dead lithium decay constant [s-1]"] == pytest.approx(expected)


def test_build_degradation_config_preset_constants_override():
    import pybamm

    from battery_oracle.degradation import _build_degradation_config

    pv = pybamm.ParameterValues("Chen2020")
    custom = {
        "plating_kinetic_rate_constant_m_s": 5e-7,
        "dead_lithium_decay_constant_s": 2e-5,
        "initial_plated_lithium_concentration_mol_m3": 0.0,
    }
    _, pv2 = _build_degradation_config("accelerated", pv, preset_constants=custom)
    assert pv2["Lithium plating kinetic rate constant [m.s-1]"] == pytest.approx(5e-7)
    assert pv2["Dead lithium decay constant [s-1]"] == pytest.approx(2e-5)


def test_sei_rate_scale_multiplies_ec_diffusivity_only():
    """The parameter set's EC diffusivity times sei_rate_scale, with no other factor; the
    SEI kinetic rate constant is untouched."""
    import pybamm

    from battery_oracle.degradation import PRESETS, _build_degradation_config

    pv = pybamm.ParameterValues("Chen2020")
    base_d, base_k = float(pv["EC diffusivity [m2.s-1]"]), float(pv["SEI kinetic rate constant [m.s-1]"])
    for preset in PRESETS:
        for scale in (1.0, 0.37):
            _, pv2 = _build_degradation_config(preset, pv, sei_rate_scale=scale)
            assert float(pv2["EC diffusivity [m2.s-1]"]) == pytest.approx(base_d * scale)
            assert float(pv2["SEI kinetic rate constant [m.s-1]"]) == pytest.approx(base_k)
    assert float(pv["EC diffusivity [m2.s-1]"]) == base_d          # input not mutated


def test_plating_scales_multiply_the_preset_constants():
    import pybamm

    from battery_oracle.degradation import _build_degradation_config

    pv = pybamm.ParameterValues("Chen2020")
    pc = {"plating_kinetic_rate_constant_m_s": 3e-9, "dead_lithium_decay_constant_s": 7e-7}
    _, pv2 = _build_degradation_config("accelerated", pv, plating_rate_scale=2.0, dead_li_decay_scale=0.5,
                                       preset_constants=pc)
    assert pv2["Lithium plating kinetic rate constant [m.s-1]"] == pytest.approx(6e-9)
    assert pv2["Dead lithium decay constant [s-1]"] == pytest.approx(3.5e-7)


def test_unknown_preset_is_rejected():
    import pybamm

    from battery_oracle.degradation import _build_degradation_config

    with pytest.raises(ValueError, match="Unknown degradation_preset"):
        _build_degradation_config("severe", pybamm.ParameterValues("Chen2020"))


def test_protocol_bound_kwarg_overrides_sanitisation():
    o = PyBaMMOracle(c_max_crate=0.25, c_min_crate=0.02)
    assert o._C_MAX_mA == pytest.approx(0.25 * o._nominal_mAh)
    assert o._C_MIN_mA == pytest.approx(0.02 * o._nominal_mAh)
    amps = o._sanitise_current(999_999.0, 500.0, o._C_MIN_mA, o._C_MAX_mA)
    assert amps == pytest.approx(0.25 * o._nominal_mAh / 1000.0)


def test_solver_and_ecm_kwargs_wire_to_instance_attrs():
    o = PyBaMMOracle(
        emergency_solver_rtol=1e-4, linkk_c=0.9, linkk_max_M=40,
        initial_soc=0.7, lam_ceiling=0.9,
    )
    assert o._emergency_solver_rtol == pytest.approx(1e-4)
    assert o._linkk_c == pytest.approx(0.9)
    assert o._linkk_max_M == 40
    assert o._initial_soc == pytest.approx(0.7)
    assert o._lam_ceiling == pytest.approx(0.9)


def test_randles_stub_shape():
    freq = np.logspace(-2, 4, 30)
    # Simple synthetic spectrum: ohmic + one decaying arc
    Z = 0.05 + 0.02 / (1 + 1j * freq / 10.0)
    out = randles_stub_ecm(freq, Z.real, Z.imag)
    # one DEFAULT_CIRCUIT half per state (charge, discharge)
    assert out.shape == (2 * len(param_labels_from_circuit(DEFAULT_CIRCUIT)),)
    assert np.all(np.isfinite(out))


def test_randles_stub_circuit_generic():
    """The stub derives its layout from the circuit (here a 9-param one)."""
    freq = np.logspace(-2, 4, 30)
    Z = 0.05 + 0.02 / (1 + 1j * freq / 10.0)
    out = randles_stub_ecm(freq, Z.real, Z.imag, circuit="R1-P2-[R3,P4]-[R5,P6]")
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
        "END_OF_LIFE", "SOLVER_DEGRADED",
    } == names


def test_oracle_failure_carries_failure_kind():
    exc = OracleFailure("boom", protocol=np.zeros(6), failure_kind=FailureKind.SOLVER_TRUNCATION)
    assert exc.failure_kind is FailureKind.SOLVER_TRUNCATION
    with pytest.raises(TypeError):
        OracleFailure("no kind")


# ── Phase 4: multi-chemistry (#14) ─────────────────────────────────────────

def _other_chemistry() -> str:
    from battery_oracle.parameter_sets import SUPPORTED_PARAMETER_SETS
    return next(n for n in SUPPORTED_PARAMETER_SETS if n != OracleConfig().chemistry)


def test_chemistry_kwarg_resolves_and_validates():
    from battery_oracle.parameter_sets import resolve_parameter_values
    assert PyBaMMOracle()._chemistry == OracleConfig().chemistry  # default
    name = _other_chemistry()
    o = PyBaMMOracle(chemistry=name)
    assert o._chemistry == name
    nominal = float(resolve_parameter_values(name)["Nominal cell capacity [A.h]"])
    assert float(o._pv["Nominal cell capacity [A.h]"]) == pytest.approx(nominal)
    with pytest.raises(ValueError):
        PyBaMMOracle(chemistry="bogus")


def test_explicit_parameter_values_wins_over_chemistry():
    import pybamm
    pv = pybamm.ParameterValues("Chen2020")
    pv["Nominal cell capacity [A.h]"] = 1.25
    name = _other_chemistry()
    o = PyBaMMOracle(chemistry=name, parameter_values=pv)
    # chemistry recorded for provenance, but the explicit pv is used.
    assert o._chemistry == name
    assert float(o._pv["Nominal cell capacity [A.h]"]) == pytest.approx(1.25)


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


# ── Phase 1: DFN solver settings + fallback (#2, #4) ───────────────────────

def test_dfn_solver_kwargs_wire_to_instance_attrs():
    o = PyBaMMOracle(dfn_solver_rtol=1e-7, dfn_solver_atol=1e-9,
                     dfn_solver_dt_max_s=0.5)
    assert o._dfn_solver_rtol == pytest.approx(1e-7)
    assert o._dfn_solver_atol == pytest.approx(1e-9)
    assert o._dfn_solver_dt_max_s == pytest.approx(0.5)


def test_current_bounds_are_c_rates_of_the_cells_own_capacity():
    """The same C-rate bounds give mA bounds in proportion to each cell's nominal
    capacity (a 5 Ah pouch and a ~42 mAh coin cell)."""
    cfg = OracleConfig()
    for chemistry in (OracleConfig().chemistry, "LiR2032"):
        o = PyBaMMOracle(chemistry=chemistry)
        q = o._nominal_mAh
        assert (o._C_MIN_mA, o._C_MAX_mA, o._C2_MIN_mA, o._C2_MAX_mA) == pytest.approx(
            (cfg.c_min_crate * q, cfg.c_max_crate * q, cfg.c2_min_crate * q, cfg.c2_max_crate * q))
    lo = o._sanitise_current(1e-6, 1.0, o._C_MIN_mA, o._C_MAX_mA)
    assert 1000.0 * lo == pytest.approx(cfg.c_min_crate * o._nominal_mAh)   # floor below 1C on a coin cell


@pytest.mark.slow
def test_one_cycle_runs_with_the_primary_solver(one_cycle):
    """A real DFN cycle runs end-to-end with the primary solver
    and records one history row with the cycle's ECM fit and SOH."""
    oracle, res = one_cycle
    assert len(res.state) == oracle.state_vector_len
    assert len(oracle.history) == 1
    h = oracle.history[-1]
    assert h["failure_kind"] is None and "fidelity" not in h
    assert 0.0 < h["end_soh"] <= 1.0
    assert len(h["ecm_params_charge"]) == len(oracle._ecm_param_names)


def test_dfn_solver_failure_fails_loudly():
    """When both solvers fail on the DFN, the oracle raises OracleFailure: it never
    swaps in a reduced model (the DFN -> SPMe fallback was removed 2026-10-16)."""
    o = PyBaMMOracle(ecm_model_fn=randles_stub_ecm, capacity_check=False)
    o.reset()

    class _Boom:
        def solve(self, *a, **k):
            raise RuntimeError("forced DFN solver failure")

    o._solver = _Boom()
    o._solver_emerg = _Boom()
    with pytest.raises(OracleFailure):
        o(PROTOCOL)
    assert not hasattr(o, "_degraded_to_spme")


# ── Phase 0: STATE_VECTOR_SCHEMA + ECM σ (#6) ──────────────────────────────

def test_state_vector_schema_is_derived():
    # 4 blocks of the default circuit's parameter count, all derived (no literals).
    n = len(param_labels_from_circuit(DEFAULT_CIRCUIT))
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
def test_returned_state_matches_schema_and_std_is_nan_for_stub(one_cycle):
    """Under the Randles stub the returned width equals state_vector_len and
    the σ slots are NaN (no AutoEIS posterior). Asserts against derived length."""
    oracle, res = one_cycle
    state = res.state
    assert state.shape == (oracle.state_vector_len,)
    lo, hi = oracle.state_vector_schema["std_charge"]
    lo2, hi2 = oracle.state_vector_schema["std_discharge"]
    assert np.isnan(state[lo:hi]).all()
    assert np.isnan(state[lo2:hi2]).all()
    assert np.isfinite(state[: oracle.state_vector_schema["means_discharge"][1]]).all()
    h = oracle.history[-1]
    assert "ecm_std_charge" in h and "ecm_std_discharge" in h


@pytest.mark.slow
def test_lumped_thermal_populates_t_cell():
    """A lumped-thermal cycle surfaces T_cell_K in history + the state vector,
    and the state width equals the schema (schema-derived, no magic number)."""
    o = PyBaMMOracle(thermal="lumped", T_ambient_K=308.15,
                     ecm_model_fn=randles_stub_ecm, capacity_check=False)
    o.reset()
    try:
        state = o(PROTOCOL)
    except OracleFailure as exc:
        pytest.skip(f"lumped-thermal cycle failed: {exc}")
    assert len(state) == o.state_vector_len
    lo, hi = o.state_vector_schema["T_cell_K"]
    t_cell = o.history[-1]["T_cell_K"]
    assert state[lo] == pytest.approx(t_cell)
    # Cell self-heats to at or above ambient.
    assert t_cell >= 308.15 - 1.0


@pytest.mark.slow
def test_radial_mesh_kwarg_builds_the_fit_mesh():
    """The oracle replays protocols on the chain fit's converged particle mesh
    (verify/radial_mesh_check.py: 40 points, clustered at the surface)."""
    import pybamm

    o = PyBaMMOracle(radial_mesh={"points": 40, "kind": "surface", "stretch": 3.0})
    model = pybamm.lithium_ion.DFN()
    var_pts = o._model_var_pts(model)
    assert var_pts["r_n"] == 40 and var_pts["r_p"] == 40
    sub = o._model_submesh(model)
    assert sub["positive particle"].submesh_type is pybamm.Exponential1DSubMesh
    sim = pybamm.Simulation(model, parameter_values=o._pv, var_pts=var_pts, submesh_types=sub)
    sim.build()
    edges = sim.mesh["positive particle"].edges
    assert len(edges) == 41
    widths = np.diff(edges)
    assert widths[-1] < widths[0] / 5          # clustered at the particle surface
    assert PyBaMMOracle()._model_submesh(model) is None   # default: PyBaMM's mesh


def test_radial_mesh_rejects_unknown_kind():
    with pytest.raises(ValueError, match="radial mesh kind"):
        PyBaMMOracle(radial_mesh={"points": 40, "kind": "bogus"})


def test_mesh_setting_x_factor_refines_the_electrolyte_mesh():
    import pybamm

    o = PyBaMMOracle(radial_mesh={"points": 80, "kind": "uniform", "x_factor": 2})
    model = pybamm.lithium_ion.DFN()
    vp = o._model_var_pts(model)
    d = model.default_var_pts
    assert vp["r_n"] == 80 and vp["x_n"] == 2 * d["x_n"] and vp["x_s"] == 2 * d["x_s"]
    assert o._model_submesh(model) is None          # uniform: PyBaMM's submeshes


def test_dod_lam_follows_the_simulated_discharge():
    """The DoD-LAM increment uses what the cycle discharged (a discharge stopped at the
    voltage limit counts only what it passed), not the requested current x duration."""
    def stats(frac):
        return [{"kind": "dis", "frac": frac, "crate": 1.0}, {"kind": "rest"},
                {"kind": "chg", "frac": 0.2, "crate": 1.0}, {"kind": "chg"}, {"kind": "rest"}]

    full, cut = PyBaMMOracle(dod_lam_scale=1.0), PyBaMMOracle(dod_lam_scale=1.0)
    full._accumulate_dod_lam(stats(0.8))
    cut._accumulate_dod_lam(stats(0.2))
    assert cut._cumulative_dod_throughput_ah == pytest.approx(0.2)
    assert 0.0 < cut._cumulative_dod_lam_frac < full._cumulative_dod_lam_frac
    off = PyBaMMOracle()
    off._accumulate_dod_lam(stats(0.8))
    assert off._cumulative_dod_lam_frac == 0.0


def test_reset_restores_the_ambient_a_7d_protocol_set():
    """A 7-D protocol's ambient slot rewrites the cell's ambient and initial temperature;
    reset() returns them to the constructed cell's."""
    from battery_oracle import Protocol

    o = PyBaMMOracle(thermal="lumped", use_temperature_protocol=True, T_ambient_K=298.15)
    before = (float(o._pv["Ambient temperature [K]"]), float(o._pv["Initial temperature [K]"]))
    p7 = np.r_[PROTOCOL, 313.15]
    o._apply_ambient(p7, Protocol.from_array(p7), first_call=True)
    assert float(o._pv["Ambient temperature [K]"]) == pytest.approx(313.15)
    o.reset()
    assert (float(o._pv["Ambient temperature [K]"]), float(o._pv["Initial temperature [K]"])) == before
    assert o._T_ambient_K == pytest.approx(298.15)
