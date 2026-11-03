"""Every setting of :class:`battery_oracle.oracle.PyBaMMOracle` and its one default.

``PyBaMMOracle(config, **settings)`` takes an :class:`OracleConfig` and/or field names as
keyword overrides; the YAML mappers (:mod:`battery_oracle.experiment`) read each field from
the place in ``config_oracle_*.yml`` that :data:`YAML_KEYS` names, so a field absent from
the YAML takes the default here and nowhere else. Run-time objects (``ecm_model_fn``,
``parameter_values``) are constructor arguments, not settings.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class OracleConfig:
    n_cycles: int = 1
    frequencies: np.ndarray | None = None
    # Multi-chemistry (#14): validated parameter-set name. Ignored if an explicit
    # parameter_values object is passed. Non-Chen2020 chemistries need their own
    # calibration YAML (bounds/preset) — see parameter_sets.SUPPORTED_PARAMETER_SETS.
    chemistry: str = "Chen2020"
    degradation_preset: str = "accelerated"
    eol_capacity_fraction: float = 0.80
    capacity_check: bool = False
    temperature_K: float = 298.15
    # -- Thermal submodel (#9): "isothermal" (default) | "lumped" (Timms 2021).
    # Lumped adds one ODE coupling cell temperature to the electrochemistry;
    # SEI/plating/electrolyte conductivity are T-dependent so the degradation
    # trajectory shifts for non-ambient protocols. T_cell_K is surfaced in the
    # state vector / history / CSV. --
    thermal: str = "isothermal"
    T_ambient_K: float = 298.15
    h_total_W_per_m2K: float = 10.0
    # Opt-in 7-D protocol (#10): appends a per-cycle ambient temperature slot
    # (u[6] = T_ambient_K). Requires thermal="lumped". 6-D protocols are the
    # unchanged default; a 7-D vector overrides the ambient per call.
    use_temperature_protocol: bool = False
    # -- Temperature-dependent EIS (#11): Arrhenius scaling of the charge-transfer
    # arc (E_a) and the electrolyte/ohmic intercept (E_a_electrolyte) using T_cell. --
    E_a_J_per_mol: float = 30e3
    E_a_electrolyte_J_per_mol: float = 15e3
    eis_noise_level: float = 0.02
    eis_noise_model: str = "combined"
    # -- EIS measurement model: "equilibrium" (stock, linearised at a relaxed
    #    state) | "nonstationary" (the cycling model itself linearised at the
    #    simulated states inside the sweep that follows each 20 min rest; see
    #    _nonstationary_spectrum) --
    eis_mode: str = "equilibrium"
    eis_sweep_overhead_s: float = 1.15
    eis_sweep_periods: float = 1.0
    eis_amplitude_mA: float = 5.0
    # Seedable randomness (#A4): seeds the synthetic EIS noise generator only
    # (see the docstring's `seed` entry for what this does/does not cover).
    seed: int | None = None
    real_cell_capacity_mah: float = 200.0
    kinetics_scale: float = 1.0
    sei_rate_scale: float = 1.0
    dead_li_decay_scale: float = 1.0
    plating_rate_scale: float = 1.0
    dod_lam_scale: float = 0.0
    # -- Beginning-of-life (BOL) fit (config_oracle_defaults.yml: bol.*) --
    # Multipliers / absolute values applied to the PyBaMM parameter set after
    # the degradation preset (see battery_oracle.params). Produced by
    # the stage-A chain fit (battery_oracle.bol_chain) to the real cell's first cycles.
    parameter_scales: dict | None = None
    parameter_overrides: dict | None = None
    # Series contact resistance [Ohm] in the 5 Ah PyBaMM frame; None = off.
    contact_resistance_ohm: float | None = None
    # "cutoff": discharge always runs to v_discharge_min (>= 1 h timeout) --
    # the jones2022 protocol. "protocol": discharge runs for the protocol's
    # own duration (or until v_discharge_min), for time-limited protocols.
    discharge_mode: str = "cutoff"
    # Particle radial mesh of the chain fit (FitConfig.radial_mesh_setting):
    # {"points": .., "kind": "uniform" | "surface", "stretch": .., "x_factor": ..}
    # (x_factor multiplies PyBaMM's electrode/separator points); None = PyBaMM default.
    radial_mesh: dict | None = None
    rest_s: float = 1200.0
    circuit: str | None = None
    action_names: list[str] | None = None
    # -- Solver tolerances (config_oracle_defaults.yml: solver.*) --------
    emergency_solver_rtol: float = 1e-2
    emergency_solver_atol: float = 1e-5
    # -- DFN-specific solver settings (config_oracle_defaults.yml: solver.dfn) --
    # DFN is a stiff coupled DAE; it needs tight IDAKLU tolerances,
    # and its Casadi fallback tiers use a low dt_max so IDA can backstep before
    # a catastrophic failure at the CC->CV switch rather than silently truncate.
    dfn_solver_rtol: float = 1e-6
    dfn_solver_atol: float = 1e-8
    dfn_solver_dt_max_s: float = 1.0
    # _protocol_to_experiment caps the current at this C-rate (DFN stability).
    dfn_max_crate: float = 1.5
    # -- Protocol sanitisation bounds (config_oracle_defaults.yml: protocol_bounds.*) --
    c_min_mA: float = 50.0
    c_max_mA: float = 10_000.0
    c2_min_mA: float = 20.0
    c2_max_mA: float = 10_000.0
    dur_min_s: float = 60.0
    dur_max_s: float = 28_800.0
    v_charge_max: float = 4.3
    v_discharge_min: float = 3.0
    charge_stage_max_s: float = 900.0
    # -- First-call SOC and internal clip bounds (config_oracle_defaults.yml: cycling.initial_soc, eis.soc_clip_*/lam_ceiling) --
    initial_soc: float = 0.8
    soc_clip_min: float = 0.05
    soc_clip_max: float = 0.99
    lam_ceiling: float = 0.95
    # -- Combined-noise split (config_oracle_defaults.yml: eis.noise_combined_*_frac) --
    noise_combined_flicker_frac: float = 0.75
    noise_combined_white_frac: float = 0.25
    # -- linKK validation params (config_oracle_defaults.yml: eis.linkk.*) --
    linkk_c: float = 0.85
    linkk_max_M: int = 50
    # -- Degradation preset physics (config_oracle_defaults.yml: degradation.*) --
    ec_diffusivity_base_factor: float = 0.25
    preset_constants: dict | None = None
    # -- ECM CPE fit seeds + AutoEIS sampler (config_oracle_defaults.yml: ecm.*) --
    cpe_w_seed: dict | None = None
    cpe_n_seed: dict | None = None
    cpe_w_default: float = 0.1
    cpe_n_default: float = 0.80
    ecm_rescale_target_r0: float | None = 0.1334
    autoeis_num_warmup: int = 500
    autoeis_num_samples: int = 200

    def replace(self, **settings) -> "OracleConfig":
        """A copy with ``settings`` (field names) changed; an unknown name raises TypeError."""
        unknown = set(settings) - FIELD_NAMES
        if unknown:
            raise TypeError(f"unknown oracle setting(s): {sorted(unknown)}")
        return dataclasses.replace(self, **settings)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(OracleConfig))

#: where each field lives in config_oracle_*.yml (section path, key). Fields not listed
#: have no YAML key; frequencies, chemistry and preset_constants are resolved by
#: :func:`settings_from_yaml` itself.
YAML_KEYS: dict[str, tuple[str, ...]] = {
    **{k: ("cycling", k) for k in ("n_cycles", "temperature_K", "rest_s", "initial_soc", "thermal",
                                   "T_ambient_K", "h_total_W_per_m2K", "use_temperature_protocol",
                                   "discharge_mode")},
    **{k: ("bol", k) for k in ("parameter_scales", "parameter_overrides", "contact_resistance_ohm")},
    "degradation_preset": ("degradation", "preset"),
    **{k: ("degradation", k) for k in ("eol_capacity_fraction", "capacity_check", "ec_diffusivity_base_factor",
                                       "lam_ceiling", "dod_lam_scale")},
    **{k: ("protocol_scaling", k) for k in ("real_cell_capacity_mah", "kinetics_scale", "sei_rate_scale",
                                            "dead_li_decay_scale", "plating_rate_scale")},
    "eis_noise_level": ("eis", "noise_level"),
    "eis_noise_model": ("eis", "noise_model"),
    "eis_mode": ("eis", "mode"),
    "eis_amplitude_mA": ("eis", "amplitude_mA"),
    "eis_sweep_overhead_s": ("eis", "sweep_overhead_s"),
    "eis_sweep_periods": ("eis", "sweep_periods"),
    "radial_mesh": ("bol", "radial_mesh"),
    **{k: ("eis", k) for k in ("seed", "E_a_J_per_mol", "E_a_electrolyte_J_per_mol", "noise_combined_flicker_frac",
                               "noise_combined_white_frac", "soc_clip_min", "soc_clip_max")},
    "linkk_c": ("eis", "linkk", "c"),
    "linkk_max_M": ("eis", "linkk", "max_M"),
    "circuit": ("ecm", "circuit"),
    "cpe_w_seed": ("ecm", "cpe_seeds", "w"),
    "cpe_n_seed": ("ecm", "cpe_seeds", "n"),
    "cpe_w_default": ("ecm", "cpe_seeds", "w_default"),
    "cpe_n_default": ("ecm", "cpe_seeds", "n_default"),
    "ecm_rescale_target_r0": ("ecm", "rescale_target_r0"),
    "autoeis_num_warmup": ("ecm", "autoeis", "num_warmup"),
    "autoeis_num_samples": ("ecm", "autoeis", "num_samples"),
    "emergency_solver_rtol": ("solver", "emergency", "rtol"),
    "emergency_solver_atol": ("solver", "emergency", "atol"),
    "dfn_solver_rtol": ("solver", "dfn", "rtol"),
    "dfn_solver_atol": ("solver", "dfn", "atol"),
    "dfn_solver_dt_max_s": ("solver", "dfn", "dt_max_s"),
    **{k: ("protocol_bounds", k) for k in ("c_min_mA", "c_max_mA", "c2_min_mA", "c2_max_mA", "dur_min_s",
                                           "dur_max_s", "v_charge_max", "v_discharge_min", "charge_stage_max_s",
                                           "dfn_max_crate")},
}
#: YAML keys read by the mappers that are not a field of their own (the EIS grid, the
#: parameter set, the per-preset constants) or that belong to a consumer of the oracle
#: (battery-forecast's exp2 capacity mapping and candidate-grid ceiling)
OTHER_YAML_KEYS = frozenset({
    ("cycling", "parameter_set"), ("cycling", "chemistry"),
    ("eis", "frequencies_hz"), ("eis", "freq_min_hz"), ("eis", "freq_max_hz"), ("eis", "n_freq_points"),
    ("degradation", "preset_constants"),
    ("protocol_scaling", "real_cell_capacity"), ("protocol_scaling", "charge_envelope_crate"),
    ("protocol_scaling", "c_rate_range_mA"), ("protocol_scaling", "c_rate_ceiling_fraction"),
})
_SECTIONS = frozenset(p[0] for p in YAML_KEYS.values())
#: values that are themselves mappings (not walked for unknown keys)
_MAPPING_VALUES = frozenset({("bol", "parameter_scales"), ("bol", "parameter_overrides"), ("bol", "radial_mesh"),
                             ("ecm", "cpe_seeds", "w"), ("ecm", "cpe_seeds", "n"),
                             ("degradation", "preset_constants")})


def _cast(name: str, value):
    """``value`` as the field's type (YAML gives ints for floats, strings for numbers in e)."""
    default = getattr(_DEFAULT, name)
    if value is None or isinstance(default, (dict, str, type(None))) or default is None:
        return value
    if isinstance(default, bool):
        return bool(value)
    return type(default)(value)


def unknown_yaml_keys(cfg: dict) -> list[str]:
    """Keys under the oracle's sections that no field or mapper reads (a typo, or a
    setting that no longer exists)."""
    known = set(YAML_KEYS.values()) | OTHER_YAML_KEYS
    out = []

    def walk(d, path):
        for k, v in d.items():
            p = path + (k,)
            if isinstance(v, dict) and p not in _MAPPING_VALUES and p not in known:
                walk(v, p)
            elif p not in known:
                out.append(".".join(p))

    for sec in _SECTIONS:
        if isinstance(cfg.get(sec), dict):
            walk(cfg[sec], (sec,))
    return sorted(out)


def settings_from_yaml(cfg: dict, preset: str | None = None) -> dict:
    """The fields a config_oracle_*.yml dict sets (absent ones are left to the defaults).
    ``preset`` overrides ``degradation.preset``. Unknown keys under the oracle's sections
    raise ValueError."""
    bad = unknown_yaml_keys(cfg)
    if bad:
        raise ValueError(f"unknown oracle config key(s): {bad}")
    out = {}
    for name, path in YAML_KEYS.items():
        node = cfg
        for k in path:
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                break
        else:
            out[name] = _cast(name, node)
    cycling = cfg.get("cycling") or {}
    if cycling.get("chemistry") or cycling.get("parameter_set"):
        out["chemistry"] = cycling.get("chemistry") or cycling.get("parameter_set")
    if preset:
        out["degradation_preset"] = preset
    constants = (cfg.get("degradation") or {}).get("preset_constants") or {}
    if constants:
        out["preset_constants"] = constants.get(out.get("degradation_preset", _DEFAULT.degradation_preset))
    eis = cfg.get("eis") or {}
    if eis.get("frequencies_hz"):       # an explicit grid (e.g. the instrument's) wins over min/max/n
        out["frequencies"] = np.asarray(eis["frequencies_hz"], float)
    elif any(k in eis for k in ("freq_min_hz", "freq_max_hz", "n_freq_points")):
        out["frequencies"] = np.logspace(np.log10(float(eis.get("freq_min_hz", 0.01))),
                                         np.log10(float(eis.get("freq_max_hz", 1e4))),
                                         int(eis.get("n_freq_points", 60)))
    return out


_DEFAULT = OracleConfig()
