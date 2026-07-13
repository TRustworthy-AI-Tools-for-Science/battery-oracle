"""Load an experiment-protocol YAML and build/run a :class:`PyBaMMOracle` from it.

The one shipped YAML that the oracle actually *executes*. ``config_oracle_defaults.yml``
documents the oracle's internal defaults; this module instead reads a protocol
definition (see ``config_experiment_defaults.yml``) and turns it into a running
oracle.

The layer is deliberately split so the parse + kwargs-mapping functions are
PyBaMM-free and unit-testable in isolation:

  * :func:`load_experiment_config`   — parse + validate (PyYAML only)
  * :func:`oracle_kwargs_from_config`— YAML field -> ``PyBaMMOracle`` kwarg (numpy only)
  * :func:`protocols_from_config`    — 6-D protocol vectors (numpy only)
  * :func:`build_oracle_from_config` — construct the oracle (imports PyBaMM lazily)
  * :func:`run_experiment`           — build + run every protocol, return the history

Example
-------
    from battery_oracle import run_experiment
    history = run_experiment("config_experiment_defaults.yml")
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from battery_oracle._circuit import DEFAULT_CIRCUIT

if TYPE_CHECKING:  # pragma: no cover - typing only
    from battery_oracle.oracle import PyBaMMOracle

_DEFAULT_CONFIG_NAME = "config_experiment_defaults.yml"
_ORACLE_CONFIG_NAME = "config_oracle_defaults.yml"
# Used only if config_oracle_defaults.yml lacks an ecm.circuit field. Aliases the
# package's canonical circuit constant so every default path (direct construction,
# oracle-config, experiment-config) resolves to the same string; the ECM layout is
# still derived from the string, so nothing downstream hard-codes an element count.
_FALLBACK_ECM_CIRCUIT = DEFAULT_CIRCUIT

# 6-D protocol field order — matches PyBaMMOracle / ACTION_FEATURE_NAMES:
#   [C_rate_1_mA, C_rate_2_mA, dur_1_h, dur_2_h, D_rate_mA, dur_d_h]
_PROTOCOL_FIELDS = (
    "C_rate_1_mA", "C_rate_2_mA", "dur_1_h", "dur_2_h", "D_rate_mA", "dur_d_h",
)
_VALID_MODELS = ("SPMe", "SPM", "DFN")
_VALID_PRESETS = ("nominal", "accelerated", "severe")
# Packaged per-dataset calibration configs (#12), selectable via config_dataset=.
# Each ships as config_oracle_{name}.yml alongside config_oracle_defaults.yml.
_VALID_DATASETS = ("calce", "oxford", "matr")
# ECM fitter. 'randles' is the fast analytic stub (available on the core install);
# 'autoeis' is the Bayesian fit and needs the [autoeis] extra. Default 'randles'
# so a bare `run_experiment` works out of the box.
_VALID_ECM_FITTERS = ("randles", "autoeis")
_DEFAULT_ECM_FITTER = "randles"
# Optional per-instance calibration scales (default 1.0 when omitted).
_SCALE_FIELDS = (
    "kinetics_scale", "sei_rate_scale", "dead_li_decay_scale", "plating_rate_scale",
)


def load_experiment_config(path: str | Path | None = None) -> dict:
    """Parse and validate an experiment-protocol YAML into a plain ``dict``.

    Parameters
    ----------
    path : str or Path, optional
        Path to the YAML file.  ``None`` (default) loads the packaged
        ``config_experiment_defaults.yml`` via :mod:`importlib.resources`, so the
        function returns a valid config with no external file.

    Returns
    -------
    dict
        The validated config.

    Raises
    ------
    ValueError
        On an unknown ``model.type`` / ``degradation.preset``, an empty or missing
        ``protocols`` list, or a protocol missing any of the six required fields.
    KeyError
        On a missing required top-level section.

    Notes
    -----
    PyYAML only — no PyBaMM import, so this is fast and unit-testable.
    """
    if path is None:
        with resources.files("battery_oracle").joinpath(
            _DEFAULT_CONFIG_NAME
        ).open("r") as fh:
            cfg = yaml.safe_load(fh)
    else:
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
    _validate_config(cfg)
    return cfg


def load_oracle_config(path: str | Path | None = None) -> dict:
    """Parse an oracle-defaults YAML into a plain ``dict``.

    Parameters
    ----------
    path : str or Path, optional
        Path to a ``config_oracle_*.yml`` file (e.g. a tune-oracle-skill
        calibration output). ``None`` (default) loads the packaged
        ``config_oracle_defaults.yml``.

    Returns
    -------
    dict
        The parsed config, unvalidated -- every consumer reads fields via
        ``.get(key, <PyBaMMOracle's own default>)``, so a partial or
        hand-edited file degrades gracefully rather than raising.

    Notes
    -----
    PyYAML only -- no PyBaMM import, mirrors :func:`load_experiment_config`.
    """
    if path is None:
        with resources.files("battery_oracle").joinpath(
            _ORACLE_CONFIG_NAME
        ).open("r") as fh:
            return yaml.safe_load(fh) or {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _resolve_dataset_config(name: str) -> dict:
    """Load a packaged per-dataset calibration config (#12) by short name.

    ``name`` is one of :data:`_VALID_DATASETS` (calce/oxford/matr); resolves to
    the packaged ``config_oracle_{name}.yml`` via ``importlib.resources`` (same
    mechanism as :func:`load_oracle_config`). PyYAML only — no PyBaMM import.

    If no dataset-specific file is packaged yet, returns ``{}`` so that
    :func:`oracle_kwargs_from_oracle_config` falls back to the
    ``config_oracle_defaults.yml`` / PyBaMMOracle Python-literal defaults.
    """
    if name not in _VALID_DATASETS:
        raise ValueError(
            f"Unknown config_dataset {name!r}; choose one of {_VALID_DATASETS}."
        )
    try:
        with resources.files("battery_oracle").joinpath(
            f"config_oracle_{name}.yml"
        ).open("r") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


def oracle_kwargs_from_oracle_config(oracle_cfg: dict, preset: str | None = None) -> dict:
    """Map an oracle-defaults YAML dict to a full ``PyBaMMOracle(**kwargs)`` dict.

    Every field is read with ``.get(key, <PyBaMMOracle's own Python literal
    default>)`` so an absent field falls back to PyBaMMOracle's constructor
    default rather than raising -- this is the base layer of the three-layer
    precedence (PyBaMMOracle Python defaults < oracle YAML < experiment YAML;
    see :func:`oracle_kwargs_from_config`).

    ``preset`` overrides the resolved ``degradation.preset`` (used when an
    experiment YAML specifies its own preset); defaults to the oracle YAML's
    own ``degradation.preset``. Also resolves ``degradation.preset_constants``
    down to the numeric-only dict for the resolved preset.

    PyBaMM-free (PyYAML/numpy only), matching :func:`oracle_kwargs_from_config`'s
    design.
    """
    model_cfg = oracle_cfg.get("model", {}) or {}
    cycling = oracle_cfg.get("cycling", {}) or {}
    solver = oracle_cfg.get("solver", {}) or {}
    solver_primary = solver.get("primary", {}) or {}
    solver_emergency = solver.get("emergency", {}) or {}
    solver_dfn = solver.get("dfn", {}) or {}
    bounds = oracle_cfg.get("protocol_bounds", {}) or {}
    eis = oracle_cfg.get("eis", {}) or {}
    linkk = eis.get("linkk", {}) or {}
    ecm = oracle_cfg.get("ecm", {}) or {}
    cpe_seeds = ecm.get("cpe_seeds", {}) or {}
    autoeis = ecm.get("autoeis", {}) or {}
    degradation = oracle_cfg.get("degradation", {}) or {}
    c2_stress = degradation.get("c2_stress", {}) or {}
    preset_constants_all = degradation.get("preset_constants", {}) or {}
    protocol_scaling = oracle_cfg.get("protocol_scaling", {}) or {}

    resolved_preset = preset or degradation.get("preset", "accelerated")
    rescale_r0 = ecm.get("rescale_target_r0", 0.1334)

    freq_min = float(eis.get("freq_min_hz", 0.01))
    freq_max = float(eis.get("freq_max_hz", 10000.0))
    n_freq = int(eis.get("n_freq_points", 60))

    return {
        "model": model_cfg.get("type", "SPMe"),
        "n_cycles": int(cycling.get("n_cycles", 1)),
        "temperature_K": float(cycling.get("temperature_K", 298.15)),
        "parameter_set": cycling.get("parameter_set", "Chen2020"),
        # #14: declared chemistry (defaults to the parameter_set name). Validated
        # against parameter_set in build_oracle_from_config to catch a mismatched
        # calibration YAML (e.g. LFP scales loaded onto an NMC cell).
        "chemistry": cycling.get("chemistry", cycling.get("parameter_set", "Chen2020")),
        "real_cell_capacity_mah": float(
            protocol_scaling.get("real_cell_capacity_mah_legacy_default", 200.0)
        ),
        "rest_s": float(cycling.get("rest_s", 1200.0)),
        "initial_soc": float(cycling.get("initial_soc", 0.8)),
        "thermal": cycling.get("thermal", "isothermal"),
        "T_ambient_K": float(cycling.get("T_ambient_K", 298.15)),
        "h_total_W_per_m2K": float(cycling.get("h_total_W_per_m2K", 10.0)),
        "use_temperature_protocol": bool(cycling.get("use_temperature_protocol", False)),

        "degradation_preset": resolved_preset,
        "eol_capacity_fraction": float(degradation.get("eol_capacity_fraction", 0.80)),
        "capacity_check": bool(degradation.get("capacity_check", False)),
        "ec_diffusivity_base_factor": float(degradation.get("ec_diffusivity_base_factor", 0.25)),
        "lam_ceiling": float(degradation.get("lam_ceiling", 0.95)),
        "dod_lam_scale": float(degradation.get("dod_lam_scale", 0.0)),
        "c2_stress_scale": float(c2_stress.get("scale", 0.0)),
        "c2_stress_slope_mah_per_ma": float(c2_stress.get("slope_mah_per_ma", 0.0794)),
        "c2_stress_ref_ma": float(c2_stress.get("ref_ma", 75.27)),
        "preset_constants": preset_constants_all.get(resolved_preset),
        "kinetics_scale": float(protocol_scaling.get("kinetics_scale", 1.0)),
        "sei_rate_scale": float(protocol_scaling.get("sei_rate_scale", 1.0)),
        "dead_li_decay_scale": float(protocol_scaling.get("dead_li_decay_scale", 1.0)),
        "plating_rate_scale": float(protocol_scaling.get("plating_rate_scale", 1.0)),

        "frequencies": np.logspace(np.log10(freq_min), np.log10(freq_max), n_freq),
        "eis_noise_level": float(eis.get("noise_level", 0.02)),
        "eis_noise_model": eis.get("noise_model", "combined"),
        "E_a_J_per_mol": float(eis.get("E_a_J_per_mol", 30e3)),
        "E_a_electrolyte_J_per_mol": float(eis.get("E_a_electrolyte_J_per_mol", 15e3)),
        "eis_drift_scale": float(eis.get("drift_scale", 0.0)),
        "eis_drift_tau_s": float(eis.get("drift_tau_s", 600.0)),
        "eis_drift_n_periods": float(eis.get("drift_n_periods", 4.0)),
        "noise_combined_flicker_frac": float(eis.get("noise_combined_flicker_frac", 0.75)),
        "noise_combined_white_frac": float(eis.get("noise_combined_white_frac", 0.25)),
        "soc_clip_min": float(eis.get("soc_clip_min", 0.05)),
        "soc_clip_max": float(eis.get("soc_clip_max", 0.99)),
        "linkk_c": float(linkk.get("c", 0.85)),
        "linkk_max_M": int(linkk.get("max_M", 50)),

        "circuit": ecm.get("circuit") or _FALLBACK_ECM_CIRCUIT,
        "cpe_w_seed": cpe_seeds.get("w"),
        "cpe_n_seed": cpe_seeds.get("n"),
        "cpe_w_default": float(cpe_seeds.get("w_default", 0.1)),
        "cpe_n_default": float(cpe_seeds.get("n_default", 0.80)),
        "ecm_rescale_target_r0": float(rescale_r0) if rescale_r0 is not None else None,
        "autoeis_num_warmup": int(autoeis.get("num_warmup", 500)),
        "autoeis_num_samples": int(autoeis.get("num_samples", 200)),

        "solver_rtol": float(solver_primary.get("rtol", 1e-3)),
        "solver_atol": float(solver_primary.get("atol", 1e-6)),
        "solver_dt_max_s": float(solver_primary.get("dt_max_s", 60.0)),
        "emergency_solver_rtol": float(solver_emergency.get("rtol", 1e-2)),
        "emergency_solver_atol": float(solver_emergency.get("atol", 1e-5)),
        "emergency_solver_dt_max_s": float(solver_emergency.get("dt_max_s", 10.0)),
        "dfn_solver_rtol": float(solver_dfn.get("rtol", 1e-6)),
        "dfn_solver_atol": float(solver_dfn.get("atol", 1e-8)),
        "dfn_solver_dt_max_s": float(solver_dfn.get("dt_max_s", 1.0)),

        "c_min_mA": float(bounds.get("c_min_mA", 50.0)),
        "c_max_mA": float(bounds.get("c_max_mA", 10_000.0)),
        "c2_min_mA": float(bounds.get("c2_min_mA", 20.0)),
        "c2_max_mA": float(bounds.get("c2_max_mA", 10_000.0)),
        "dur_min_s": float(bounds.get("dur_min_s", 60.0)),
        "dur_max_s": float(bounds.get("dur_max_s", 28_800.0)),
        "v_charge_max": float(bounds.get("v_charge_max", 4.3)),
        "v_discharge_min": float(bounds.get("v_discharge_min", 3.0)),
        "charge_stage_max_s": float(bounds.get("charge_stage_max_s", 900.0)),
        "dfn_max_crate": float(bounds.get("dfn_max_crate", 1.5)),
    }


def load_default_ecm_circuit() -> str:
    """Return the default ECM circuit string from ``config_oracle_defaults.yml``.

    Reads ``ecm.circuit`` from the packaged oracle config so the ECM structure is
    defined in YAML, not hard-coded. Falls back to the canonical 9-parameter circuit
    only if the field is absent. PyYAML only — no PyBaMM import.
    """
    try:
        with resources.files("battery_oracle").joinpath(
            _ORACLE_CONFIG_NAME
        ).open("r") as fh:
            cfg = yaml.safe_load(fh) or {}
        circuit = (cfg.get("ecm") or {}).get("circuit")
        if isinstance(circuit, str) and circuit.strip():
            return circuit.strip()
    except Exception:
        pass
    return _FALLBACK_ECM_CIRCUIT


def _validate_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("Experiment config must be a mapping (YAML dict).")

    for section in ("model", "cycling", "degradation", "eis", "protocols"):
        if section not in cfg:
            raise KeyError(f"Experiment config missing required section: {section!r}")

    model_type = cfg["model"].get("type")
    if model_type not in _VALID_MODELS:
        raise ValueError(
            f"Unknown model.type {model_type!r}; choose one of {_VALID_MODELS}."
        )

    preset = cfg["degradation"].get("preset")
    if preset not in _VALID_PRESETS:
        raise ValueError(
            f"Unknown degradation.preset {preset!r}; choose one of {_VALID_PRESETS}."
        )

    # 'ecm' is optional; validate the fitter name if present.
    fitter = cfg.get("ecm", {}).get("fitter", _DEFAULT_ECM_FITTER)
    if fitter not in _VALID_ECM_FITTERS:
        raise ValueError(
            f"Unknown ecm.fitter {fitter!r}; choose one of {_VALID_ECM_FITTERS}."
        )

    # 'oracle_config' is optional: a path to a custom oracle-defaults YAML
    # used as the base layer instead of the packaged config_oracle_defaults.yml.
    oracle_config = cfg.get("oracle_config")
    if oracle_config is not None and not isinstance(oracle_config, (str, Path)):
        raise ValueError("Experiment config 'oracle_config' must be a path string or null.")

    protocols = cfg["protocols"]
    if not isinstance(protocols, list) or not protocols:
        raise ValueError("Experiment config 'protocols' must be a non-empty list.")
    for i, proto in enumerate(protocols):
        if not isinstance(proto, dict):
            raise ValueError(f"protocols[{i}] must be a mapping of the 6 protocol fields.")
        missing = [f for f in _PROTOCOL_FIELDS if f not in proto]
        if missing:
            raise ValueError(f"protocols[{i}] is missing field(s): {missing}")


def oracle_kwargs_from_config(cfg: dict, oracle_cfg: dict | None = None) -> dict:
    """Map a validated config ``dict`` to :class:`PyBaMMOracle` ``**kwargs``.

    Three-layer precedence: starts from ``oracle_kwargs_from_oracle_config``'s
    base layer (``oracle_cfg``, defaulting to the packaged
    ``config_oracle_defaults.yml`` if not supplied), then overlays every field
    the experiment config ``cfg`` actually specifies. Fields ``cfg`` omits fall
    back to the base layer's value rather than raising, so a minimal
    experiment YAML (just the five required sections) still works.

    PyBaMM-free: the parameter set and ECM fitter are carried as plain strings
    (``"parameter_set"`` e.g. ``"Chen2020"``; ``"ecm_fitter"`` e.g. ``"randles"``)
    rather than resolved objects — :func:`build_oracle_from_config` pops and
    resolves them (``parameter_set`` -> ``pybamm.ParameterValues``, ``ecm_fitter``
    -> the ``ecm_model_fn`` callable).
    """
    cycling = cfg["cycling"]
    degradation = cfg["degradation"]
    eis = cfg["eis"]

    if oracle_cfg is None:
        oracle_cfg = load_oracle_config()
    kwargs = oracle_kwargs_from_oracle_config(oracle_cfg, preset=degradation.get("preset"))

    kwargs["model"] = cfg["model"]["type"]
    kwargs["n_cycles"] = int(cycling.get("n_cycles", kwargs["n_cycles"]))
    kwargs["temperature_K"] = float(cycling.get("temperature_K", kwargs["temperature_K"]))
    kwargs["parameter_set"] = cycling.get("parameter_set", kwargs["parameter_set"])
    kwargs["chemistry"] = cycling.get("chemistry", kwargs["chemistry"])
    kwargs["real_cell_capacity_mah"] = float(
        cycling.get("real_cell_capacity_mah", kwargs["real_cell_capacity_mah"])
    )
    kwargs["rest_s"] = float(cycling.get("rest_s", kwargs["rest_s"]))
    kwargs["initial_soc"] = float(cycling.get("initial_soc", kwargs["initial_soc"]))
    kwargs["thermal"] = cycling.get("thermal", kwargs["thermal"])
    kwargs["T_ambient_K"] = float(cycling.get("T_ambient_K", kwargs["T_ambient_K"]))
    kwargs["h_total_W_per_m2K"] = float(
        cycling.get("h_total_W_per_m2K", kwargs["h_total_W_per_m2K"])
    )
    kwargs["use_temperature_protocol"] = bool(
        cycling.get("use_temperature_protocol", kwargs["use_temperature_protocol"])
    )

    kwargs["degradation_preset"] = degradation["preset"]
    kwargs["eol_capacity_fraction"] = float(
        degradation.get("eol_capacity_fraction", kwargs["eol_capacity_fraction"])
    )
    kwargs["capacity_check"] = bool(degradation.get("capacity_check", kwargs["capacity_check"]))
    kwargs["dod_lam_scale"] = float(degradation.get("dod_lam_scale", kwargs["dod_lam_scale"]))
    kwargs["c2_stress_scale"] = float(
        degradation.get("c2_stress_scale", kwargs["c2_stress_scale"])
    )
    kwargs["c2_stress_slope_mah_per_ma"] = float(
        degradation.get("c2_stress_slope_mah_per_ma", kwargs["c2_stress_slope_mah_per_ma"])
    )
    kwargs["c2_stress_ref_ma"] = float(
        degradation.get("c2_stress_ref_ma", kwargs["c2_stress_ref_ma"])
    )
    for field in _SCALE_FIELDS:
        kwargs[field] = float(degradation.get(field, kwargs[field]))

    freq_min = float(eis.get("freq_min_hz", oracle_cfg.get("eis", {}).get("freq_min_hz", 0.01)))
    freq_max = float(eis.get("freq_max_hz", oracle_cfg.get("eis", {}).get("freq_max_hz", 10000.0)))
    n_freq = int(eis.get("n_freq_points", oracle_cfg.get("eis", {}).get("n_freq_points", 60)))
    kwargs["frequencies"] = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freq)
    kwargs["eis_noise_level"] = float(eis.get("noise_level", kwargs["eis_noise_level"]))
    kwargs["eis_noise_model"] = eis.get("noise_model", kwargs["eis_noise_model"])
    kwargs["E_a_J_per_mol"] = float(eis.get("E_a_J_per_mol", kwargs["E_a_J_per_mol"]))
    kwargs["E_a_electrolyte_J_per_mol"] = float(
        eis.get("E_a_electrolyte_J_per_mol", kwargs["E_a_electrolyte_J_per_mol"])
    )
    kwargs["eis_drift_scale"] = float(eis.get("drift_scale", kwargs["eis_drift_scale"]))
    kwargs["eis_drift_tau_s"] = float(eis.get("drift_tau_s", kwargs["eis_drift_tau_s"]))
    kwargs["eis_drift_n_periods"] = float(eis.get("drift_n_periods", kwargs["eis_drift_n_periods"]))

    # Carried as a string (like parameter_set); build_oracle_from_config
    # resolves it to the ecm_model_fn callable, keeping this layer PyBaMM-free.
    kwargs["ecm_fitter"] = cfg.get("ecm", {}).get("fitter", _DEFAULT_ECM_FITTER)
    # ECM structure -> PyBaMMOracle(circuit=...). Experiment YAML's ecm.circuit
    # wins; otherwise falls back to the oracle-config base layer's circuit.
    kwargs["circuit"] = cfg.get("ecm", {}).get("circuit") or kwargs["circuit"]
    return kwargs


def protocols_from_config(cfg: dict) -> list[np.ndarray]:
    """Return the config's protocols as shape-``(6,)`` float64 vectors.

    Order matches :data:`_PROTOCOL_FIELDS` /
    :data:`battery_oracle.ACTION_FEATURE_NAMES`.
    """
    return [
        np.array([float(proto[f]) for f in _PROTOCOL_FIELDS], dtype=np.float64)
        for proto in cfg["protocols"]
    ]


def build_oracle_from_config(
    cfg: dict | str | Path, config_dataset: str | None = None
) -> "PyBaMMOracle":
    """Construct a :class:`PyBaMMOracle` from a config dict (or a path to load first).

    The base oracle layer is resolved in this order:
      * ``config_dataset`` (this kwarg, or the experiment YAML's top-level
        ``config_dataset`` field) -> the packaged ``config_oracle_{name}.yml``
        for one of :data:`_VALID_DATASETS` (#12); OR
      * ``cfg``'s optional top-level ``oracle_config`` field -> a custom
        ``config_oracle_*.yml`` (e.g. a tune-oracle calibration output); OR
      * the packaged ``config_oracle_defaults.yml``.

    ``config_dataset`` and ``oracle_config`` are mutually exclusive (both select
    the base layer) -- setting both raises. Resolves the ``parameter_set`` string
    to a ``pybamm.ParameterValues`` and imports :class:`PyBaMMOracle` lazily so the
    parse/mapping layer stays PyBaMM-free.
    """
    if not isinstance(cfg, dict):
        cfg = load_experiment_config(cfg)

    config_dataset = config_dataset or cfg.get("config_dataset")
    oracle_config_path = cfg.get("oracle_config")
    if config_dataset and oracle_config_path:
        raise ValueError(
            "config_dataset and oracle_config both select the oracle base layer; "
            "set only one."
        )
    if config_dataset:
        oracle_cfg = _resolve_dataset_config(config_dataset)
    else:
        oracle_cfg = load_oracle_config(oracle_config_path)
    kwargs = oracle_kwargs_from_config(cfg, oracle_cfg)
    pset = kwargs.pop("parameter_set", None)

    from battery_oracle.oracle import (
        _SUPPORTED_CHEMISTRIES,
        PyBaMMOracle,
        _autoeis_ecm,
        _randles_stub_ecm,
    )
    # #14: cross-layer chemistry validation — the config's declared chemistry must
    # resolve to the same PyBaMM parameter set, so an LFP-calibrated YAML can't be
    # silently paired with an NMC cell (mismatched degradation scales/bounds).
    chem = kwargs.get("chemistry", "Chen2020")
    if chem not in _SUPPORTED_CHEMISTRIES:
        raise ValueError(
            f"Unknown chemistry {chem!r}; choose one of {sorted(_SUPPORTED_CHEMISTRIES)}."
        )
    if pset:
        resolved = _SUPPORTED_CHEMISTRIES.get(chem, chem)
        if resolved != pset:
            raise ValueError(
                f"Config chemistry {chem!r} (-> {resolved}) does not match "
                f"parameter_set {pset!r}. Point the experiment at a calibration YAML "
                f"whose chemistry matches its parameter_set."
            )
        import pybamm  # local import — keeps the mapping layer PyBaMM-free
        kwargs["parameter_values"] = pybamm.ParameterValues(pset)

    fitter = kwargs.pop("ecm_fitter", _DEFAULT_ECM_FITTER)
    kwargs["ecm_model_fn"] = _autoeis_ecm if fitter == "autoeis" else _randles_stub_ecm
    return PyBaMMOracle(**kwargs)


def build_oracle_from_oracle_config(source: str | Path | dict) -> "PyBaMMOracle":
    """Build a :class:`PyBaMMOracle` straight from an oracle/calibration config.

    Unlike :func:`build_oracle_from_config`, this needs no experiment YAML /
    protocols list -- it is the convenient entry point for tooling that just wants
    a calibrated oracle. ``source`` is a packaged dataset name (one of
    :data:`_VALID_DATASETS`), a path to a ``config_oracle_*.yml``, or an already-
    parsed dict. The three-layer precedence collapses to a single oracle-config
    layer here (no experiment overlay).
    """
    if isinstance(source, dict):
        oracle_cfg = source
    elif isinstance(source, str) and source in _VALID_DATASETS:
        oracle_cfg = _resolve_dataset_config(source)
    else:
        oracle_cfg = load_oracle_config(source)

    kwargs = oracle_kwargs_from_oracle_config(oracle_cfg)
    pset = kwargs.pop("parameter_set", None)

    from battery_oracle.oracle import (
        _SUPPORTED_CHEMISTRIES,
        PyBaMMOracle,
        _randles_stub_ecm,
    )
    chem = kwargs.get("chemistry", "Chen2020")
    if chem not in _SUPPORTED_CHEMISTRIES:
        raise ValueError(
            f"Unknown chemistry {chem!r}; choose one of {sorted(_SUPPORTED_CHEMISTRIES)}."
        )
    if pset:
        resolved = _SUPPORTED_CHEMISTRIES.get(chem, chem)
        if resolved != pset:
            raise ValueError(
                f"Config chemistry {chem!r} (-> {resolved}) does not match "
                f"parameter_set {pset!r}."
            )
        import pybamm
        kwargs["parameter_values"] = pybamm.ParameterValues(pset)

    # This layer carries an ecm.circuit-derived `circuit`; the fitter is not part
    # of the oracle config, so default to the fast Randles stub.
    kwargs.pop("ecm_fitter", None)
    kwargs["ecm_model_fn"] = _randles_stub_ecm
    return PyBaMMOracle(**kwargs)


def run_experiment(path: str | Path, *, reset: bool = True) -> list[dict]:
    """Load a config, build the oracle, and run every protocol in order.

    Parameters
    ----------
    path : str or Path
        Path to the experiment YAML.
    reset : bool
        If ``True`` (default) call :meth:`PyBaMMOracle.reset` before the first
        protocol to start from a fresh cell.

    Returns
    -------
    list of dict
        The oracle history (``oracle._history``).  If the oracle reaches
        end-of-life (:class:`OracleFailure`) partway through, the loop stops and
        the history accumulated so far is returned.
    """
    import numpy as np

    from battery_oracle.oracle import OracleFailure

    cfg = load_experiment_config(path)
    oracle = build_oracle_from_config(cfg)
    if reset:
        oracle.reset()

    for protocol in protocols_from_config(cfg):
        try:
            oracle(protocol)
        except OracleFailure as exc:
            # Record a terminal audit row so the failure mode is preserved in the
            # returned history (a successful step appends its own row; a failing
            # step raises before appending). save_to_csv .get(...)-guards the ECM
            # fields, so this row exports as an ECM-zero row carrying failure_kind.
            fk = exc.failure_kind
            oracle._history.append({
                "call_idx":     len(oracle._history),
                "model":        getattr(oracle, "_model", None),
                "failed":       True,
                "failure_kind": fk,
                "fidelity":     "failed",
                "protocol": (
                    np.asarray(exc.protocol).copy()
                    if exc.protocol is not None
                    else np.asarray(protocol).copy()
                ),
            })
            break
    return oracle._history
