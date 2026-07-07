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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from battery_oracle.oracle import PyBaMMOracle

_DEFAULT_CONFIG_NAME = "config_experiment_defaults.yml"
_ORACLE_CONFIG_NAME = "config_oracle_defaults.yml"
# Used only if config_oracle_defaults.yml lacks an ecm.circuit field. This is the
# canonical 9-parameter circuit; the ECM layout is still derived from the string,
# so nothing downstream hard-codes an element count.
_FALLBACK_ECM_CIRCUIT = "R1-P2-[R3,P4]-[R5,P6]"

# 6-D protocol field order — matches PyBaMMOracle / ACTION_FEATURE_NAMES:
#   [C_rate_1_mA, C_rate_2_mA, dur_1_h, dur_2_h, D_rate_mA, dur_d_h]
_PROTOCOL_FIELDS = (
    "C_rate_1_mA", "C_rate_2_mA", "dur_1_h", "dur_2_h", "D_rate_mA", "dur_d_h",
)
_VALID_MODELS = ("SPMe", "SPM", "DFN")
_VALID_PRESETS = ("nominal", "accelerated", "severe")
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

    protocols = cfg["protocols"]
    if not isinstance(protocols, list) or not protocols:
        raise ValueError("Experiment config 'protocols' must be a non-empty list.")
    for i, proto in enumerate(protocols):
        if not isinstance(proto, dict):
            raise ValueError(f"protocols[{i}] must be a mapping of the 6 protocol fields.")
        missing = [f for f in _PROTOCOL_FIELDS if f not in proto]
        if missing:
            raise ValueError(f"protocols[{i}] is missing field(s): {missing}")


def oracle_kwargs_from_config(cfg: dict) -> dict:
    """Map a validated config ``dict`` to :class:`PyBaMMOracle` ``**kwargs``.

    PyBaMM-free: the parameter set and ECM fitter are carried as plain strings
    (``"parameter_set"`` e.g. ``"Chen2020"``; ``"ecm_fitter"`` e.g. ``"randles"``)
    rather than resolved objects — :func:`build_oracle_from_config` pops and
    resolves them (``parameter_set`` -> ``pybamm.ParameterValues``, ``ecm_fitter``
    -> the ``ecm_model_fn`` callable).  The ``eis`` block is converted to a
    ``frequencies`` array here (numpy is a core dependency).  Optional calibration
    scales default to ``1.0``.
    """
    cycling = cfg["cycling"]
    degradation = cfg["degradation"]
    eis = cfg["eis"]

    freqs = np.logspace(
        np.log10(float(eis["freq_min_hz"])),
        np.log10(float(eis["freq_max_hz"])),
        int(eis["n_freq_points"]),
    )

    kwargs: dict = {
        "model": cfg["model"]["type"],
        "n_cycles": int(cycling["n_cycles"]),
        "temperature_K": float(cycling["temperature_K"]),
        "parameter_set": cycling.get("parameter_set", "Chen2020"),
        "real_cell_capacity_mah": float(cycling["real_cell_capacity_mah"]),
        "degradation_preset": degradation["preset"],
        "eol_capacity_fraction": float(degradation["eol_capacity_fraction"]),
        "capacity_check": bool(degradation["capacity_check"]),
        "frequencies": freqs,
        "eis_noise_level": float(eis["noise_level"]),
        "eis_noise_model": eis["noise_model"],
        # Carried as a string (like parameter_set); build_oracle_from_config
        # resolves it to the ecm_model_fn callable, keeping this layer PyBaMM-free.
        "ecm_fitter": cfg.get("ecm", {}).get("fitter", _DEFAULT_ECM_FITTER),
        # ECM structure -> PyBaMMOracle(circuit=...). Loaded from the experiment
        # config's ecm.circuit, or the oracle config default. No layout is assumed.
        "circuit": cfg.get("ecm", {}).get("circuit") or load_default_ecm_circuit(),
    }
    for field in _SCALE_FIELDS:
        kwargs[field] = float(degradation.get(field, 1.0))
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


def build_oracle_from_config(cfg: dict | str | Path) -> "PyBaMMOracle":
    """Construct a :class:`PyBaMMOracle` from a config dict (or a path to load first).

    Resolves the ``parameter_set`` string to a ``pybamm.ParameterValues`` and
    imports :class:`PyBaMMOracle` lazily so the parse/mapping layer stays
    PyBaMM-free.
    """
    if not isinstance(cfg, dict):
        cfg = load_experiment_config(cfg)

    kwargs = oracle_kwargs_from_config(cfg)
    pset = kwargs.pop("parameter_set", None)
    if pset:
        import pybamm  # local import — keeps the mapping layer PyBaMM-free
        kwargs["parameter_values"] = pybamm.ParameterValues(pset)

    from battery_oracle.oracle import PyBaMMOracle, _autoeis_ecm, _randles_stub_ecm
    fitter = kwargs.pop("ecm_fitter", _DEFAULT_ECM_FITTER)
    kwargs["ecm_model_fn"] = _autoeis_ecm if fitter == "autoeis" else _randles_stub_ecm
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
    from battery_oracle.oracle import OracleFailure

    cfg = load_experiment_config(path)
    oracle = build_oracle_from_config(cfg)
    if reset:
        oracle.reset()

    for protocol in protocols_from_config(cfg):
        try:
            oracle(protocol)
        except OracleFailure:
            break
    return oracle._history
