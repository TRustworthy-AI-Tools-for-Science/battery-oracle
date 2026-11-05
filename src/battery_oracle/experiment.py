"""Load an experiment-protocol YAML and build/run a :class:`~battery_oracle.PyBaMMOracle` from it.

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
    history = run_experiment()  # runs the packaged config_experiment_defaults.yml
"""
from __future__ import annotations

import copy
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from battery_oracle.circuit import DEFAULT_CIRCUIT
from battery_oracle.oracle_config import OracleConfig, settings_from_yaml
from battery_oracle.protocol import PROTOCOL_FIELD_NAMES

if TYPE_CHECKING:  # pragma: no cover - typing only
    from battery_oracle.oracle import PyBaMMOracle

_DEFAULT_CONFIG_NAME = "config_experiment_defaults.yml"
_ORACLE_CONFIG_NAME = "config_oracle_defaults.yml"
# Used only if config_oracle_defaults.yml lacks an ecm.circuit field. Aliases the
# package's canonical circuit constant so every default path (direct construction,
# oracle-config, experiment-config) resolves to the same string; the ECM layout is
# still derived from the string, so nothing downstream hard-codes an element count.

# 6-D protocol field order in the experiment-YAML's field-name spelling. The
# ORDER is not redefined here -- it is derived by iterating
# battery_oracle.protocol.PROTOCOL_FIELD_NAMES (itself locked to
# circuit.ACTION_FEATURE_NAMES by an assertion in protocol.py), so there is
# exactly one place the 6-D slot order is decided; this dict only supplies the
# YAML's own field-name spelling for each canonical slot.
_YAML_FIELD_NAME_BY_SLOT = {
    "charge_current_1_mA":  "C_rate_1_mA",
    "charge_current_2_mA":  "C_rate_2_mA",
    "charge_duration_1_h":  "dur_1_h",
    "charge_duration_2_h":  "dur_2_h",
    "discharge_current_mA": "D_rate_mA",
    "discharge_duration_h": "dur_d_h",
}
_PROTOCOL_FIELDS = tuple(_YAML_FIELD_NAME_BY_SLOT[slot] for slot in PROTOCOL_FIELD_NAMES)
_VALID_PRESETS = ("nominal", "accelerated")   # = battery_oracle.degradation.PRESETS (no pybamm import here)
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
        On an unknown ``degradation.preset``, an empty or missing
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


def oracle_kwargs_from_oracle_config(oracle_cfg: dict, preset: str | None = None) -> dict:
    """Every ``PyBaMMOracle`` setting from an oracle YAML dict: the keys it sets (their
    places: :data:`battery_oracle.oracle_config.YAML_KEYS`) over ``OracleConfig``'s
    defaults, plus ``parameter_set`` (a name; the builders resolve it). ``preset``
    overrides ``degradation.preset``. Unknown keys under the oracle's sections raise.

    PyBaMM-free (PyYAML/numpy only).
    """
    kw = OracleConfig().replace(**settings_from_yaml(oracle_cfg, preset)).to_dict()
    kw["parameter_set"] = (oracle_cfg.get("cycling") or {}).get("parameter_set", "Chen2020")
    return kw


def load_default_ecm_circuit() -> str:
    """Return the default ECM circuit: ``ecm.circuit`` of the packaged
    ``config_oracle_defaults.yml``, or :data:`DEFAULT_CIRCUIT` when that field is
    absent. PyYAML only — no PyBaMM import.
    """
    circuit = (load_oracle_config().get("ecm") or {}).get("circuit")
    if isinstance(circuit, str) and circuit.strip():
        return circuit.strip()
    return DEFAULT_CIRCUIT


def _validate_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("Experiment config must be a mapping (YAML dict).")

    for section in ("cycling", "degradation", "eis", "protocols"):
        if section not in cfg:
            raise KeyError(f"Experiment config missing required section: {section!r}")

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
    """Map a validated experiment config ``dict`` to :class:`~battery_oracle.PyBaMMOracle` ``**kwargs``.

    Precedence: ``OracleConfig`` defaults < the oracle YAML (``oracle_cfg``, default the
    packaged ``config_oracle_defaults.yml``) < the experiment YAML's ``cycling`` /
    ``degradation`` / ``eis`` / ``ecm.circuit``, which use the oracle YAML's keys except
    ``cycling.real_cell_capacity_mah`` and ``degradation.<calibration scale>`` (the
    oracle YAML's ``protocol_scaling``).

    PyBaMM-free: the parameter set and ECM fitter are carried as plain strings
    (``"parameter_set"`` e.g. ``"Chen2020"``; ``"ecm_fitter"`` e.g. ``"randles"``)
    rather than resolved objects — :func:`build_oracle_from_config` pops and
    resolves them (``parameter_set`` -> ``pybamm.ParameterValues``, ``ecm_fitter``
    -> the ``ecm_model_fn`` callable).
    """
    if oracle_cfg is None:
        oracle_cfg = load_oracle_config()
    merged = copy.deepcopy(oracle_cfg)
    for sec in ("cycling", "degradation", "eis"):
        merged[sec] = {**(merged.get(sec) or {}), **copy.deepcopy(cfg.get(sec) or {})}
    scaling = merged["protocol_scaling"] = dict(merged.get("protocol_scaling") or {})
    if "real_cell_capacity_mah" in merged["cycling"]:
        scaling["real_cell_capacity_mah"] = merged["cycling"].pop("real_cell_capacity_mah")
    for field in _SCALE_FIELDS:
        if field in merged["degradation"]:
            scaling[field] = merged["degradation"].pop(field)
    circuit = (cfg.get("ecm") or {}).get("circuit")
    if circuit:
        merged["ecm"] = {**(merged.get("ecm") or {}), "circuit": circuit}
    kwargs = oracle_kwargs_from_oracle_config(merged, preset=cfg["degradation"]["preset"])
    # Carried as a string (like parameter_set); build_oracle_from_config
    # resolves it to the ecm_model_fn callable, keeping this layer PyBaMM-free.
    kwargs["ecm_fitter"] = (cfg.get("ecm") or {}).get("fitter", _DEFAULT_ECM_FITTER)
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
    """Construct a :class:`~battery_oracle.PyBaMMOracle` from a config dict (or a path to load first).

    The base oracle layer is ``cfg``'s optional top-level ``oracle_config`` field
    (a ``config_oracle_*.yml``, e.g. a tune-oracle calibration output), or else the
    packaged ``config_oracle_defaults.yml``. Resolves the ``parameter_set`` string
    to a ``pybamm.ParameterValues`` and imports :class:`~battery_oracle.PyBaMMOracle` lazily so the
    parse/mapping layer stays PyBaMM-free.
    """
    if not isinstance(cfg, dict):
        cfg = load_experiment_config(cfg)

    oracle_cfg = load_oracle_config(cfg.get("oracle_config"))
    kwargs = oracle_kwargs_from_config(cfg, oracle_cfg)
    pset = kwargs.pop("parameter_set", None)

    from battery_oracle.ecm import autoeis_ecm, randles_stub_ecm
    from battery_oracle.oracle import PyBaMMOracle
    from battery_oracle.parameter_sets import check_chemistry, resolve_parameter_values
    # #14: cross-layer chemistry validation — the config's declared chemistry must be
    # its parameter set, so a YAML calibrated on one cell can't be silently paired
    # with another (mismatched degradation scales/bounds).
    check_chemistry(kwargs.get("chemistry", "Chen2020"), pset)
    if pset:
        # local import — keeps the mapping layer PyBaMM-free; custom sets such as
        # "LiR2032" resolve through battery_oracle.parameter_sets.
        kwargs["parameter_values"] = resolve_parameter_values(pset)

    fitter = kwargs.pop("ecm_fitter", _DEFAULT_ECM_FITTER)
    kwargs["ecm_model_fn"] = autoeis_ecm if fitter == "autoeis" else randles_stub_ecm
    return PyBaMMOracle(**kwargs)


def build_oracle_from_oracle_config(source: str | Path | dict, *, ecm_model_fn=None) -> "PyBaMMOracle":
    """Build a :class:`~battery_oracle.PyBaMMOracle` straight from an oracle/calibration config.

    Unlike :func:`build_oracle_from_config`, this needs no experiment YAML /
    protocols list -- it is the convenient entry point for tooling that just wants
    a calibrated oracle. ``source`` is a path to a ``config_oracle_*.yml``
    or an already-parsed dict. The three-layer precedence collapses to a single oracle-config
    layer here (no experiment overlay).

    The ECM fitter is not part of an oracle config: ``ecm_model_fn`` (default the fast
    Randles stub, :func:`battery_oracle.ecm.randles_stub_ecm`; pass
    :func:`battery_oracle.ecm.autoeis_ecm` for Bayesian fits).
    """
    if isinstance(source, dict):
        oracle_cfg = source
    else:
        oracle_cfg = load_oracle_config(source)

    kwargs = oracle_kwargs_from_oracle_config(oracle_cfg)
    pset = kwargs.pop("parameter_set", None)

    from battery_oracle.ecm import randles_stub_ecm
    from battery_oracle.oracle import PyBaMMOracle
    from battery_oracle.parameter_sets import check_chemistry, resolve_parameter_values
    check_chemistry(kwargs.get("chemistry", "Chen2020"), pset)
    if pset:
        kwargs["parameter_values"] = resolve_parameter_values(pset)

    kwargs.pop("ecm_fitter", None)
    kwargs["ecm_model_fn"] = randles_stub_ecm if ecm_model_fn is None else ecm_model_fn
    return PyBaMMOracle(**kwargs)


def run_experiment(path: str | Path | None = None, *, reset: bool = True) -> list[dict]:
    """Load a config, build the oracle, and run every protocol in order.

    Parameters
    ----------
    path : str, Path, or None, optional
        Path to the experiment YAML. ``None`` (default) loads the packaged
        ``config_experiment_defaults.yml`` via :func:`load_experiment_config`,
        so the function runs a valid default experiment with no external file.
    reset : bool
        If ``True`` (default) call :meth:`PyBaMMOracle.reset` before the first
        protocol to start from a fresh cell.

    Returns
    -------
    list of dict
        The oracle history (``oracle.history``).  If the oracle reaches
        end-of-life (:class:`~battery_oracle.OracleFailure`) partway through, the loop stops and
        the history accumulated so far is returned.
    """
    import numpy as np

    from battery_oracle.failures import OracleFailure

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
            # step raises before appending). battery-forecast's save_oracle_csv .get(...)-guards the ECM
            # fields, so this row exports as an ECM-zero row carrying failure_kind.
            fk = exc.failure_kind
            oracle.history.append({
                "call_idx":     len(oracle.history),
                "failed":       True,
                "failure_kind": fk,
                "protocol": (
                    np.asarray(exc.protocol).copy()
                    if exc.protocol is not None
                    else np.asarray(protocol).copy()
                ),
            })
            break
    return oracle.history
