"""battery_oracle — a standalone PyBaMM/SPMe battery *oracle* (simulator).

The simulated counterpart to a physical battery / BattMAP lab interface: given a
6-D charge/discharge protocol it runs a PyBaMM SPMe simulation, synthesises an
EIS spectrum, and fits an equivalent-circuit model — returning the same kind of
featurised state a real cell would. Drop-in usable as the "battery source" in an
active-learning loop.

    from battery_oracle import PyBaMMOracle, make_pybamm_candidates

    oracle = PyBaMMOracle(degradation_preset="accelerated")  # ready to call immediately
    for protocol in make_pybamm_candidates():
        oracle(protocol)

Optional extras:
  * ``[autoeis]`` — Bayesian ECM fitting (falls back to a fast Randles stub without it)
  * ``[drt]``     — distribution-of-relaxation-times peaks (hybrid-drt)
  * ``[tune]``    — Optuna calibration engine (``battery_oracle.tune``)
"""
from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Configure JAX BEFORE any import below that pulls it in transitively
# (battery_oracle.oracle -> pybamm -> jax). This package is designed to be
# "imported and reused independently" (see module docstring), so it can't
# rely on every caller setting this up first the way battery_forecast's
# entry-point scripts currently have to. Without it, a CPU-only process (no
# --gres=gpu) hard-crashes: jax[cuda12]'s eager CUDA backend registration
# raises "FAILED_PRECONDITION: No visible GPU devices" instead of falling
# back, and JAX_PLATFORMS (plural) -- not the legacy JAX_PLATFORM_NAME -- is
# the only setting that actually stops jax from probing the cuda plugin at
# all. GPU presence is inferred from the environment alone (no subprocess
# probe) -- CUDA_VISIBLE_DEVICES is set by the scheduler (Slurm --gres=gpu,
# Docker --gpus) whenever a GPU is actually allocated.
if os.environ.get('CUDA_VISIBLE_DEVICES'):
    _HAS_GPU = True
    os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu')
else:
    _HAS_GPU = False
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from battery_oracle._circuit import (
    ACTION_FEATURE_NAMES,
    DEFAULT_CIRCUIT,
    ECM_PARAM_NAMES,
)
from battery_oracle.experiment import (
    build_oracle_from_config,
    build_oracle_from_oracle_config,
    load_default_ecm_circuit,
    load_experiment_config,
    load_oracle_config,
    oracle_kwargs_from_config,
    oracle_kwargs_from_oracle_config,
    protocols_from_config,
    run_experiment,
)
from battery_oracle.oracle import (
    STATE_VECTOR_SCHEMA,
    CycleResult,
    FailureKind,
    OracleFailure,
    PyBaMMOracle,
    autoeis_ecm,
    make_pybamm_candidates,
    randles_stub_ecm,
    state_vector_schema,
)
from battery_oracle.protocol import Protocol
from battery_oracle.tune import (
    calibrate_drift,
    calibrate_oracle,
    compute_real_targets,
    write_calibration_summary,
    write_oracle_config,
)
from battery_oracle.tune_plots import plot_eis_comparison, plot_tune_oracle_summary

try:
    __version__ = _pkg_version("battery-oracle")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "PyBaMMOracle",
    "Protocol",
    "CycleResult",
    "OracleFailure",
    "FailureKind",
    "STATE_VECTOR_SCHEMA",
    "state_vector_schema",
    "make_pybamm_candidates",
    "randles_stub_ecm",
    "autoeis_ecm",
    "DEFAULT_CIRCUIT",
    "ECM_PARAM_NAMES",
    "ACTION_FEATURE_NAMES",
    "calibrate_oracle",
    "calibrate_drift",
    "write_oracle_config",
    "write_calibration_summary",
    "compute_real_targets",
    "plot_tune_oracle_summary",
    "plot_eis_comparison",
    "load_experiment_config",
    "oracle_kwargs_from_config",
    "protocols_from_config",
    "build_oracle_from_config",
    "build_oracle_from_oracle_config",
    "run_experiment",
    "load_default_ecm_circuit",
    "load_oracle_config",
    "oracle_kwargs_from_oracle_config",
]

# Deprecation shim (A3.6): the underscore names `_randles_stub_ecm`/
# `_autoeis_ecm` were previously exported from this module (and in __all__).
# They still resolve here -- via module __getattr__, not a plain assignment,
# so every access (not just the first) warns -- but emit a DeprecationWarning
# and will be removed in 0.5.0. Use the public `randles_stub_ecm`/`autoeis_ecm`
# names instead.
_DEPRECATED_ALIASES = {
    "_randles_stub_ecm": "randles_stub_ecm",
    "_autoeis_ecm": "autoeis_ecm",
}


def __getattr__(name: str):
    if name in _DEPRECATED_ALIASES:
        import warnings

        new_name = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"battery_oracle.{name} is deprecated; use {new_name} instead. "
            "The underscore alias will be removed in 0.5.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
