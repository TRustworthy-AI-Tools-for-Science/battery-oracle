"""battery_oracle — a standalone PyBaMM/SPMe battery *oracle* (simulator).

The simulated counterpart to a physical battery / BattMAP lab interface: given a
6-D charge/discharge protocol it runs a PyBaMM SPMe simulation, synthesises an
EIS spectrum, and fits an equivalent-circuit model — returning the same kind of
featurised state a real cell would. Drop-in usable as the "battery source" in an
active-learning loop.

    from battery_oracle import PyBaMMOracle, make_pybamm_candidates

    oracle = PyBaMMOracle(degradation_preset="accelerated")
    oracle.reset()
    for protocol in make_pybamm_candidates():
        oracle(protocol)

Optional extras:
  * ``[autoeis]`` — Bayesian ECM fitting (falls back to a fast Randles stub without it)
  * ``[drt]``     — distribution-of-relaxation-times peaks (hybrid-drt)
  * ``[tune]``    — Optuna calibration engine (``battery_oracle.tune``)
"""
from __future__ import annotations

import os
import subprocess

# Configure JAX BEFORE any import below that pulls it in transitively
# (battery_oracle.oracle -> pybamm -> jax). This package is designed to be
# "imported and reused independently" (see module docstring), so it can't
# rely on every caller setting this up first the way battery_forecast's
# entry-point scripts currently have to. Without it, a CPU-only process (no
# --gres=gpu) hard-crashes: jax[cuda12]'s eager CUDA backend registration
# raises "FAILED_PRECONDITION: No visible GPU devices" instead of falling
# back, and JAX_PLATFORMS (plural) -- not the legacy JAX_PLATFORM_NAME -- is
# the only setting that actually stops jax from probing the cuda plugin at
# all.
try:
    _HAS_GPU = bool(os.environ.get('CUDA_VISIBLE_DEVICES')) and (
        subprocess.run(['nvidia-smi'], capture_output=True, check=True) is not None
    )
except (subprocess.CalledProcessError, FileNotFoundError):
    _HAS_GPU = False

if _HAS_GPU:
    os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu')
else:
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from battery_oracle._circuit import (
    ACTION_FEATURE_NAMES,
    DEFAULT_CIRCUIT,
    ECM_PARAM_NAMES,
)
from battery_oracle.experiment import (
    build_oracle_from_config,
    load_default_ecm_circuit,
    load_experiment_config,
    load_oracle_config,
    oracle_kwargs_from_config,
    oracle_kwargs_from_oracle_config,
    protocols_from_config,
    run_experiment,
)
from battery_oracle.oracle import (
    OracleFailure,
    PyBaMMOracle,
    _autoeis_ecm,
    _randles_stub_ecm,
    make_pybamm_candidates,
)
from battery_oracle.tune import (
    calibrate_drift,
    calibrate_oracle,
    compute_real_targets,
    write_calibration_summary,
    write_oracle_config,
)
from battery_oracle.tune_plots import plot_tune_oracle_summary

__version__ = "0.3.0"

__all__ = [
    "PyBaMMOracle",
    "OracleFailure",
    "make_pybamm_candidates",
    "_randles_stub_ecm",
    "_autoeis_ecm",
    "DEFAULT_CIRCUIT",
    "ECM_PARAM_NAMES",
    "ACTION_FEATURE_NAMES",
    "calibrate_oracle",
    "calibrate_drift",
    "write_oracle_config",
    "write_calibration_summary",
    "compute_real_targets",
    "plot_tune_oracle_summary",
    "load_experiment_config",
    "oracle_kwargs_from_config",
    "protocols_from_config",
    "build_oracle_from_config",
    "run_experiment",
    "load_default_ecm_circuit",
    "load_oracle_config",
    "oracle_kwargs_from_oracle_config",
]
