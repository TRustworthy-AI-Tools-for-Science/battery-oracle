"""battery_oracle — a standalone PyBaMM (DFN) battery *oracle* (simulator).

The simulated counterpart to a physical battery / BattMAP lab interface: given a
6-D charge/discharge protocol it runs a PyBaMM DFN simulation, synthesises an
EIS spectrum, and fits an equivalent-circuit model — returning the same kind of
featurised state a real cell would. Drop-in usable as the "battery source" in an
active-learning loop.

    import numpy as np
    from battery_oracle import PyBaMMOracle

    oracle = PyBaMMOracle(degradation_preset="accelerated")  # ready to call immediately
    for c_mA in (68.0, 100.0, 140.0):   # [C_rate_1, C_rate_2 mA, dur_1, dur_2 h, D_rate mA, dur_d h]
        oracle(np.array([c_mA, c_mA / 2, 1.0, 0.5, 100.0, 1.0]))

Optional extras:
  * ``[autoeis]`` — Bayesian ECM fitting (falls back to a fast Randles stub without it)
  * ``[drt]``     — distribution-of-relaxation-times peaks (hybrid-drt)

On a CPU-only machine with jax[cuda12] installed, set ``JAX_PLATFORMS=cpu`` before
importing (jax otherwise fails probing the CUDA plugin).
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from battery_oracle.circuit import (
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
from battery_oracle.ecm import autoeis_ecm, randles_stub_ecm
from battery_oracle.failures import FailureKind, OracleFailure
from battery_oracle.oracle import PyBaMMOracle
from battery_oracle.state import CycleResult, STATE_VECTOR_SCHEMA, state_vector_schema
from battery_oracle.protocol import Protocol

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
    "randles_stub_ecm",
    "autoeis_ecm",
    "DEFAULT_CIRCUIT",
    "ECM_PARAM_NAMES",
    "ACTION_FEATURE_NAMES",
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
