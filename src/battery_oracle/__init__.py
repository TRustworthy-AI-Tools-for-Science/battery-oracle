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
    write_oracle_config,
)

__version__ = "0.1.0"

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
    "compute_real_targets",
    "load_experiment_config",
    "oracle_kwargs_from_config",
    "protocols_from_config",
    "build_oracle_from_config",
    "run_experiment",
    "load_default_ecm_circuit",
    "load_oracle_config",
    "oracle_kwargs_from_oracle_config",
]
