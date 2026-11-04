API reference
=============

The public API is ``battery_oracle.__all__`` (the names below re-exported from
:mod:`battery_oracle`) and the modules listed after it. The stage-A fit,
:mod:`battery_oracle.fit`, needs the ``[fit]`` extra (PyBOP).

Oracle
------
.. autosummary::
   :toctree: generated
   :nosignatures:

   battery_oracle.PyBaMMOracle
   battery_oracle.Protocol
   battery_oracle.CycleResult
   battery_oracle.OracleFailure
   battery_oracle.FailureKind

State-vector layout
-------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   battery_oracle.state_vector_schema

``battery_oracle.STATE_VECTOR_SCHEMA`` is ``state_vector_schema`` of the default circuit:
a dict of block name to ``(start, stop)`` slice of the state vector.

ECM fitters
-----------
Pass either as ``ecm_model_fn`` to :class:`~battery_oracle.PyBaMMOracle`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   battery_oracle.autoeis_ecm
   battery_oracle.randles_stub_ecm

Circuit and protocol names
--------------------------
.. autosummary::
   :toctree: generated

   battery_oracle.DEFAULT_CIRCUIT
   battery_oracle.ECM_PARAM_NAMES
   battery_oracle.ACTION_FEATURE_NAMES

YAML configuration and experiments
----------------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   battery_oracle.load_oracle_config
   battery_oracle.oracle_kwargs_from_oracle_config
   battery_oracle.build_oracle_from_oracle_config
   battery_oracle.load_experiment_config
   battery_oracle.oracle_kwargs_from_config
   battery_oracle.protocols_from_config
   battery_oracle.build_oracle_from_config
   battery_oracle.run_experiment
   battery_oracle.load_default_ecm_circuit

Oracle modules
--------------
.. autosummary::
   :toctree: generated

   battery_oracle.oracle
   battery_oracle.oracle_config
   battery_oracle.experiment
   battery_oracle.state
   battery_oracle.failures
   battery_oracle.protocol
   battery_oracle.circuit
   battery_oracle.ecm
   battery_oracle.eis
   battery_oracle.degradation
   battery_oracle.parameter_sets
   battery_oracle.params
   battery_oracle.model_options
   battery_oracle.patches

Physical validity
-----------------
.. autosummary::
   :toctree: generated

   battery_oracle.validity
   battery_oracle.physicality

Stage-A fit (``[fit]`` extra)
-----------------------------
.. autosummary::
   :toctree: generated

   battery_oracle.fit.chain
   battery_oracle.fit.config
   battery_oracle.fit.pybop_ext
   battery_oracle.fit.checks

EIS helpers
-----------
Internal modules used by the oracle: the linKK check (``linkk_rmse``), the
white/flicker noise terms, and the DRT fit (needs hybrid-drt).

.. autosummary::
   :toctree: generated

   battery_oracle._eis.kk
   battery_oracle._eis.noise
   battery_oracle._eis.drt
