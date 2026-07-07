API reference
=============

Public objects re-exported from :mod:`battery_oracle` (see ``__all__``).

Oracle
------
.. autosummary::
   :toctree: generated
   :nosignatures:

   battery_oracle.PyBaMMOracle
   battery_oracle.OracleFailure
   battery_oracle.make_pybamm_candidates

Circuit constants
-----------------
.. autosummary::
   :toctree: generated

   battery_oracle.DEFAULT_CIRCUIT
   battery_oracle.ECM_PARAM_NAMES
   battery_oracle.ACTION_FEATURE_NAMES

ECM fitting functions
---------------------
.. note::

   ``_randles_stub_ecm`` and ``_autoeis_ecm`` are exported in
   ``battery_oracle.__all__`` and are part of the public API, but keep their
   leading underscore to signal they are lower-level building blocks passed as the
   ``ecm_model_fn`` argument to :class:`~battery_oracle.PyBaMMOracle`. Prefer the
   high-level oracle for normal use.

.. autosummary::
   :toctree: generated
   :nosignatures:

   battery_oracle._randles_stub_ecm
   battery_oracle._autoeis_ecm

Experiment configuration
------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   battery_oracle.load_experiment_config
   battery_oracle.oracle_kwargs_from_config
   battery_oracle.protocols_from_config
   battery_oracle.build_oracle_from_config
   battery_oracle.run_experiment
   battery_oracle.load_default_ecm_circuit

Calibration
-----------
.. autosummary::
   :toctree: generated
   :nosignatures:

   battery_oracle.calibrate_oracle
   battery_oracle.calibrate_drift
   battery_oracle.write_oracle_config
   battery_oracle.compute_real_targets
