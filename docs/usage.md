# Using the oracle

## Building an oracle

{class}`~battery_oracle.PyBaMMOracle` takes an
{class}`~battery_oracle.oracle_config.OracleConfig` and/or its field names as keyword
overrides. `OracleConfig` holds the one default of every setting; an unknown keyword
raises `TypeError`. Two run-time objects are constructor arguments, not settings:
`ecm_model_fn` (the ECM fitter) and `parameter_values` (a `pybamm.ParameterValues`
that replaces the one resolved from `chemistry`).

```python
from battery_oracle import PyBaMMOracle, randles_stub_ecm
from battery_oracle.oracle_config import OracleConfig

oracle = PyBaMMOracle(degradation_preset="nominal", seed=0)       # keyword settings
cfg = OracleConfig(chemistry="Chen2020", eis_noise_level=0.01)
oracle = PyBaMMOracle(cfg, n_cycles=2, ecm_model_fn=randles_stub_ecm)
oracle.config                                                     # the OracleConfig in use
```

From YAML:

- `PyBaMMOracle.from_config(source)` builds from an oracle YAML
  (`config_oracle_*.yml`; `None` loads the packaged `config_oracle_defaults.yml`; a
  parsed `dict` is also accepted). It always uses the Randles stub as the ECM fitter.
- {func}`~battery_oracle.build_oracle_from_config` builds from an experiment YAML
  layered over an oracle YAML; its `ecm.fitter` chooses the fitter (see
  [Experiment protocols](protocol.md)).

`OracleConfig.to_dict()` gives every setting; `oracle_config.YAML_KEYS` maps each
field to its place in the oracle YAML.

## Semantics of a call

The oracle is a stateful cell. A call runs `n_cycles` cycles of the 5-step protocol
(discharge, rest, two CC charge stages, rest; see [protocol](protocol.md)) starting
from the PyBaMM solution left by the previous call (`starting_solution`). The first
call starts at `initial_soc` (0.8). After the solve the oracle:

1. reads the state of charge at the end of the discharged rest (step 1) and the
   charged rest (step 4);
2. synthesises one impedance spectrum at each rest (equilibrium or non-stationary
   EIS, see [the model](models.md));
3. fits the ECM to each spectrum independently;
4. updates the state-of-health bookkeeping and appends one row to `history`.

```python
import numpy as np
from battery_oracle import PyBaMMOracle, Protocol

oracle = PyBaMMOracle(degradation_preset="accelerated")
p = Protocol(charge_current_1_mA=200.0, charge_current_2_mA=100.0,
             charge_duration_1_h=0.25, charge_duration_2_h=0.25,
             discharge_current_mA=100.0, discharge_duration_h=1.0)
result = oracle.run_cycle(p)        # CycleResult
state = oracle(p.to_array())        # the state vector only; same as run_cycle(...).state
```

`run_cycle` and `__call__` accept a {class}`~battery_oracle.Protocol`, a length-6
array, or a length-7 array when the oracle was built with
`use_temperature_protocol=True`. Any other length raises `ValueError`.

A new `PyBaMMOracle(...)` is a fresh cell. `reset()` returns an existing instance to
a fresh cell: it clears the solution, history and degradation bookkeeping, re-seeds
the noise generator from `seed`, and rebuilds the PyBaMM model and solvers. A
`reset()` followed by the same protocols with the same `seed` reproduces the
trajectory, measurement noise included (`tests/test_statefulness.py`).

### Failures

{class}`~battery_oracle.OracleFailure` carries `protocol` and a
{class}`~battery_oracle.FailureKind` in `failure_kind`:

| `FailureKind` | Raised when |
|---|---|
| `END_OF_LIFE` | SOH falls below `eol_capacity_fraction` (default 0.80) |
| `SOLVER_TRUNCATION` | the emergency solver also returned a cycle with fewer steps than requested |
| `SOLVER_FAILURE` | both solvers raised |
| `VOLTAGE_INFEASIBLE` | both solvers raised and the error message mentions the voltage |
| `SOLVER_DEGRADED` | never raised: recorded in `CycleResult.failure_kind` and the history row when the emergency solver was needed |

The oracle does not roll back on a failure: the solution of an end-of-life cycle is
kept, and no history row is appended for the failing call. Treat `OracleFailure` as
the end of the cell; continuing to call the same instance keeps cycling the failed
cell.

## The returned state vector

For a circuit with $P$ parameters (labels from the circuit string, AutoEIS
convention: `Rx` resistor, `Pxw`/`Pxn` CPE admittance/exponent, `Cx` capacitor),
the state vector is

$$
s = \big[\theta^{\mathrm{chg}}_{1..P},\; \theta^{\mathrm{dis}}_{1..P},\;
         \sigma^{\mathrm{chg}}_{1..P},\; \sigma^{\mathrm{dis}}_{1..P}\;(,\, T_{\mathrm{cell}})\big],
$$

where $\theta$ are the posterior medians, $\sigma$ the posterior standard deviations,
and $T_{\mathrm{cell}}$ [K] (at the charged rest) is appended only with
`thermal="lumped"`. Its length is $4P$ or $4P+1$: 28 for the default circuit
`R1-[R2,P3]-[R4,P5]` ($P=7$). The $\sigma$ slots are NaN when the fit is not AutoEIS
(Randles stub, an AutoEIS failure, or a custom fitter).

Slice by name, not by index: `oracle.state_vector_schema` maps each block
(`means_charge`, `means_discharge`, `std_charge`, `std_discharge`, `T_cell_K`) to
its `(lo, hi)` range; `oracle.state_vector_len` is the length;
`CycleResult.block(name)` returns one block. {data}`~battery_oracle.STATE_VECTOR_SCHEMA`
is the schema of the default circuit without `T_cell_K`, and
{func}`~battery_oracle.state_vector_schema` builds one for any circuit.

Pass the circuit your experiment is featurised with (`PyBaMMOracle(circuit=...)` or
`ecm.circuit` in a YAML). Without one the oracle uses
{data}`~battery_oracle.DEFAULT_CIRCUIT`.

## CycleResult

| Field | Content |
|---|---|
| `state`, `schema` | the state vector and its schema; `np.asarray(result)` gives `state` |
| `soh` | state of health after this call (1.0 = fresh) |
| `soc_charge`, `soc_discharge` | SOC at the end of the charged / discharged rest |
| `capacity_ah` | discharge capacity of the call [A h] (C/20 check capacity with `capacity_check`) |
| `failure_kind` | `FailureKind.SOLVER_DEGRADED` (the emergency solver finished the cycle) or `None` |
| `protocol` | the protocol as received |
| `protocol_applied` | the sanitised protocol actually simulated, as a `Protocol` in the simulated cell's mA/h |
| `eis_charge`, `eis_discharge` | `(frequencies_hz, Z)` with complex `Z` [Ohm] |
| `ecm_charge`, `ecm_discharge`, `ecm_std_charge`, `ecm_std_discharge` | `{label: value}` |
| `raw` | read-only view of this call's history row |

Other properties: `history`, `last_result`, `last_solution` (the PyBaMM solution of
the last call), `last_parameter_values` (the parameter values it was solved with, as
{func}`battery_oracle.validity.check_solution` needs them),
`circuit`, `action_names`, `degradation_options` (the PyBaMM model options),
`dod_lam_scale`.

## The history record

`oracle.history` holds one dict per successful call. Fields:

| Key | Content |
|---|---|
| `call_idx`, `chemistry`, `protocol` | provenance |
| `failure_kind` | solver status of the call |
| `eis_mode_used` | `"equilibrium"`, `"nonstationary"`, or `"equilibrium (non-stationary failed)"` |
| `eis_leakage_charge`, `eis_leakage_discharge` | lock-in drift leakage included in Z [Ohm] (non-stationary EIS only) |
| `end_soh` | SOH from the loss budget (see [Degradation](degradation.md)) |
| `capacity_ah`, `T_cell_K` | discharge capacity; cell temperature at the charged rest |
| `cumulative_sei_loss_ah`, `cumulative_plating_ah`, `cumulative_lli_total_ah`, `delta_lli_ah` | PyBaMM lithium-loss integrators and the loss since the first call |
| `lam_pct`, `lam_cap_loss_ah`, `cumulative_dod_lam_ah` | active-material loss terms |
| `sei_thickness_nm`, `dead_li_nm` | x-averaged film thicknesses that feed the EIS ohmic correction |
| `step_stats` | per protocol step: `kind`, `current_A`, `crate`, `duration_s`, `throughput_ah`, `frac`, `v_end`, `hit_cutoff` |
| `Z_charge_real`, `Z_charge_neg_imag` (and `Z_discharge_*`) | spectra in the jones2022 convention: a Nyquist plot is `plot(Z_real, Z_neg_imag)` |
| `ecm_params_charge`, `ecm_params_discharge`, `ecm_std_charge`, `ecm_std_discharge` | the state-vector blocks |
| `ecm_samples_*`, `ecm_variables_*` | AutoEIS posterior samples (`{variable: samples}`) and variable names; `None` / `[]` for the stub |
| `linkk_rmse` | linKK residual RMS of the charge spectrum (Kramers–Kronig consistency) |
| `max_cv`, `converged`, `ecm_params` | AutoEIS diagnostics of the charge fit |
| `drt_peaks` | relaxation times [s] of the DRT peaks of the charge spectrum (empty without hybrid-drt) |

## ECM fitters

- {func}`~battery_oracle.autoeis_ecm` (default when AutoEIS is installed): NumPyro
  NUTS inference per spectrum {cite:p}`omurabaki2023autoeis,phan2019`, 500 warm-up and
  200 samples by default. The spectrum is rescaled to a target ohmic resistance
  before fitting (`ecm_rescale_target_r0`, 0.1334 Ohm) and the resistor samples are
  divided back afterwards; admittances and exponents are reported at the rescaled
  scale. On an inference error it falls back to the Randles stub. The oracle seeds
  each fit from its own random stream, so `seed` also fixes the MCMC.
- {func}`~battery_oracle.randles_stub_ecm`: the high-frequency real part sets the
  ohmic resistor, the low-frequency limit minus it is split over the arcs (0.6/0.4
  for two arcs, evenly otherwise), each CPE admittance is $1/(R\,\omega_{\mathrm{peak}})$
  and each exponent comes from the seed map. Milliseconds, no JAX.

Any callable `(frequencies, Z_real, Z_imag) -> array` is accepted as `ecm_model_fn`;
the oracle keeps its first $P$ entries for each spectrum.

`battery_oracle.circuit` parses circuits (`param_labels_from_circuit`,
`circuit_roles`, `randles_pairs_from_circuit`) and provides
`fix_parameter_flips_dicts`, the sequence-level correction of arc labels that
AutoEIS swaps between cycles for a symmetric circuit. The oracle does not apply it
to `ecm_params_*`; a consumer applies it to the retained samples over a whole run.

## Concurrency and lifetime

One oracle is one cell; use it sequentially. A lock serialises the PyBaMM solve, but
the history, the random stream and the bookkeeping are updated outside it, so
sharing an instance across threads is not supported. AutoEIS inference is serialised by a module-level lock.

## Downstream

battery-forecast reads `oracle.history` for its CSV export (`save_oracle_csv`),
plots, the arc-flip correction over a run, and calibration of the degradation
scales.
