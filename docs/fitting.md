# Stage-A fitting

{mod}`battery_oracle.fit` fits the beginning-of-life (BOL) parameters of the oracle's
DFN to a cell's measured history and checks that the fitted cell is physically valid.
It is the `[fit]` extra: it needs PyBOP at the commit pinned in `pyproject.toml`
([Installation](install.md)); `import battery_oracle.fit` raises `ImportError`
without it. Calibrating the degradation scales to measured ageing (stage B) is done
downstream, in `battery-forecast`.

| Module | Content |
|---|---|
| {mod}`battery_oracle.fit.chain` | the chain model and {func}`~battery_oracle.fit.chain.fit_cells` |
| {mod}`battery_oracle.fit.config` | {class}`~battery_oracle.fit.config.FitConfig`, recorded in every report |
| {mod}`battery_oracle.fit.pybop_ext` | PyBOP patches, the fork-safe solver, the CMA-ES warm restart |
| {mod}`battery_oracle.fit.checks` | stage-A physical-validity checks |

## One continuous history per cell

Per cell and per candidate parameter set, one model is integrated once through the
cell's whole measured current history: formation, every cycling record, every rest
and every EIS sweep, in order, from the as-received cell (the only truly relaxed
state). From that one solution:

- each cycling or formation record's simulated voltage is compared with its
  measurement;
- each EIS spectrum is the model linearised at the simulated state where each
  frequency was measured, plus the lock-in drift leakage
  ({mod}`battery_oracle.eis`, the same functions as the oracle's non-stationary EIS).

The jones2022 records run back to back: a cycling record starts minutes after the
previous record's rest and EIS sweep, while the voltage still drifts. A continuous
integration keeps those gradients and never re-anchors the state of charge.

### The chain model

`CHAIN_OPTIONS`: the DFN with PyBOP's EIS voltage/current variables, the double layer
(`surface form: differential`), contact resistance, lumped thermal with the whole-cell
heat capacity (`use lumped thermal capacity`), and the oracle's `ec reaction limited`
SEI, so the fitted balance carries the lithium the SEI consumes. The model's own
voltage window is widened to 2.5–4.6 V (`WIDE_WINDOW`) so the cycler limits, not the
model's cut-offs, end each step.

**Cycler limits** come with the data and are imposed by
{func}`~battery_oracle.fit.chain.apply_voltage_limits`: the current equation
$I = I_{\mathrm{app}}(t)$ is replaced by $I = I_{\mathrm{app}}(1-\mathrm{cut})$
(times a smooth CV factor during a CV charge). Hard stops are exact: the solve runs in
segments ({meth}`~battery_oracle.fit.chain.ChainModel.solve_full`); an event stops the
integration where the voltage crosses its limit in the current's direction, the solve
continues with the current cut, and a second event releases the cut when the measured
step ends. A model cell that reaches a limit earlier than the real one therefore stops
(or holds, in CV) where the real cycler would have, instead of being forced on.

**Electrolyte floor.** A solver event stops a candidate whose minimum electrolyte
concentration falls below `ce_floor` × its initial value (default `CE_FAIL_REL` = 0.1);
the truncated history is penalised.

**Initial state.** The cell's first voltage is taken as an open-circuit voltage; with
the cyclable lithium it fixes both electrode stoichiometries
(`initial_stoichiometries`). A simulated surface stoichiometry beyond the OCP data
range by more than `OCP_GUARD_TOL` (0.02) fails the candidate; smaller excursions are
counted and reported.

### Data layout

`fit_cells` takes one data dict per cell. Times are in seconds from the start of
each group, currents in A with the PyBaMM sign (discharge positive):

```python
data = {
    "cell_id": "PJ121",
    "cell_capacity_mah": 42.0,        # measured capacity
    "initial_voltage": 3.9,           # as-received open-circuit voltage [V]
    "discharge_limit_v": 3.0,         # cycler hard stop on every discharge [V]
    "amplitude_a": 5e-3,              # EIS current amplitude [A] (default 5e-3)
    "groups": [                       # measurement sequences, in order
        {"name": "000", "duration_s": ..., "gap_before": None, "formation": True,
         "charge_limit_v": 4.2, "charge_cv": True},     # CV hold at the limit
        {"name": "002", "duration_s": ..., "gap_before": "fit", "formation": False,
         "charge_limit_v": 4.3, "charge_cv": False},    # hard stop at the limit
    ],
    "current": [{"group": 0, "t": t, "I": i}, ...],     # the applied current
    "records": [{"group": 0, "label": "PJ121/000_01", "kind": "formation", "t": t, "V": v}, ...],
    "spectra": [{"group": 0, "label": "PJ121/000_02", "f": f, "Z": z,
                 "t_start": lo, "t_end": hi}, ...],      # window of each frequency
}
```

`gap_before` is the unrecorded rest before a group: `None`/0, a number of seconds, or
`"fit"` to fit it (a parameter `Rest before <group>`, log-uniform in 60 s – 30 days,
centred on the warm value). {func}`battery_oracle.eis.windows_from_stamps` turns an
EIS file's time stamps into `t_start`/`t_end`.

## The fit

{func}`~battery_oracle.fit.chain.fit_cells` fits one cell or several:

```python
fit_cells(datas, pv, *, warm=None, max_iterations=150, max_unchanged=30, seed=0,
          ambient_k=None, bounds=None, ec_diffusivity_factor=1.0, config=None)
```

 `datas` maps cell id to data dict and `pv` is
the template parameter set (normally `parameter_sets.lir2032()`). Two steps per cell:

1. **Electrode balance.** {func}`~battery_oracle.fit.chain.ocv_balance` fits the
   positive capacity, N:P ratio and cyclable lithium (a least-squares fit of the
   closed-form OCV) to the measured
   rest voltages (`history_rest_points`) and the main dQ/dV feature of a formation
   cycle (`measured_ocv_feature`), inside the literature ranges. These are then held
   fixed; capacities differ between cells.
2. **DFN fit.** CMA-ES (PyBOP) over priors uniform or log-uniform on
   `LITERATURE_BOUNDS`: multipliers of the eight kinetic, transport and double-layer
   parameters (`CHAIN_SCALES`: both exchange-current densities, both particle
   diffusivities, electrolyte conductivity and diffusivity, both double-layer
   capacities), contact resistance, an electrode-thickness scale, the negative/positive
   thickness ratio (in place of the negative active fraction, which follows from the
   fixed N:P), negative tortuosity, separator MacMullin number, heat-transfer
   coefficient, both particle radii, and each cell's unrecorded rests.

With several cells the kinetic, transport and geometric parameters are shared and
scored on every cell's records and spectra; each cell keeps its own balance and rests.
A fit of one cell can tune the shared physics to that cell's history; a fit of several
cannot. The cost is a weighted sum of root-mean-square errors: voltage with weight
$1/(10\ \mathrm{mV})$ (`SIGMA_V`), each spectrum with $1/(0.05\,\mathrm{median}|Z|)$
(`SIGMA_Z_REL`). `bounds` overrides entries of `LITERATURE_BOUNDS`;
`ec_diffusivity_factor` multiplies the template EC diffusivity as the oracle does
(its `sei_rate_scale`).

`fit_cells` returns `{cell_id: report}`. A report holds `best_inputs`,
`fitted_inputs`, `fixed_inputs`, `ocv_balance`, `param_bounds`, `log_params`, `cost`,
`fit_config`, `seed`, `runtime_s`, `worker_pool`, `n_iterations`, and the output of
{func}`~battery_oracle.fit.chain.evaluate_chain`: measured and simulated curves and
spectra per record, `group_offsets_s`, and `diagnostics`, which include the validity
records (`diagnostics["validity"]`).

`evaluate_chain` solves the best candidate again with full storage on the validation
mesh (`validate_x_factor`, the oracle's) and on that mesh refined 2×, and runs the
two-mesh gate of [Physical validity](validity.md): the exact laws (with the
tolerance ladder when one is above its limit), continuity convergence, relevance, and
the agreement of the fit mesh with the validation mesh (warn above 1 mV p99).

### Envelope probe

The electrolyte floor protects only the measured history, but the oracle is asked for
the whole protocol space. With `FitConfig.envelope_ma` set, every candidate's history
is followed by the most demanding protocol of that space
({func}`~battery_oracle.fit.chain.with_envelope_probe`): a CC charge to the cycling
charge limit (two stages of up to 900 s each with three values), a 20 min rest, a CC
discharge to the discharge limit and a 20 min rest. A candidate whose electrolyte
reaches the floor in the probe, or whose solve stops there, is rejected.

## FitConfig

{class}`~battery_oracle.fit.config.FitConfig` is a frozen dataclass built once by the
caller and stored in each report (`fit_config`), so a report is re-evaluated with the
settings it was made with (`FitConfig.from_report`). `FitConfig.from_env()` reads the
`ORACLE_FIT_*` variables of the battery-forecast launchers.

| Field | Default | Meaning |
|---|---|---|
| `r_points`, `r_mesh`, `r_stretch` | `None`, `"uniform"`, 1.15 | particle radial mesh (`"surface"`: exponential, clustered at $r=R$) |
| `x_factor`, `validate_x_factor` | 1, 2 | electrode/separator mesh factor of the fit and of the final evaluation, gate and oracle |
| `envelope_ma` | `None` | `(charge, discharge)` or `(c1, discharge, c2)` mA of the envelope probe |
| `ce_floor` | 0.1 | electrolyte floor, fraction of $c_0$ |
| `workers`, `candidate_timeout_s` | 1, 600 | forked candidate workers and the per-candidate time limit |
| `checkpoint`, `from_checkpoint` | `None`, `False` | best-so-far checkpoint file; report from it instead of optimising |
| `warm_json`, `warm_sigma`, `free` | `None`, 0.3, `()` | warm start from a report's `best_inputs`, CMA-ES step × prior's, parameters left free (the rest held at their warm values) |
| `dry_run` | `False` | build the problems and return their sizes |

`radial_mesh_setting()` gives the mesh as the oracle's `radial_mesh` setting.

## From a fit to an oracle

- {func}`~battery_oracle.fit.chain.aggregate_chain` (`aggregate_chain(cell_reports, pv)`) takes the
  median over cells (geometric for log-scale parameters) and converts it with
  {func}`~battery_oracle.fit.chain.overrides_from_chain`: geometry as one square layer
  of the fitted area, thickness, N:P and the cyclable lithium (as initial
  concentrations at 100 % SOC) as absolute `parameter_overrides`; 0 % and 100 % SOC
  open-circuit voltages kept inside the OCP data (`ocv_limits_in_data`); the nominal
  capacity set to the measured one; the fitted multipliers as `parameter_scales`; the
  contact resistance; `thermal: lumped` and the fitted `h_total_W_per_m2K`; and the
  fit's `radial_mesh`. Rests are reported, not carried over.
- {func}`battery_oracle.physicality.fitted_oracle_kwargs`
  (`fitted_oracle_kwargs(fit_report, oracle_kwargs)`) returns the oracle keywords of the deployed oracle from that aggregate.

In an oracle YAML, `parameter_overrides`, `parameter_scales`,
`contact_resistance_ohm` and `radial_mesh` go in `bol`; `thermal` and
`h_total_W_per_m2K` go in `cycling`.

## Stage-A checks

{mod}`battery_oracle.fit.checks` judges validity, never accuracy:

- `cell_checks(cell_report)`: the report's validity records; the whole history
  simulated; surface stoichiometries inside the OCP data (warn up to 0.02 outside);
  temperature rise (pass below 10 K, given `ambient_k`); electrolyte depletion.
- `population_checks(fit_report, pv_template, oracle_kwargs)`: electrode area, active
  fraction and thickness against the template; volume fractions (active + pores ≤ 1,
  porosity ≥ 0.15); cyclable lithium; heat-transfer coefficient; capacity between the
  OCV limits against the measured capacity; the SOC window inside the OCP data;
  `microstructure_checks` (tortuosities, separator MacMullin number, positive
  utilisation, practical N:P); the plausibility windows and electrode-SOH solve; and
  {func}`~battery_oracle.physicality.check_dfn_cycle` on the deployed oracle.

## PyBOP extensions

{mod}`battery_oracle.fit.pybop_ext` patches two PyBOP assumptions that do not hold for
cells measured out of equilibrium; `pybop_ext.apply()` installs both, `revert()`
removes them, `applied()` lists them. Each patch checks the installed versions
(`TESTED_PYBAMM` 26.10.1.0, PyBOP 26.3) and target signatures, and raises
`PatchError` (or, with `strict=False`, warns and skips).

- `graded_failure`: `pybop.ErrorMeasure.evaluate_batch` gives every simulation that
  stops early one flat failure cost, so a population of infeasible candidates is a
  flat landscape. Patched: a truncated trajectory is scored on the simulated part plus
  a per-point penalty on the missing part.
- `parallel_batches`: `pybop.MetaProblem.evaluate_batch` evaluates candidates one after
  another. Patched: after `problem.set_run(workers=n)` the candidates are split over
  `n` forked processes (built models are inherited, not pickled), each under a hard
  time limit (`set_run(timeout_s=...)`); a candidate that exceeds it is killed and
  scored `inf`. With one worker, or with sensitivities, the stock path runs.
  Solvers used with it must be single-threaded (`single_thread_solver`).
- `restart_population(optim, sigma_factor)`: PyBOP starts population optimisers at the
  prior mean; this restarts CMA-ES at the parameters' initial values with the prior's
  step sizes × `sigma_factor`. It uses PyBOP private attributes and raises
  `PatchError` when they are missing.
