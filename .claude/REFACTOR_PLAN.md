# battery-oracle refactor plan (2026-10-27)

Baseline: branch `killarney` @ 5113627 ("things seem to be working"), the working
implementation the repo is rebased to. Line numbers below refer to that commit.

## Goal

A lightweight package that gives what a generic PyBaMM setup does not:

1. **Statefulness.** A cell that carries its state across calls: `_prev_solution`
   hand-off, cumulative LLI/LAM, SOH/EOL, degradation state, and the EIS/ECM features
   of that state (state vector + posterior std).
2. **Physicality.** Physically consistent parameter sets (LiR2032 Marquis LiCoO2 +
   thermo-D, Chen2020), the two-mesh validity gate (exact laws, convergence, relevance),
   and the physical checks (`check_dfn_cycle`, eSOH, microstructure).

Everything else either goes downstream to battery-forecast (dataset calibration,
tuning, plotting, CSV export), goes behind an optional `[fit]` extra (PyBOP fitting
machinery), or is deleted (rejected experiments, dead code, debug scripts).

Size: `src/` is ~13.6k lines today. Target: a core of ~4k plus a `[fit]` extra of
~1.5k. About 2.3k lines are deleted outright and ~3k move downstream.

## Target layout

```
battery_oracle/
  __init__.py        public API only; no eager tune import; no env mutation at import
  oracle.py          PyBaMMOracle: init, __call__/run_cycle orchestration, reset, history
  state.py           history row and state-vector schema (STATE_VECTOR_SCHEMA), CycleResult
  degradation.py     presets + _build_degradation_config (from oracle.py:591-877)
  ecm.py             AutoEIS fit + Randles stub + the autoeis/mpire patches (oracle.py:76-176, 334-564)
  eis.py             ONE EIS implementation: equilibrium + non-stationary
                     (merges eis_nonstationary.py and oracle._nonstationary_spectrum)
  circuit.py         public (was _circuit.py); fixes the legacy-pairs default
  protocol.py        Protocol + sanitisation + _protocol_to_experiment
  model_options.py   mesh/radial/x-factor/thermal/chain_options/psd/hysteresis builders
                     (moved out of bol_chain so the oracle no longer imports the fit module)
  config.py          OracleConfig dataclass = the single source of defaults; from_yaml/to_yaml
                     (replaces experiment.py's two mappers and 3-4 copies of each default)
  parameter_sets.py  Chen2020 + LiR2032 (Marquis LiCoO2 + thermo-D) only
  validity.py        measure() returns raw values; judge(); relevance(); public _layout/_entries
  checks.py          physical checks from physicality.py (check_dfn_cycle, check_esoh, microstructure)
  patches.py         nonuniform_edge only, and only while PyBaMM 26.7 is supported
  noise.py           measurement noise (white/flicker); was _eis/noise.py
  fit/               OPTIONAL extra [fit] (pybop):
    chain.py         ChainModel (history simulator) + evaluate_chain (validity integration)
    fit.py           fit_cells(cells, config): single-cell = joint with one cell
    balance.py       ocv_balance, measured_ocv_feature, history_rest_points
    probe.py         envelope probe
    config.py        FitConfig dataclass (replaces 22 ORACLE_FIT_* env reads)
    pybop_ext.py     graded_failure, parallel_batches/run_candidates, checkpoint
    bounds.py        LITERATURE_BOUNDS, scale names, aggregate_chain/overrides_from_chain
```

## Phase 0: safety net (before any change)

1. Tag the baseline in both repos (`pre-refactor-2026-10-27`) before rebasing, so
   archived evidence scripts stay runnable against it.
2. **Golden outputs (SLURM job).** Record them at fixed seeds, using PyBaMM 26.10:
   - oracle: 5 cycles of 2 protocols, saving the state vectors and history rows;
   - chain: one solve of the v26j checkpoint on one cell, saving the voltage trace and
     the validity measures;
   - one `check_dfn_cycle`.

   Every refactor phase must reproduce these within solver tolerance. A phase that
   changes them is a behaviour change, and goes in Phase 8.
3. Enable CI on `killarney`; ci.yml currently triggers on `main` only. Add pybop to the
   dev group, so importorskip stops silently skipping tests.

## Phase 1: delete rejected experiments and dead code (pure deletion)

Rejected physics, with their tests and registrations:

| Remove | Where | Tests removed |
|---|---|---|
| polish (L-BFGS) | polish.py; bol_chain 1389-1424, 1443; `ORACLE_FIT_POLISH*` | test_polish (7) |
| two_phase MSMR | two_phase.py (531) | test_two_phase (15) |
| PlateauWall OCP | parameter_sets 282-314, 370, 471-475, sets `*_Wall*`; oracle.py:225-226 | test_ocp_plateau_wall (4) |
| vacancy D | parameter_sets 224-240, 442-444; `*_VacancyD` sets | test_parameter_units vacancy cases |
| MIT conductivity | parameter_sets 317-343, 476-481; `*_MIT` | test_parameter_units mit |
| Ai2020-table thermo-D | parameter_sets 180-221, 437-439; `LiR2032_MohtatGraphite_ThermoD` | |
| knee / graphite knee / j0 shape | `LICO2_DS_*`, `GRAPHITE_DS_*`, `J0_SHAPE_*` (no-op `x**0` nodes in every model); bol_chain 414-430, 843, 1248, 1763 | the knee cases |
| porosity / solid-fraction modes | bol_chain 444-459, 797-811, 1248, 1327-1332, 1511-1516 | test_chain_porosity (4; keep the N:P identity) |
| hysteresis, PSD (rejected v21) | bol_chain 94-104, 189-193, 260-268, 339-366, 1334-1340, 1517-1522; oracle.py 1198-1205, 1561-1579 | |
| harmonic_flux patch | patches 419-492; oracle.py:1603-1604 | test_harmonic_flux (2; move the checkpoint test) |
| eis_nonstationary PyBOP patch | patches 123-180; `eis_nonstationary.History`/`nonstationary_impedance` | test_nonequilibrium patched-EIS (2) |
| bol_fit, bol_joint (superseded) | relocate their live constants first: `EIS_SCALES`, `ocp_data_range`, `CONTACT_R`, `EPS_N`, `N_PARALLEL`, `OCV0/100`, `SIGMA_V`, `SIGMA_Z_REL`, `_centred`, `widened`; replace `oracle_parameter_values` (builds an oracle to read `_pv`) with `parameter_sets.resolve` | test_bol_fit (5), joint_fit_recovers |

- The harmonic patch is deleted only after a firing count on the oracle's DFN and the
  chain model under 26.10: zero firings means removal. BF `diag/validity_probe.py` once
  logged two firings.
- Dead oracle code (from the review), removed with its YAML keys and mapper lines:
  - the non-DFN branches (`_MODEL_CLASSES` has a single entry), the `model` kwarg,
    `REMOVED_MODELS`, `solver_rtol/atol/dt_max_s`, `emergency_solver_dt_max_s`;
  - the never-produced `fidelity="reduced"`, plus `FailureKind.ECM_NONCONVERGENCE` and
    `THERMAL_RUNAWAY`;
  - `_CAP_SCALE`, `_N_CAP_CHECK_STEPS`, `_eis_lin_ok`, the first `_c20_A`, and the
    `"multiply"` override op;
  - the crack-borrow block (46 lines) and the crack/LAM plot series (mechanics are off in
    every preset);
  - history keys with no consumer: `state_raw`, `protocol_applied`, `end_soc_*`,
    `crack_sei_nm`, `l_crack_nm`, `cumulative_dod_throughput_ah`;
  - `_VALID_DATASETS` (calce/oxford/matr, unshipped), the `__init__` deprecation shim,
    `_HAS_GPU`, and the unused `_plotting` helpers.
- Make the always-on fit paths unconditional. Every launch from v20 to v26 sets
  `VOLTAGE_LIMITS=1`, `SEI=ec reaction limited`, `BALANCE=ocv_features` and
  `CE_FLOOR=0.1`, so delete the non-segmented `sim.solve`, `balance="free"/"ocv"` and
  `sei="constant"` paths.
- Strip every dated "2026-10-xx" history narration (about 32 in src), every comment
  that cites assistant memory or plan names, and every stale "SPMe" reference (about
  23). The facts go to the commit message or docs.

## Phase 2: decouple the oracle from fitting; no env-var configuration

1. Move `x_var_pts`, `radial_mesh`, `chain_options`, thermal and `PSD_POINTS` users
   into `model_options.py`. After this the oracle imports nothing from the fit code.
   Today `_solve` imports bol_chain on every call.
2. Replace the env reads with a frozen `FitConfig`, built once by the caller and stored
   in the fit report. That covers the 22 `ORACLE_FIT_*` names in this repo (import-time
   and run-time) and ~10 in BF `tune_oracle.py`. `chain_for_fit` and `aggregate_chain`
   rebuild from the report's config, not the current environment; see Phase 8,
   reproducibility.
3. `__init__` no longer mutates `JAX_PLATFORMS` / `CUDA_VISIBLE_DEVICES`; that moves
   to BF `_env.sh`.

## Phase 3: split oracle.py and make defaults single-source

- **Split** oracle.py (3312 lines) into oracle/state/degradation/ecm/eis/protocol.
  - `run_cycle` (564 lines, 4 closures) becomes ~7 named steps: read SOC/T, prepare
    LAM/EIS parameters, ohmic corrections, EIS, capacity/SOH bookkeeping, ECM fit,
    history row.
  - `__init__` (545 lines, 75 kwargs) becomes `PyBaMMOracle(config: OracleConfig)`,
    plus a thin `from_yaml`.
- **One default per setting.** `OracleConfig` replaces the three or four copies: the
  signature literal, `.get(key, default)` in two mappers, the YAML, and module
  constants. The YAML is generated from or validated against it. The mapper gains
  the bol/model-option keys it drops today: `hysteresis_half_gap_V`,
  `size_distribution`, `radial_mesh`, `eis_mode`, `frequencies`, `eis_noise`.
- **One implementation each:**
  - EIS: the oracle and the chain share `eis.py` (two dv/dt estimators today);
  - ECM label derivation: one helper (three copies today);
  - chemistry validation: derived from `CUSTOM_PARAMETER_SETS` (three copies plus
    `_SUPPORTED_CHEMISTRIES` today);
  - callable scaling: `params._scaled`.
- Turn the ~45 `except Exception: pass` blocks into explicit errors or logged
  warnings. A missing PyBaMM variable must not silently become 0 or NaN.

## Phase 4: move calibration, plotting and export downstream (to battery-forecast)

| Move | Lines | BF destination |
|---|---|---|
| tune.py, tune_plots.py, `battery-oracle-tune` entry point | ~2,430 | `battery_forecast/oracle_tune/` (BF already has its own CLI) |
| `save_to_csv` (BattMAP format), `plot_diagnostics` | ~370 | BF utils/plots |
| `make_pybamm_candidates` | 52 | BF candidates |
| `_plotting.py`, `_eis/drt.py`, the `kk_*` pipeline, `ecm_kk_validate` | ~400 | BF analysis/eis (it already re-exports them) |
| `experiment.run_experiment`, config_experiment_defaults.yml | ~200 | BF (it has its own `run_experiment`) |

- Keep `linkk_rmse` in the core: it's a per-cycle physical check.
- **Public API for what BF uses:**
  - expose `history` instead of `_history`;
  - expose `ecm.autoeis_ecm` / `ecm.randles_stub` instead of the underscore names;
  - make the `circuit.*` helpers public;
  - expose `validity.layout` / `entries` and `ChainModel.solve_full` instead of
    `_solve`;
  - expose `lico2_monotone_table`.
- Coordinated BF commit: update its imports in the same change.

## Phase 5: the `[fit]` extra

- `fit_cells(cells, config)`. A single-cell fit is the joint fit with one cell. About
  95 of the joint function's 138 lines repeat the single fit, and the two have already
  drifted apart (free/held-at-warm handling, `EPS_N` popping).
- The CMA-ES warm restart pokes private PyBOP attributes (`_optimiser`,
  `_pints_optimiser`, `_std0`). Use a public PyBOP option if one exists; if not, isolate
  it in `pybop_ext.py` with a version guard.
- `validity.check_solution` returns raw measures. `measure()` stops parsing check names
  back by string matching; today a renamed check silently becomes "skip".
- `physicality.check_chain`: its physical half (state, volume fractions) goes to
  `checks.py`; its accuracy half and `check_tune` go downstream. Use one electrolyte
  threshold set across fit, gate and checks: today it is 0.1 / 0.2 / 0.05 / 0.1.

## Phase 6: tests

Counts are from the test inventory: 226 tests, of which 109 KEEP, 8 REMOVE-DEBUG,
24 REWRITE, 34 TIED-TO-REMOVED, and 51 MOVE-DOWNSTREAM.

- **Remove the 42 debug and removed-code tests.**
  - Debug: `run_experiment_no_arg` (a duplicate), `reduced_model(s)_rejected` ×2,
    `calibration_model_is_dfn`, `eol_target_cycles_sourced_from_oracle_yaml`,
    `leakage_is_about_one_percent`, `electrode_capacity_matches_nominal_scale`,
    `fitted_values_and_esoh_on_the_template_pass`.
  - Removed code: the 34 tests tied to it.
- **Move the 51 calibration, tuning, plotting and pool tests with their code.**
- **Rewrite the 24 dataset-pinned tests as relationships.** Examples:
  - the SOC window from PJ248 values;
  - the Ai2020 range pins;
  - 50 pinned YAML literals, replaced by one parity test of YAML against `OracleConfig`;
  - jones2022 currents;
  - 42 mAh / 10-20 cm² pins, replaced by area scaling.
- **Add tests for the core claim; none exist today.**
  - State carries across calls: SOH is non-increasing over cycles, and LLI is
    cumulative and monotone.
  - A one-call-per-cycle oracle equals a single PyBaMM experiment over the same steps,
    which certifies the `_prev_solution` hand-off.
  - `reset()` returns to the fresh cell (state, not only the RNG).
  - The same seed gives the same trajectory.
- Mark the heavy tests that solve without `@slow`: parallel_batches, runaway kill,
  nonuniform_edge leak, the DFN charge-balance test, and `radial_mesh` sim.build.
  Unmark `oracle_lumped_thermal`, which doesn't solve.
- Consolidate the three overlapping one-cycle smoke tests into a single fixture.
- Merge the two modules concatenated in test_parameter_units.py.

## Phase 7: docs, scripts, repo hygiene

- **battery-oracle docs.** All of the following still describe SPMe as the default
  and need rewriting for a DFN-only package: models.md, protocol.md, index, numerics,
  divergences, README, notebooks/README. None of them documents bol_chain, validity,
  non-stationary EIS or the LiR2032 set yet, so add those. docs/api/index.rst lists
  deprecated private names. Rewrite `.claude/TODO.md`, which still has the SPMe Tier-1
  items.
- **Notebooks.**
  - 02_demonstrations (SPM vs SPMe vs DFN) no longer runs: rewrite it as a demo of
    state and validity.
  - 01_oracle_tuning_calibration and notebooks/config_oracle_jones2022.yml (a stale
    Chen2020 tune) move to BF.
- **bin/.** Both scripts generate docs figures. Keep them, but switch them to public
  APIs.
- **Debug scripts** (these live in battery-forecast `scripts/oracle_recal`, 153
  tracked files):
  - **Delete the debug-only scripts:**
    - 31 in diag/: crawl_*, ds_probe*, psd_*, hyst_psd_probe, ie_tree, joint_timing*,
      memory_profile*, oracle_charge_balance*, oracle_opts_smoke, pybamm_upgrade_diag*,
      sei_probe, state_at, vlim_*, formation_end_v, hardstop_api_test, ai2020_ocp_end;
    - in verify/: radial_ref_why*, onset_smoke*, oracle_cycle_validity*;
    - ad-hoc runners: 00c, 00d, exp2_envelope_check, unit_tests_new, worker_pool_test,
      radial_mesh_test. Keep one env-selectable runner (oracle_tests.sh) and update
      00_tests.sh.
  - **Delete the superseded launchers** v20–v25w, which drive removed code. Keep v24 in
    the archive as provenance for the kept model.
  - **Archive the evidence for rejected options** (wall, vacancy, MIT, two_phase,
    porosity, polish scripts). Its code is deleted, so it stays reachable only through
    the baseline tag.
  - **Keep the evidence for kept decisions** in `scripts/oracle_recal/evidence/`,
    ported to the public API:
    - from diag/: particle_conservation, validity_probe, cutoff/hardstop,
      protocol_envelope/coverage, ageing_magnitude, ocv_pair_peaks;
    - from verify/: synthetic_*, mesh/radial/li_conservation, lean_solve,
      validity_check, charge_balance_*, rejudge, probe_check, physics_coverage*,
      overfit, version_compare, the rx_/skin_law chain, thermod_*, figures.
  - Rename `vacancyd_check.py`, the shared driver of the kept thermo-D runs, and
    change its default chemistry.

## Phase 8: behaviour fixes (separate commits; each intentionally changes the golden outputs)

Found by the reviews. They are listed here because they matter for exp2/stage B, not
for the refactor's own correctness:

1. exp2 silently drops the fitted model options (mapper never emits them; see Phase 3).
2. The equilibrium-EIS model is built without the cycling options (SEI, plating,
   contact resistance, thermal) or the mesh (oracle.py:2534). Its `_last_cycle_pv` is
   stored but never used.
3. On 26.7, the oracle doesn't apply the nonuniform_edge patch, so a surface-clustered
   radial mesh leaks particle lithium. The oracle's PSD mesh also follows a fit env var.
4. `candidate_values` silently ignores inputs outside its `direct` list.
   `LICO2_WALL_Y` and `LICO2_SIGMA_RATIO` were never applied. Raise on unknown inputs.
5. Re-evaluation depends on the current env, not on the report (Phase 2 fixes the
   mechanism).
6. DoD-LAM uses the requested discharge duration, not the simulated one.
7. AutoEIS means are ordered by `variables` but labelled by circuit order.
   `fix_parameter_flips_dicts(…, None)` raises a KeyError for the 7-param circuit.
8. Tuning scores solver failures as EOL and hardcodes 0.20.
   `build_oracle_from_oracle_config` forces the Randles stub.
9. The checkpoint is written only in the parallel path, and `_BEST` is process-global.
10. The thread-safety lock covers the solve only, while history, RNG and counters are
    mutated outside it. 7-D protocols permanently mutate `_pv` temperatures, and
    `reset()` doesn't restore them.
11. Undeclared extras: pybop is used by non-stationary EIS, tqdm by tune.

## Decisions (user, 2026-10-27)

1. **Fitting code is a `[fit]` extra** (`battery_oracle.fit`, which needs pybop). The
   jones2022-specific pieces (cycler constants, data-dict layout) move to BF.
2. **Drop PyBaMM 26.7.** 26.10 is the only supported version. This deletes the
   nonuniform_edge patch and its tests, `TESTED_PYBAMM`'s 26.7 entry, and the Ai2020
   unit shim. `_env.sh` defaults to the 26.10 venv. Risk 3 in Phase 8 goes away.
3. **Stage-B levers.**
   - Delete `c2_stress_*`: its fade-vs-C2 slope target was measurement noise.
   - Delete `eis_drift_*` / `add_relaxation_drift`: the non-stationary EIS covers
     drift physically. Also delete `tune.calibrate_drift`, `_linkk_lowhigh_ratio` and
     BF `tune_oracle`'s drift wiring.
   - Keep `dod_lam_*` as a labelled empirical active-material-loss term, off until a
     literature value or a deep-DoD dataset sets its size.
   - Keep its EIS feed: the loss reduces eps_s, so it shows in the state vector.
   - Remove the C2 slope probe from tuning (`run_crate2_slope_probe`,
     `slope_match_error`). Its only target was the noise slope.
4. **No knee form.** Delete `lico2_ds="knee"`, the graphite knee and the j0 shape
   exponent. The fitted set already uses none of them: the graphite knee and j0 shape
   are installed at drop = 0 and γ = 0, so their factors are exactly 1. Also delete
   the Ai2020 monotone LiCoO2 table, which Marquis2019 overwrites. `"LiR2032"` means
   the Marquis LiCoO2 + thermo-D + Mohtat graphite set. Removing these factors must
   leave the golden outputs unchanged, which also verifies the deletion.
5. **Evidence scripts:** split them.
   - The algorithmic content goes into the fit extra as dataset-agnostic functions plus
     tests: synthetic recovery (truth → simulate → refit on any schedule), tolerance
     scaling and mesh convergence (already in the gate), and probe checks.
   - The scripts that run on jones2022 cells and write `_results` stay in BF
     `scripts/oracle_recal/evidence/` as experiment provenance. They import
     battery_forecast, so they cannot live in the package.
6. **Phase 8 fixes come after the refactor.**

## Order and size

| Phase | Depends on | Size |
|---|---|---|
| 0 | none | ~half a day, plus ~2 h of SLURM |
| 1 | 0 | about a day; deletion only |
| 2 | 1 | about a day |
| 3 | 2 | 2–3 days; the main readability work |
| 4 | 1 | about a day; BF is changed in the same commit |
| 5 | 2 | about a day |
| 6, 7 | alongside each phase | |
| 8 | after 3 | |

Rerun the golden comparison and the fast test suite after every phase, and the slow
suite plus one physics-coverage check after phases 3 and 5.
