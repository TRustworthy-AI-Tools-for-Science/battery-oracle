# To-Do List: Feature Additions by Repository

---

## battery-oracle

### Tier 1 (required for paper experiments)

- [ ] **`model` kwarg on `PyBaMMOracle.__init__`**: `model="SPMe"` (default) | `"DFN"`. DFN is the Doyle-Fuller-Newman model; resolves electrolyte concentration gradients across separator and both electrodes, making it accurate above ~1C where SPMe loses fidelity. Add `model` to the YAML config and to `oracle_kwargs_from_config` in `experiment.py`. The kwarg must thread through `_build_degradation_config` and `_build_native_state` so the correct PyBaMM model class is instantiated.

- [ ] **DFN solver settings**: DFN is stiff because the solid-phase diffusion ODEs (fast) couple to the electrolyte PDEs (slow), creating a stiff coupled DAE. The `IDA_ERR_FAIL` events already caught by the truncation-detection logic occur at the CC→CV switch where electrolyte concentration spikes. Required settings for DFN path:
  - Solver: `IDAKLUSolver` with `rtol=1e-6, atol=1e-8` (tighter than SPMe default `rtol=1e-3`)
  - Set a low initial step size via `IDAKLUSolver(dt_max=1.0)` to allow IDA to backstep before catastrophic failure rather than silently truncating
  - C-rate ceiling: **1.5C** for DFN (distinct from SPMe's ~2C ceiling in `_sanitise_current`); add `_dfn_max_crate` as a class-level constant, and branch `_sanitise_current` on `self._model`
  - The existing `_is_truncated` step-count check applies unchanged — if DFN truncates silently, the same early-exit path fires

- [ ] **SPMe warm-start for DFN**: Run the first 1–2 cycles with SPMe, use the terminal electrode stoichiometries, SEI thickness, and plating accumulation from `_prev_solution` as the initial condition for DFN via PyBaMM's `starting_solution` kwarg. This avoids integrating through the cold-start transient (the primary DFN failure point). Extend `_prev_solution` carry-over so that a mode switch from SPMe to DFN at `n_warmup_cycles` does not reset state. Expose `n_warmup_cycles=2` as a constructor kwarg.

- [ ] **Solver fallback hierarchy for DFN**: `DFN + IDAKLU (tight)` → `DFN + CasadiSolver (loose tolerances)` → `SPMe fallback`. Each level logged to `OracleFailure.failure_kind` with `SOLVER_DEGRADED`. The SPMe fallback must return a result object with a `fidelity="reduced"` flag so the audit hook can track per-step fidelity in MLflow without breaking the AL loop. Wrap the three-level try/except inside `__call__` after the existing truncation check.

- [ ] **DFN used only for final confirmation, not calibration inner loop**: `tune.py` runs hundreds of oracle calls; keep `model="SPMe"` for all Optuna calibration trials. Reserve DFN for the final fidelity check of shortlisted protocols. Add a `calibration_model` override kwarg (default `"SPMe"`) that `tune.py` uses to instantiate the oracle independently of the experiment-time `model` setting.

- [ ] **ECM posterior σ in returned state vector**: The `_autoeis_ecm` method already stores `ecm_samples_charge` and `ecm_samples_discharge` in the per-cycle history dict (each a `(n_samples, n_params)` array of AutoEIS posterior draws). Surface the per-parameter posterior standard deviation as explicit entries in the numpy state array returned by `__call__`. Specific design:
  - Compute `ecm_std_charge = ecm_samples_charge.std(axis=0)` and `ecm_std_discharge = ecm_samples_discharge.std(axis=0)` (shape `(n_params,)` each)
  - Append both vectors to the existing state array after the current scalar entries (SOH, voltage, capacity, etc.)
  - Document the new array indices in a `STATE_VECTOR_SCHEMA` dict at module level so downstream code (DMDc, audit hook) can slice by name rather than by magic index
  - When AutoEIS is unavailable (Randles stub fallback), fill ECM std slots with `np.nan`; DMDc must handle `nan` columns by dropping or imputing before fitting

- [ ] **Structured `OracleFailure.failure_kind` enum**: Replace the current free-text message with a `FailureKind(str, Enum)`:
  ```
  SOLVER_TRUNCATION     # step count mismatch detected by _is_truncated
  VOLTAGE_INFEASIBLE    # experiment voltage window violated at first step
  END_OF_LIFE           # SOH < eol_capacity_fraction
  ECM_NONCONVERGENCE    # AutoEIS and Randles stub both fail
  THERMAL_RUNAWAY       # T_cell exceeds thermal_runaway_K threshold (future)
  SOLVER_DEGRADED       # DFN fell back to SPMe (reduced fidelity, not a hard failure)
  ```
  Store `failure_kind` on the `OracleFailure` exception and in the per-cycle history dict. Expose it in `save_to_csv` as a dedicated column. The audit hook reads this column to correlate failure modes with calibration check outcomes.

- [x] **Detrended state output mode — moved to traits-audit**: this is no longer battery-oracle's responsibility. The per-protocol-group EMA detrending previously implemented inline here (`detrend=True` on `__call__`) has been generalized and moved to `traits_audit.RegimeDetrender` (`traits-audit/src/traits_audit/detrend.py`) — it accepts an arbitrary-length regime vector (not hardcoded to the 6-D protocol; the 7-D `use_temperature_protocol` gap is fixed as part of the move), so it is no longer battery-physics-specific. battery-oracle does **not** depend on traits-audit — per-cycle `state_raw` is kept in `PyBaMMOracle`'s history (see `__call__`) for whatever orchestrates the digital twin (`battery-forecast`, not yet built) to feed into `RegimeDetrender` alongside the oracle's `_action_names`-length protocol vectors.

- [ ] **`Timms2021` thermal submodel**: Add `thermal="isothermal"` (default) | `"lumped"` kwarg. The Timms 2021 lumped model adds `T_ambient_K` and `h_total_W_per_m2K` to the parameter values and couples cell temperature to the electrochemical model via a single ODE. SEI growth, Li plating rate, and electrolyte conductivity are all temperature-dependent in Chen2020, so the thermal coupling changes the degradation trajectory for non-ambient protocols. Implementation:
  - `pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})` (or DFN equivalent)
  - Add `T_ambient_K=298.15` and `h_total_W_per_m2K=10.0` as constructor kwargs with YAML-configurable defaults
  - Lumped thermal adds one ODE; stiffness increase is modest (not DFN-level) and IDAKLU handles it without tolerance changes
  - Surface `T_cell_K` (volume-averaged) in the returned state vector and in `save_to_csv`

- [ ] **`T_ambient` protocol vector slot**: Extend 6-D to 7-D when `thermal="lumped"` and `use_temperature_protocol=True`: `u = (I_CC, V_CV, t_CV, I_dis, SoC_lo, SoC_hi, T_ambient_K)`. The 6-D default is unchanged. `_sanitise_current` and `_protocol_to_experiment` branch on `len(protocol)`. `make_pybamm_candidates` gains a `T_ambient_range` kwarg (default `[278.15, 318.15]`, i.e. 5–45°C). Update `STATE_VECTOR_SCHEMA` to document slot 6.

- [ ] **Temperature-dependent EIS synthesis**: When `thermal="lumped"`, update `_autoeis_ecm` to pass `T_cell_K` from the post-cycle solution to the EIS synthesis step. Charge-transfer resistance scales as `R_ct ∝ exp(E_a / R T)` (Arrhenius) and electrolyte conductivity shifts the high-frequency intercept `R_0`. Without this, ECM fits at non-ambient temperatures are biased, corrupting the uncertainty vector. Concretely: multiply the synthesised `R_ct` by `exp(E_a/R * (1/T_cell - 1/T_ref))` with `E_a=30e3` J/mol (typical for LiPF6 electrolyte) and `T_ref=298.15 K`; expose `E_a_J_per_mol` as a constructor kwarg.

---

### Tier 2 (multi-dataset / benchmarking)

- [ ] **`config_oracle_calce.yml`, `config_oracle_oxford.yml`, `config_oracle_matr.yml`**: Calibrated hyperparameter YAML files for three public battery datasets. Each file contains `kinetics_scale`, `sei_rate_scale`, `dead_li_decay_scale`, `plating_rate_scale` (and optionally `c2_stress_scale`) obtained by running `tune.py` against the respective dataset. Procedure:
  - CALCE: CS2 cells, 1C CC-CV charge, 0.5C discharge; download from calce.umd.edu; fit against cycles 1–100
  - Oxford: Kokam 740 mAh cells, 1C charge/discharge, 40°C; fit against capacity fade curve
  - MATR: 124-cell dataset from Severson et al. 2019 (*Nature Energy*); use the first 5 cycles of each cell as warm-start; fit against the 80% capacity-fade cycle count
  - Add a `config_dataset` kwarg to `build_oracle_from_config` and a `--dataset` flag to the CLI

- [ ] **GIFTERS self-assessment script**: CLI entrypoint `battery-oracle gifters` that evaluates the oracle against the 7 GIFTERS dimensions (Generalizable, Interpretable, Fair, Transparent, Explainable, Robust, Stable) per arXiv 2512.01080. Concretely:
  - **Generalizable**: run calibrated oracle against held-out dataset not used for tuning; report RMSE of SOH at cycle 100 vs. ground truth
  - **Interpretable**: report that all oracle parameters are physically named and bounded (pass/fail checklist)
  - **Fair**: report that calibration was performed on cells from ≥2 manufacturers/chemistries (pass/fail)
  - **Transparent**: verify that `config_oracle_defaults.yml` documents every parameter with units and physical meaning (automated doc coverage check)
  - **Explainable**: report that each degradation mechanism (SEI, cracking, plating, DoD-LAM) contributes a logged fraction to total SOH loss (already implemented in history dict; check it's present)
  - **Robust**: run oracle with ±10% perturbation to each of the 4 calibration scales; report sensitivity index per parameter
  - **Stable**: run DMDc on a 100-cycle trajectory under a fixed protocol; report `ρ(A)` of the state-only dynamics (should be <1 for a physically stable simulation)
  - Output: JSON report with one score (0/1) per dimension and supporting metrics

- [ ] **Multi-chemistry switching**: `chemistry` kwarg on `PyBaMMOracle.__init__` accepting `"Chen2020"` (LG M50, default) | `"Xu2019"` (NMC523) | `"Prada2013"` (LFP A123). Maps to the corresponding PyBaMM parameter set via `pybamm.ParameterValues(chemistry)`. Each chemistry requires its own calibration YAML (see above). The degradation model scales (`sei_rate_scale`, etc.) are chemistry-specific; the YAML must include a `chemistry` field that is validated against the kwarg at startup to prevent mismatched configs.
