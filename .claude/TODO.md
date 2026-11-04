# battery-oracle: open items

Judged from the code on branch `refactor-lightweight`. Items marked **(unverified)**
could not be confirmed from the code alone.

## Behaviour issues (Phase 8 of `.claude/REFACTOR_PLAN.md`)

Phase 8 lists the known behaviour issues; fix them as separate commits after the
refactor, each one changing the golden outputs on purpose. Still visible in the code:

- [ ] 8.2: equilibrium EIS (`eis.equilibrium_impedance`) builds its DFN with only
  `surface form: differential`: no SEI, plating, contact resistance or thermal option,
  and PyBaMM's default mesh, not the oracle's `radial_mesh`.
- [ ] 8.4: `fit.chain.candidate_values` silently ignores inputs outside its known
  lists; raise on unknown inputs.
- [ ] 8.6: the DoD-LAM term (`PyBaMMOracle._accumulate_dod_lam`) uses the requested
  discharge current x duration, not the simulated discharge.
- [ ] 8.7: `ecm.autoeis_ecm` returns medians in circuit-label order, while
  `PyBaMMOracle._ecm_std_half` takes the std in AutoEIS variable order; the two can be
  misaligned. `circuit.fix_parameter_flips_dicts` needs a circuit string (no default).
- [ ] 8.8 (oracle side): `build_oracle_from_oracle_config` / `PyBaMMOracle.from_config`
  always use the Randles stub.
- [ ] 8.9: `fit.pybop_ext` writes the checkpoint only on the parallel path, and its
  best-so-far `_BEST` is process-global.
- [ ] 8.10: the oracle lock covers the solve only; history, RNG and counters change
  outside it. A 7-D protocol writes the ambient temperature into `_pv` permanently and
  `reset()` does not restore it.
- [ ] 8.1, 8.5 **(unverified)**: whether exp2 still drops the fitted model options and
  whether re-evaluation still depends on the environment is a battery-forecast
  question; the mappers here read `bol.radial_mesh` and `cycling.thermal`, and
  `FitConfig.from_report` rebuilds a report's settings.

## Oracle

- [ ] `physicality.check_dfn_cycle` (and `fit.checks.population_checks`) run the
  demanding 3.8C/2C/3.8C cycle through the oracle's protocol bounds, so with the
  default `dfn_max_crate = 1.5` it runs at 1.5C unless the caller raises it. The
  default `protocol_bounds` (`c_min_mA = 50`, `c2_min_mA = 20`) are also above 1C for a
  42 mAh LiR2032 cell. Decide whether bounds should be C-rate based.
- [ ] `fidelity` is always `"full"`; the non-stationary EIS guard
  `self._last_fidelity == "full"` in `_cycle_spectrum` is therefore always true, also
  after the emergency solver. Either drop the field or make the guard use
  `failure_kind`.
- [ ] `harmonic_flux_coefficients` (`battery_oracle.patches`) stays until upstream PR
  #5856 ships: it changes 8 operators in the stage-A oracle and 7 in the default oracle
  (the electrolyte current), none in the chain fit's model. Drop it then.
- [ ] Re-check on PyBaMM 26.10 that particle cracking still fails DAE initialisation
  with the Chen2020 graphite OCP (the code records PyBaMM 26.6.2.0).
- [ ] The degradation presets and their `target_eol_cycles_at_1c` labels were never
  checked against the DFN; there is no calibration of them for LiR2032.
- [ ] `dod_lam_scale` needs a literature value or a deep-DoD dataset before use.
- [ ] `config_oracle_defaults.yml`: the `preset_constants` switches and the
  `target_eol_cycles_at_1c` labels are not read by the code.
- [ ] A top-level `model:` section in an experiment YAML is silently ignored; consider
  rejecting unknown top-level keys.

## Docs, CI, scripts

- [ ] Build the docs with `sphinx-build -W` after the rewrite (not run). The API page
  now autosummarises every module, so docstring markup warnings may surface.
- [ ] Regenerate the figures in `docs/_static/degradation` and `docs/_static/numerics`
  with the DFN (they were made with SPMe and are no longer referenced by the docs),
  then reference them again from `degradation.md` / `numerics.md`.
- [ ] CI (`ci.yml`, `docs.yml`) triggers on `main` only (Phase 0, item 3).
- [ ] The CPE seeds and `ecm_rescale_target_r0` are jones2022 training statistics;
  document or move them with the featurisation they belong to **(unverified intent)**.
