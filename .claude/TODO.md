# battery-oracle: open items

Judged from the code on branch `refactor-lightweight`. Items marked **(unverified)**
could not be confirmed from the code alone.

## Oracle

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
