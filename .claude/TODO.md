# battery-oracle: open items

Judged from the code on branch `refactor-lightweight`. Items marked **(unverified)**
could not be confirmed from the code alone.

## Oracle

- [ ] `harmonic_flux_coefficients` (`battery_oracle.patches`) stays until upstream PR
  #5856 ships: it changes 8 operators in the stage-A oracle and 7 in the default oracle
  (the electrolyte current), none in the chain fit's model. Drop it then.
- [ ] Re-check on PyBaMM 26.10 that particle cracking still fails DAE initialisation
  with the Chen2020 graphite OCP (the code records PyBaMM 26.6.2.0).
- [ ] The degradation presets were never checked against the DFN; there is no
  calibration of them for LiR2032 (battery-forecast Stage B is being rebuilt on
  lithium-loss targets, 2026-11-04).
- [ ] `dod_lam_scale` needs a literature value or a deep-DoD dataset before use.
- [ ] `config_oracle_defaults.yml`: the `preset_constants` submodel switches are not
  read by the code.
- [ ] A top-level `model:` section in an experiment YAML is silently ignored; consider
  rejecting unknown top-level keys.

## Docs, CI, scripts

- [ ] Build the docs with `sphinx-build -W` after the rewrite (not run). The API page
  now autosummarises every module, so docstring markup warnings may surface.
- [ ] `bin/generate_numerics_data.py`: on PyBaMM 26.10 and the DFN the silent-truncation
  and JAX-cache demos no longer show the effect their sections describe, and the
  cracking demo fails on missing crack parameters (Chen2020 has none) before the
  initialisation failure it is meant to show; rework them before citing them.
- [ ] CI (`ci.yml`, `docs.yml`) triggers on `main` only (Phase 0, item 3).
- [ ] The CPE seeds and `ecm_rescale_target_r0` are jones2022 training statistics;
  document or move them with the featurisation they belong to **(unverified intent)**.
