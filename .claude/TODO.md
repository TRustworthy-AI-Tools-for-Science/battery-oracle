# battery-oracle: open items

Judged from the code on branch `refactor-lightweight`. Items marked **(unverified)**
could not be confirmed from the code alone.

## Oracle

- [ ] `harmonic_flux_coefficients` (`battery_oracle.patches`) stays until upstream PR
  #5856 ships: it changes 8 operators in the stage-A oracle and 7 in the default oracle
  (the electrolyte current), none in the chain fit's model. Drop it then.
- [ ] Re-check on PyBaMM 26.10 that particle cracking still fails DAE initialisation
  with the Chen2020 graphite OCP (the code records PyBaMM 26.6.2.0). With the LiR2032
  set every mechanics variant solves on 26.10 (battery-forecast B5d, 2026-11-05); they
  stay off for the physical reasons in `battery_oracle.degradation`.
- [ ] The LiR2032 degradation rates are calibrated by battery-forecast Stage B (published
  values and bounds set 2026-11-05, decision D9); record the fitted scales here when B6 is done.
- [ ] `dod_lam_scale` needs a literature value or a deep-DoD dataset before use.
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
