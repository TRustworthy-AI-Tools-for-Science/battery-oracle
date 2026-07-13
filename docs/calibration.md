# Calibration (tune-oracle)

The `[tune]` extra fits the oracle's degradation hyperparameters to a target cell's
measured EIS/capacity behaviour. It is dataset-agnostic; it consumes a plain
cache dict of per-cycle ECM fits + protocols and precomputed target metrics. It is also circuit-agnostic, with no assumed ECM layout (see below).

## Calibration scales

Calibration searches log-uniformly over four multiplicative corrections to the
preset physics. They were chosen to be as close to orthogonal levers on the
observable targets as the underlying physics permits; the residual couplings are the
important part:

| Scale | Multiplies | Primary effect | Confounding side effect |
|---|---|---|---|
| `kinetics_scale` ∈ [0.1, 0.5] | both electrodes' $j_0(c_e, c_s, T)$ (function-wrapped) | grows the relative charge-transfer arcs $(R_3{+}R_5)/R_1$ — research coin cells sit ~10× above the automotive-grade Chen2020 | lower $j_0$ ⇒ higher $\eta$ at fixed current ⇒ exponentially faster SEI growth ($j_{\mathrm{SEI}} \propto e^{-F\eta/2RT}$) |
| `sei_rate_scale` ∈ [0.01, 1] | $D_{\mathrm{EC}}$ (on top of the universal ×0.25) | slows SEI/R1 growth — exists precisely to undo the side effect above | strongly sublinear once growth is kinetics-limited (measured 133/111/82.5 % R1 growth at 0.5/0.25/0.1×); large excursions saturate |
| `plating_rate_scale` ∈ [0.01, 10] | $k_{\mathrm{pl}}$ | lithium plated per cycle | moves R1 growth and EOL together (same sign) |
| `dead_li_decay_scale` ∈ [0.1, 1000] | $\gamma$ (dead-Li decay) | partitions plated Li between dead film and dissolution: R1 growth without the EOL shift | — the designed complement to `plating_rate_scale` |

The (`plating_rate_scale`, `dead_li_decay_scale`) pair spans the (R1-growth, EOL)
plane; the (`kinetics_scale`, `sei_rate_scale`) pair spans (arc ratio, R1-growth).
The overlap on R1-growth is the fundamental identifiability limit — see below.

## Targets and scoring

{func}`~battery_oracle.compute_real_targets` reduces the cache to three scalar
targets, chosen because they are protocol-independent health signals:

- **`mean_arc_ratio`** = mean $(\sum_i R_{\mathrm{arc},i})/R_{\mathrm{ohmic}}$ over
  all cycles and both charge/discharge states — the relative size of the
  charge-transfer arcs;
- **`r1_growth_pct`** = ohmic-resistance growth first→last cycle (charge-state ECM),
  the film-growth signature;
- **`soh_fade_per_cycle`** = mean per-cycle real-SOH loss over the window (from the
  cache's per-cycle `real_soh`, or `real_capacity_mah` ÷ reference capacity) — the
  capacity-fade signature.

The first two need EIS/ECM data; `soh_fade_per_cycle` needs only capacity/cycling
data. `score_candidate` therefore runs in one of **two modes**, selected
automatically from what the cache provides:

- **EIS/ECM mode** (`mean_arc_ratio`/`r1_growth_pct` present): arc-ratio + R1-growth
  relative errors, plus an EOL-plausibility penalty anchored to the preset's
  documented cycle-life midpoint — without it, the BO happily matches arc-ratio and
  R1-growth with cells that die in 5 cycles or live 500.
- **Capacity-fade mode** (no ECM target — e.g. the EIS-less CALCE/Oxford/MATR
  datasets, whose caches carry `real_soh` with null ECMs): the fit is driven by the
  log-ratio of oracle vs. real `soh_fade_per_cycle`, which *replaces* the preset EOL
  anchor with a real-data target. `kinetics_scale` is left near its chemistry
  default here — with no charge-transfer-arc signal, nothing constrains it.

Ohmic vs arc resistor positions are derived from the circuit string
(ohmic = the resistor outside any `[R,CPE]` branch), so both ECM metrics work
unchanged for any circuit. Fits with $R_{\mathrm{ohmic}} < 10^{-6}\,\Omega$ are
discarded as AutoEIS blow-up artefacts. The default single-objective engine
({func}`~battery_oracle.calibrate_oracle`, Optuna TPE, log-space) scalarises the
active terms. Optional C-rate probes (`skip_crate_probe=False`) add a
sensitivity-ratio term by re-running the candidate at low/high C from fresh state.

## Identifiability, Multi-objective Approach

Because two scale pairs share the R1-growth axis, the inverse problem is
under-determined: distinct scale vectors reach indistinguishable
(arc-ratio, R1-growth) pairs, differing mainly in their mechanism attribution
(SEI-driven vs dead-Li-driven ohmic growth) and hence in extrapolation behaviour
under protocols unlike the calibration protocols. Practical guidance:

- Treat a fitted point as a representative of a level set, not a unique
  physical identification; report the full trial population
  (`out["results"]`/`out["scored"]`), which `write_oracle_config` embeds as
  provenance.
- When arc-ratio and R1-growth compete, scalarisation hides the trade-off.
  The [tuning notebook](notebooks/01_oracle_tuning_calibration) runs a genuine
  NSGA-II multi-objective study (both errors minimised simultaneously), yields
  the non-dominated set (`study.best_trials`), tracks convergence by dominated
  hypervolume, and shows the mechanism mix changing along the Pareto front. The
  arc-optimal and R1-optimal extremes are different physical stories, not different
  precisions.
- Seed-sensitivity is real (TPE and NSGA-II are both stochastic); a reliable calibration should repeat across seeds and check that the pareto front, not just a point, is stable.

## API

```python
from battery_oracle import compute_real_targets, calibrate_oracle, write_oracle_config

targets = compute_real_targets(cache)                       # circuit from cache/config
out = calibrate_oracle(cache, targets, preset="accelerated", n_trials=35)
write_oracle_config("config_oracle_mycell.yml", "mydataset", "accelerated",
                    cell_id="C01", n_cycles=len(cache["cycles"]),
                    best=out["best"], real_targets=targets, all_results=out["results"])
```

### Cache format

The cache contract (minimal EIS/ECM form; add per-cycle `real_soh` for capacity-fade
mode):

```python
cache = {
    "real_cell_capacity_mah": 42.0,       # per-cell measured capacity (sets current mapping)
    "circuit": "R1-[R2,P3]-[R4,P5]",   # optional: ECM layout of the vectors below
                                       # (legacy caches may carry the 9-param
                                       # "R1-P2-[R3,P4]-[R5,P6]")
    "cycles": ["0", "1", ...],
    "data": {"0": {"protocol": [...6...],
                   "ecm_charge": [...], "ecm_discharge": [...]}, ...},
}
```

```{note}
No ECM structure is assumed. The metrics derive resistor positions from the
circuit string: `cache["circuit"]` if present, else the `circuit=` argument, else
the packaged default from `config_oracle_defaults.yml` (`ecm.circuit`) via
{func}`~battery_oracle.load_default_ecm_circuit`. Candidate oracles are constructed
with the same circuit, so cached and simulated ECM vectors are always commensurate.
```

CLI (cache/targets from JSON):

```bash
battery-oracle-tune --cache real_ecm.json --output-config config_oracle_mycell.yml \
    --preset accelerated --n-trials 35
```

Transplant the fitted `protocol_scaling` block into an experiment config as shown in
[Worked configurations](protocol.md) ("Calibrated cell").

### Plots

After the search, the CLI automatically writes **`oracle_tuning_eis.png`** — a
Nyquist overlay of the winning candidate's synthesized EIS against the
ground-truth EIS (reconstructed from the cached ECM), at the first and last cycle.
Both spectra are normalised by their ohmic resistance R0: the oracle's 5 Ah PyBaMM
cell has a ~16x smaller absolute impedance than a real coin cell, so only the
R0-normalised arc shape — what `mean_arc_ratio` scores — is comparable (absolute R0
is annotated per curve). It lands next to `--output-config` (override with
`--plots-dir`; disable with `--no-eis-plot`), and is produced by re-running the best
candidate once, so it needs a live oracle. For an EIS-less, capacity-only cache
there is no ground-truth EIS, so the plot is skipped
(`collect_eis_comparison` returns `None`).

The Pareto / optimisation-history / real-vs-achieved *alignment* plots are separate:
they are regenerable from the `sweep_results.csv` + `calibration_summary.json`
sidecars via {func}`~battery_oracle.plot_tune_oracle_summary` (or the
`battery-oracle-tune-plot` CLI), without re-running the search.

## Multi-dataset & multi-chemistry calibration (#12, #14)

The package ships **placeholder** per-dataset calibration configs alongside
`config_oracle_defaults.yml`:

| File | Dataset | Chemistry (`parameter_set`) | Notes |
|------|---------|-----------------------------|-------|
| `config_oracle_calce.yml`  | CALCE CS2         | `Chen2020` (NMC proxy) | 1C CC-CV / 0.5C, cycles 1–100 |
| `config_oracle_oxford.yml` | Oxford Kokam      | `Chen2020` (NMC proxy) | 1C, 40 °C, 740 mAh |
| `config_oracle_matr.yml`   | MATR (Severson 2019) | `Prada2013` (**LFP**) | 2.0–3.65 V window, ~1.1 Ah |

They carry the four calibration scales at their **chemistry defaults** (all `1.0`)
and a `_calibration.status: PLACEHOLDER` marker: the scales are **not yet
calibrated**. All three datasets ship capacity/cycling but no EIS, so they calibrate
through the engine's [capacity-fade mode](#targets-and-scoring) — a real, supported
path, not a stub (see the adapter procedure below). The only remaining step is the
data itself: download the dataset, build the cache, and run the `battery-oracle-tune`
command in each file's header.

Select one by name (no experiment YAML needed):

```python
from battery_oracle import build_oracle_from_oracle_config
oracle = build_oracle_from_oracle_config("matr")   # LFP oracle with the 3.65 V window
```

…or as the base layer of an experiment config (mutually exclusive with
`oracle_config`):

```python
from battery_oracle import build_oracle_from_config
oracle = build_oracle_from_config(experiment_cfg, config_dataset="calce")
```

The config's `chemistry` is validated against its `parameter_set` at build time,
so an LFP-calibrated YAML cannot be silently paired with an NMC cell.

### Dataset → cache adapter (procedure)

None of CALCE / Oxford / MATR ships EIS spectra, so their adapters emit the
[cache schema](#cache-format) with `ecm_charge`/`ecm_discharge = null` and populate
per-cycle `real_soh` (or `real_capacity_mah`) + the 6-D `protocol`. With null ECMs,
{func}`~battery_oracle.compute_real_targets` returns `mean_arc_ratio =
r1_growth_pct = None` and a real `soh_fade_per_cycle`, so calibration runs in
[capacity-fade mode](#targets-and-scoring): the fit matches the oracle's per-cycle
SOH-loss rate to the dataset's, leaving `kinetics_scale` near its chemistry default
(no charge-transfer-arc signal to constrain it). Steps:

1. Download the raw dataset (external; not vendored).
2. Map each cycle to a 6-D `[C_rate_1, C_rate_2, dur_1, dur_2, D_rate, dur_d]`
   protocol and record its measured `real_soh` (and/or `real_capacity_mah`).
3. Write the cache JSON, then:
   ```bash
   battery-oracle-tune --cache matr.json --chemistry Prada2013 --preset nominal \
       --output-config src/battery_oracle/config_oracle_matr.yml
   ```

The written config records the fit under `_calibration.capacity_fade`
(`real_soh_fade_per_cycle` vs `achieved_soh_fade_per_cycle`); the `kinetics_scale`
provenance notes it was held at the chemistry default for want of an EIS signal.
