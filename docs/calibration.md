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

{func}`~battery_oracle.compute_real_targets` reduces the cache to two scalar
targets, chosen because they are protocol-independent health signals:

- **`mean_arc_ratio`** = mean $(\sum_i R_{\mathrm{arc},i})/R_{\mathrm{ohmic}}$ over
  all cycles and both charge/discharge states — the relative size of the
  charge-transfer arcs;
- **`r1_growth_pct`** = ohmic-resistance growth first→last cycle (charge-state ECM),
  the film-growth signature.

Ohmic vs arc resistor positions are derived from the circuit string
(ohmic = the resistor outside any `[R,CPE]` branch), so both metrics work unchanged
for any circuit. Fits with $R_{\mathrm{ohmic}} < 10^{-6}\,\Omega$ are discarded as
AutoEIS blow-up artefacts. The default single-objective engine
({func}`~battery_oracle.calibrate_oracle`, Optuna TPE, log-space) scalarises these
errors together with an EOL-plausibility penalty anchored to the preset's
documented cycle-life midpoint — without it, the BO happily matches arc-ratio and
R1-growth with cells that die in 5 cycles or live 500. Optional C-rate probes
(`skip_crate_probe=False`) add a sensitivity-ratio term by re-running the candidate
at low/high C from fresh state.

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

The cache contract:

```python
cache = {
    "real_cell_capacity_mah": 42.0,       # per-cell measured capacity (sets current mapping)
    "circuit": "R1-P2-[R3,P4]-[R5,P6]",   # optional: ECM layout of the vectors below
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
