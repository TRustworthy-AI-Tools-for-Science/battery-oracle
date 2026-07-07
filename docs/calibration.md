# Calibration (tune-oracle)

The `[tune]` extra provides an Optuna Bayesian-optimisation engine that fits the
oracle's degradation hyperparameters (the four scales) to a target cell's measured
EIS/capacity behaviour. It is dataset-agnostic: it operates on a plain **cache dict**
of measured ECM-per-cycle + capacity + protocols, plus precomputed target metrics.

```python
from battery_oracle import (
    compute_real_targets, calibrate_oracle, write_oracle_config,
)

targets = compute_real_targets(cache)                      # arc-ratio + R1-growth targets
out = calibrate_oracle(cache, targets, preset="accelerated", n_trials=35)
write_oracle_config("config_oracle_mycell.yml", "mydataset", "accelerated",
                    cell_id="C01", n_cycles=len(cache["cycles"]),
                    best=out["best"], real_targets=targets, all_results=out["results"])
```

## The cache format

```python
cache = {
    "real_cell_capacity_mah": 200.0,
    "circuit": "R1-P2-[R3,P4]-[R5,P6]",   # optional; ECM structure of the vectors below
    "cycles": ["0", "1", ...],
    "data": {
        "0": {"protocol": [...6...], "ecm_charge": [...], "ecm_discharge": [...]},
        ...
    },
}
```

```{note}
**No ECM structure is assumed.** The arc-ratio ((sum arc R)/(ohmic R)) and R1-growth
metrics derive the ohmic/arc-resistor positions from the ECM circuit. That circuit is
`cache["circuit"]` if present, else the `circuit=` argument, else the default from
`config_oracle_defaults.yml` (`ecm.circuit`) via
{func}`~battery_oracle.load_default_ecm_circuit`.
```

## CLI

```bash
battery-oracle-tune --cache real_ecm.json --output-config config_oracle_mycell.yml \
    --preset accelerated --n-trials 35
```

## Worked example

The [tuning notebook](notebooks/01_oracle_tuning_calibration) builds a self-contained
cache from the oracle itself, calibrates it, and visualises the AutoEIS fit, the
Optuna Pareto frontier, the convergence trace, and how the degradation-mechanism mix
shifts during tuning.
