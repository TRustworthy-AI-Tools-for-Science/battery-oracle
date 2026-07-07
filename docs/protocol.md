# Experiment protocols

## The 6-D protocol vector

Every oracle query is a 6-D charge/discharge protocol, in the order given by
{data}`battery_oracle.ACTION_FEATURE_NAMES`:

```
[C_rate_1_mA, C_rate_2_mA, dur_1_h, dur_2_h, D_rate_mA, dur_d_h]
```

- `C_rate_1_mA`, `C_rate_2_mA` — two-stage CC charge currents (stage 2 tapers stage 1)
- `dur_1_h`, `dur_2_h` — the two charge-stage durations (each capped at 15 min)
- `D_rate_mA`, `dur_d_h` — discharge current and duration (governed by a 3.0 V cutoff)

{func}`~battery_oracle.make_pybamm_candidates` builds a grid varying the first
charge current.

## Executable protocol config

Unlike `config_oracle_defaults.yml` (reference documentation of the oracle's
internal defaults), `config_experiment_defaults.yml` is **executable**: it defines a
protocol and is run by {mod}`battery_oracle.experiment`.

```yaml
model:
  type: SPMe                 # SPMe | SPM | DFN
cycling:
  n_cycles: 1
  temperature_K: 298.15
  parameter_set: Chen2020
  real_cell_capacity_mah: 200.0
degradation:
  preset: accelerated        # nominal | accelerated | severe
  eol_capacity_fraction: 0.80
  capacity_check: false
eis:
  freq_min_hz: 0.01
  freq_max_hz: 10000.0
  n_freq_points: 60
  noise_level: 0.02
  noise_model: combined
ecm:
  fitter: randles            # randles | autoeis
  circuit: "R1-P2-[R3,P4]-[R5,P6]"   # ECM structure; layout is derived from this string
protocols:
  - name: baseline
    C_rate_1_mA: 200.0
    C_rate_2_mA: 100.0
    dur_1_h: 1.0
    dur_2_h: 0.5
    D_rate_mA: 100.0
    dur_d_h: 1.0
```

Load and run it:

```python
from battery_oracle import run_experiment
history = run_experiment("config_experiment_defaults.yml")
```

or build the pieces yourself:

```python
from battery_oracle import (
    load_experiment_config, build_oracle_from_config, protocols_from_config,
)
cfg = load_experiment_config("config_experiment_defaults.yml")
oracle = build_oracle_from_config(cfg)
oracle.reset()
for p in protocols_from_config(cfg):
    oracle(p)
```

Every YAML field maps to a {class}`~battery_oracle.PyBaMMOracle` constructor kwarg —
see {func}`~battery_oracle.oracle_kwargs_from_config`.

## EIS noise (and noisy low frequencies)

Synthesised spectra carry measurement noise controlled by `eis.noise_level` and
`eis.noise_model` (`white` | `flicker` | `combined` | `none`). The default
`combined` model adds **1/f flicker noise** whose amplitude scales as
`noise_level · |Z| / √f`, so it grows toward low frequencies — at 0.01 Hz it is
~10× the noise at 1 Hz. This is realistic (real low-frequency EIS is dominated by
1/f noise and drift), but it makes the low-frequency tail visibly scattered.

To reduce the low-frequency scatter:

- **Flat noise:** `noise_model: white` — flat `noise_level · |Z|`, no 1/√f blow-up.
- **Lower level:** e.g. `noise_level: 0.005`.
- **Trim the noisy decade:** raise `freq_min_hz` (e.g. `0.1`).
- **Screen bad points:** Kramers–Kronig validation via
  `battery_oracle._eis.kk.linkk_rmse` (each oracle record already stores `linkk_rmse`).

Non-stationary drift (`eis_drift_scale`, off by default) is a separate low-frequency
effect coupled to the rest time; keep it at `0.0` unless modelling operando drift.
