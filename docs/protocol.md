# Experiment protocols

## The 6-D protocol vector and semantics

Every oracle query is a 6-D vector, ordered per
{data}`battery_oracle.ACTION_FEATURE_NAMES`:

```
[C_rate_1_mA, C_rate_2_mA, dur_1_h, dur_2_h, D_rate_mA, dur_d_h]
```

Currents are specified at the real cell's scale and mapped onto the Chen2020
5 Ah simulation cell by the same-C-rate-fraction rule

$$
I_{\mathrm{sim}}\,[\mathrm{A}] =
\mathrm{clip}\!\left(I_{\mathrm{real}}\,[\mathrm{mA}] \cdot
\frac{5000}{Q_{\mathrm{real}}\,[\mathrm{mAh}]},\;
50,\; 10\,000\right) / 1000,
$$

with $Q_{\mathrm{real}}$ = `real_cell_capacity_mah`. The clip enforces a validated
envelope (≤2C on the 5 Ah cell; SPMe accuracy degrades above ≈1C and the DAE solver
diverges above ≈2–3C — see [Numerical stability](numerics.md)). Durations are
clipped to [60 s, 8 h], and each CC charge stage is additionally capped at 900 s.
Non-finite entries fall back to defaults. Out-of-envelope protocols are silently
clipped, not rejected, because an active-learning proposer must always get an answer. Inspect the logged sanitized step strings when auditing.

Each call expands the vector into one experiment cycle of five steps, after the
cycling protocol of Jones, Stimming & Lee {cite:p}`jones2022`:

```
[0] Discharge at D A          for dur_d   or until 3.0 V
[1] Rest 1200 s                                            ← EIS @ discharged
[2] Charge   at C1 A (CC-1)   for dur_1   or until 4.3 V
[3] Charge   at C2 A (CC-2)   for dur_2   or until 4.3 V   (two-stage taper, no CV hold)
[4] Rest 1200 s                                            ← EIS @ charged
```

The steps are wrapped as a single PyBaMM cycle (tuple) so per-cycle degradation
integrators accumulate once per call, and EIS is synthesized at the two post-rest
(relaxed) states. The rest duration (`rest_s`, default 1200 s) also gates the
optional non-stationary EIS drift model.

## The experiment config

`config_experiment_defaults.yml` is the packaged, executable protocol definition. Every field
maps 1:1 onto a {class}`~battery_oracle.PyBaMMOracle` constructor kwarg via
{func}`~battery_oracle.oracle_kwargs_from_config`; the parse/mapping layer is
PyBaMM-free and independently testable. Schema (defaults shown):

```yaml
model:
  type: SPMe                     # SPMe | SPM | DFN         -> model=
cycling:
  n_cycles: 1                    #                          -> n_cycles=
  temperature_K: 298.15          # isothermal ambient T     -> temperature_K=
  parameter_set: Chen2020        # resolved lazily          -> parameter_values=
  real_cell_capacity_mah: 200.0  # sets the current mapping -> real_cell_capacity_mah=
degradation:
  preset: accelerated            # nominal|accelerated|severe
  eol_capacity_fraction: 0.80
  capacity_check: false
  kinetics_scale: 1.0            # the four calibration scales, default 1.0
  sei_rate_scale: 1.0
  dead_li_decay_scale: 1.0
  plating_rate_scale: 1.0
eis:
  freq_min_hz: 0.01              # -> frequencies=np.logspace(...)
  freq_max_hz: 10000.0
  n_freq_points: 60
  noise_level: 0.02
  noise_model: combined          # white | flicker | combined | none
ecm:
  fitter: randles                # randles | autoeis (needs [autoeis] extra)
  circuit: "R1-P2-[R3,P4]-[R5,P6]"   # ECM layout is DERIVED from this string
protocols:                       # ≥1 six-field entries; 'name' optional
  - name: baseline
    C_rate_1_mA: 200.0
    C_rate_2_mA: 100.0
    dur_1_h: 1.0
    dur_2_h: 0.5
    D_rate_mA: 100.0
    dur_d_h: 1.0
```

## Worked configurations

### Fast-charge stress study (plating-sensitive)

Two-stage taper at 2C/1C on a 200 mAh cell, `severe` preset to compound the plating
pathway, multiple protocols in one run:

```yaml
model: {type: SPMe}
cycling: {n_cycles: 1, temperature_K: 298.15, parameter_set: Chen2020,
          real_cell_capacity_mah: 200.0}
degradation: {preset: severe, eol_capacity_fraction: 0.80, capacity_check: false}
eis: {freq_min_hz: 0.1, freq_max_hz: 10000.0, n_freq_points: 60,
      noise_level: 0.01, noise_model: white}
ecm: {fitter: randles}
protocols:
  - {name: fast_2C,   C_rate_1_mA: 400.0, C_rate_2_mA: 200.0,
     dur_1_h: 0.25, dur_2_h: 0.25, D_rate_mA: 200.0, dur_d_h: 1.0}
  - {name: fast_2C_b, C_rate_1_mA: 400.0, C_rate_2_mA: 200.0,
     dur_1_h: 0.25, dur_2_h: 0.25, D_rate_mA: 200.0, dur_d_h: 1.0}
```

Note the EIS choices: `freq_min_hz: 0.1` drops the noisiest decade and `white`
noise avoids the $1/\sqrt{f}$ flicker blow-up (see below) — appropriate when the
study target is degradation trends rather than measurement realism.

### Isothermal temperature point

There is no thermal model — `temperature_K` sets a *constant* ambient temperature
entering all Arrhenius-corrected rate constants. A "temperature sweep" is therefore
a family of configs (or constructor calls) differing only in this field:

```yaml
cycling: {n_cycles: 1, temperature_K: 318.15, parameter_set: Chen2020,
          real_cell_capacity_mah: 200.0}
# ... everything else as in the default
```

### High-fidelity DFN reference with Bayesian ECM

DFN with the AutoEIS fitter — the expensive configuration; use a single protocol and
expect order-of-magnitude longer solves:

```yaml
model: {type: DFN}
degradation: {preset: accelerated, eol_capacity_fraction: 0.80, capacity_check: false}
ecm: {fitter: autoeis, circuit: "R1-P2-[R3,P4]-[R5,P6]"}   # requires [autoeis]
# cycling / eis / protocols as in the default
```

### Calibrated cell

After running the [calibration](calibration.md), transplant the fitted scales:

```yaml
degradation:
  preset: accelerated
  eol_capacity_fraction: 0.80
  capacity_check: false
  kinetics_scale: 0.31        # <- from config_oracle_<dataset>.yml protocol_scaling
  sei_rate_scale: 0.045
  dead_li_decay_scale: 12.0
  plating_rate_scale: 0.21
cycling: {n_cycles: 1, temperature_K: 298.15, parameter_set: Chen2020,
          real_cell_capacity_mah: 42.0}    # per-cell measured capacity
```

## Running configs

```python
from battery_oracle import run_experiment
history = run_experiment("my_experiment.yml")          # load → build → run all protocols
```

or compose the pieces — useful for parameter sweeps over a base config:

```python
from battery_oracle import (
    load_experiment_config, oracle_kwargs_from_config,
    build_oracle_from_config, protocols_from_config,
)

cfg = load_experiment_config("my_experiment.yml")      # validated dict; None -> packaged default
for T in (288.15, 298.15, 308.15, 318.15):
    cfg["cycling"]["temperature_K"] = T
    oracle = build_oracle_from_config(cfg)             # resolves parameter_set + ecm fitter
    oracle.reset()
    for p in protocols_from_config(cfg):
        oracle(p)
```

`run_experiment` stops at `OracleFailure` (end-of-life) and returns the history
accumulated to that point; treat a short history as data, not as an error.

## EIS noise (and noisy low frequencies)

Synthesized spectra carry measurement noise controlled by `eis.noise_level`
($\epsilon$) and `eis.noise_model`. The models, applied per quadrature component:

- `white`: $\sigma = \epsilon\,|Z|$ — frequency-flat;
- `flicker`: $\sigma(f) = \epsilon\,|Z|/\sqrt{f}$ — grows toward low frequency
  (at 0.01 Hz, ~10× the noise at 1 Hz);
- `combined` (default): the $\epsilon$ budget split 3:1 between flicker and white —
  realistic lab EIS, where the low-frequency tail is $1/f$-dominated.

To suppress low-frequency scatter: switch to `white`, lower $\epsilon$, raise
`freq_min_hz` (drop the worst decade), or screen points post-hoc with
Kramers–Kronig validation {cite:p}`luo2021` — every oracle record stores a
`linkk_rmse`. Non-stationary relaxation drift {cite:p}`hallemans2023`
(`eis_drift_scale`, default 0) is a separate low-frequency effect coupled to
`rest_s`; enable it only when explicitly modelling operando measurement conditions.
