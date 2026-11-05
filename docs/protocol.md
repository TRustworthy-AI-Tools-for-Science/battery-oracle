# Experiment protocols

## The protocol vector

Every oracle call takes a 6-D vector, ordered as
{data}`battery_oracle.ACTION_FEATURE_NAMES`:

```
[C_rate_1, C_rate_2, duration_1, duration_2, D_rate, duration_d]
 mA        mA        h           h           mA      h
```

Despite the names, the current slots are currents in mA at the real cell's scale.
{class}`~battery_oracle.Protocol` gives the same vector unit-explicit field names
(`charge_current_1_mA`, `charge_current_2_mA`, `charge_duration_1_h`,
`charge_duration_2_h`, `discharge_current_mA`, `discharge_duration_h`) and converts
with `to_array()` / `from_array()`. With `thermal="lumped"` and
`use_temperature_protocol=True` the vector has a seventh slot, `T_ambient_K`, the
ambient temperature of that call (clipped to 253.15–333.15 K).

## From protocol to PyBaMM experiment

### Current mapping

Currents are mapped onto the simulated cell at the same C-rate:

$$
I_{\mathrm{sim}}\,[\mathrm{mA}] = I_{\mathrm{real}}\,[\mathrm{mA}]\cdot
\frac{Q_{\mathrm{nom}}\,[\mathrm{mAh}]}{Q_{\mathrm{real}}\,[\mathrm{mAh}]},
$$

with $Q_{\mathrm{real}}$ = `real_cell_capacity_mah` (default 200 mAh) and
$Q_{\mathrm{nom}}$ the parameter set's `Nominal cell capacity [A.h]` after the
beginning-of-life overrides (5 Ah for Chen2020, 42 mAh for LiR2032). A cell
parameterised at its real size (a stage-A fit sets the nominal capacity to the
measured one) is run with `real_cell_capacity_mah` equal to it, so the mapping is the
identity.

### Sanitisation

The mapped values are bounded before the experiment is built. The current bounds are
C-rates of the simulated cell's nominal capacity $Q_{nom}$, so the same bounds suit a
5 Ah pouch and a 42 mAh coin cell:

| Quantity | Bound | Default |
|---|---|---|
| charge stage 1, discharge current | `[c_min_crate · Q_nom, c_max_crate · Q_nom]` | 0.01C, 1.5C |
| charge stage 2 current | `[c2_min_crate · Q_nom, c2_max_crate · Q_nom]` | 0.004C, 1.5C |
| every duration | `[dur_min_s, dur_max_s]` | 60 s, 8 h |
| each charge stage | at most `charge_stage_max_s` | 900 s |
| discharge duration, `discharge_mode: cutoff` | at least 3600 s, so the discharge reaches `v_discharge_min` | |
| discharge duration, `discharge_mode: protocol` | as requested (time-limited discharges) | |

A non-finite entry is replaced by a default: 0.1C for stage 1 and the discharge,
0.05C for stage 2 (of $Q_{\mathrm{nom}}$), 1 h for the discharge and 0.25 h for each
charge stage. The defaults suit the 5 Ah Chen2020 cell; a cell of another size needs
its own `protocol_bounds` (for a 42 mAh cell the 50 mA lower bound is above 1C).

Out-of-range values are clipped, not rejected, so an active-learning proposer always
gets an answer. The first occurrence of each kind of clip (field and reason) is
logged as one warning per call; repeats are logged at DEBUG. `CycleResult.protocol_applied`
holds the values simulated, and `logging.getLogger("battery_oracle.oracle")` at
DEBUG logs every call's step strings.

### The cycle

Each call is one PyBaMM cycle of five steps, after the protocol of Jones, Stimming &
Lee {cite:p}`jones2022`:

```
[0] Discharge at D  for dur_d   or until v_discharge_min (3.0 V)
[1] Rest rest_s (1200 s)                                    <- EIS, discharged state
[2] Charge at C1    for dur_1   or until v_charge_max (4.3 V)
[3] Charge at C2    for dur_2   or until v_charge_max       (two-stage CC, no CV hold)
[4] Rest rest_s                                             <- EIS, charged state
```

The steps are one tuple, so PyBaMM treats them as a single cycle and its cycle-level
degradation summaries accumulate once per call. With `n_cycles > 1` the cycle is
repeated. With `capacity_check: true` four steps are appended to each cycle: a C/20
charge to `v_charge_max`, a hold until C/50, a 300 s rest and a C/20 discharge to
`v_discharge_min`; `capacity_ah` is then the last step's discharge capacity. With
`eis_mode: nonstationary` each rest is extended by the length of the EIS sweep, so
the cycle keeps the measurement's real timing.

## Configuration files

Two YAML layers configure an oracle.

**The oracle YAML** (`config_oracle_defaults.yml`, packaged, or a
`config_oracle_*.yml` of a cell) holds oracle settings in these sections; the
field-to-key map is `battery_oracle.oracle_config.YAML_KEYS`:

| Section | Keys |
|---|---|
| `cycling` | `parameter_set`, `chemistry`, `n_cycles`, `temperature_K`, `initial_soc`, `rest_s`, `thermal`, `T_ambient_K`, `h_total_W_per_m2K`, `use_temperature_protocol`, `discharge_mode` |
| `bol` | `parameter_scales`, `parameter_overrides`, `contact_resistance_ohm`, `radial_mesh` |
| `solver` | `dfn.rtol`, `dfn.atol`, `dfn.dt_max_s`, `emergency.rtol`, `emergency.atol` |
| `protocol_bounds` | `c_min_crate`, `c_max_crate`, `c2_min_crate`, `c2_max_crate`, `dur_min_s`, `dur_max_s`, `v_charge_max`, `v_discharge_min`, `charge_stage_max_s` |
| `eis` | `freq_min_hz`, `freq_max_hz`, `n_freq_points` or `frequencies_hz`; `noise_level`, `noise_model`, `noise_combined_flicker_frac`, `noise_combined_white_frac`, `seed`; `mode`, `amplitude_mA`, `sweep_periods`, `sweep_overhead_s`; `E_a_J_per_mol`, `E_a_electrolyte_J_per_mol`; `soc_clip_min`, `soc_clip_max`; `linkk.c`, `linkk.max_M` |
| `ecm` | `circuit`, `cpe_seeds.{w,n,w_default,n_default}`, `rescale_target_r0`, `autoeis.{num_warmup,num_samples}` |
| `degradation` | `preset`, `eol_capacity_fraction`, `capacity_check`, `lam_ceiling`, `dod_lam_scale`, `preset_constants.<preset>` |
| `protocol_scaling` | `real_cell_capacity_mah`, `kinetics_scale`, `sei_rate_scale`, `dead_li_decay_scale`, `plating_rate_scale` |

A key absent from the YAML takes the `OracleConfig` default. An unknown key under
these sections raises `ValueError`. `cycling.chemistry` and `cycling.parameter_set`
must name the same set when both are given
({func}`battery_oracle.parameter_sets.check_chemistry`). A few `protocol_scaling`
keys (`real_cell_capacity`, `charge_envelope_crate`, `c_rate_range_mA`,
`c_rate_ceiling_fraction`) are accepted but read only by battery-forecast.

**The experiment YAML** (`config_experiment_defaults.yml`, packaged) adds protocols
to run. Its required sections are `cycling`, `degradation`, `eis` and `protocols`;
`ecm` and `oracle_config` are optional. Precedence, lowest first:

1. `OracleConfig` defaults;
2. the oracle YAML named by `oracle_config` (default: the packaged one);
3. the experiment's `cycling`, `degradation`, `eis` and `ecm.circuit`, which use the
   oracle YAML's keys, except that `cycling.real_cell_capacity_mah` and
   `degradation.{kinetics,sei_rate,dead_li_decay,plating_rate}_scale` go to the
   oracle YAML's `protocol_scaling`.

`ecm.fitter` is `randles` (default, core install) or `autoeis` (needs the `[autoeis]`
extra). `degradation.preset` must be `nominal` or `accelerated`. The
packaged oracle YAML sets `cycling.chemistry: Chen2020`, so an experiment that changes
`cycling.parameter_set` must set `cycling.chemistry` to the same name. There is no
model section: the cycling model is always the DFN.

```yaml
oracle_config: null              # or a path to a config_oracle_*.yml
cycling:
  n_cycles: 1
  temperature_K: 298.15
  parameter_set: Chen2020
  real_cell_capacity_mah: 200.0
degradation:
  preset: accelerated
  eol_capacity_fraction: 0.80
  capacity_check: false
  kinetics_scale: 1.0
  sei_rate_scale: 1.0
  dead_li_decay_scale: 1.0
  plating_rate_scale: 1.0
  dod_lam_scale: 0.0
eis:
  freq_min_hz: 0.01
  freq_max_hz: 10000.0
  n_freq_points: 60
  noise_level: 0.02
  noise_model: combined          # white | flicker | combined | none
ecm:
  fitter: randles                # randles | autoeis
  circuit: "R1-[R2,P3]-[R4,P5]"  # the parameter layout is derived from this string
protocols:                       # one or more; 'name' is optional
  - name: baseline
    C_rate_1_mA: 200.0
    C_rate_2_mA: 100.0
    dur_1_h: 1.0
    dur_2_h: 0.5
    D_rate_mA: 100.0
    dur_d_h: 1.0
```

## Worked configurations

### Fast-charge stress study

A 2C/1C two-stage charge of a 200 mAh cell with the `accelerated` preset (SEI + plating). On the 5 Ah
simulated cell 2C is above the 1.5C ceiling, so stage 1 is clipped to 7.5 A; raise
`protocol_bounds.c_max_crate` in the oracle YAML to run it as asked.

```yaml
cycling: {n_cycles: 1, temperature_K: 298.15, parameter_set: Chen2020,
          real_cell_capacity_mah: 200.0}
degradation: {preset: accelerated, eol_capacity_fraction: 0.80, capacity_check: false}
eis: {freq_min_hz: 0.1, freq_max_hz: 10000.0, n_freq_points: 60,
      noise_level: 0.01, noise_model: white}
ecm: {fitter: randles}
protocols:
  - {name: fast_2C, C_rate_1_mA: 400.0, C_rate_2_mA: 200.0,
     dur_1_h: 0.25, dur_2_h: 0.25, D_rate_mA: 200.0, dur_d_h: 1.0}
```

`freq_min_hz: 0.1` drops the noisiest decade and `white` noise avoids the
$1/\sqrt{f}$ growth of flicker noise (below), which suits a study of degradation
trends rather than of measurement realism.

### Temperature

Isothermal (default): `cycling.temperature_K` sets PyBaMM's constant
`Ambient temperature [K]`, which enters every temperature-dependent parameter. A
temperature sweep is a family of configs differing in this field.

Lumped thermal: `cycling.thermal: lumped` adds PyBaMM's lumped thermal ODE, with
`T_ambient_K` and `h_total_W_per_m2K`. The cell temperature then follows the heat
generated, enters the state vector as `T_cell_K`, and scales the equilibrium spectra
(see [the model](models.md)). `use_temperature_protocol: true` makes the ambient a
per-call protocol slot.

```yaml
cycling: {thermal: lumped, T_ambient_K: 288.15, h_total_W_per_m2K: 10.0,
          parameter_set: Chen2020, real_cell_capacity_mah: 200.0}
```

### High-fidelity spectra

`eis.mode: nonstationary` (needs the `[fit]` extra) computes each spectrum on the
relaxing cell during the sweep that follows the rest; `ecm.fitter: autoeis` (needs the
`[autoeis]` extra) gives posterior medians and standard deviations. Both are slow; use
few protocols.

```yaml
eis: {mode: nonstationary, amplitude_mA: 5.0, sweep_periods: 1.0, sweep_overhead_s: 1.15}
ecm: {fitter: autoeis, circuit: "R1-[R2,P3]-[R4,P5]"}
```

### A fitted LiR2032 cell

A cell fitted in stage A ([Fitting](fitting.md)) is an oracle YAML whose `bol`
section holds the fitted values, with `cycling.chemistry` and `cycling.parameter_set`
both `LiR2032`, `protocol_scaling.real_cell_capacity_mah` equal to the fitted
nominal capacity, `cycling.thermal: lumped` with the fitted `h_total_W_per_m2K`, and
`protocol_bounds` set for a 42 mAh cell (the defaults are for 5 Ah). The degradation
presets are not calibrated for this cell.

## Running configs

```python
from battery_oracle import run_experiment
history = run_experiment("my_experiment.yml")      # load, build, run every protocol once
```

`run_experiment` calls `reset()` first (`reset=False` skips it), runs the protocols
in order, and stops at the first {class}`~battery_oracle.OracleFailure`. It then
appends a terminal row (`failed: True`, `failure_kind`,
`protocol`) and returns the history: a short history is data, not an error.

The pieces compose, for example for a sweep over a base config:

```python
from battery_oracle import (
    load_experiment_config, build_oracle_from_config, protocols_from_config,
)

cfg = load_experiment_config("my_experiment.yml")   # validated dict; None -> the packaged file
for T in (288.15, 298.15, 308.15, 318.15):
    cfg["cycling"]["temperature_K"] = T
    oracle = build_oracle_from_config(cfg)          # resolves parameter_set and ecm.fitter
    for p in protocols_from_config(cfg):
        oracle(p)
```

{func}`~battery_oracle.oracle_kwargs_from_config` and
{func}`~battery_oracle.oracle_kwargs_from_oracle_config` return the constructor
keywords without building (PyBaMM-free; `parameter_set` and `ecm_fitter` are left
as strings). {func}`~battery_oracle.load_oracle_config` parses an oracle YAML and
{func}`~battery_oracle.load_default_ecm_circuit` returns the packaged `ecm.circuit`.

## EIS noise

Synthesised spectra carry measurement noise set by `eis.noise_level` ($\epsilon$,
default 0.02) and `eis.noise_model`. Gaussian noise is added independently to the
real and imaginary parts, with standard deviation

- `white`: $\epsilon\,|Z|$, flat in frequency;
- `flicker`: $\epsilon\,|Z|/\sqrt{f}$, growing toward low frequency (at 0.01 Hz,
  10× the 1 Hz value);
- `combined` (default): flicker with $0.75\,\epsilon$ plus white with $0.25\,\epsilon$
  (`noise_combined_flicker_frac`, `noise_combined_white_frac`);
- `none`: no noise.

To reduce low-frequency scatter: use `white`, lower $\epsilon$, or raise
`freq_min_hz`. Every history row stores `linkk_rmse`, the linKK
Kramers–Kronig residual of the charge spectrum {cite:p}`luo2021`, for screening.
