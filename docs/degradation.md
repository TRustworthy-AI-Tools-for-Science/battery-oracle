# Degradation presets

The `degradation_preset` kwarg selects which physical aging mechanisms are active.
All presets apply a universal ×0.25 EC-diffusivity correction (a more passivating
SEI than the Chen2020 baseline {cite:p}`reniers2019`); they differ in which
mechanisms are enabled.

| Preset        | Mechanisms                                    | Target EOL @ 1C |
|---------------|-----------------------------------------------|-----------------|
| `nominal`     | SEI only                                       | ~200–400 cycles |
| `accelerated` | SEI + lithium plating {cite:p}`okane2022`      | ~40–70 cycles   |
| `severe`      | SEI + faster lithium plating                   | ~20–50 cycles   |

```python
oracle = PyBaMMOracle(degradation_preset="severe")
```

The SEI film {cite:p}`christensen2004,pinson2013` is treated as a pure ohmic
contribution to the impedance; lithium plating and dead-lithium accumulation provide
C-rate sensitivity. Particle cracking {cite:p}`ai2020` is **disabled** in all presets
— see [Numerical stability](numerics.md) for why.

## End-of-life

`eol_capacity_fraction` (default `0.80`) sets the SOH threshold below which the
oracle raises {class}`~battery_oracle.OracleFailure`. Treat this as battery
end-of-life: do not reset the oracle and end the loop.

## Accurate capacity

By default the oracle reads discharge capacity from the voltage-limited cycling step.
Set `capacity_check=True` to append a C/20 reference discharge each call for a more
accurate (but ~20× slower) capacity measurement.

## Calibrating degradation

The four per-instance scales (`kinetics_scale`, `sei_rate_scale`,
`dead_li_decay_scale`, `plating_rate_scale`) tune the degradation rates to match a
target cell — see [Calibration](calibration.md).
