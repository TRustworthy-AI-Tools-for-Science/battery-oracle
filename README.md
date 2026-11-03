<p align="center">
  <img src="logo.png" alt="battery-oracle logo" width="360">
</p>

# battery-oracle

[![CI](https://github.com/TRustworthy-AI-Tools-for-Science/battery-oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/TRustworthy-AI-Tools-for-Science/battery-oracle/actions/workflows/ci.yml)
[![Docs](https://github.com/TRustworthy-AI-Tools-for-Science/battery-oracle/actions/workflows/docs.yml/badge.svg)](https://trustworthy-ai-tools-for-science.github.io/battery-oracle/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Built with PyBaMM](https://img.shields.io/badge/built%20with-PyBaMM-orange)](https://www.pybamm.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI](https://img.shields.io/pypi/v/battery-oracle)](https://pypi.org/project/battery-oracle/)

A standalone PyBaMM/SPMe battery oracle. Given a 6-D charge/discharge protocol
it runs a PyBaMM single-particle-with-electrolyte (SPMe) simulation, synthesises
an EIS spectrum, and fits an equivalent-circuit model (ECM). This mimics the same
kind of featurised state a real cell would, allowing the battery oracle to act as a
digital twin.

## Install

```bash
pip install battery-oracle                    # core (PyBaMM + Randles-stub ECM)
pip install "battery-oracle[autoeis]"          # + Bayesian ECM fitting
```

For development (editable install, all extras), managed with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync                          # core (PyBaMM + Randles-stub ECM)
uv sync --extra autoeis          # + Bayesian ECM fitting (AutoEIS)
uv sync --extra drt              # + DRT peaks (hybrid-drt)
```

### Extras

| Extra      | Enables                                                        | Without it                                   |
|------------|---------------------------------------------------------------|----------------------------------------------|
| `autoeis`  | Bayesian ECM fitting via AutoEIS                              | falls back to the fast analytic Randles stub |
| `drt`      | distribution-of-relaxation-times peaks (`hybrid-drt`)         | DRT peaks omitted                            |

## Usage

```python
from battery_oracle import PyBaMMOracle, Protocol

oracle = PyBaMMOracle(degradation_preset="accelerated")  # ready to call immediately

protocol = Protocol(
    charge_current_1_mA=1000.0, charge_current_2_mA=500.0,
    charge_duration_1_h=0.25, charge_duration_2_h=0.25,
    discharge_current_mA=1000.0, discharge_duration_h=1.0,
)
result = oracle.run_cycle(protocol)   # runs SPMe → EIS → ECM
print(result.soh)
```

`oracle.run_cycle(...)` returns a `CycleResult` with named fields (`soh`,
`soc_charge`, `soc_discharge`, `capacity_ah`, `eis_charge`/`eis_discharge`,
`ecm_charge`/`ecm_discharge`, `protocol_applied`, ...) instead of a bare array.
The original array-in/array-out contract (`oracle(protocol) -> np.ndarray`,
also accepting a raw `np.ndarray`/sequence) is unchanged and still supported —
it's what a downstream active-learning/DMDc loop consumes:

```python
u = protocol.to_array()             # 6-D protocol as a plain array
state = oracle(u)                    # -> np.ndarray, unchanged contract
```

The ECM circuit is configurable (`PyBaMMOracle(circuit="R1-[R2,P3]-[R4,P5]", ...)`);
in practice take it from the experiment config (`ecm.circuit`), so the oracle's state
vector matches the featurization of the real experiment. Without one it falls back to
`battery_oracle.DEFAULT_CIRCUIT` (the 7-parameter circuit above). A fully-configured oracle can
also be built directly from a calibration YAML with
`PyBaMMOracle.from_config()` (packaged defaults), `.from_config("calce")`
(a packaged per-dataset calibration), or `.from_config("my_cell.yml")`.

The reduced-order model is selectable with `model="SPMe"` (default), `"SPM"`, or
`"DFN"` — see the [model docs](https://trustworthy-ai-tools-for-science.github.io/battery-oracle/models.html)
and the [numerical-stability notes](https://trustworthy-ai-tools-for-science.github.io/battery-oracle/numerics.html).

## Experiments (from a YAML protocol)

`config_experiment_defaults.yml` defines an experiment protocol (model, cycling,
degradation, EIS, ECM, and one or more 6-D protocols) that the package can execute:

```python
from battery_oracle import run_experiment
history = run_experiment()   # runs the packaged config_experiment_defaults.yml
```

Pass a path (`run_experiment("my_experiment.yml")`) to run a custom protocol file.

## Documentation

Full docs (guides, API reference, and the SPM/SPMe/DFN + numerical-stability
notebooks) are published at
**<https://trustworthy-ai-tools-for-science.github.io/battery-oracle/>**.

## Calibration

Fitting the oracle's degradation scales to a measured cell (the Optuna search and
its plots) lives in `battery-forecast` (`battery_forecast.oracle_tune`); see its
`docs/oracle_calibration.md`.

## Physics scope

Be aware of what the oracle does and doesn't model before trusting it beyond a
demonstration/AL-loop role:

- The shipped degradation presets (`nominal`/`accelerated`/`severe`) are
  calibrated against the **Chen2020 (LG M50, NMC811/graphite) parameter set**
  running **SPMe**. The other supported parameter sets (the LiR2032 coin cells,
  `parameter_sets.SUPPORTED_PARAMETER_SETS`) run with an uncalibrated degradation
  trajectory — pair them with their own calibration YAML rather than trusting the
  default preset's cycle-to-EOL count. Other PyBaMM sets are rejected: they lack
  the SEI parameters the oracle's degradation model needs.
- **Particle cracking / active-material-loss (LAM) submodels are currently
  disabled in every preset.** This works around an upstream PyBaMM regression
  in the particle-mechanics ("swelling and cracking") submodel — see the
  comment in `_build_degradation_config` (`src/battery_oracle/oracle.py`) for
  the full writeup. Capacity fade in the shipped presets comes from SEI growth
  and lithium plating only; mechanical-stress-driven fade is not represented
  (an optional, explicitly-disabled-by-default `dod_lam_scale` term
  approximates depth-of-discharge-driven LAM as a stand-in — see its
  docstring in `PyBaMMOracle.__init__` before enabling it).

## License

MIT
