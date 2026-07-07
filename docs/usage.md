# Usage

## Quickstart

The oracle is a **stateful** callable: each call runs `n_cycles` charge/discharge
cycles from the cell state left by the previous call, so degradation accumulates.
Call {meth}`~battery_oracle.PyBaMMOracle.reset` between experiments for a fresh cell.

```python
from battery_oracle import PyBaMMOracle, make_pybamm_candidates

oracle = PyBaMMOracle(degradation_preset="accelerated")
oracle.reset()
for protocol in make_pybamm_candidates():   # 6-D protocol grid
    state = oracle(protocol)                 # 18-D ECM state vector
    print(oracle._history[-1]["end_soh"])
```

Each call returns the concatenated **charge + discharge ECM parameter vector** and
appends a rich record to `oracle._history` (SOH, capacity, per-mechanism loss, the
raw EIS spectra `Z_charge_real`/`Z_charge_neg_imag`, the fitted ECM, and more).

## The equivalent-circuit model

The ECM circuit is configurable and its parameter layout is derived from the
circuit string (nothing assumes a fixed element count):

```python
oracle = PyBaMMOracle(circuit="R1-P2-[R3,P4]-[R5,P6]")
```

The default is `battery_oracle.DEFAULT_CIRCUIT`; the canonical calibration circuit is
loaded from the YAML config with {func}`~battery_oracle.load_default_ecm_circuit`.

## ECM fitters

Without the `autoeis` extra, pass the fast analytic stub explicitly:

```python
from battery_oracle import PyBaMMOracle
from battery_oracle.oracle import _randles_stub_ecm

oracle = PyBaMMOracle(ecm_model_fn=_randles_stub_ecm)
```

With the `autoeis` extra, the default `ecm_model_fn` performs a full Bayesian ECM fit.

## Choosing a model

See [Battery models](models.md) — the `model` kwarg selects `"SPMe"` (default),
`"SPM"`, or `"DFN"`.
