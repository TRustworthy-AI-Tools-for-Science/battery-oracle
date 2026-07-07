```{image} _static/logo.png
:alt: battery-oracle
:width: 320px
:align: center
```

# battery-oracle

A standalone **PyBaMM/SPMe battery oracle** — the *simulated* counterpart to a
physical battery (BattMAP) lab interface. Given a 6-D charge/discharge protocol it
runs a PyBaMM single-particle-with-electrolyte (SPMe) simulation, synthesises an EIS
spectrum, and fits an equivalent-circuit model (ECM), returning the same kind of
featurised state a real cell would. It is a drop-in "battery source" for an
active-learning loop, so a simulated cell can be swapped for a real one behind the
same interface.

```{code-block} python
from battery_oracle import PyBaMMOracle, make_pybamm_candidates

oracle = PyBaMMOracle(degradation_preset="accelerated")
oracle.reset()
for protocol in make_pybamm_candidates():   # 6-D protocol grid
    oracle(protocol)                         # runs SPMe → EIS → ECM
    print(oracle._history[-1]["end_soh"])
```

```{toctree}
:maxdepth: 2
:caption: Guides

install
usage
protocol
models
degradation
calibration
numerics
```

```{toctree}
:maxdepth: 1
:caption: Notebooks

notebooks/01_oracle_tuning_calibration
notebooks/02_demonstrations
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
references
```
