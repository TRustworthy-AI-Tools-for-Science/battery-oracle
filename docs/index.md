```{image} _static/logo.png
:alt: battery-oracle
:width: 320px
:align: center
```

# battery-oracle

A stateful PyBaMM-based battery *oracle*: the simulated counterpart to a physical
cell behind a lab (BattMAP) interface. Per query it maps a 6-D charge/discharge
protocol through the pipeline

$$
\text{protocol} \;\xrightarrow{\text{SPM/SPMe/DFN cycling}}\;
\text{degraded state} \;\xrightarrow{\text{linearised EIS + film corrections}}\;
Z(\omega) \;\xrightarrow{\text{Bayesian ECM fit}}\;
s \in \mathbb{R}^{2P},
$$

returning the same featurised observable a real cell would, with degradation
(SEI growth, lithium plating, dead-lithium accumulation) integrated across calls.
It is a drop-in battery source for active-learning loops, and ships an Optuna
calibration engine (single-objective TPE and a multi-objective NSGA-II workflow)
for fitting the degradation physics to a measured cell.

These docs assume familiarity with porous-electrode theory, DAE integration, and
impedance analysis. Three pages carry the load-bearing caveats: the
[model-hierarchy assumptions](models.md), the catalogue of
[divergences from stock PyBaMM](divergences.md), and the
[numerical-stability](numerics.md) failure modes with their empirical
reproductions.

```{code-block} python
from battery_oracle import PyBaMMOracle, make_pybamm_candidates

oracle = PyBaMMOracle(degradation_preset="accelerated")
oracle.reset()
for protocol in make_pybamm_candidates():   # 6-D protocol grid
    oracle(protocol)                         # cycling → EIS → ECM
    print(oracle._history[-1]["end_soh"])
```

```{toctree}
:maxdepth: 2
:caption: Guides

install
usage
protocol
models
divergences
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
