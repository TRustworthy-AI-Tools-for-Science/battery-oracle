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
It is a drop-in battery source for active-learning loops. Calibrating the
degradation physics to a measured cell is done downstream, in `battery-forecast`.

These docs assume familiarity with porous-electrode theory, DAE integration, and
impedance analysis. Three pages carry the load-bearing caveats: the
[model-hierarchy assumptions](models.md), the catalogue of
[divergences from stock PyBaMM](divergences.md), and the
[numerical-stability](numerics.md) failure modes with their empirical
reproductions.

```{code-block} python
import numpy as np
from battery_oracle import PyBaMMOracle

oracle = PyBaMMOracle(degradation_preset="accelerated")
oracle.reset()
for c_mA in (68.0, 100.0, 140.0):          # 6-D protocol: I1, I2 [mA], t1, t2 [h], I_d [mA], t_d [h]
    oracle(np.array([c_mA, c_mA / 2, 1.0, 0.5, 100.0, 1.0]))   # cycling → EIS → ECM
    print(oracle.history[-1]["end_soh"])
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
numerics
```

```{toctree}
:maxdepth: 1
:caption: Notebooks

notebooks/02_demonstrations
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
references
```
