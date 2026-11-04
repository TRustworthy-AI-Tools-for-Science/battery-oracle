```{image} _static/logo.png
:alt: battery-oracle
:width: 320px
:align: center
```

# battery-oracle

A stateful PyBaMM battery *oracle*: the simulated counterpart of a physical cell
behind a cycler and potentiostat. Each call takes a 6-D charge/discharge protocol
and maps it through

$$
\text{protocol} \;\xrightarrow{\text{DFN cycle}}\;
\text{degraded state} \;\xrightarrow{\text{EIS}}\;
Z(\omega) \;\xrightarrow{\text{ECM fit}}\;
s \in \mathbb{R}^{4P\,(+1)},
$$

returning the featurised observable a real cell would give: the fitted
equivalent-circuit parameters of the post-charge and post-discharge spectra and their
posterior standard deviations. The cycling model is PyBaMM's Doyle–Fuller–Newman
(DFN) model {cite:p}`doyle1993,sulzer2021`. Each call is one cycle that continues
from the solution of the previous call, so SEI growth, lithium plating and
dead-lithium accumulation integrate across calls. The oracle is meant as the battery
source of an active-learning loop.

The package has two parts:

- the **oracle** (core install): {class}`~battery_oracle.PyBaMMOracle`, its
  configuration, the experiment-YAML runner, the EIS and ECM steps, and the physical
  checks of a parameterised oracle;
- the **stage-A fit** (`[fit]` extra, needs PyBOP): {mod}`battery_oracle.fit`, which
  fits the beginning-of-life parameters of the DFN to a cell's whole measured history
  and checks that the fitted cell is physically valid.

Calibrating the degradation scales to measured ageing, CSV export, plots and the
Kramers–Kronig analysis pipeline live downstream, in `battery-forecast`.

The pages that carry the modelling caveats are [the model](models.md),
[divergences from stock PyBaMM](divergences.md), [degradation](degradation.md) and
[numerical stability](numerics.md).

```{code-block} python
import numpy as np
from battery_oracle import PyBaMMOracle, randles_stub_ecm

oracle = PyBaMMOracle(degradation_preset="accelerated", ecm_model_fn=randles_stub_ecm)
for c_mA in (68.0, 100.0, 140.0):   # [C_rate_1, C_rate_2 mA, dur_1, dur_2 h, D_rate mA, dur_d h]
    result = oracle.run_cycle(np.array([c_mA, c_mA / 2, 1.0, 0.5, 100.0, 1.0]))
    print(result.soh, result.block("means_charge"))
```

```{toctree}
:maxdepth: 2
:caption: Guides

install
usage
protocol
models
degradation
divergences
numerics
validity
fitting
```

```{toctree}
:maxdepth: 1
:caption: Notebooks

notebooks/02_demonstrations
```

The notebook is rendered from its committed outputs; the docs builder does not
execute it.

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
references
```
