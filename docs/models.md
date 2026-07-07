# Battery models: SPM, SPMe, DFN

The `model` kwarg selects the reduced-order PyBaMM model used for **both** the
cycling simulation and the internal EIS simulation:

```python
from battery_oracle import PyBaMMOracle
oracle = PyBaMMOracle(model="SPMe")   # "SPM" | "SPMe" (default) | "DFN"
```

| Model  | Electrolyte | Accuracy | Cost / stiffness | Notes |
|--------|-------------|----------|------------------|-------|
| `SPM`  | none        | lowest   | fastest, least stiff | no electrolyte overpotential |
| `SPMe` | reduced     | medium   | balanced (default) | degradation presets are calibrated here |
| `DFN`  | full        | highest  | slowest, stiffest  | full Doyle–Fuller–Newman {cite}`doyle1993` |

**SPM** (single particle) omits electrolyte transport, so it misses the electrolyte
overpotential — visible as a gap in the terminal-voltage trace versus SPMe/DFN.
**SPMe** adds a reduced electrolyte model and is the default. **DFN** resolves the
full electrode/electrolyte physics {cite}`doyle1993`; it is the most accurate but the
slowest and most numerically demanding (see [Numerical stability](numerics.md)).

Because the degradation presets are calibrated against SPMe, the same preset can
reach end-of-life on a different cycle count under SPM or DFN. The
[demonstrations notebook](notebooks/02_demonstrations) overlays all three on one
protocol.

```{note}
The Chen2020 parameter set works with all three models. DFN is considerably slower,
so the demonstration runs it on a single short protocol.
```
