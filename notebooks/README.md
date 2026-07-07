# Notebooks

Demonstration notebooks for `battery-oracle`. Both are committed **with outputs
pre-rendered** and are **not** executed in CI or on the docs builder (they run real
PyBaMM / AutoEIS and are slow).

| Notebook | What it shows | Extras needed |
|----------|---------------|---------------|
| [`01_oracle_tuning_calibration.ipynb`](01_oracle_tuning_calibration.ipynb) | AutoEIS EIS + ECM fit, Optuna calibration, Pareto/convergence/mechanism-shift diagnostics | `autoeis`, `tune` |
| [`02_demonstrations.ipynb`](02_demonstrations.ipynb) | Charge-rate sweep, temperature sweep, SPM vs SPMe vs DFN, degradation presets | core only |

Refresh the rendered outputs locally (do **not** run in CI):

```bash
uv run --extra autoeis --extra tune jupyter nbconvert --to notebook --execute --inplace \
    notebooks/01_oracle_tuning_calibration.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_demonstrations.ipynb
```
