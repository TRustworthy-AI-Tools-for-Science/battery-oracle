# Notebooks

Demonstration notebook for `battery-oracle`. It is committed **with outputs
pre-rendered** and is **not** executed in CI or on the docs builder (it runs real
PyBaMM and is slow).

| Notebook | What it shows | Extras needed |
|----------|---------------|---------------|
| [`02_demonstrations.ipynb`](02_demonstrations.ipynb) | Charge-rate sweep, temperature sweep, SPM vs SPMe vs DFN, degradation presets | core only |

Refresh the rendered outputs locally (do **not** run in CI):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_demonstrations.ipynb
```
