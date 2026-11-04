# Notebooks

| Notebook | What it shows | Extras needed |
|----------|---------------|---------------|
| [`02_demonstrations.ipynb`](02_demonstrations.ipynb) | State and validity: building an oracle and running cycles, what carries across calls, `reset()` and the seed, and the physical-validity checks of a solution (`battery_oracle.validity`, `battery_oracle.physicality`) | core + matplotlib (`notebook` extra) |

The notebook runs real DFN solves and is slow. It is committed with its outputs and is
not executed in CI or on the docs builder, which renders the committed outputs.

Refresh the outputs locally (not in CI):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_demonstrations.ipynb
```
