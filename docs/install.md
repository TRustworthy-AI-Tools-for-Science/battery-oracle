# Installation

Requires **Python 3.12+**. Managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # core (PyBaMM + Randles-stub ECM)
uv sync --extra autoeis          # + Bayesian ECM fitting (AutoEIS)
uv sync --extra drt              # + DRT peaks (hybrid-drt)
uv sync --extra docs             # + this documentation site's build deps
```

or with pip:

```bash
pip install "battery-oracle[autoeis] @ git+https://github.com/TRustworthy-AI-Tools-for-Science/battery-oracle.git"
```

## Extras

| Extra      | Enables                                                        | Without it                                   |
|------------|---------------------------------------------------------------|----------------------------------------------|
| `autoeis`  | Bayesian ECM fitting via AutoEIS                              | falls back to the fast analytic Randles stub |
| `drt`      | distribution-of-relaxation-times peaks (`hybrid-drt`)         | DRT peaks omitted                            |
| `docs`     | Sphinx + furo + myst-nb to build these docs                   | —                                            |

Heavy optional dependencies (AutoEIS/JAX, Optuna, DRT git packages) are
**lazy-imported** in the source, so `import battery_oracle` needs only the core
dependencies.

On a CPU-only machine with `jax[cuda12]` installed, set `JAX_PLATFORMS=cpu` before
importing; otherwise jax fails while probing the CUDA plugin. The package does not
set it for you.
