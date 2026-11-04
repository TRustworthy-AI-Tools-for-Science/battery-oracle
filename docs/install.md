# Installation

Requires Python 3.12 or later and PyBaMM 26.10 (`pybamm>=26.10,<27`; no other
PyBaMM version is supported). The project is managed with
[uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # core + the dev group (pytest, ruff, the [fit] and [notebook] extras)
uv sync --extra autoeis          # + Bayesian ECM fitting (AutoEIS)
uv sync --extra drt              # + tqdm for the DRT peaks (hybrid-drt is installed by hand, below)
uv sync --extra docs             # + this documentation site's build dependencies
```

A bare `uv sync` installs the `dev` dependency group, which pulls in the `fit` and
`notebook` extras. `uv sync` removes anything it does not resolve, so pass every
extra you need on each call.

With pip:

```bash
pip install "battery-oracle @ git+https://github.com/TRustworthy-AI-Tools-for-Science/battery-oracle.git"
pip install "battery-oracle[autoeis] @ git+https://github.com/TRustworthy-AI-Tools-for-Science/battery-oracle.git"
```

## Core dependencies

`numpy`, `scipy`, `pybamm`, `pyyaml` and `impedance` (the linKK check). pandas and
matplotlib are not dependencies of the core package; the `notebook` extra brings
matplotlib.

## Extras

| Extra      | Enables | Without it |
|------------|---------|------------|
| `autoeis`  | Bayesian ECM fitting with AutoEIS ({func}`~battery_oracle.autoeis_ecm`) | the oracle uses the analytic Randles stub and logs a warning; posterior-std slots of the state vector are NaN |
| `drt`      | `tqdm`, used by the DRT peak fit | DRT peaks are empty (a warning is logged each cycle) |
| `fit`      | PyBOP: the stage-A fit ({mod}`battery_oracle.fit`) and the oracle's non-stationary EIS (`eis_mode="nonstationary"`) | `import battery_oracle.fit` raises `ImportError`; a non-stationary spectrum fails and the oracle falls back to the equilibrium spectrum |
| `notebook` | matplotlib, ipykernel, ipywidgets for `notebooks/` | — |
| `docs`     | Sphinx, furo, myst-nb, sphinx-copybutton, sphinxcontrib-bibtex | — |

### DRT libraries

The DRT fit needs `hybrid-drt` (imported as `hybdrt`) and `Mittag-Leffler`, which
are not on PyPI. Install them by hand:

```bash
pip install "hybrid-drt @ git+https://github.com/jdhuang-csm/hybrid-drt.git" \
            "Mittag-Leffler @ git+https://github.com/jdhuang-csm/mittag-leffler.git"
```

### PyBOP version

The PyBOP extensions in {mod}`battery_oracle.fit.pybop_ext` touch PyBOP internals and
are tested against one commit of PyBOP's develop branch (reported version 26.3).
`pyproject.toml` pins that commit under `[tool.uv.sources]`, so `uv sync` installs
it. A pip install of the `fit` extra resolves `pybop` from PyPI instead; install the
pinned commit yourself:

```bash
pip install "pybop @ git+https://github.com/pybop-team/PyBOP.git@25d0b5be0ed3f6abaaf1ec388ae3cea54dbf4c84"
```

`pybop_ext.apply()` checks the installed PyBaMM (`26.10.1.0`) and PyBOP (`26.3.*`)
versions and raises `PatchError` on any other.

## Imports and the environment

Optional dependencies (AutoEIS/JAX, mpire, hybrid-drt, PyBOP) are imported lazily,
so `import battery_oracle` needs only the core dependencies. Importing the package
does not change environment variables or patch third-party libraries; the
AutoEIS/mpire patches are installed on the first AutoEIS fit.

On a CPU-only machine with `jax[cuda12]` installed, set `JAX_PLATFORMS=cpu` before
importing; otherwise JAX fails while probing the CUDA plugin. The package does not
set it.
