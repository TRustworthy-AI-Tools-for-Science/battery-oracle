# Numerical stability

PyBaMM battery simulations are stiff DAEs, and the oracle drives them across many
protocols, degradation mechanisms, and models (SPM/SPMe/DFN). A number of the
oracle's design choices exist to keep these solves and the downstream Bayesian ECM
fit from failing or corrupting memory. This page describes each instability and the
workaround, and shows the failure vs. the fix using data produced by
`bin/generate_numerics_data.py` (written to `docs/_static/numerics/`).

```{note}
Reproducibility of the *failures* is version- and platform-dependent (PyBaMM,
SUNDIALS, JAX). The generator records the **actual** outcome on the machine it runs
on, so the committed figures reflect this environment; the workaround rationale holds
regardless.
```

## Model choice and the C-rate envelope

SPM (no electrolyte) is the least stiff and fastest; SPMe {cite:p}`sulzer2021` is the
default; DFN {cite:p}`doyle1993` is the full model — most accurate, slowest, and
stiffest. SPMe accuracy degrades above ~1C and the DAE solver diverges above ~2–3C,
which is why the oracle clamps protocol currents to a validated envelope
(`oracle.py`, `_C_MIN_mA … _C_MAX_mA`). See [Battery models](models.md).

## CC→CV control-mode switch: solver family

The algebraic constraint at a constant-current → constant-voltage boundary
reproducibly trips `IDA_ERR_FAIL` in the IDAKLU solver {cite:p}`hindmarsh2005`
irrespective of tolerance. The oracle therefore keeps an emergency solver from a
**different family** — `CasadiSolver(mode="safe")` {cite:p}`andersson2019` — which
integrates through the boundary, and retries with it on any primary-solver failure.

```{figure} _static/numerics/solver_family.png
:width: 90%

The same CC→CV experiment solved with each solver family (data:
`bin/generate_numerics_data.py --only solver_family`).
```

## Silent truncation

On an internal solver error mid-experiment, PyBaMM catches it on its own callback
path and returns whatever it managed to integrate **without raising** — so a
truncated cycle looks "successful". The oracle guards against this by comparing the
number of completed steps to the number requested and raising
{class}`~battery_oracle.OracleFailure` when they differ.

```{figure} _static/numerics/silent_truncation.png
:width: 70%

Requested vs. completed steps in the final cycle
(`--only silent_truncation`).
```

## Particle cracking + Chen2020: initialization failure

Enabling `"particle mechanics": "swelling and cracking"` on the Chen2020
{cite:p}`chen2020` parameter set fails DAE initialization (an `IDA_BAD_K` / missing
crack-parameter error, depending on version), because the cracking stress submodel
{cite:p}`ai2020` evaluates `dOCP/dc` at initial conditions where the Chen2020 graphite
OCP is inconsistent. Particle cracking is therefore **disabled in all presets**;
C-rate sensitivity is provided by lithium plating {cite:p}`okane2022` instead. The
generator confirms cracking fails while the no-cracking configuration completes
(`numerics_data.json`, `cracking_regression`).

## AutoEIS / JAX: XLA compilation cache growth

Each NumPyro {cite:p}`phan2019` inference JIT-compiles fresh XLA kernels
{cite:p}`jax2018` that are not released, so repeated ECM fits grow the process's
memory until it segfaults (~40 inferences). The oracle calls `jax.clear_caches()` in
a `finally` after every fit, trading ~0.5 s recompilation for bounded memory. The
generator (with the `autoeis` extra) plots max-RSS vs. inference count with and
without the workaround (`--only jax_cache`).

## mpire fork vs. spawn

AutoEIS uses `mpire`, whose default `fork` start method copies the parent's
SUNDIALS/CasADi JIT state into workers at the same virtual addresses; touching it
post-fork can segfault and corrupt shared memory back into the parent. The oracle
patches `mpire.WorkerPool` to `start_method="spawn"` at import. This corruption is
nondeterministic, so the generator documents it rather than asserting a crash
(`--only mpire_fork`).

## Regenerating the data

```bash
uv run --extra autoeis python bin/generate_numerics_data.py --all
# core-only (skips the autoeis-dependent demos):
uv run python bin/generate_numerics_data.py --all --skip-crashy
```

See the [references](references.md) for the papers behind these methods.
