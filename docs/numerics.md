# Numerical stability

Discretising any of the [model hierarchy](models.md) by the method of lines yields a
stiff, semi-explicit index-1 differential-algebraic system

$$
M\,\dot{y} = f(y, t), \qquad 0 = g(y, t),
$$

where the algebraic block $g$ carries the through-cell current balance (unless the
`surface form: differential` option converts it to a double-layer ODE), terminal
voltage under CV control, and event functions for step termination. The oracle
drives hundreds of such solves per campaign, across degradation option sets and
accumulated state, and layers a JAX/NumPyro Bayesian fitter on top. Every subsection
below documents one failure mode, its mechanism at the solver level, the workaround
implemented, and — where the failure is deterministically reproducible — an
empirical with/without demonstration produced by `bin/generate_numerics_data.py`
(records in `docs/_static/numerics/numerics_data.json`).

```{note}
Reproducibility of the *failures* is version- and platform-dependent (PyBaMM,
SUNDIALS build, JAX). The generator records the actual outcome on the machine it
runs on; the committed records reflect this environment. Notably, the minimal CC→CV
reproduction below completes under both solver families on this PyBaMM/SUNDIALS
build — the failure that motivated the workaround manifests under the full
degradation option set with threaded `starting_solution` state, where consistent
re-initialisation is materially harder.
```

## Model choice and the C-rate envelope

Stiffness and DAE dimension grow down the hierarchy SPM → SPMe → DFN
{cite:p}`doyle1993,marquis2019`. On the Chen2020 parameterisation, SPMe accuracy
degrades above ≈1C (the first-order electrolyte correction leaves its asymptotic
regime) and the DAE solver diverges above ≈2–3C. The oracle therefore clamps
protocol currents to a validated envelope rather than solving unvalidated inputs
(see [protocol semantics](protocol.md)).
This is a harness decision: silent clipping keeps an active-learning loop alive at
the cost of a flat response beyond the clamp boundary.

## CC→CV control-mode switches: cross-family solver fallback

SUNDIALS IDA {cite:p}`hindmarsh2005` integrates the DAE with variable-order (1–5)
BDF and a modified Newton iteration, and requires consistent initial conditions
$(y_0, \dot y_0)$ satisfying both $f$ and $g$ (computed via `IDACalcIC`) at the
start of every experiment step. A constant-current → constant-voltage transition
swaps the active algebraic constraint discontinuously (from $I = I_{\mathrm{app}}$
to $V = V_{\mathrm{hold}}$): the BDF history is invalid across the switch and the
new algebraic Jacobian is near-singular precisely at the voltage limit. The
signature is `IDA_ERR_FAIL` — repeated local error-test failures driving
$h \to h_{\min}$ — and it is tolerance-independent: it was verified to persist
at rtol $10^{-2}$, $10^{-3}$, and $10^{-6}$, which is diagnostic of a structural
initialisation problem rather than a precision problem.

The implemented mitigation is a fallback to a different integrator family, not a
tolerance retry: `CasadiSolver(mode="safe")` advances in short fixed windows
(`dt_max`) with event checks and re-initialisation after each window, so it
traverses control-mode boundaries that defeat IDA's single consistent-IC
computation. IDAKLU remains the primary (it is substantially faster on the smooth
interior of a step); the emergency path pays ~2–5× per solve but only on failures.

```{figure} _static/numerics/solver_family.png
:width: 90%

The same CC→CV experiment under each family (`--only solver_family`). On this build
both complete — see the note above; the recorded solve times show the IDAKLU/CasADi
cost ordering that motivates keeping IDAKLU primary.
```

## Silent truncation: partial solutions returned as success

PyBaMM's experiment loop catches internal `SolverError`s on its own callback path
and returns whatever was integrated without raising an error or warning. A cycle whose CV hold and
final rest were dropped is indistinguishable from success by return type. Because
the oracle reads SOC and film state from specific step indices of the last cycle,
consuming a truncated solution would silently corrupt every downstream quantity
(EIS SOC, degradation integrators, SOH). The guard is structural:
`len(cycle.steps)` is compared against `experiment.cycle_lengths[-1]` for every
cycle a call adds, and a shortfall raises
{class}`~battery_oracle.OracleFailure` — on both the primary *and* emergency paths.

```{figure} _static/numerics/silent_truncation.png
:width: 70%

Requested vs completed step count in the final cycle (`--only silent_truncation`) —
the quantity the guard compares.
```

## Particle cracking + Chen2020: initialization failure

Enabling `"particle mechanics": "swelling and cracking"` {cite:p}`ai2020` with the
Chen2020 parameter set fails at $t=0$ with `IDAGetDky: IDA_BAD_K` (or, on newer
versions, a missing-crack-parameter error before that point). Mechanism: the
cracking stress submodel evaluates $\partial U/\partial c$ at the initial
stoichiometry; Chen2020's interpolated LGM50 graphite OCP returns derivative values
inconsistent enough that `IDACalcIC` cannot produce consistent $(y_0,\dot y_0)$,
and the subsequent derivative-order query fails. The root cause was isolated by
parameter bisection: replacing only `Negative electrode OCP [V]` with Chen2020's
function in an otherwise-Ai2020 set reproduces the failure.

Workaround: cracking is disabled in all presets and C-rate sensitivity is
re-supplied through plating kinetics — an explicit mechanism substitution whose
interpretive consequences are discussed in [Degradation](degradation.md)
("Mechanism 3 — particle cracking / LAM: deliberately disabled").
A historical dead end is documented in the code for future maintainers: lowering the
critical stress $\sigma_c$ does not create a C-rate threshold (the LAM rate
$\beta(\sigma_h/\sigma_c)^m$ is a smooth power law — $\sigma_c$ cancels from the
C-rate sensitivity *ratio*) while amplifying the Jacobian entry
$\partial(\text{rate})/\partial\sigma \propto \sigma_c^{-2}$ 16-fold, which was
itself causing IDA convergence failures.

## AutoEIS / JAX: unbounded XLA compilation cache

Each NumPyro {cite:p}`phan2019` inference JIT-compiles fresh XLA executables
{cite:p}`jax2018`; the process-wide compilation cache retains them, and in a
long-running oracle campaign (~40 inferences) the accumulated executables exhaust
the C++ execution engine's address space — terminating the process with an
allocation failure *inside native code*, unrecoverable from Python. The fit wrapper
therefore calls `jax.clear_caches()` in a `finally` after every inference,
trading ~0.5 s of recompilation per call for bounded memory. With the `[autoeis]`
extra installed, `--only jax_cache` measures max-RSS vs inference count with and
without the workaround (the crash-prone arm runs in a subprocess and is recorded by
exit code if it dies).

## Process hygiene: fork vs spawn, and native-object lifetimes

Two related lifetime problems with native state:

- **`mpire` fork inheritance.** AutoEIS parallelises with `mpire`, whose Linux
  default start method is `fork`. A forked child inherits the parent's address
  space — including SUNDIALS contexts, CasADi JIT-compiled code pages, and any held
  native mutexes — at identical virtual addresses. First write triggers
  copy-on-write against inconsistent native state; the child segfaults, and shared
  memory channels can propagate corruption *back into the parent*.
  `battery_oracle` monkeypatches `mpire.WorkerPool` to `start_method="spawn"` at
  import time (before AutoEIS can import mpire), paying interpreter-restart cost
  for clean state. This is nondeterministic to reproduce, so `--only mpire_fork`
  documents rather than asserts it.
- **IDAKLU/CasADi accumulation.** PyBaMM's discretisation attaches processed
  symbols to model objects across rebuilds, and IDAKLU's workspace does not fully
  release prior compiled CasADi function handles on rebinding. Over hundreds of
  solves in one process these accumulate and eventually corrupt memory —
  segfaulting inside `idaklu_solver` with no Python traceback. The oracle rebuilds
  model + both solvers in `reset()` (once per experiment/policy), bounding native
  lifetimes to a single campaign segment.

## Regenerating the evidence

```bash
uv run --extra autoeis python bin/generate_numerics_data.py --all
uv run python bin/generate_numerics_data.py --all --skip-crashy   # core-only
```

Crash-prone demos run in isolated subprocesses with incremental checkpointing, so a
worker segfault yields a recorded exit code and partial data rather than aborting
the run. See [references](references.md) for the solver and tooling literature.
