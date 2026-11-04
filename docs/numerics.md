# Numerical stability

The DFN discretised by the method of lines is a stiff, semi-explicit index-1
differential-algebraic system

$$
M\,\dot{y} = f(y, t), \qquad 0 = g(y, t),
$$

where the algebraic block $g$ holds the potentials and the through-cell current
balance (unless `surface form: differential` turns the interfacial balance into a
double-layer ODE) and the experiment's step-termination events. The oracle runs many
such solves per campaign, with accumulated degradation state, and can run a JAX
Bayesian fitter in the same process. Each section below gives one failure mode, its
mechanism, and what the code does about it. Whether a solution that completes is
physically valid is a separate question, answered in [Physical validity](validity.md).

```{note}
Reproducing the failures depends on the PyBaMM, SUNDIALS and JAX builds. Statements
below that cite a PyBaMM version are what the code records for that version.
```

## Solvers

SUNDIALS IDA {cite:p}`hindmarsh2005` integrates the DAE with variable-order BDF and
a modified Newton iteration, and needs consistent initial conditions
$(y_0, \dot y_0)$ at the start of every experiment step (`IDACalcIC`). The oracle
builds two solvers ({meth}`~battery_oracle.PyBaMMOracle.reset` rebuilds both):

| Solver | Settings (YAML `solver`) | Role |
|---|---|---|
| `IDAKLUSolver` | `dfn.rtol` $10^{-6}$, `dfn.atol` $10^{-8}$ | primary |
| `CasadiSolver(mode="safe")` {cite:p}`andersson2019` | `emergency.rtol` $10^{-2}$, `emergency.atol` $10^{-5}$, `dfn.dt_max_s` 1 s | retry after any primary failure |

A switch of control mode at a voltage limit (constant current to constant voltage)
swaps the active algebraic constraint discontinuously: the BDF history is invalid
across the switch and the new algebraic Jacobian is near-singular at the limit. The
signature is `IDA_ERR_FAIL`, repeated error-test failures driving the step to its
minimum. The code records that this failure persisted at rtol $10^{-2}$, $10^{-3}$
and $10^{-6}$: a structural initialisation problem, not a precision one. The retry
therefore uses a different integrator family, not looser IDA tolerances: the safe-mode
CasADi solver advances in short windows (`dt_max`) with event checks and
re-initialisation after each, and crosses control-mode boundaries that defeat a single
`IDACalcIC`. The oracle's cycle is two-stage CC with no CV hold; the C/20 capacity
check (`capacity_check`) has one.

A call that needed the retry is returned normally, with
`failure_kind = FailureKind.SOLVER_DEGRADED`. If the retry also fails the call raises
{class}`~battery_oracle.OracleFailure` (`SOLVER_FAILURE`, or `VOLTAGE_INFEASIBLE`
when the error mentions the voltage). The model is never replaced by a simpler one.

## The C-rate ceiling

Stiffness and the cost of consistent initialisation grow with the applied current.
The oracle clips protocol currents to `dfn_max_crate` (1.5C) of the simulated cell's
nominal capacity, in addition to `c_max_mA` ([protocol](protocol.md)). This is a
harness decision: clipping keeps an active-learning loop alive at the cost of a flat
response beyond the bound. A cell whose measured protocols exceed 1.5C (the jones2022
coin cells are cycled at up to ~3.8C) needs `protocol_bounds.dfn_max_crate` raised.

## Silent truncation

PyBaMM's experiment loop catches solver errors on its own callback path and returns
what was integrated, without raising or warning. A cycle missing its last steps is
indistinguishable from success by return type. The oracle reads SOC, film state and
the EIS start from fixed step indices of the last cycle, so a truncated solution would
corrupt every downstream quantity. The guard (`_is_truncated`) compares
`len(cycle.steps)` of every cycle the call added with
`experiment.cycle_lengths[-1]`. A shortfall on the primary solver triggers the retry;
a shortfall on the retry raises `OracleFailure` with `SOLVER_TRUNCATION`.

## Non-stationary EIS solves

The double-layer model of the non-stationary EIS ([the model](models.md)) starts at
the beginning of a rest, where the current has just stopped. It is solved with
`IDAKLUSolver(rtol=1e-6, atol=1e-6)` single-threaded, then, on a `SolverError`, with
the algebraic error test suppressed. If both fail, the first 30 s of the rest are
integrated with the cycling model (no double layer; it has already solved this rest)
and the double-layer model starts from there. The double layer relaxes in
milliseconds and the first EIS window opens after the rest, so the measured spectrum
does not change. Any remaining failure is caught: the oracle uses the equilibrium
spectrum for that rest and records it in `eis_mode_used`.

## Particle cracking with Chen2020

With `"particle mechanics": "swelling and cracking"` {cite:p}`ai2020` and the
Chen2020 parameter set, the code records a failure at $t=0$ with
`IDAGetDky: IDA_BAD_K` on PyBaMM 26.6.2.0. Mechanism: the cracking stress submodel
evaluates $\partial U/\partial c$ at the initial stoichiometry, and the interpolated
LG M50 graphite OCP gives derivatives inconsistent enough that `IDACalcIC` cannot
find consistent initial conditions. Parameter bisection isolated it: replacing only
`Negative electrode OCP [V]` with Chen2020's function in an otherwise-Ai2020 set
reproduces the failure. Cracking is disabled in every preset; see
[Degradation](degradation.md) for the consequences. The code also records that
lowering the critical stress $\sigma_c$ does not create a C-rate threshold (the LAM
rate $\beta(\sigma_h/\sigma_c)^m$ is a smooth power law, so $\sigma_c$ cancels from
the C-rate sensitivity ratio) while it amplifies the Jacobian entry
$\partial(\text{rate})/\partial\sigma \propto \sigma_c^{-2}$, which caused IDA
convergence failures.

## AutoEIS and JAX: unbounded compilation cache

Each NumPyro {cite:p}`phan2019` inference JIT-compiles new XLA executables
{cite:p}`jax2018`, and the process-wide cache keeps them. Over a long campaign the
accumulated executables exhaust the address space of the execution engine and the
process dies with an allocation failure inside native code, which Python cannot
catch. {func}`~battery_oracle.autoeis_ecm` calls `jax.clear_caches()` after every
inference, trading recompilation time for bounded memory.

## Process hygiene: fork, spawn and native lifetimes

- **mpire fork inheritance.** AutoEIS parallelises with mpire, whose Linux default
  start method is `fork`. A forked child inherits the parent's SUNDIALS contexts,
  CasADi JIT code pages and held native mutexes at the same virtual addresses; writes
  after copy-on-write can segfault the child and corrupt shared state. On the first
  AutoEIS fit the package patches `mpire.WorkerPool` to `start_method="spawn"`.
- **IDAKLU/CasADi accumulation.** PyBaMM's discretisation attaches processed symbols
  to model objects across builds, and IDAKLU does not fully release earlier compiled
  CasADi functions when rebound. Over hundreds of solves in one process this can
  corrupt memory and segfault inside the solver with no Python traceback. Every call
  builds a fresh `pybamm.Simulation`; `reset()` also rebuilds the model and both
  solvers, which bounds native lifetimes to one campaign.
- **Threads.** NumPyro's parameter store and tracing state are not thread-safe, so
  AutoEIS inference is serialised by a module-level lock. The oracle's own lock covers
  the PyBaMM solve only.
- **Forked fit workers.** The stage-A fit evaluates candidates in forked processes.
  Forking after PyBaMM's multi-threaded (OpenMP) IDAKLU has run deadlocks the workers,
  so the fit's solvers are single-threaded
  ({func}`battery_oracle.fit.pybop_ext.single_thread_solver`), and each candidate runs
  under a hard time limit ([Fitting](fitting.md)).

## Discretisation patch

`PyBaMMOracle()` applies the `harmonic_flux_coefficients` patch of
{mod}`battery_oracle.patches` (non-strict: on a PyBaMM other than 26.10.1.0 it warns
and stock averaging is used). PyBaMM's finite volumes put a node coefficient on cell
edges with the harmonic mean only when it multiplies a bare gradient; the DFN
electrolyte current multiplies its effective conductivity by a combination of
gradients, $(\kappa\,\mathrm{tor})(\chi\nabla c_e - \nabla\phi_e)$, which is averaged
arithmetically at the electrode|separator interfaces where $\varepsilon^b$ jumps. The
patch uses the harmonic mean for a coefficient that multiplies any combination of
gradient terms, as upstream PyBaMM PR #5856 does. The stage-A chain model (double
layer on) has no such term.
