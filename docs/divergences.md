# Divergences from stock PyBaMM

`battery_oracle` departs from out-of-the-box PyBaMM in deliberate ways, each to solve
a fidelity, stability or calibration problem. They are listed here so a result can be
read with a clear view of what is *model* and what is *harness*. Code references are
to `src/battery_oracle/oracle.py` unless stated.

## 1. Stateful cycling across calls

A stock `pybamm.Simulation` is stateless. The oracle passes the previous call's
solution as `solve(starting_solution=...)`, so the cell state and the degradation
integrators (lithium loss, film thicknesses) accumulate over a whole campaign.

- **One call is one PyBaMM cycle.** The five steps are wrapped in a tuple. A flat step
  list would make every step a cycle and inflate PyBaMM's cycle-level summaries.
- **The first call starts at `initial_soc=0.8`.** Chen2020's default negative
  stoichiometry gives an open-circuit voltage close to the charge limit, which would
  stop the first charge at once.
- **No rollback.** A failed call leaves the cell as the solver left it.
- **The SOC reported** is the average negative stoichiometry at the end of a rest over
  the parameter set's initial negative stoichiometry, clipped to [0.05, 0.99].

## 2. Degradation reaches the impedance through corrections

`pybamm.EISSimulation` linearises a model at a relaxed state; stock PyBaMM has no path
from a degraded cycling solution into it.

- **Equilibrium EIS** (default) linearises a fresh DFN with only
  `surface form: differential`: none of the cycling options (SEI, plating, contact
  resistance, thermal) and PyBaMM's default mesh. Degradation enters by
  (1) a film resistance in series, from the x-averaged SEI and dead-lithium
  thicknesses {cite:p}`christensen2004` ([Degradation](degradation.md)), added to
  $\mathrm{Re}\,Z$ at every frequency; (2) the loss of active material, as a reduced
  negative active-material fraction before linearisation; (3) with lumped thermal, an
  Arrhenius scaling of the spectrum to the cell temperature ([the model](models.md)).
  Assumptions: the films are purely resistive (no film capacitance, so no new time
  constant); the state is frozen during the measurement; x-averaged thicknesses
  suffice, which ignores the through-thickness heterogeneity the DFN resolves.
- **Non-stationary EIS** (`eis_mode="nonstationary"`) linearises the cycling model
  (its options, the double layer, the deployed mesh) at the simulated state of each
  frequency's window during the sweep after the rest, and adds the lock-in drift
  leakage {cite:p}`hallemans2023`. Only the dead-lithium resistance is added
  afterwards.

## 3. Particle cracking disabled everywhere

O'Kane et al. {cite:p}`okane2022` couple particle cracking, SEI on crack faces and
active-material loss. The oracle turns particle mechanics off in every preset because
the cracking stress submodel fails DAE initialisation with the Chen2020 graphite OCP
(recorded on PyBaMM 26.6.2.0; see [Numerical stability](numerics.md)). C-rate
sensitivity comes from plating kinetics instead ([Degradation](degradation.md)).

## 4. Parameter changes

The parameter set is modified at construction (`degradation._build_degradation_config`,
then `params.apply_parameter_scales` / `apply_parameter_overrides`):

| Modification | Mechanism | Rationale |
|---|---|---|
| `EC diffusivity [m2.s-1]` × 0.25 × `sei_rate_scale` | direct override | commercial electrolytes with film-forming additives form a denser SEI than the LP30 baseline {cite:p}`reniers2019`; applied in every preset |
| both exchange-current densities × `kinetics_scale` | function wrapper | the $j_0(c_e, c_s^{\mathrm{surf}}, c_{s,\max}, T)$ are callables; wrapping keeps their dependence and rescales the magnitude |
| plating parameters from OKane2022 | copied only where absent | Chen2020 has none; values already in the set win |
| `Initial plated lithium concentration [mol.m-3]` = 0 | override | a fresh cell; OKane2022's formation plating would strip and drive the plating loss negative |
| plating rate constant and dead-lithium decay constant × their scales | override | the two plating calibration scales |
| `bol.parameter_scales`, `bol.parameter_overrides` | multiplier / absolute value, after the preset | beginning-of-life values of a stage-A fit; scales compose with `kinetics_scale` |
| `Contact resistance [Ohm]` | added with the `contact resistance` option | `bol.contact_resistance_ohm` |
| `Ambient temperature [K]` | override when not 298.15 K, or from `T_ambient_K` with lumped thermal | ambient setting |

`LiR2032` is a custom set built on Ai2020 ([the model](models.md)).

## 5. Protocol semantics and current scaling

Stock PyBaMM runs whatever experiment string it is given. The oracle imposes:

- **Same-C-rate mapping.** Protocol currents are at the real cell's scale and are
  mapped onto the simulated cell as
  $I_{\mathrm{sim}} = I_{\mathrm{real}}\,Q_{\mathrm{nom}}/Q_{\mathrm{real}}$
  (`real_cell_capacity_mah`). This assumes degradation and relative impedance respond
  to C-rate, not to absolute current.
- **A bounded envelope.** Currents are clipped to C-rate bounds of the simulated cell
  (`c_*_crate`, default 0.01C to 1.5C), durations to [60 s, 8 h], each CC
  charge stage to 900 s, and (with `discharge_mode: cutoff`) the discharge is given at
  least 1 h so it reaches the cut-off. Requests outside are clipped, not rejected, so
  an active-learning loop always gets an answer.
- **A fixed 5-step cycle** (discharge, rest, CC1, CC2, rest) after Jones, Stimming &
  Lee {cite:p}`jones2022`, with EIS at the two rests.

## 6. Solver orchestration

Three behaviours differ from a plain `sim.solve()` ([Numerical stability](numerics.md)):

- **Cross-family retry.** IDAKLU (rtol $10^{-6}$, atol $10^{-8}$) first; on any
  failure, `CasadiSolver(mode="safe")` on the same model and cell state.
- **Silent-truncation guard.** PyBaMM's experiment loop can return a partial solution
  without raising. The oracle compares the steps completed with the steps requested
  and treats a shortfall as a failure.
- **Native rebuild in `reset()`.** The model and both solvers are rebuilt per campaign
  to bound the lifetime of IDAKLU/CasADi native allocations.

## 7. Measurement realism and the ECM fit

Stock PyBaMM gives a noiseless impedance. The oracle adds white and/or $1/f$ flicker
noise ([protocol](protocol.md)) and, in non-stationary mode, the drift leakage. The
AutoEIS fit {cite:p}`omurabaki2023autoeis` is modified ({mod}`battery_oracle.ecm`):

- **Scale-equivariant rescaling.** AutoEIS is scale-equivariant
  ($Z \to sZ \Rightarrow R \to sR,\; Q \to Q/s,\; n \to n$). The spectrum is rescaled
  so its high-frequency real part equals `ecm_rescale_target_r0` (0.1334 Ohm) before
  fitting, so CPE admittances land in the range of the downstream featurisation; the
  resistor samples are divided back afterwards, the admittances and exponents are
  reported at the rescaled scale. `null` disables it.
- **Posterior median, not mean.** The series-CPE admittance posterior is heavy-tailed.
- **Narrower priors.** `autoeis.utils.initialize_priors` is replaced: exponents
  Uniform(0, 1), every other parameter LogNormal around its initial guess with
  $\pm 3\sigma$ spanning a factor of 100.
- **Initial guesses** from the circuit: the ohmic resistor from the high-frequency
  real part, arc resistances 0.6/0.4 of the remaining low-frequency resistance, CPE
  seeds from `ecm.cpe_seeds`.
- **Fallback.** An inference error falls back to the Randles stub for that spectrum
  (posterior std NaN, `converged = False`).
- **Process hygiene.** `jax.clear_caches()` after every fit; mpire worker pools
  forced to `start_method="spawn"`; inference serialised by a lock.

## 8. What is stock PyBaMM

The DFN equations, the SEI and plating submodel equations, the Butler–Volmer forms,
`EISSimulation`, the parameter values not listed in §4 and, outside the patch listed
in [Numerical stability](numerics.md), the discretisation are stock PyBaMM. When
auditing a result, look first at §2 (EIS coupling), §4 (parameter changes) and §5
(protocol clipping): these are where the package's numbers differ from a hand-written
PyBaMM script given the same inputs.
