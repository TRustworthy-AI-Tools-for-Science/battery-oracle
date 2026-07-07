# Divergences from stock PyBaMM

`battery_oracle` makes a series of deliberate departures
from out-of-the-box PyBaMM behaviour for the purposes of solving a specific fidelity, stability,
or calibration problem. These departures are catalogued here so results can be interpreted and challenged with full comprehension of what is *model* and what is *harness*.

Code references are to `src/battery_oracle/oracle.py` unless stated.

## 1. Stateful cycling across calls

A stock `pybamm.Simulation` is stateless; the oracle threads
`solve(starting_solution=...)` through successive calls so degradation integrators
(LLI, film thicknesses) accumulate over an entire active-learning campaign.
Consequences:

- **Cycle-level summary variables are accumulated per call.** Each call's steps are
  wrapped as a single experiment *cycle* (a tuple), because a flat step list would
  make PyBaMM treat every step as its own cycle and inflate the LLI integrators.
- **The first call initialises at `initial_soc=0.8`**. Chen2020's default negative
  stoichiometry (0.901) gives an OCV ≈ 4.10 V that instantly violates the charge
  voltage limit and triggers a spurious infeasibility event.
- **State is protected by a lock**. The oracle is sequential by construction.

## 2. Degradation → EIS coupling is post-hoc (the largest divergence)

`pybamm.EISSimulation` linearises a fresh, undegraded model. Stock PyBaMM
provides no path from a degraded cycling solution into an impedance simulation. The
oracle bridges this with two controlled injections per call (`_eis_and_correct`):

1. **Film resistances in series.** X-averaged SEI, crack-SEI, and dead-lithium
   thicknesses are read from the cycling solution and converted to a lumped ohmic
   shift $\Delta R_{\mathrm{ohm}}$ via the Christensen–Newman film form
   {cite:p}`christensen2004` (formulas in
   [Degradation](degradation.md#from-hidden-state-to-observable-impedance)), then
   added to $\mathrm{Re}\,Z$ at all frequencies.
2. **LAM as a parameter override.** Cumulative active-material loss reduces the
   negative-electrode active-material volume fraction in the EIS parameter set
   before linearisation, shrinking $a_s$ and hence growing the charge-transfer arc.

**Required assumptions:** films are purely resistive (no film capacitance, so
no new time constant appears in the spectrum); the degradation state is frozen
during the "measurement"; X-averaged thicknesses suffice (consistent with the
SPMe's uniform-reaction assumption, inconsistent with a DFN's through-thickness
heterogeneity). A fully coupled alternative — linearising the degradation-equipped
model — would capture film capacitance and SOC-dependent film effects, at the cost
of a much stiffer linearisation and no guarantee of well-posedness with the plating
submodel active.

## 3. Particle cracking disabled everywhere

PyBaMM's `"particle mechanics": "swelling and cracking"` fails DAE initialisation
against Chen2020's LGM50 graphite OCP (the stress submodel evaluates
$\partial U/\partial c$ at $t=0$, where the interpolated OCP returns derivatives
inconsistent enough to defeat consistent-initial-condition computation; root cause
isolated by parameter bisection — swapping only the negative OCP reproduces the
failure). Stock behaviour would be to enable it per O'Kane et al.
{cite:p}`okane2022`; the oracle instead re-supplies C-rate sensitivity via plating
kinetics. See [Numerical stability](numerics.md) for the reproduction and
[Degradation](degradation.md) for the interpretive consequences.

## 4. Parameter surgery on Chen2020

The parameter set is modified at construction (`_build_degradation_config`):

| Modification | Mechanism | Rationale |
|---|---|---|
| `EC diffusivity [m2.s-1]` × 0.25 (× `sei_rate_scale`) | direct override | FEC/VC-containing commercial electrolytes form a denser SEI than Chen2020's LP30 baseline {cite:p}`reniers2019`; applied to *all* presets — electrolyte quality is a cell property, not a severity setting |
| Exchange-current densities × `kinetics_scale` | **function wrapper**, not scalar overwrite | Chen2020's $j_0(c_e, c_s^{\mathrm{surf}}, T)$ are callables; wrapping preserves concentration/temperature dependence while rescaling magnitude. Needed because research coin cells show ~10× larger relative charge-transfer arcs than the automotive-grade Chen2020 pouch |
| Plating parameter block borrowed from OKane2022 | copy-if-absent | Chen2020 lacks plating constants entirely; only genuinely missing keys are copied so caller-supplied values win |
| `Initial plated lithium concentration` = 0 | override | OKane2022's formation-cycle plated Li strips during early low-C cycling and drives the plating-loss integrator negative on a fresh cell |
| Plating rate / dead-Li decay constants × calibration scales | override | the two orthogonalised calibration levers (see [Calibration](calibration.md)) |

> Note: scaling `SEI kinetic rate constant [m.s-1]`
(the parameter the `ec reaction limited` submodel reads for kinetics) was
tried as a second SEI lever and reverted. At 0.1× it made 15-cycle R1 growth
worse (236 % vs 82.5 %), i.e. the submodel's parameter sensitivity is not the
naïve monotone relationship. R1-growth-rate calibration remains only partially
solved.

## 5. Protocol semantics and current scaling

Stock PyBaMM accepts whatever experiment string you write. The oracle imposes:

- **Same-C-rate-fraction mapping.** Protocol currents are specified at the *real*
  cell's scale (mA) and mapped onto the 5 Ah simulation cell as
  $I_{\mathrm{sim}} = I_{\mathrm{real}} \cdot Q_{\mathrm{sim}}/Q_{\mathrm{real}}$
  (`real_cell_capacity_mah`). The embedded assumption is that degradation and
  (relative) impedance respond to C-rate, not absolute current; geometric
  differences (coin vs pouch areal current density) are absorbed empirically by
  `kinetics_scale` and by the ECM-fit rescaling (§7).
- **Sanitisation to a validated envelope.** Currents are clamped to
  [50, 10 000] mA·(scaled) — 2C upper bound on the 5 Ah cell — and durations to
  [60 s, 8 h] with 15-min caps per CC charge stage, because SPMe accuracy degrades
  above ≈1C and the DAE solver diverges above ≈2–3C. Out-of-envelope requests are
  silently clipped, not rejected: in an active-learning loop the oracle must
  return an answer for any proposed protocol.
- **A fixed 5-step faithful cycle** (discharge → rest → CC1 → CC2 → rest), after
  Jones, Stimming & Lee {cite:p}`jones2022`, with EIS read at the two post-rest
  states.

## 6. Solver orchestration (stock PyBaMM would silently mislead)

Three behaviours differ from a naïve `sim.solve()` (details and reproductions in
[Numerical stability](numerics.md)):

- **Cross-family fallback**: IDAKLU primary; on failure, retry with
  `CasadiSolver(mode="safe")` — deliberately a different integrator family, since
  the CC→CV failure is tolerance-independent.
- **Silent-truncation guard**: PyBaMM's experiment loop catches internal solver
  errors and returns the partially-integrated solution without raising an error message. The oracle
  compares steps completed against steps requested and raises `OracleFailure`.
- **Periodic native rebuild** in `reset()`: model + solver objects are reconstructed
  per experiment to bound the lifetime of IDAKLU/CasADi native allocations, which
  otherwise accumulate over hundreds of solves and eventually corrupt memory.

## 7. Measurement realism and the ECM fit

Stock PyBaMM produces a noiseless impedance. The oracle layers on: white and/or
$1/f$ flicker noise (amplitude $\propto |Z|/\sqrt{f}$ — the source of realistic
low-frequency scatter; see [protocol.md](protocol.md#eis-noise-and-noisy-low-frequencies)),
and optionally Hallemans-style OCV-relaxation drift {cite:p}`hallemans2023` coupled
to the protocol's rest duration. The AutoEIS Bayesian ECM fit is itself modified.

- **Scale-equivariant rescaling ("Option B").** The 5 Ah simulation cell's
  impedance is ~16× smaller than the ~43 mAh coin cells the downstream featurisation
  was trained on, pushing fitted CPE admittances ($\sim 1/Z$) out of distribution.
  Since AutoEIS is scale-equivariant ($Z \to sZ \Rightarrow R \to sR,\; Q \to Q/s,\;
  n \to n$), spectra are rescaled to a target ohmic resistance before fitting and
  resistances divided back afterwards.
- **Posterior median, not mean** — the series-CPE admittance posterior is
  heavy-tailed; the mean is a chain-unstable ~64× outlier statistic.
- **Prior monkeypatch** on `autoeis.utils.initialize_priors` (excessively wide
  default priors), a Randles-stub analytic fallback on inference failure, and
  process-hygiene patches (`jax.clear_caches()` per fit, `mpire` forced to spawn).

## 8.Settings Consistent with PyBaMM

To bound the list: the discretisations, the SPM/SPMe/DFN equations themselves, the
Butler–Volmer forms, the SEI/plating submodel equations, and all Chen2020 values not
listed in §4 are stock PyBaMM. Anyone auditing a result should therefore focus on
§2 (EIS coupling), §4 (parameter surgery), and §5 (protocol clipping) — these are
where this package's numbers can differ from a hand-rolled PyBaMM script given "the
same" inputs.
