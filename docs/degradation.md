# Degradation: mechanisms, presets and their observable signatures

`degradation_preset` selects which ageing submodels the DFN carries and with which
constants ({func}`battery_oracle.degradation._build_degradation_config`). This page
gives each mechanism as implemented, the preset definitions, how the hidden
degradation state reaches the impedance, and how state of health is computed.

## Mechanism 1: SEI growth (`"SEI": "ec reaction limited"`)

Ethylene carbonate (EC) is reduced at the inner SEI interface after diffusing
through the film. With $c_{\mathrm{EC}}^{s}$ the EC concentration at that interface,
transport and reaction balance:

$$
\underbrace{\frac{D_{\mathrm{EC}}\,(c_{\mathrm{EC}}^{0}-c_{\mathrm{EC}}^{s})}{L_{\mathrm{SEI}}}}_{\text{transport}}
=
\underbrace{k_{\mathrm{SEI}}\, c_{\mathrm{EC}}^{s}\,
\exp\!\left(-\frac{F\,\eta_{\mathrm{SEI}}}{2RT}\right)}_{\text{reaction}},
\qquad
\frac{\mathrm{d}L_{\mathrm{SEI}}}{\mathrm{d}t}
  = -\frac{\bar V_{\mathrm{SEI}}}{2F}\, j_{\mathrm{SEI}},
$$

a mixed-control law {cite:p}`reniers2019,pinson2013`. Two properties matter for the
calibration scales:

1. **Overpotential sensitivity.** $j_{\mathrm{SEI}}$ is exponential in
   $\eta_{\mathrm{SEI}}$, which contains the negative-electrode overpotential.
   Anything that raises it at fixed current accelerates SEI growth, including
   `kinetics_scale < 1`, which lowers the exchange-current density. `sei_rate_scale`
   exists to offset that side effect.
2. **Regime switching.** `sei_rate_scale` multiplies $D_{\mathrm{EC}}$. Lowering
   $D_{\mathrm{EC}}$ slows growth only while transport limits it; once the reaction
   limits, further reduction has sharply diminishing effect. Scaling the SEI kinetic
   rate constant instead does not change growth monotonically, so the oracle does not
   scale it.

Every preset multiplies the parameter set's `EC diffusivity [m2.s-1]` by
`ec_diffusivity_base_factor` (0.25) × `sei_rate_scale`. The base factor represents a
commercial electrolyte with film-forming additives, which forms a denser SEI than the
LP30 baseline of the parameter sets {cite:p}`reniers2019`; it is a cell property,
applied identically in every preset.

## Mechanism 2: lithium plating (`"lithium plating": "partially reversible"`)

Metallic lithium deposits with Tafel-type kinetics in the plating direction
{cite:p}`okane2022`:

$$
j_{\mathrm{pl}} = -F\,k_{\mathrm{pl}}\, c_{e}\,
  \exp\!\left(-\frac{\alpha F\,\eta_{\mathrm{pl}}}{RT}\right),
\qquad \eta_{\mathrm{pl}} = \phi_s - \phi_e \;(\text{vs. Li/Li}^+),
$$

with a stripping branch that returns reversibly plated lithium to the inventory, and a
first-order transfer of plated lithium into a disconnected dead-lithium pool with
decay constant $\gamma$ (`Dead lithium decay constant [s-1]`).

- `plating_rate_scale` multiplies $k_{\mathrm{pl}}$: it sets how much lithium plates
  per cycle, and moves dead-lithium accumulation (hence ohmic growth) and end of life
  in the same direction.
- `dead_li_decay_scale` multiplies $\gamma$: it moves plated lithium between the dead
  pool and dissolution. A larger $\gamma$ gives less net dead-lithium film and less
  ohmic growth.

Plating is exponential in the local overpotential, so it is the mechanism that makes
the presets C-rate sensitive. The DFN resolves the overpotential through the
electrode thickness; plating starts first at the separator side of the negative
electrode.

Chen2020 has no plating parameters. Keys absent from the parameter set are copied from
PyBaMM's `OKane2022` set (plating and stripping exchange-current densities, initial
and typical plated-lithium concentrations, transfer coefficient, rate constant, dead
lithium decay constant and rate, lithium-metal partial molar volume); keys already
present are kept. The preset then sets the rate constant, the decay constant and an
initial plated-lithium concentration of zero (a fresh cell: OKane2022's formation
plating would strip during early low-rate cycling and drive the plating-loss
integrator negative).

## Mechanism 3: particle cracking and active-material loss: disabled

PyBaMM's `"particle mechanics": "swelling and cracking"` submodel {cite:p}`ai2020`,
and SEI on cracks, are off in every preset. The code records the reason: with the
Chen2020 LG M50 graphite OCP the cracking stress submodel fails DAE initialisation at
$t=0$ (`IDAGetDky: IDA_BAD_K`) on PyBaMM 26.6.2.0 (see [Numerical stability](numerics.md)).
Whether this still holds on PyBaMM 26.10 has not been re-checked. Consequences:

- there is no stress-driven loss of active material: the PyBaMM LAM variable and the
  crack-SEI loss are zero, unless the optional `dod_lam_scale` term (below) is on;
- C-rate sensitivity that the literature attributes to cracking comes from plating
  kinetics in `accelerated` and `severe`. This reproduces faster fade at higher rate
  with a different microscopic cause.

## Mechanism 4: depth-of-discharge LAM term (`dod_lam_scale`, off by default)

An empirical, additive loss of active material driven by depth of discharge and a
charge-induced stress proxy, integrated per call from the requested protocol:

$$
d_i = \frac{I_d\,t_d}{Q_{\mathrm{real}}},\quad
s_i = \sqrt{\frac{I_{c1}}{Q_{\mathrm{real}}}\, d_i},\quad
T_i = T_{i-1} + d_i,\quad
\Delta\mathrm{LAM}_i = k_{\mathrm{DoD}}\;\frac{s_i\, d_i}{2\sqrt{T_i}},
$$

with $k_{\mathrm{DoD}}$ = `dod_lam_scale` and $I_d, t_d, I_{c1}$ the protocol's
discharge current and duration and first charge current, so the cumulative loss grows as the square root of throughput. The
cumulative fraction reduces the negative active-material fraction in the EIS (it grows
the charge-transfer arcs) and enters the state-of-health budget as a fraction of the
nominal capacity. It is a labelled empirical term: leave it at 0 until a literature
value or a deep-discharge dataset sets its size.

## Preset definitions

| | `nominal` | `accelerated` (default) | `severe` |
|---|---|---|---|
| SEI (`ec reaction limited`) | yes | yes | yes |
| Lithium plating (`partially reversible`) | no | yes | yes |
| $k_{\mathrm{pl}}$ [m s⁻¹] | — | $10^{-8}$ | $10^{-7}$ |
| $\gamma$ [s⁻¹] | — | $4\times10^{-6}$ | $10^{-4}$ |
| Particle cracking, SEI on cracks | no | no | no |

The numeric constants come from `degradation.preset_constants.<preset>` of the oracle
YAML (`plating_kinetic_rate_constant_m_s`, `dead_lithium_decay_constant_s`,
`initial_plated_lithium_concentration_mol_m3`), with the values above as fallbacks.
The YAML also lists the submodel switches and a `target_eol_cycles_at_1c` per preset;
the code does not read those keys (the submodels are fixed per preset in the code), and
the targets have not been checked against the DFN.

Severity presets are not the same physics run faster: they change the mechanism mix.
`nominal` is SEI only, with protocol-insensitive growth; `accelerated` and `severe` add
a protocol-sensitive plating channel whose observable signature is dead-lithium ohmic
growth.

## From hidden state to observable impedance

In equilibrium EIS the impedance model has no degradation submodels, so the cycling
solution's degradation state is injected into each spectrum. With the beginning-of-life
negative electrode's specific surface area $a_s = 3\varepsilon_s/r_p$ and volume
$V_{el} = L_n W H$:

$$
\Delta R_{\mathrm{ohm}} =
\frac{L_{\mathrm{SEI}}}{\sigma_{\mathrm{SEI}}\, a_s V_{el}}
+ \frac{\rho_{\mathrm{dead}}\, L_{\mathrm{dead}}}{a_s V_{el}},
\qquad
\varepsilon_s \leftarrow \varepsilon_s\,(1-\mathrm{LAM}),
$$

the film-resistance form of Christensen and Newman {cite:p}`christensen2004`, with
$\sigma_{\mathrm{SEI}} = 1/$`SEI resistivity [Ohm.m]` of the parameter set,
$\rho_{\mathrm{dead}} = 0.05\ \Omega\,\mathrm{m}$, and the x-averaged SEI and
dead-lithium thicknesses at the end of the call. $\Delta R_{\mathrm{ohm}}$ shifts the
spectrum along the real axis (it grows the fitted ohmic resistance). LAM is the PyBaMM
negative-electrode LAM plus the DoD term, clipped to `lam_ceiling` (0.95), and is
applied when above $10^{-4}$; shrinking $a_s$ grows the charge-transfer arcs
($R_{ct} \propto 1/(a_s j_0)$). Degradation therefore reaches the ECM state the
surrogate observes, not only the hidden SOH. In non-stationary EIS the model carries
the SEI film itself and only the dead-lithium term is added. The limits of this
coupling are listed in [Divergences](divergences.md).

## End-of-life accounting

SOH is computed from the irreversible-loss budget, not from the voltage-limited
discharge capacity, which conflates protocol with health:

$$
\mathrm{SOH} = 1 - \frac{\Delta\mathrm{LLI} + \mathrm{LAM}_{\mathrm{PyBaMM}}\,Q_{\mathrm{nom}} + \mathrm{LAM}_{\mathrm{DoD}}\,Q_{\mathrm{nom}}}{Q_{\mathrm{nom}}},
$$

where $Q_{\mathrm{nom}}$ is the parameter set's nominal capacity and
$\Delta\mathrm{LLI}$ is the lithium inventory lost since the first call: PyBaMM's
cumulative `Loss of capacity to negative SEI`, `... SEI on cracks` and
`... lithium plating` (the last clamped at ≥ 0, since stripping returns lithium to the
inventory), minus their value after the first call. The integrators carry across calls
through the threaded solution. {class}`~battery_oracle.OracleFailure`
(`END_OF_LIFE`) is raised when SOH falls below `eol_capacity_fraction` (0.80).

`capacity_check=True` appends a C/20 reference charge and discharge to every cycle
and reports that discharge's capacity as `capacity_ah` (much slower; sensitive to both
lithium loss and impedance rise). It does not change the SOH definition.
