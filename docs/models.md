# The model

The oracle's cycling model is PyBaMM's Doyle–Fuller–Newman model
(`pybamm.lithium_ion.DFN`) {cite:p}`doyle1993,sulzer2021`. There is no reduced-order
alternative and no fallback to one: a cycle that does not solve raises
{class}`~battery_oracle.OracleFailure`. This page states the equations, the model
options and parameter sets the oracle uses, how it computes impedance, and the
assumptions every result inherits.

## Equations

The DFN couples concentrated-solution theory in the electrolyte to Fickian diffusion
in spherical particles resolved through the electrode thickness (a pseudo-2D
$x \times r$ domain). Per electrode $k \in \{\mathrm{n}, \mathrm{p}\}$:

$$
\frac{\partial c_{s,k}}{\partial t}
  = \frac{1}{r^{2}}\frac{\partial}{\partial r}
    \!\left(r^{2} D_{s,k}\frac{\partial c_{s,k}}{\partial r}\right),
\qquad
-D_{s,k}\frac{\partial c_{s,k}}{\partial r}\bigg|_{r=R_k} = \frac{j_k}{F},
$$

$$
\varepsilon_k \frac{\partial c_e}{\partial t}
  = \frac{\partial}{\partial x}\!\left(D_e^{\mathrm{eff}}\frac{\partial c_e}{\partial x}\right)
  + \frac{1-t_+^0}{F}\, a_k j_k,
\qquad
\frac{\partial i_e}{\partial x} = a_k j_k, \quad i_s + i_e = i_{\mathrm{app}},
$$

with the MacInnes equation for the electrolyte potential (including the
concentration-gradient term with the thermodynamic factor) and Ohm's law
$i_s = -\sigma^{\mathrm{eff}}\, \partial\phi_s/\partial x$ in the solid. The
interfacial current density follows symmetric Butler–Volmer kinetics,

$$
j_k = 2\,j_{0,k}\sinh\!\left(\frac{F\eta_k}{2RT}\right),
\qquad
\eta_k = \phi_s - \phi_e - U_k(c_{s,k}^{\mathrm{surf}}) - \Delta\phi_{\mathrm{film}},
$$

with the exchange-current density $j_{0,k}(c_e, c_{s,k}^{\mathrm{surf}}, T)$ of the
parameter set, and $\Delta\phi_{\mathrm{film}}$ the drop across the SEI film that the
SEI submodel grows (see [Degradation](degradation.md)). Effective transport
coefficients use Bruggeman corrections $\theta^{\mathrm{eff}} = \theta\,\varepsilon^{b}$.

## Model options

The cycling model is built from the option dict of the degradation preset
(`PyBaMMOracle.degradation_options`), extended by the oracle's settings:

| Option | Value | Set by |
|---|---|---|
| `SEI` | `ec reaction limited` | every preset |
| `lithium plating` | `partially reversible` | `accelerated` |
| `contact resistance` | `true` | `contact_resistance_ohm` not `None` |
| `thermal` | `lumped` | `thermal="lumped"` |
| `use lumped thermal capacity` | `true` | lumped thermal with a set that has `Cell heat capacity [J.K-1.m-3]` (LiR2032) |

The cycling model keeps PyBaMM's algebraic surface form (no double layer). The EIS
models add `surface form: differential` (below).

The mesh is PyBaMM's default unless `radial_mesh` is set:
`{"points": n, "kind": "uniform" | "surface", "stretch": s, "x_factor": k}` sets the
particle radial points, an exponential mesh clustered at the particle surface
(`kind: surface`), and multiplies the electrode and separator points by `k`
({mod}`battery_oracle.model_options`). A stage-A fit records the mesh it converged on,
and the oracle built from the fit uses it.

## Parameter sets

`chemistry` (or `cycling.chemistry` / `cycling.parameter_set`) is one of
`battery_oracle.parameter_sets.SUPPORTED_PARAMETER_SETS`. Other PyBaMM sets are
rejected: the SEI model needs EC-diffusivity and SEI parameters they lack.

**`Chen2020`** (default): PyBaMM's LG M50 5 Ah NMC811/graphite–SiOₓ cell
{cite:p}`chen2020`. The default protocol bounds, the degradation presets and the
first-call `initial_soc` are set for this cell.

**`LiR2032`** (alias `LiR2032_MohtatGraphite_MarquisLCO_ThermoD`, the name the
stage-A fits were run under): the LiCoO2/graphite coin cell of the jones2022 dataset
{cite:p}`jones2022`, at its real size. It is built by
{func}`battery_oracle.parameter_sets.lir2032` on PyBaMM's `Ai2020` set
{cite:p}`ai2020`, which supplies geometry, porosities, active fractions, particle
radii, Bruggeman exponents, electrolyte transport, conductivities, thermal and SEI
parameters. Changes from the template:

- one electrode layer, of the template's area per capacity scaled to the nominal
  capacity (an effective area; a stage-A fit sets it);
- nominal capacity 42 mAh (measured on the jones2022 cells; datasheet 35 mAh), voltage
  window 3.0–4.3 V, zero contact resistance;
- coin-can volume, cooling area and whole-cell heat capacity (2.6 J/K), heat-transfer
  coefficient 10 W m⁻² K⁻¹;
- activation energies of the exchange-current densities and particle diffusivities
  from PyBaMM's Ramadass2004 set, replacing Ai2020's 5 kJ/mol;
- positive OCP of PyBaMM's `Marquis2019` set {cite:p}`marquis2019` (a flat plateau
  ending in a steep wall near full lithiation), with zero entropic term;
- positive particle diffusivity $D = D_0\,\Theta(y, T)$, with the thermodynamic factor
  $\Theta = -\frac{F}{RT}\,y(1-y)\,\frac{\mathrm{d}U_p}{\mathrm{d}y}$ of that OCP
  (floored at $10^{-3}$), so only $D_0$ is free;
- negative OCP of `Mohtat2020` (`graphite_ocp=` also accepts `Ecker2015`, `Chen2020`,
  `Ramadass2004`, `Ai2020`).

The degradation presets are not calibrated for LiR2032.

`parameter_scales` and `parameter_overrides` (the `bol` section) are applied after
the degradation preset: a scale multiplies a numeric or functional parameter
({mod}`battery_oracle.params`), an override sets an absolute value. The nominal
capacity used for the current mapping is read after them.

## Impedance

The oracle synthesises two spectra per call, at the end of the discharged rest and of
the charged rest, by one of two methods (`eis_mode`). Both linearise a DFN with the
double layer on (`surface form: differential`): the interfacial charge balance is
closed by a double-layer ODE,

$$
C_{\mathrm{dl}}\frac{\partial(\phi_s-\phi_e)}{\partial t} = \frac{i_{\mathrm{app}}}{a\,\delta} - j,
$$

which turns the algebraic current-balance constraint into an ODE (making the
linearisation well posed) and gives the high-frequency capacitive arc.

### Equilibrium EIS (default)

{func}`battery_oracle.eis.equilibrium_impedance` runs `pybamm.EISSimulation` on a DFN
built with only `surface form: differential`, at the state of charge read from the
rest, and assumes the cell does not change during the sweep. The EIS model has none of
the cycling options (no SEI, plating, contact resistance or thermal submodel) and uses
PyBaMM's default mesh. Degradation and temperature enter as corrections:

1. the negative active-material fraction is reduced by the total active-material loss
   before linearisation;
2. with lumped thermal, the spectrum is solved at $T_{\mathrm{ref}} = 298.15$ K and
   scaled to the cell temperature $T$ at the rest:
   $Z \leftarrow R_0\,a_{\mathrm{el}} + a_{\mathrm{ct}}\,(Z - R_0)$, with
   $a = \exp\!\big(\tfrac{E_a}{R}(\tfrac{1}{T} - \tfrac{1}{T_{\mathrm{ref}}})\big)$,
   $E_a$ = `E_a_J_per_mol` (30 kJ/mol) for $a_{\mathrm{ct}}$ and
   `E_a_electrolyte_J_per_mol` (15 kJ/mol) for $a_{\mathrm{el}}$, and $R_0$ the
   real part at the highest frequency;
3. the SEI and dead-lithium film resistance $\Delta R_{\mathrm{ohm}}$ is added to the
   real part ([Degradation](degradation.md));
4. measurement noise is added ([protocol](protocol.md)).

The SOC passed to the EIS is the average negative-particle stoichiometry at the end of
the rest divided by the parameter set's initial negative stoichiometry, clipped to
[`soc_clip_min`, `soc_clip_max`] = [0.05, 0.99].

### Non-stationary EIS

The jones2022 spectra are measured after a 20 min rest at open circuit following a
1–4C step, with a ~300 s sweep from high to low frequency: the cell is still relaxing.
With `eis_mode="nonstationary"` ({mod}`battery_oracle.eis`; needs PyBOP) the oracle
computes the spectrum the way the measurement was made:

1. A DFN with the oracle's cycling options, the double layer and PyBOP's EIS
   voltage/current variables ({func}`~battery_oracle.eis.eis_model`) starts from the
   first state of the rest (current already zero) and is integrated at zero current
   through the rest and the sweep.
2. Frequency $f_k$ is measured in a window $[t_k^{\mathrm{lo}}, t_k^{\mathrm{hi}}]$
   after the rest ({func}`~battery_oracle.eis.sweep_windows`: each window lasts
   `eis_sweep_periods`$/f_k$ + `eis_sweep_overhead_s`, from high to low frequency). The
   model is linearised at the simulated state at the window centre:
   $Z_k = -x_V/x_I$ with $(j\omega_k M - J(y(t_k)))\,x = b$
   ({func}`~battery_oracle.eis.linearised_impedance`).
3. The lock-in leakage of the voltage drift is added: over whole periods of a
   sinusoidal current of amplitude $A$, a linear drift $\mathrm{d}V/\mathrm{d}t$ adds
   $-2\,(\mathrm{d}V/\mathrm{d}t)/(\omega A)$ to the impedance
   ({func}`~battery_oracle.eis.drift_leakage`), the leading term of the treatment of
   Hallemans et al. {cite:p}`hallemans2023`. $A$ is `eis_amplitude_mA` (5 mA) mapped to
   the simulated cell, and the drift is the mean slope over each window.

The model carries temperature, the SEI film and contact resistance itself, so only the
dead-lithium resistance and the noise are added afterwards. The rest step is extended
by the sweep length so the cycle keeps the real timing. When the double-layer solve
fails at the start of the rest, the first 30 s are integrated with the cycling model
and the measurement starts from there. If the non-stationary spectrum fails (or
PyBOP is not installed), the oracle logs a warning, uses the
equilibrium spectrum, and records `eis_mode_used = "equilibrium (non-stationary failed)"`.
The stage-A fit uses the same functions ([Fitting](fitting.md)).

## Assumptions inherited by every result

| Assumption | Where it enters | When it breaks |
|---|---|---|
| Single particle size per electrode | solid diffusion; $a_s = 3\varepsilon_s/r_p$ in the film-resistance conversion | wide particle-size distributions |
| Bruggeman effective media | electrolyte and solid transport | anisotropic or heterogeneous microstructures |
| Isothermal at `temperature_K` (default) | every Arrhenius-dependent parameter | self-heating at high rate; use `thermal="lumped"` |
| x-averaged film state for the EIS correction | equilibrium EIS ohmic shift | through-thickness SEI or plating heterogeneity, which the DFN resolves but the correction averages |
| Equilibrium EIS model without degradation submodels | the equilibrium spectrum | film capacitance and SOC-dependent film effects; see [Divergences](divergences.md) |
| Protocol currents clipped to `c_max_crate` | every cycle | protocols above 1.5C, unless the bound is raised |
| Parameter set | everything | a chemistry other than the set's; the calibration scales absorb some cell-to-cell variation, not a change of chemistry |
