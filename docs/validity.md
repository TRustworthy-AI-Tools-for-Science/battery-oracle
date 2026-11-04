# Physical validity

A parameterised oracle need not reproduce a cell exactly, but its internal solution
must obey the continuum equations it discretises, and its parameters must describe a
physically sensible cell. Two modules judge this; neither looks at the fit error.

- {mod}`battery_oracle.validity`: checks of one solved DFN history, and the two-mesh /
  tolerance-ladder judgement of them.
- {mod}`battery_oracle.physicality`: checks of a parameterised oracle: plausibility
  windows, the electrode-SOH solve, and one demanding DFN cycle that includes the
  validity gate.

Every check is a record `{"name", "status", "value", "limit", "note"}` with `status`
`pass`, `warn` (inspect before trusting), `fail` (unphysical: do not use) or `skip`
(not evaluable). {func}`battery_oracle.physicality.summarise` reduces a list to the
worst status and counts per status. The stage-A checks of a fit
({mod}`battery_oracle.fit.checks`) build on these ([Fitting](fitting.md)).

## Measures of one solution

{func}`battery_oracle.validity.measure` returns raw values (no pass/fail) from one
PyBaMM solution and the parameter values it was solved with, at every saved time:

- **flux continuity** across both electrode|separator interfaces: the electrolyte
  current density $i_e$, molar flux $N_e$, concentration $c_e$ and potential
  $\phi_e$, each extrapolated linearly to the interface from the two nearest samples
  on either side; a jump means the interface is not resolved or the coefficients
  there are inconsistent;
- **charge balance**: $i_e + i_s = i_{\mathrm{app}}$ in both electrodes,
  $i_e = i_{\mathrm{app}}$ in the separator, no electrolyte current at the current
  collectors, and an integral form of the balance over windows of at least
  `CHARGE_WINDOW_S` (1 s), which includes the change of the charge stored in the
  double layer when the model has one (`surface form: differential`);
- **mesh oscillation**: the longest run of alternating-sign second differences of
  $c_e$ and $\phi_e$ inside a domain (an odd–even mode of the finite volumes);
- **lithium conservation**: lithium in particles and electrolyte plus lithium consumed
  by side reactions stays at its initial value; `side_loss_rel` is the lithium consumed
  by side reactions relative to the total.

Fluxes are scaled by the larger of the applied and the 1C current density. Continuity
is not judged within `STEP_WINDOW_S` (2 s) of a step in the applied current: there
the double layer redistributes charge through a boundary layer at the separator that
starts with zero thickness, which no mesh resolves at the step itself.
{func}`~battery_oracle.validity.charge_balance_residual` gives the pointwise balance
over time per domain. `check_solution` is the one-mesh version with fixed limits
(`LIMITS`), kept for diagnostics.

## The judgement

Fixed tolerances on interface jumps would be arbitrary: in finite volumes the discrete
interface flux is single-valued by construction, and the one-sided reconstruction
measures how well the field is resolved there. {func}`~battery_oracle.validity.judge`
compares the measures of the deployed mesh with those of a mesh refined by `factor`
(2), in tiers:

1. **Exact laws**, with limits from the solver tolerance: charge balance (pointwise
   and integral) at most $10\,\mathrm{rtol}$; lithium drift at most 1 % of the lithium
   consumed by side reactions, or $10\,\mathrm{rtol}$ of the total when there is none.
   A residual above its limit still passes if it is a tolerance error: the deployed
   mesh is solved again with tolerances 10× and 100× tighter (`TOLERANCE_LADDER`); the
   residual must shrink at least `MIN_DECADE_RATIO` (2×) per decade, and its Aitken
   extrapolation to zero tolerance must be within the limit. A real violation levels
   off and fails. A tighter solve that fails ends the ladder.
2. **Continuity**: each jump must shrink at least by the refinement factor, or already
   be at round-off ($\le 10\,\mathrm{rtol}$) on the refined mesh. A jump that does not
   shrink is a real inconsistency and fails.
3. **Oscillation**: a sign-alternating run of 4 or more that does not shrink under
   refinement is a warning (no tolerance-free test exists).

{func}`~battery_oracle.validity.relevance` adds a third question for a fit: does the
deployed mesh resolve what is reported? Refining it must change the voltage by less
than the instrument noise (`VOLTAGE_NOISE_V`, 1 mV, p99 over the records), the
impedance by less than the EIS noise (`EIS_NOISE_REL`, 1 %), and the number of cycler
stops by zero. A failure here means the oracle must run the refined mesh, not that the
cell is unphysical. {func}`~battery_oracle.validity.voltage_difference` compares two
voltage traces outside the current steps of either.

## Checks of a parameterised oracle

{mod}`battery_oracle.physicality`:

- {func}`~battery_oracle.physicality.check_fitted_values`: kinetics and transport
  evaluated at a reference state (stoichiometry 0.5, $c_e$ = 1000 mol m⁻³, 298.15 K)
  against `PLAUSIBILITY_WINDOWS`, about an order of magnitude either side of published
  LiCoO2/graphite values (pass inside, warn within 10× outside, fail beyond); a guard
  against runaway fits, not a literature fit. Also flags fitted multipliers at a bound.
- {func}`~battery_oracle.physicality.check_esoh`: PyBaMM's electrode-SOH solve
  succeeds and $x_0 < x_{100}$, $y_{100} < y_0$ lie strictly inside (0, 1).
- {func}`~battery_oracle.physicality.check_dfn_cycle`: one demanding cycle on the
  oracle built from `oracle_kwargs` (default 3.8C first charge stage, 2C second, 3.8C
  discharge, at the C-rates of the oracle's own nominal capacity, Randles-stub ECM;
  {func}`~battery_oracle.physicality.run_demanding_cycle`). It checks that the cycle
  completes at full fidelity; the electrolyte is not depleted (minimum $c_e$ over its
  initial value: fail below `CE_FAIL_REL` = 0.1, warn below `CE_WARN_REL` = 0.2); the
  surface stoichiometries stay inside (0.005, 0.995); the positive stoichiometry stays
  inside the OCP data range; lithium is conserved (PyBaMM's total loss equals its loss
  to side reactions within $10^{-3}$); and it runs the two-mesh gate on this cycle (the
  electrode/separator mesh refined 2×, with the tolerance ladder when an exact law is
  above $10\,\mathrm{rtol}$, and the voltage-resolution check). The currents pass
  through the oracle's protocol bounds, so `oracle_kwargs` must raise
  `dfn_max_crate` (1.5C by default) for the cycle to run at the stated rates.
- {func}`~battery_oracle.physicality.fitted_oracle_kwargs`: the oracle keywords of a
  stage-A fit report (fitted values, thermal submodel, mesh).

The same electrolyte thresholds set the floor of the fit
(`battery_oracle.fit.config.CE_FLOOR_DEFAULT` = `CE_FAIL_REL`), so a fit cannot accept
what the checks reject.
