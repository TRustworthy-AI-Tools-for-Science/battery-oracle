# Degradation: mechanisms, presets, and their observable signatures

`degradation_preset` selects which aging mechanisms are active in the cycling model
and with what kinetic constants. This page gives the governing equations of each
mechanism as implemented, the exact preset definitions, how the hidden degradation
state is mapped into the *observable* impedance, and simulated evidence of how the
mechanism mix changes with severity.

## Mechanism 1 — SEI growth (`"SEI": "ec reaction limited"`)

Solvent (ethylene carbonate) is reduced at the electrode/SEI interface; EC must
first diffuse through the existing film. With $c_{\mathrm{EC}}^{s}$ the EC
concentration at the inner interface, the through-film flux and interfacial reaction
rate balance in steady state:

$$
\underbrace{\frac{D_{\mathrm{EC}}\,(c_{\mathrm{EC}}^{0}-c_{\mathrm{EC}}^{s})}{L_{\mathrm{SEI}}}}_{\text{transport}}
=
\underbrace{k_{\mathrm{SEI}}\, c_{\mathrm{EC}}^{s}\,
\exp\!\left(-\frac{F\,\eta_{\mathrm{SEI}}}{2RT}\right)}_{\text{reaction}},
\qquad
\frac{\mathrm{d}L_{\mathrm{SEI}}}{\mathrm{d}t}
  = -\frac{\bar V_{\mathrm{SEI}}}{2F}\, j_{\mathrm{SEI}},
$$

a series (mixed-control) law {cite:p}`reniers2019,pinson2013`. Two properties drive
the package's design decisions:

1. **Exponential overpotential sensitivity.** $j_{\mathrm{SEI}} \propto
   e^{-F\eta/2RT}$ with $\eta_{\mathrm{SEI}}$ containing the negative-electrode
   overpotential. Anything that raises $\eta$ at fixed current — including the
   calibration knob `kinetics_scale`, which *deliberately* lowers the
   exchange-current density — accelerates SEI growth as a side effect. This
   confounding is why a separate `sei_rate_scale` exists (see
   [Calibration](calibration.md)).
2. **Regime switching.** `sei_rate_scale` multiplies $D_{\mathrm{EC}}$ (on top of a
   universal ×0.25 electrolyte-quality correction; see
   [Divergences](divergences.md)). Suppressing $D_{\mathrm{EC}}$ only slows growth
   while transport is rate-limiting; once the interfacial reaction becomes the
   bottleneck, further reduction has sharply diminishing returns. Empirically on
   this parameterisation, $D_{\mathrm{EC}}$ scales of 0.5/0.25/0.1 produced 15-cycle
   ohmic-growth responses of 133 %/111 %/82.5 % — strongly sublinear. R1-growth
   calibration is therefore *not* fully solved by this knob; treat large
   `sei_rate_scale` excursions as saturated.

## Mechanism 2 — lithium plating (`"lithium plating": "partially reversible"`)

Metallic deposition with Tafel-type kinetics in the plating direction
{cite:p}`okane2022`:

$$
j_{\mathrm{pl}} = -F\,k_{\mathrm{pl}}\, c_{e}\,
  \exp\!\left(-\frac{\alpha F\,\eta_{\mathrm{pl}}}{RT}\right),
\qquad \eta_{\mathrm{pl}} = \phi_s - \phi_e \;(\text{vs. Li/Li}^+),
$$

with a stripping branch returning *reversibly* plated lithium to the active
inventory, and a first-order transfer of plated lithium into an electrochemically
disconnected **dead-lithium** pool governed by the decay constant $\gamma$
(`Dead lithium decay constant [s-1]`). Operationally, as exploited by the
calibration:

- `plating_rate_scale` (× $k_{\mathrm{pl}}$) sets how much lithium plates per cycle —
  it moves **both** dead-Li accumulation (hence ohmic growth) **and** end-of-life,
  in the same direction.
- `dead_li_decay_scale` (× $\gamma$) redistributes a given amount of plated lithium
  between the dead pool and dissolution — larger $\gamma$ → less net dead-Li film →
  less ohmic growth, largely *without* the EOL shift. The two knobs are therefore
  (approximately) independent levers on (R1-growth, EOL).

Because $j_{\mathrm{pl}}$ is exponential in the local overpotential, plating is the
mechanism that carries **C-rate sensitivity** in this package — by design, since the
stress-driven cracking/LAM pathway is disabled (next section).


## Mechanism 3 — particle cracking / LAM: deliberately disabled

Stock O'Kane-style coupling (crack growth from diffusion-induced stress
{cite:p}`ai2020` → fresh SEI on crack faces → accelerated LAM) is off in every
preset because the cracking stress submodel fails DAE initialisation against the
Chen2020 graphite OCP on current PyBaMM (see
{ref}`Numerical stability <particle-cracking-chen2020-initialization-failure>`).
Two consequences you must account for when interpreting results:

- there is no stress-driven LAM channel, so `lam_pct` and the crack-SEI film are
  identically zero unless the optional throughput-based `dod_lam_scale` term is
  enabled (off by default, and deliberately so — it must be calibrated against a
  deep-DoD dataset before use);
- C-rate sensitivity that the literature attributes to cracking is *re-supplied*
  through plating kinetics in the `accelerated`/`severe` presets. This reproduces
  the phenomenology (faster fade at higher C) with a different microscopic cause —
  a modelling substitution, not a claim about mechanism.

## Preset definitions

| | `nominal` | `accelerated` (default) | `severe` |
|---|---|---|---|
| SEI (`ec reaction limited`) | ✓ | ✓ | ✓ |
| Lithium plating (partially reversible) | — | ✓ | ✓ |
| $k_{\mathrm{pl}}$ [m s⁻¹] | — | $10^{-8}$ | $10^{-7}$ |
| $\gamma$ [s⁻¹] | — | $4\times10^{-6}$ | $10^{-4}$ |
| Particle cracking / SEI-on-cracks | — (disabled) | — (disabled) | — (disabled) |
| Target EOL at 1C [cycles] | ~200–400 | ~40–70 | ~20–50 |

All presets share the ×0.25 EC-diffusivity correction (electrolyte quality treated
as a cell property, not a severity setting) and zero initial plated-lithium
concentration (the oracle models a fresh cell; OKane2022's formation-cycle plated Li
would otherwise strip during early low-C cycling and drive the plating-loss
integrator negative).

## Simulated mechanism mix vs. severity

The figures below were produced by `bin/generate_degradation_figures.py` (8 oracle
calls of the same 1.5C-charge protocol per preset; data in
`docs/_static/degradation/degradation_data.json`).

```{figure} _static/degradation/degradation_soh.png
:width: 75%

SOH trajectories. Over this short window SEI dominates total lithium loss, so the
presets separate only slowly in SOH — after 8 cycles: nominal 0.9781, accelerated
0.9781, severe 0.9772. The plating pathway compounds: its per-cycle contribution
grows with the accumulating film while SEI growth self-limits as
$L_{\mathrm{SEI}}$ thickens.
```

```{figure} _static/degradation/degradation_mechanisms.png
:width: 100%

Cumulative capacity loss by mechanism (note the shared axis). The SEI channel is
nearly *identical* across presets (~0.156 Ah after 8 cycles) — severity presets do
not touch SEI constants. What changes is the plating channel: zero for `nominal`,
2.4 mAh for `accelerated`, 8.7 mAh for `severe` at the same protocol. SEI-on-cracks
is identically zero (mechanism disabled).
```

```{figure} _static/degradation/degradation_films.png
:width: 95%

Interphase films — the quantities that feed the impedance. SEI thickness growth is
preset-independent; the dead-lithium film differs by ~30× between `accelerated`
(0.031 nm) and `severe` (0.94 nm) after 8 cycles, because `severe` combines 10×
plating kinetics with a decay constant that still leaves substantial net
accumulation.
```

Interpretation: severity presets are not "the same physics, faster". Instead, they change
the mechanism mix. A surrogate trained on `nominal` sees a pure-SEI world
(monotone, protocol-insensitive R1 growth); `accelerated`/`severe` add a
protocol-sensitive channel whose observable signature is dead-Li-driven ohmic
growth. Regenerate with more cycles to see EOL separation:

```bash
uv run python bin/generate_degradation_figures.py --cycles 20
```

## From hidden state to observable impedance

The EIS model contains no degradation submodels; the cycling solution's degradation
state is injected into each synthesized spectrum as follows (specific surface area
$a_s = 3\varepsilon_s/r_p$, electrode volume $V_{el} = L\,W\,H$):

$$
\Delta R_{\mathrm{ohm}} =
\underbrace{\frac{L_{\mathrm{SEI}}}{\sigma_{\mathrm{SEI}}\, a_s V_{el}}}_{\text{SEI film}}
+ \underbrace{\frac{L_{\mathrm{SEI,cr}}}{\sigma_{\mathrm{SEI}}\, 2 N_{\mathrm{cr}} V_{el}\, l_{\mathrm{cr}}}}_{\text{crack SEI }(\equiv 0)}
+ \underbrace{\frac{\rho_{\mathrm{dead}}\, L_{\mathrm{dead}}}{a_s V_{el}}}_{\text{dead Li}},
\qquad
\varepsilon_s \leftarrow \varepsilon_s\,(1-\mathrm{LAM}),
$$

following the film-resistance form of Christensen & Newman
{cite:p}`christensen2004` with $\rho_{\mathrm{dead}}$ $\approx 0.05$ $\Omega$,m. The
$\Delta R_{\mathrm{ohm}}$ shifts the whole spectrum along the real axis (it grows
the fitted $R_1$); the LAM injection shrinks $a_s$, which grows the charge-transfer
arc ($R_{ct} \propto 1/(a_s j_0)$) — i.e. degradation reaches the 18-D ECM state the
surrogate observes, not just the hidden SOH. Assumptions and validity limits of this
post-hoc coupling are catalogued in [Divergences](divergences.md).

## End-of-life accounting

`OracleFailure` is raised when SOH falls below `eol_capacity_fraction` (default
0.80). SOH is computed from the **cumulative lithium-inventory-loss integrators**
(SEI + SEI-on-cracks + plating), which carry across calls via the threaded solver
state, *not* from the voltage-limited discharge capacity — the latter conflates
protocol choice with health. The plating integrator is clamped at ≥ 0: stripping
returns lithium to the active pool and must not register as capacity *gain*. Set
`capacity_check=True` to append a C/20 reference discharge each call and measure
usable capacity directly (~20× slower; sensitive to both LLI and impedance rise).
