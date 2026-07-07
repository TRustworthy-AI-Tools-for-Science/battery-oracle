# Model hierarchy: DFN, SPMe, SPM

The `model` kwarg selects which member of the standard porous-electrode model
hierarchy backs both the cycling simulation and the internal EIS simulation:

```python
oracle = PyBaMMOracle(model="SPMe")   # "SPM" | "SPMe" (default) | "DFN"
```

All three are PyBaMM implementations {cite:p}`sulzer2021` sharing one parameter set
(Chen2020, an LGM50 5 Ah NMC811/graphite–SiOₓ cell {cite:p}`chen2020`). Switching
models changes the level of physical resolution, not the chemistry. 

This page
states what each model solves, the asymptotic assumptions that generate the reduced
models, and where those assumptions fail in ways that matter for this package's
outputs (degradation trajectories and impedance spectra).

## The full model (DFN)

The Doyle–Fuller–Newman model {cite:p}`doyle1993` couples concentrated-solution
theory in the electrolyte to Fickian diffusion in a continuum of spherical particles,
resolved through the electrode thickness (a "pseudo-2D" $x \times r$ domain). Per
electrode $k \in \{\mathrm{n}, \mathrm{p}\}$:

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
j_{0,k} = m_k(T)\, c_e^{1/2}\, (c_{s,k}^{\mathrm{surf}})^{1/2} (c_{s,k}^{\max}-c_{s,k}^{\mathrm{surf}})^{1/2},
\qquad
\eta_k = \phi_s - \phi_e - U_k(c_{s,k}^{\mathrm{surf}}) - \Delta\phi_{\mathrm{film}},
$$

where $\Delta\phi_{\mathrm{film}}$ carries the interphase-film resistances that the
degradation submodels grow (see [Degradation](degradation.md)). Effective transport
coefficients use Bruggeman corrections $\theta^{\mathrm{eff}} = \theta\,\varepsilon^{b}$.

## SPMe

The SPMe is not an ad-hoc simplification; PyBaMM's implementation follows the
systematic asymptotic derivation of Marquis et al. {cite:p}`marquis2019`. The
expansion parameter is the ratio of the typical electrolyte potential drop to the
thermal voltage (equivalently, of the electrolyte transport timescale to the
discharge timescale). At leading order the interfacial current density is uniform
through each electrode thickness, so one representative particle per electrode,
driven by the $x$-averaged current, suffices. The first-order correction restores the
electrolyte concentration profile, and contributes electrolyte ohmic and
concentration overpotentials to the terminal voltage:

$$
V = U_p(c_{s,p}^{\mathrm{surf}}) - U_n(c_{s,n}^{\mathrm{surf}})
  + \eta_p - \eta_n + \eta_e + \Delta\phi_{e}^{\mathrm{Ohm}} + \Delta\phi_{s}^{\mathrm{Ohm}},
$$

where $\eta_e \propto \ln\!\big(c_e(x_p)/c_e(x_n)\big)$ is the concentration
overpotential evaluated from the reduced electrolyte problem. The SPM truncates
the same expansion at leading order: electrolyte dynamics are dropped entirely and
$V$ contains only the OCPs and the (uniform) reaction overpotentials.

### Assumptions inherited by every result in this package

| Assumption | Where it enters | When it breaks |
|---|---|---|
| Uniform through-thickness reaction distribution | SPM/SPMe cycling and the degradation state (SEI/plating are $x$-averaged) | High C-rate, thick/low-porosity electrodes: reaction localises at the separator, under-predicting local plating onset |
| Single representative particle, single $R_k$ | Solid diffusion; $a_s = 3\varepsilon_s/r_p$ used in all film-resistance conversions | Wide particle-size distributions; local SOC heterogeneity |
| Electrolyte correction is first-order only (SPMe) | Terminal voltage; EIS diffusion tail | Above ≈1C on this parameterisation the correction degrades; the oracle clamps protocols to a validated envelope (see [Numerical stability](numerics.md)) |
| Isothermal at `temperature_K` | All rate constants via their Arrhenius forms evaluated at the fixed ambient $T$ | Self-heating at high C-rate; a temperature *sweep* here means a family of isothermal simulations, not a thermal model |
| Bruggeman effective media | Electrolyte transport | Anisotropic/heterogeneous microstructures |
| Chen2020 parameterisation | Everything | Chemistry other than NMC811/graphite; the calibration scales ([Calibration](calibration.md)) absorb *some* cell-to-cell variation, not chemistry changes |

## Impedance and EIS Computation

`pybamm.EISSimulation` linearises the chosen model about an equilibrated state at
the requested SOC and evaluates the small-signal impedance $Z(\omega) = \delta
V/\delta I$ over the frequency grid. Linearisation requires every equation to be
differential, which is why the oracle constructs the EIS model with
`{"surface form": "differential"}`: interfacial charge balance is closed by an
explicit double-layer ODE,

$$
C_{\mathrm{dl}}\frac{\partial(\phi_s-\phi_e)}{\partial t} = \frac{i_{\mathrm{app}}}{a\,\delta} - j,
$$

which simultaneously (i) converts the algebraic current-balance constraint into an
ODE (an index reduction, making the linearisation well-posed) and (ii) supplies the
high-frequency capacitive semicircle. Consequences of the model choice:

- **SPM** impedance lacks the electrolyte diffusion branch entirely — the
  low-frequency tail is purely solid-state diffusion.
- **SPMe** resolves *one* dominant charge-transfer arc per electrode pair. Distinct
  anode/cathode arcs that a 2-RQ equivalent circuit nominally separates are only
  weakly identifiable — this is the "single-arc fidelity limit" that caps how well
  the second arc ($R_5$) can ever be matched during calibration.
- **DFN** adds through-thickness electrolyte relaxation, at roughly an order of
  magnitude more states and a much stiffer linearisation.

```{important}
The EIS model is rebuilt **fresh** each call and does *not* include the degradation
submodels. Degradation enters the spectrum through deliberate post-hoc corrections
(film resistances added in series; LAM injected as a reduced active-material
fraction). This is a controlled divergence from stock PyBaMM — see
[Divergences from stock PyBaMM](divergences.md) for the formulas and their validity
limits.
```

## Cost and Stiffness

Empirically on this parameter set (one 5-step cycle, IDAKLU where possible): SPM is
the cheapest and least stiff (no electrolyte PDE); SPMe is ~1.5–3× SPM; DFN is
~10–30× SPMe with a substantially harder DAE initialisation, and is the most likely
to fall back to the emergency solver at control-mode switches. The degradation
presets were calibrated against SPMe; under SPM the missing electrolyte
overpotential lowers $\eta$ at fixed current (slower SEI/plating), while under DFN
the local overpotential at the separator edge exceeds the $x$-average (earlier
plating onset). The same preset therefore reaches end-of-life at a different cycle
count in each model — the [demonstrations notebook](notebooks/02_demonstrations)
quantifies this on a common protocol.
