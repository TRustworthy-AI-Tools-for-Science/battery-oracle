# Using the oracle

## Semantics of a call

{class}`~battery_oracle.PyBaMMOracle` is a stateful callable emulating a
physical cell on a potentiostat. Each call executes `n_cycles` charge/discharge
cycles continuing from the electrochemical state left by the previous call (the
PyBaMM solution is threaded via `starting_solution`), synthesizes EIS at the two
relaxed states, fits the ECM, and returns the featurised state. Degradation
therefore accumulates across calls as though it were an experimental
campaign.

```python
from battery_oracle import PyBaMMOracle, make_pybamm_candidates

oracle = PyBaMMOracle(degradation_preset="accelerated")
oracle.reset()                                # fresh cell (and native solver rebuild)
for protocol in make_pybamm_candidates():     # 6-D protocol grid
    state = oracle(protocol)                  # -> shape (2·P,) ECM state vector
```

Failure contract: {class}`~battery_oracle.OracleFailure` is raised when SOH crosses
`eol_capacity_fraction` **or** when both solver paths fail / silently truncate (see
[Numerical stability](numerics.md)). Treat it as end-of-life: do *not* reset and
continue — the physical analogue is a dead cell, and the state is intentionally not
rolled back.

## The state vector

For a circuit with $P$ parameters (labels from the circuit string; AutoEIS
convention `Rx` = resistor, `Pxw`/`Pxn` = CPE admittance/exponent), a call returns
the length-$2P$ concatenation

$$
s = \big[\underbrace{\theta^{\mathrm{chg}}_1, \dots, \theta^{\mathrm{chg}}_P}_{\text{post-charge fit}},\;
         \underbrace{\theta^{\mathrm{dis}}_1, \dots, \theta^{\mathrm{dis}}_P}_{\text{post-discharge fit}}\big],
$$

the two spectra being fitted independently. For the packaged default circuit
`R1-P2-[R3,P4]-[R5,P6]` ($P{=}9$) this is the 18-D state. Parameter
summaries are posterior medians (see [Divergences](divergences.md) §7 for why
not means).

## The history record

Each call appends a dict to `oracle._history` — the primary interface for analysis.
Key fields:

| Key | Content |
|---|---|
| `end_soh`, `capacity_ah` | LLI-integrator SOH; voltage-limited (or C/20 if `capacity_check`) capacity |
| `cumulative_{sei_loss,plating,crack_sei}_ah`, `cumulative_lli_total_ah` | per-mechanism lithium-inventory loss since cell birth |
| `sei_thickness_nm`, `dead_li_nm`, `crack_sei_nm`, `lam_pct` | film/LAM state feeding the EIS corrections |
| `Z_charge_real`, `Z_charge_neg_imag` (+ discharge) | raw synthesized spectra, jones2022 convention — Nyquist is `plot(Z_real, Z_neg_imag)` |
| `ecm_params_charge/_discharge` | fitted parameter medians (length $P$) |
| `ecm_samples_*`, `ecm_variables_*` | raw MCMC posterior samples per element (AutoEIS path; `None` for the stub) |
| `linkk_rmse`, `max_cv`, `converged`, `drt_peaks` | fit diagnostics: Kramers–Kronig residual, posterior dispersion, DRT relaxation times (`[drt]` extra) |
| `model`, `protocol`, `call_idx` | provenance |

## ECM fitters

- {func}`~battery_oracle._autoeis_ecm` (default when the `[autoeis]` extra is
  present): Bayesian inference (NumPyro NUTS) per spectrum, with the
  scale-equivariant rescaling and process-hygiene patches described in
  [Divergences](divergences.md). ~seconds per spectrum.
- {func}`~battery_oracle._randles_stub_ecm`: analytic asymptotics — HF intercept →
  ohmic $R$, LF limit → total polarisation split 60/40 across arcs, peak
  $-\mathrm{Im}\,Z$ frequency → CPE admittances. Milliseconds, no JAX. Use it
  whenever the study target is PyBaMM physics rather than fit fidelity:

```python
from battery_oracle.oracle import _randles_stub_ecm
oracle = PyBaMMOracle(ecm_model_fn=_randles_stub_ecm)
```

Any callable `(frequencies, Z_real, Z_imag) -> (2P,) array` is accepted.

## Persistence

`PyBaMMOracle.save_to_csv(history, path, cell_id=...)` writes the
jones2022-featurised record format. Before writing it applies the sequence-level
symmetric-arc flip correction. AutoEIS cannot distinguish the two Randles arcs
of a symmetric circuit, so their labels can swap between cycles. The correction
anchors cycle 0 by CPE-exponent ordering and then picks, per cycle, the arc
permutation minimising a log-space continuity distance to the previous corrected
cycle, recomputed from the retained MCMC samples. This is only resolvable at the
sequence level, which is why per-call `ecm_params_*` are left uncorrected.

## Concurrency and lifetime

The oracle serialises calls with an internal lock but is designed for sequential
use: one oracle ≈ one cell. Call `reset()` between campaigns. Besides clearing
state, it rebuilds the model and both solvers to bound native-memory lifetimes
(see [Numerical stability](numerics.md#process-hygiene-fork-vs-spawn-and-native-object-lifetimes)).
For YAML-driven construction and worked configurations, see
[Experiment protocols](protocol.md).
