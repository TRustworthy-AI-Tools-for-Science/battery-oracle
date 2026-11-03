"""Equivalent-circuit fits of a synthesised spectrum: the Randles stub (fast, any
circuit) and the AutoEIS Bayesian fit, with the CPE seeds and spectrum rescaling they use."""
from __future__ import annotations

import logging
import threading

import numpy as np

from battery_oracle._circuit import DEFAULT_CIRCUIT as _PROJECT_CIRCUIT
from battery_oracle._circuit import _param_labels_from_circuit
from battery_oracle._circuit import randles_pairs_from_circuit as _randles_pairs_from_circuit

log = logging.getLogger(__name__)

try:
    import autoeis as _autoeis
    import autoeis.utils as _ae_utils
    _AUTOEIS_AVAILABLE = True
except ImportError:
    _autoeis = None
    _ae_utils = None
    _AUTOEIS_AVAILABLE = False

# AutoEIS's numpyro/JAX inference keeps global (param-store) and tracing state that
# is NOT thread-safe: two threads fitting concurrently collide with "all sites must
# have unique names" / JAX tracer-escape errors, silently degrading a fit to the
# Randles stub. The tune engine now parallelises by process (each has its own JAX
# state), but this module-level lock defends any in-process threaded caller (e.g. a
# threaded active-learning loop) by serialising the inference call. Within a single
# process it is uncontended, so it costs nothing on the common path.
_AUTOEIS_INFERENCE_LOCK = threading.Lock()

_AUTOEIS_PATCHED = False


def _init_autoeis_patches() -> None:
    """Apply the mpire/AutoEIS monkeypatches, once, on first AutoEIS use.

    Deferred out of module import (previously these ran at ``import
    battery_oracle`` time) so importing the package no longer mutates
    third-party libraries as a side effect. Called from :func:`autoeis_ecm`
    before any AutoEIS use; a no-op if already patched or AutoEIS isn't
    installed.
    """
    global _AUTOEIS_PATCHED
    if _AUTOEIS_PATCHED or not _AUTOEIS_AVAILABLE:
        return

    # Force mpire (used by AutoEIS) to spawn worker processes instead of forking.
    # On Linux the default is fork, which copies the parent's SUNDIALS/CasADi JIT
    # state to children at the same virtual addresses.  When the child touches
    # those addresses after the OS copy-on-write triggers, it segfaults — and the
    # shared-memory corruption can propagate back, killing the parent too.  Spawn
    # starts a fresh interpreter with no inherited native JIT state.
    # Must run before AutoEIS is used, as mpire is pulled in at autoeis import.
    import mpire as _mpire
    _orig_wp_init = _mpire.WorkerPool.__init__
    def _spawn_wp_init(self, *args, **kwargs):
        kwargs.setdefault('start_method', 'spawn')
        _orig_wp_init(self, *args, **kwargs)
    _mpire.WorkerPool.__init__ = _spawn_wp_init
    # Note: _orig_wp_init must stay reachable — _spawn_wp_init looks it up
    # via closure over this function's locals, so it must not be
    # garbage-collected while the patched __init__ can still be called.

    # Monkeypatch autoeis.utils.initialize_priors to fix its excessively wide prior
    # (the same fix as inference.py's training-path patch). Without it the admittance
    # terms draw from a +-12-decade log-normal and converge 10-500x away from the
    # training distribution.
    def _corrected_initialize_priors(p0):
        priors = {}
        variables = [k for k in p0.keys() if _ae_utils.parser.validate_parameter(k, raises=False)]
        for var in variables:
            value = p0[var]
            if "n" in var:
                priors[var] = _ae_utils.dist.Uniform(0, 1)
            else:
                mean = _ae_utils.jnp.log(value)
                std_dev = _ae_utils.jnp.log(100) / 3.0  # ±3σ spans [p0/100, p0*100]
                priors[var] = _ae_utils.dist.LogNormal(mean, std_dev)
        return priors

    _ae_utils.initialize_priors = _corrected_initialize_priors

    _AUTOEIS_PATCHED = True

# The oracle fits the SAME AutoEIS circuit as the rest of the pipeline
# (``battmap._DEFAULT_CIRCUIT``, from config/datasets.yaml) and returns its
# parameters under the canonical AutoEIS labels (``battmap.ECM_PARAM_NAMES``) —
# there is NO oracle-specific circuit alias and NO remapping step.

# CPE initial-guess seeds, keyed by the canonical AutoEIS label. Admittance seeds
# are the jones2022 training-partition medians — they keep the poorly-identified
# CPE terms near the training scale (without them the wide LogNormal prior lets
# the posterior mean wander orders of magnitude off). Exponents are typical for
# Li-ion cells at moderate SOC. Both circuits are seeded: the package's 7-param
# R1-[R2,P3]-[R4,P5] and battery-forecast's 9-param R1-P2-[R3,P4]-[R5,P6] (what its
# surrogates are trained on); the arcs carry the same values in both, P2 is the
# 9-param series CPE. Unknown labels fall back to the defaults.
_CPE_W_SEED    = {"P3w": 0.071, "P5w": 0.043,                 # 7-param arcs
                  "P2w": 7.32, "P4w": 0.071, "P6w": 0.043}    # 9-param
_CPE_N_SEED    = {"P3n": 0.80, "P5n": 0.75,                   # 7-param arcs
                  "P2n": 0.85, "P4n": 0.80, "P6n": 0.75}      # 9-param
_CPE_W_DEFAULT = 0.1
_CPE_N_DEFAULT = 0.80

# Spectrum-rescaling target ohmic resistance for the AutoEIS fit (Option B).
# The oracle's PyBaMM cell (5 Ah) has ~16x smaller impedance than the real coin
# cell (~43 mAh), which pushes the fitted CPE admittances (~1/Z) ~16x above the
# training distribution. AutoEIS is scale-equivariant (Z -> s*Z gives R -> s*R,
# Q -> Q/s, n -> n), so fitting the spectrum rescaled to this target ohmic R lands
# the admittances/exponents in distribution; resistances are divided back by s
# afterwards to report the oracle's native scale. Value = pooled training-median
# R1 (ohmic) across jones2022 featurized records. Set to None to disable.
# Rescaling does not change arc collapse (a scale-invariant property of the spectrum).
_ECM_RESCALE_TARGET_R0 = 0.1334


# ---------------------------------------------------------------------------
# ECM fitting helpers
# ---------------------------------------------------------------------------

def randles_stub_ecm(
    frequencies: np.ndarray,
    Z_real: np.ndarray,
    Z_imag: np.ndarray,
    *,
    circuit: str | None = None,
    cpe_n_seed: dict | None = None,
    cpe_n_default: float = _CPE_N_DEFAULT,
) -> np.ndarray:
    """Fast Randles stub — fallback when AutoEIS is unavailable or fails.

    Estimates the ohmic R from the high-frequency asymptote, splits the
    remaining low-frequency resistance across the circuit's [R,CPE] arcs
    (0.6/0.4 for two arcs, evenly otherwise), and seeds each CPE admittance
    from 1/(R·ω_peak). The half-vector layout is derived from *circuit*
    (default: the package circuit ``DEFAULT_CIRCUIT``) — the same labels
    :func:`autoeis_ecm` and the real-experiment featurization use — and is
    duplicated for charge and discharge.

    ``cpe_n_seed`` (default ``None`` -> the module ``_CPE_N_SEED``) supplies
    the CPE exponent seeds by label, with ``cpe_n_default`` as the fallback;
    ``PyBaMMOracle`` passes its own (YAML-overridable, ``ecm.cpe_seeds.n`` in
    config_oracle_defaults.yml) map.
    """
    circuit = circuit or _PROJECT_CIRCUIT
    cpe_n_seed = cpe_n_seed or _CPE_N_SEED
    # argmax/argmin give the highest/lowest frequency regardless of sort order
    hf_idx = np.argmax(frequencies)   # HF limit → ohmic resistance R_∞
    lf_idx = np.argmin(frequencies)   # LF limit → R_∞ + sum(R_RC)
    R_ohmic = float(Z_real[hf_idx])
    R_total_rc = max(float(Z_real[lf_idx]) - R_ohmic, 0.0)
    peak_idx = int(np.argmax(-Z_imag))
    omega_peak = 2 * np.pi * frequencies[peak_idx]

    labels = _param_labels_from_circuit(circuit)
    pairs = _randles_pairs_from_circuit(circuit)
    arc_rs = {r for r, _ in pairs}
    arc_ps = {p for _, p in pairs}
    ohmic_label = next((l for l in labels if l.startswith("R") and l not in arc_rs),
                       labels[0])
    weights = [0.6, 0.4] if len(pairs) == 2 else [1.0 / max(len(pairs), 1)] * len(pairs)

    values = {ohmic_label: R_ohmic}
    for (r, p), w in zip(pairs, weights):
        values[r] = R_total_rc * w
        values[f"{p}w"] = 1.0 / max(values[r] * omega_peak, 1e-9)
        values[f"{p}n"] = cpe_n_seed.get(f"{p}n", cpe_n_default)
    for l in labels:                                   # series CPE(s): P not in any arc
        if l.startswith("P") and l.endswith("w") and l[:-1] not in arc_ps:
            values[l] = 1.0 / max(R_ohmic * omega_peak, 1e-9)
            values[f"{l[:-1]}n"] = cpe_n_seed.get(f"{l[:-1]}n", cpe_n_default)

    half = np.array([values[l] for l in labels], dtype=np.float64)
    return np.concatenate([half, half])


def autoeis_ecm(
    frequencies: np.ndarray,
    Z_real: np.ndarray,
    Z_imag: np.ndarray,
    *,
    circuit: str | None = None,
    _diag: dict | None = None,
    return_samples: bool = False,
    cpe_w_seed: dict | None = None,
    cpe_n_seed: dict | None = None,
    cpe_w_default: float = _CPE_W_DEFAULT,
    cpe_n_default: float = _CPE_N_DEFAULT,
    rescale_target_r0: float | None = _ECM_RESCALE_TARGET_R0,
    num_warmup: int = 500,
    num_samples: int = 200,
    seed: int | None = None,
) -> np.ndarray:
    """Fit ECM parameters to an EIS spectrum using AutoEIS Bayesian inference.

    Fits *circuit* directly (default: the package circuit ``DEFAULT_CIRCUIT``,
    which mirrors the study's ``datasets.yaml default_circuit``) and returns its
    parameters under the canonical AutoEIS labels — no oracle-specific circuit
    alias and no remapping step. For the default 7-param circuit
    ``R1-[R2,P3]-[R4,P5]`` this is the half-vector
    ``[R1, R2, P3w, P3n, R4, P5w, P5n]``, duplicated for charge and discharge
    into the 14-D state. The initial guess ``p0`` is built generically from the
    circuit (ohmic R, arc [R,P] pairs, series CPEs), so any circuit works
    unchanged (e.g. battery-forecast's 9-param ``R1-P2-[R3,P4]-[R5,P6]``).

    ``cpe_w_seed``/``cpe_n_seed``/``cpe_w_default``/``cpe_n_default``/
    ``rescale_target_r0``/``num_warmup``/``num_samples`` default to the module
    constants (mirroring config_oracle_defaults.yml's ``ecm`` section);
    ``PyBaMMOracle`` passes its own (YAML-overridable) values here.

    ``seed`` is forwarded to ``autoeis.perform_bayesian_inference``'s own
    ``seed`` kwarg (its NumPyro/JAX MCMC sampler *does* accept one, despite
    this module's noise-only ``PyBaMMOracle(seed=...)`` docs elsewhere
    describing AutoEIS as unseeded — ``PyBaMMOracle`` draws a fresh int from
    its noise RNG for each fit so the two stay in the same reproducible
    stream). ``None`` (default) leaves the sampler unseeded.

    Falls back to :func:`randles_stub_ecm` if AutoEIS inference fails.
    """
    if not _AUTOEIS_AVAILABLE:
        raise ImportError(
            "autoeis is required for autoeis_ecm. "
            "Install it or use randles_stub_ecm as the ecm_model_fn."
        )
    _init_autoeis_patches()

    circuit = circuit or _PROJECT_CIRCUIT
    cpe_w_seed = cpe_w_seed or _CPE_W_SEED
    cpe_n_seed = cpe_n_seed or _CPE_N_SEED

    # Option B: rescale the spectrum to the training ohmic-R scale before fitting
    # so the CPE admittances land in the training distribution (see
    # rescale_target_r0). The fit then runs at real-cell scale; resistances
    # are restored to the oracle's native scale after fitting.
    _R0_native = float(Z_real[np.argmax(frequencies)])
    if rescale_target_r0 and _R0_native > 1e-9:
        _scale = float(rescale_target_r0) / _R0_native
    else:
        _scale = 1.0
    # Rescaled copies feed the fit; native Z_real/Z_imag are kept for the stub
    # fallback (which produces native-scale resistances directly).
    Z_real_fit = Z_real * _scale
    Z_imag_fit = Z_imag * _scale
    Z = Z_real_fit + 1j * Z_imag_fit

    R0_est = float(Z_real_fit[np.argmax(frequencies)])
    R_rc   = max(float(Z_real_fit[np.argmin(frequencies)]) - R0_est, 0.0)
    # Build p0 generically from the circuit (canonical AutoEIS labels), so it
    # works for the default circuit and any migrated one with no remapping:
    #   - ohmic resistor (the R not inside a parallel [R,P] arc) <- HF intercept
    #   - each arc resistor <- share of the RC resistance (0.6/0.4 for two arcs,
    #     else split evenly); its CPE admittance/exponent from the seed maps
    #   - any series CPE (a P not paired with an R) from the seed maps
    labels = _param_labels_from_circuit(circuit)
    pairs  = _randles_pairs_from_circuit(circuit)      # [("R3","P4"), ("R5","P6")] for C9
    arc_rs = [r for r, _ in pairs]
    arc_ps = {p for _, p in pairs}
    ohmic  = next((l for l in labels if l.startswith("R") and l not in arc_rs), "R1")
    weights = [0.6, 0.4] if len(pairs) == 2 else [1.0 / max(len(pairs), 1)] * len(pairs)
    p0 = {ohmic: max(R0_est, 1e-4)}
    for (r, p), w in zip(pairs, weights):
        p0[r] = R_rc * w
        p0[f"{p}w"] = cpe_w_seed.get(f"{p}w", cpe_w_default)
        p0[f"{p}n"] = cpe_n_seed.get(f"{p}n", cpe_n_default)
    for l in labels:                                   # series CPE(s): P not in any arc
        if l.startswith("P") and l.endswith("w") and l[:-1] not in arc_ps:
            p0[l] = cpe_w_seed.get(l, cpe_w_default)
            p0[f"{l[:-1]}n"] = cpe_n_seed.get(f"{l[:-1]}n", cpe_n_default)

    _raw_samples: dict | None = None
    _variables: list[str] = []
    try:
        # Serialise the numpyro/JAX inference: its global param-store + tracing
        # state is not thread-safe (see _AUTOEIS_INFERENCE_LOCK). Uncontended in the
        # single-threaded-per-process tune workers; guards any threaded caller.
        with _AUTOEIS_INFERENCE_LOCK:
            results = _autoeis.perform_bayesian_inference(
                circuit, frequencies, Z,
                p0=p0,
                num_warmup=num_warmup, num_samples=num_samples,
                seed=seed,
                progress_bar=False, parallel=False,
            )
        _raw_samples = results[0].samples
        _variables = list(results[0].variables)
        # Restore resistances to the oracle's native impedance scale. Admittances
        # (P*w) and exponents (P*n) are kept at the rescaled = training scale,
        # where they are in distribution; only resistors (names starting "R")
        # carry the cell-size scale and are divided back by _scale.
        if _scale != 1.0:
            for _k in _variables:
                if _k.startswith("R"):
                    _raw_samples[_k] = np.asarray(_raw_samples[_k]) / _scale
        # Emit the posterior MEDIAN, not the mean: the featurized training data
        # (jones2022-featurized-robust) summarises each parameter by its posterior
        # median, and for the heavy-tailed series-CPE admittance (P2w) the mean is
        # a ~64x chain-unstable statistic that lands far in the training tail while
        # the median is stable. Matching the training statistic keeps the oracle's
        # observed state in-distribution (fixes the P2w VarianceAlignment collapse).
        half = np.array(
            [float(np.median(_raw_samples[k])) for k in _variables],
            dtype=np.float64,
        )
        if _diag is not None:
            cvs = [
                np.std(_raw_samples[k]) / max(abs(np.median(_raw_samples[k])), 1e-10)
                for k in _variables
            ]
            _diag["max_cv"]    = float(max(cvs)) if cvs else float("nan")
            _diag["converged"] = True
            _diag["ecm_params"] = {
                "elements": _variables,
                "values":   {k: float(np.median(_raw_samples[k])) for k in _variables},
            }
    except Exception as exc:
        log.warning("[AutoEIS] inference failed (%s); using Randles stub fallback", exc)
        half = randles_stub_ecm(frequencies, Z_real, Z_imag,
                                 circuit=circuit, cpe_n_seed=cpe_n_seed)[:len(labels)]
        if _diag is not None:
            _diag["max_cv"]    = float("nan")
            _diag["converged"] = False
            _diag["ecm_params"] = None  # stub doesn't produce named element values
    finally:
        # NumPyro (via JAX/XLA) JIT-compiles a new set of LLVM kernels for each
        # inference call and never releases them in a long-running process.  After
        # ~40 iterations the XLA compilation cache exhausts available address space
        # and the process segfaults with "Cannot allocate memory" from execution_engine.
        # clear_caches() frees all compiled XLA kernels; the next call re-compiles
        # (adds ~0.5 s) but prevents memory growth.
        try:
            import jax as _jax
            _jax.clear_caches()
        except Exception:
            pass

    out = np.concatenate([half, half])
    if return_samples:
        samples_out = (
            {k: np.asarray(v) for k, v in _raw_samples.items()}
            if _raw_samples is not None else None
        )
        return out, samples_out, _variables
    return out
