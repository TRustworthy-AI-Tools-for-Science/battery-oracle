"""Battery oracle: PyBaMM SPMe simulations + EIS + ECM fitting.

Public
------
OracleFailure           — raised when simulation fails; caller should end the AL loop
PyBaMMOracle            — stateful battery oracle (SPMe → EIS → ECM)
make_pybamm_candidates  — build a 6-D protocol candidate grid for the CLI

Internal
--------
_randles_stub_ecm  — fast analytic Randles fallback (no AutoEIS required)
_autoeis_ecm       — ECM fit via AutoEIS Bayesian inference; falls back to Randles stub
"""
from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm as _pb
from scipy.signal import find_peaks as _find_peaks

from battery_oracle._circuit import (
    ACTION_FEATURE_NAMES as _ACTION_NAMES,
)
from battery_oracle._circuit import (
    DEFAULT_CIRCUIT as _PROJECT_CIRCUIT,
)
from battery_oracle._circuit import (
    ECM_PARAM_NAMES as _ECM_PARAM_NAMES,
)
from battery_oracle._circuit import (
    _param_labels_from_circuit as _param_labels_from_circuit,
)
from battery_oracle._circuit import (
    randles_pairs_from_circuit as _randles_pairs_from_circuit,
)
from battery_oracle._eis.kk import linkk_rmse as _linkk_rmse
from battery_oracle._eis.noise import (
    add_flicker_noise as _add_flicker_noise,
)
from battery_oracle._eis.noise import (
    add_relaxation_drift as _add_relaxation_drift,
)

# DRT (get_drt_impedance) needs hybrid-drt (the optional [drt] extra); it is
# imported lazily inside build_record's try/except so DRT peaks are simply
# omitted when the extra is absent.
from battery_oracle._eis.noise import (
    add_white_noise as _add_white_noise,
)
from battery_oracle._plotting import SLIPSTREAM_COLORS, slipstream

log = logging.getLogger(__name__)

# Force mpire (used by AutoEIS) to spawn worker processes instead of forking.
# On Linux the default is fork, which copies the parent's SUNDIALS/CasADi JIT
# state to children at the same virtual addresses.  When the child touches
# those addresses after the OS copy-on-write triggers, it segfaults — and the
# shared-memory corruption can propagate back, killing the parent too.  Spawn
# starts a fresh interpreter with no inherited native JIT state.
# Must run before AutoEIS is imported, as mpire is pulled in at autoeis import.
try:
    import mpire as _mpire
    _orig_wp_init = _mpire.WorkerPool.__init__
    def _spawn_wp_init(self, *args, **kwargs):
        kwargs.setdefault('start_method', 'spawn')
        _orig_wp_init(self, *args, **kwargs)
    _mpire.WorkerPool.__init__ = _spawn_wp_init
    # Note: _orig_wp_init must stay defined at module scope — _spawn_wp_init
    # looks it up via global lookup (not a closure, since it's defined here
    # at module level, not nested in an enclosing function), so deleting it
    # breaks the patched __init__ with NameError as soon as it's called.
except Exception:
    pass

try:
    import autoeis as _autoeis
    import autoeis.utils as _ae_utils
    _AUTOEIS_AVAILABLE = True
except ImportError:
    _autoeis = None
    _ae_utils = None
    _AUTOEIS_AVAILABLE = False

# Monkeypatch autoeis.utils.initialize_priors to fix excessively wide prior —
# same fix as inference.py's training-path patch (see that module for the
# full bug writeup). Without this, _autoeis_ecm's admittance terms (P0w/P1w/
# P2w below) draw from a ±12-decade log-normal and routinely converge 10-500x
# away from the training distribution; this patch was previously applied to
# the training featurization path only, never to the oracle's own AutoEIS
# call, which is the root cause of that mismatch (see
# project_oracle_p2_prior_mismatch memory / oracle-p2-prior-mismatch-fix plan).
if _AUTOEIS_AVAILABLE:
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

# The oracle fits the SAME AutoEIS circuit as the rest of the pipeline
# (``battmap._DEFAULT_CIRCUIT``, from config/datasets.yaml) and returns its
# parameters under the canonical AutoEIS labels (``battmap.ECM_PARAM_NAMES``) —
# there is NO oracle-specific circuit alias and NO remapping step.

# CPE initial-guess seeds, keyed by the canonical AutoEIS label. Admittance seeds
# are the jones2022 training-partition medians — they keep the poorly-identified
# CPE terms near the training scale (without them the wide LogNormal prior lets
# the posterior mean wander orders of magnitude off). Exponents are typical for
# Li-ion SPMe at moderate SOC (Chen2020). Unknown labels (e.g. a migrated
# circuit) fall back to the defaults.
_CPE_W_SEED    = {"P2w": 7.32, "P4w": 0.071, "P6w": 0.043}   # series P2w; arc P4w/P6w
_CPE_N_SEED    = {"P2n": 0.85, "P4n": 0.80, "P6n": 0.75}
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
# NOTE: rescaling does NOT fix the R5 arc collapse (scale-invariant) -- that is an
# SPMe single-arc fidelity limit. See project_oracle_p2_prior_mismatch memory.
_ECM_RESCALE_TARGET_R0 = 0.1334


# Reduced-order PyBaMM model, selectable via ``PyBaMMOracle(model=...)``. Applied
# identically to the cycling model (``_build_native_state``) and the internal EIS
# model (``_eis_and_correct``). All three build with both the degradation option
# dict from ``_build_degradation_config`` and the EIS ``surface form: differential``
# option (verified against PyBaMM 26.5.0).
#   SPM  — single particle, NO electrolyte (fastest, least stiff; no electrolyte overpotential)
#   SPMe — single particle + electrolyte (default; degradation presets are calibrated here)
#   DFN  — full Doyle–Fuller–Newman (most accurate, slowest, stiffest — see the numerics docs)
_MODEL_CLASSES = {
    "SPM":  _pb.lithium_ion.SPM,
    "SPMe": _pb.lithium_ion.SPMe,
    "DFN":  _pb.lithium_ion.DFN,
}


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class OracleFailure(RuntimeError):
    """Raised when the PyBaMM simulation fails and no recovery is possible.

    The caller should treat this as battery end-of-life: do not reset the
    oracle, do not substitute synthetic data — end the active learning loop.
    The last protocol that triggered the failure is stored in
    ``self.protocol`` for diagnostics.
    """
    def __init__(self, message: str, protocol: np.ndarray | None = None) -> None:
        super().__init__(message)
        self.protocol = protocol


# ---------------------------------------------------------------------------
# ECM fitting helpers
# ---------------------------------------------------------------------------

def _randles_stub_ecm(
    frequencies: np.ndarray,
    Z_real: np.ndarray,
    Z_imag: np.ndarray,
) -> np.ndarray:
    """Fast Randles stub — fallback when AutoEIS is unavailable or fails.

    Estimates R1 (ohmic), R3/R5 (charge-transfer), and CPE admittances from
    high/low-frequency asymptotes and the peak imaginary frequency.  Returns
    the same 9-parameter half-vector duplicated for charge and discharge
    (18-D state vector).
    """
    # argmax/argmin give the highest/lowest frequency regardless of sort order
    hf_idx = np.argmax(frequencies)   # HF limit → ohmic resistance R_∞
    lf_idx = np.argmin(frequencies)   # LF limit → R_∞ + sum(R_RC)
    R1 = float(Z_real[hf_idx])
    R_total_rc = max(float(Z_real[lf_idx]) - R1, 0.0)
    R3 = R_total_rc * 0.6
    R5 = R_total_rc * 0.4
    peak_idx = int(np.argmax(-Z_imag))
    omega_peak = 2 * np.pi * frequencies[peak_idx]
    P2w = 1.0 / max(R1 * omega_peak, 1e-9)
    P4w = 1.0 / max(R3 * omega_peak, 1e-9)
    P6w = 1.0 / max(R5 * omega_peak, 1e-9)
    half = np.array([R1, P2w, _CPE_N_SEED["P2n"], R3, P4w, _CPE_N_SEED["P4n"],
                     R5, P6w, _CPE_N_SEED["P6n"]], dtype=np.float64)
    return np.concatenate([half, half])


def _autoeis_ecm(
    frequencies: np.ndarray,
    Z_real: np.ndarray,
    Z_imag: np.ndarray,
    *,
    circuit: str | None = None,
    _diag: dict | None = None,
    return_samples: bool = False,
) -> np.ndarray:
    """Fit ECM parameters to an EIS spectrum using AutoEIS Bayesian inference.

    Fits the canonical project circuit directly (``battmap._DEFAULT_CIRCUIT``,
    from config/datasets.yaml) and returns its parameters under the AutoEIS
    labels ``battmap.ECM_PARAM_NAMES`` — no oracle-specific circuit alias and no
    remapping step. For the default circuit ``R1-P2-[R3,P4]-[R5,P6]`` this is the
    9-D half-vector ``[R1, P2w, P2n, R3, P4w, P4n, R5, P6w, P6n]``, duplicated for
    charge and discharge into the 18-D state. The initial guess ``p0`` is built
    generically from the circuit (ohmic R, arc [R,P] pairs, series CPEs), so a
    migrated circuit (e.g. the 7-param ``R1-[R2,P3]-[R4,P5]``) works unchanged.

    Falls back to :func:`_randles_stub_ecm` if AutoEIS inference fails.
    """
    if not _AUTOEIS_AVAILABLE:
        raise ImportError(
            "autoeis is required for _autoeis_ecm. "
            "Install it or use _randles_stub_ecm as the ecm_model_fn."
        )

    circuit = circuit or _PROJECT_CIRCUIT

    # Option B: rescale the spectrum to the training ohmic-R scale before fitting
    # so the CPE admittances land in the training distribution (see
    # _ECM_RESCALE_TARGET_R0). The fit then runs at real-cell scale; resistances
    # are restored to the oracle's native scale after fitting.
    _R0_native = float(Z_real[np.argmax(frequencies)])
    if _ECM_RESCALE_TARGET_R0 and _R0_native > 1e-9:
        _scale = float(_ECM_RESCALE_TARGET_R0) / _R0_native
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
        p0[f"{p}w"] = _CPE_W_SEED.get(f"{p}w", _CPE_W_DEFAULT)
        p0[f"{p}n"] = _CPE_N_SEED.get(f"{p}n", _CPE_N_DEFAULT)
    for l in labels:                                   # series CPE(s): P not in any arc
        if l.startswith("P") and l.endswith("w") and l[:-1] not in arc_ps:
            p0[l] = _CPE_W_SEED.get(l, _CPE_W_DEFAULT)
            p0[f"{l[:-1]}n"] = _CPE_N_SEED.get(f"{l[:-1]}n", _CPE_N_DEFAULT)

    _raw_samples: dict | None = None
    _variables: list[str] = []
    try:
        results = _autoeis.perform_bayesian_inference(
            circuit, frequencies, Z,
            p0=p0,
            num_warmup=500, num_samples=200,
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
        half = _randles_stub_ecm(frequencies, Z_real, Z_imag)[:len(labels)]
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


# ---------------------------------------------------------------------------
# Degradation configuration helper
# ---------------------------------------------------------------------------

def _build_degradation_config(
    preset: str,
    pv,
    kinetics_scale: float = 1.0,
    sei_rate_scale: float = 1.0,
    dead_li_decay_scale: float = 1.0,
    plating_rate_scale: float = 1.0,
) -> tuple[dict, object]:
    """Return (model_options, pv_modified) for the requested degradation preset.

    All presets apply a ×0.25 correction to the Chen2020 EC diffusivity, calibrated
    to match commercial CR2032 cells with FEC/VC-containing electrolytes that form a
    more passivating SEI than the LP30 baseline (Reniers et al., 2019).  Preset
    lifetimes differ only through which physical mechanisms are active.

    ``kinetics_scale`` (default 1.0, no change) multiplies both electrodes'
    exchange-current density. Chen2020 is parameterised from a high-power
    automotive NMC/graphite pouch cell; jones2022's small research coin cells
    have measurably slower charge-transfer kinetics (real R3/R1 EIS ratios are
    roughly an order of magnitude larger than the oracle's at kinetics_scale=1.0
    — see jones_oracle_study.py comparisons). Pass a value < 1 to slow the
    kinetics and grow the relative charge-transfer arc.

    ``sei_rate_scale`` (default 1.0, no change) multiplies the EC diffusivity
    correction below — additional factor on top of the existing ×0.25, so the
    final factor is ``0.25 * sei_rate_scale``. Has sharply diminishing effect on
    R1 growth once D_EC is low enough that growth becomes reaction-limited
    (confirmed empirically on PJ121: 0.5/0.25/0.1 only reduced 15-cycle R1
    growth 133%/111%/82.5%, far from linear, against a real target of ~10%).
    Scaling "SEI kinetic rate constant [m.s-1]" (the actual reaction-kinetics
    parameter for "ec reaction limited") was tried as a second lever and
    reverted — it moved R1 growth in the *wrong* direction (236% at 0.1x), so
    R1/SEI growth-rate calibration is not yet fully solved by this knob alone.
    Added because lowering
    ``kinetics_scale`` has a side effect on R1/SEI growth rate that's independent
    of its intended effect on the EIS arc shape: lower exchange-current density
    raises the electrode overpotential needed for the same applied current, and
    the SEI growth law is exponential in that overpotential (η_SEI; see
    ORACLE.md §2.1), so kinetics_scale alone *also* accelerates SEI growth as an
    unintended side effect. sei_rate_scale lets R1 growth rate be tuned back down
    independently, without re-perturbing the arc-shape calibration kinetics_scale
    was actually tuned for.

    ``"nominal"``
        SEI (``"ec reaction limited"``) + particle cracking, using Ai2020's
        unmodified critical stress (60 MPa).  PyBaMM's LAM rate is a smooth power
        law ``beta_LAM * (sigma_h/sigma_critical)^m_LAM`` with no threshold — there
        is no C-rate "onset"; sensitivity comes entirely from how sigma_hyd scales
        with C-rate, which Ai2020's stress model already provides.  No plating,
        no SEI-on-cracks.
        Target EOL: ~80–150 cycles at 1C (commercial CR2032-calibrated).

    ``"accelerated"``  *(default)*
        SEI + cracking (Ai2020 default critical stress) + partially-reversible
        lithium plating.  No "SEI on cracks": the three-way crack→SEI-on-cracks→
        plating coupling causes IDA solver stiffness (h → 0); cracking + plating
        together already provide C-rate sensitivity.
        Plating kinetic rate 1e-8 m/s, dead-Li decay 4e-6 s⁻¹.
        Target EOL: ~40–70 cycles at 1C.

    ``"severe"``
        Same model options as accelerated; faster plating (1e-7 m/s, decay 1e-4 s⁻¹).
        Target EOL: ~20–50 cycles at 1C.

    An earlier version of this function reduced critical stress to 15 MPa for all
    presets, reasoning that this would create a C-rate "activation threshold" for
    cracking.  That reasoning was wrong: the LAM rate has no threshold/switch at
    sigma_critical (it is a smooth power law), so the C-rate sensitivity RATIO
    (sigma_1C/sigma_critical)^2 / (sigma_0.3C/sigma_critical)^2 =
    (sigma_1C/sigma_0.3C)^2 is invariant to sigma_critical — it cancels out.
    Lowering sigma_critical 4x (60->15 MPa) bought zero C-rate differentiation
    while amplifying the Jacobian sensitivity d(rate)/d(sigma) ~ 1/sigma_critical^2
    by 16x, which is what was causing repeated IDA_ERR_FAIL convergence failures
    in the IDA solver.  Reverted to Ai2020 defaults for all presets.

    Physical basis: Reniers et al. (2019) for SEI; Ai et al. (2020) for crack/LAM
    parameters and the underlying sigma_hyd(C-rate) stress model; OKane et al.
    (2022) for plating kinetics.
    """

    if preset not in ("nominal", "accelerated", "severe"):
        raise ValueError(
            f"Unknown degradation_preset {preset!r}; "
            "choose 'nominal', 'accelerated', or 'severe'."
        )

    pv = copy.deepcopy(pv)

    # Universal EC diffusivity correction: commercial-grade electrolyte (FEC/VC additives)
    # forms a denser, more passivating SEI, reducing effective EC transport through the
    # film by ~4× relative to Chen2020's LP30 baseline (Reniers et al., 2019;
    # Pinson & Bazant, 2013).  Applied identically to all presets because electrolyte
    # quality is a cell property, not a degradation severity setting.
    try:
        pv["EC diffusivity [m2.s-1]"] = float(pv["EC diffusivity [m2.s-1]"]) * 0.25 * sei_rate_scale
    except Exception:
        pass

    # An attempt was made to also scale "SEI kinetic rate constant [m.s-1]" (the
    # parameter "ec reaction limited" actually reads for reaction kinetics, NOT
    # "SEI reaction exchange current density [A.m-2]" despite both existing in
    # Chen2020's ParameterValues — verified against
    # pybamm.lithium_ion.SPMe(options=opts).parameters). That attempt was
    # reverted: lowering it to 0.1x made 15-cycle R1 growth on PJ121 WORSE
    # (236% vs the EC-diffusivity-only behavior's 82.5%, against a real target of
    # ~10%) — i.e. growth got *faster*, the wrong direction, indicating SEI
    # growth rate's dependence on this PyBaMM submodel's parameters is not the
    # simple monotonic relationship assumed. sei_rate_scale therefore only
    # scales EC diffusivity for now; R1/SEI growth-rate calibration remains
    # unresolved (see TODO.md) and likely needs the PyBaMM SEI submodel
    # equations read directly rather than further blind parameter sweeps.

    # Charge-transfer kinetics correction (see kinetics_scale docstring above).
    # Wraps Chen2020's concentration-dependent exchange-current-density functions
    # rather than overwriting with a scalar, since they're functions of
    # (c_e, c_s_surf, c_s_max, T), not plain floats.
    if kinetics_scale != 1.0:
        for key in (
            "Negative electrode exchange-current density [A.m-2]",
            "Positive electrode exchange-current density [A.m-2]",
        ):
            try:
                orig_fn = pv[key]
                pv[key] = (lambda fn, s: lambda *a, **kw: fn(*a, **kw) * s)(orig_fn, kinetics_scale)
            except Exception:
                pass

    if preset == "nominal":
        # PyBaMM 26.6.2.0 regression: particle mechanics ('swelling and cracking')
        # fails at t=0 with IDAGetDky: IDA_BAD_K when the Chen2020 LGM50 graphite
        # OCP ('graphite_LGM50_ocp_Chen2020') is used. The cracking stress submodel
        # internally evaluates dOCP/dc at initial conditions; the LGM50 function
        # returns inconsistent derivatives at the initial stoichiometry that make
        # the IDA DAE initialization fail. Ai2020's OCP works; Chen2020's does not.
        # Root cause confirmed by binary search (see _build_degradation_config
        # comments, 2026-06-23): replacing only 'Negative electrode OCP [V]' with
        # Chen2020's value causes the failure; all other Ai2020 params are fine.
        # Workaround: drop 'particle mechanics' from all presets; use plating
        # kinetics for C-rate sensitivity in accelerated/severe instead.
        opts = {
            "SEI": "ec reaction limited",
        }
        _override: dict = {}

    elif preset == "accelerated":
        # SEI + plating.  Particle cracking dropped (see nominal comment above for
        # the PyBaMM 26.6.2.0 regression). Partially-reversible lithium plating
        # provides strong C-rate sensitivity: plating rate scales exponentially
        # with the local overpotential, which rises steeply at high C-rates.
        opts = {
            "SEI": "ec reaction limited",
            "lithium plating": "partially reversible",
        }
        _override = {
            "Lithium plating kinetic rate constant [m.s-1]": ("set", 1e-8 * plating_rate_scale),
            "Dead lithium decay constant [s-1]": ("set", 4e-6 * dead_li_decay_scale),
            # Zero out OKane2022's formation-cycle plated Li: the oracle starts with
            # a fresh cell.  Non-zero initial concentration strips during early cycling
            # at low C-rates and drives "Loss of capacity to ... plating" negative.
            "Initial plated lithium concentration [mol.m-3]": ("set", 0.0),
        }

    else:  # severe
        opts = {
            "SEI": "ec reaction limited",
            "lithium plating": "partially reversible",
            # Particle cracking also dropped for severe — same PyBaMM 26.6.2.0 OCP
            # regression applies here (see nominal preset comment above).
        }
        _override = {
            "Lithium plating kinetic rate constant [m.s-1]": ("set", 1e-7 * plating_rate_scale),
            "Dead lithium decay constant [s-1]": ("set", 1e-4 * dead_li_decay_scale),
            "Initial plated lithium concentration [mol.m-3]": ("set", 0.0),
        }

    # Borrow plating parameters that Chen2020 lacks from OKane2022.
    # Only copy keys that are genuinely absent so caller-supplied values win.
    _PLATING_BORROW_KEYS = [
        "Exchange-current density for plating [A.m-2]",
        "Exchange-current density for stripping [A.m-2]",
        "Initial plated lithium concentration [mol.m-3]",
        "Typical plated lithium concentration [mol.m-3]",
        "Lithium plating transfer coefficient",
        "Lithium plating kinetic rate constant [m.s-1]",
        "Dead lithium decay constant [s-1]",
        "Dead lithium decay rate [s-1]",
    ]
    try:
        pv_ref = _pb.ParameterValues("OKane2022")
        for k in _PLATING_BORROW_KEYS:
            try:
                _ = pv[k]   # already present — leave it
            except (KeyError, Exception):
                try:
                    pv[k] = pv_ref[k]
                except Exception:
                    pass
    except Exception:
        pass   # OKane2022 unavailable — plating parameters stay as-is

    # Borrow particle mechanics parameters that Chen2020 lacks from Ai2020.
    # "swelling and cracking" requires partial molar volumes, crack geometry,
    # cracking rate functions, and elastic constants — none in Chen2020.
    # "Lithium metal partial molar volume" is absent from all standard sets;
    # use 1.3e-5 m³/mol (standard value from OKane2022/Ecker2015).
    _CRACK_BORROW_KEYS = [
        "Negative electrode partial molar volume [m3.mol-1]",
        "Positive electrode partial molar volume [m3.mol-1]",
        "Negative electrode volume change",
        "Positive electrode volume change",
        "Negative electrode reference concentration for free of deformation [mol.m-3]",
        "Positive electrode reference concentration for free of deformation [mol.m-3]",
        "Negative electrode initial crack length [m]",
        "Negative electrode initial crack width [m]",
        "Negative electrode number of cracks per unit area [m-2]",
        "Negative electrode cracking rate",
        "Negative electrode activation energy for cracking rate [J.mol-1]",
        "Negative electrode critical stress [Pa]",
        "Negative electrode Young's modulus [Pa]",
        "Negative electrode Poisson's ratio",
        "Negative electrode Paris' law constant b",
        "Negative electrode Paris' law constant m",
        "Positive electrode initial crack length [m]",
        "Positive electrode initial crack width [m]",
        "Positive electrode number of cracks per unit area [m-2]",
        "Positive electrode cracking rate",
        "Positive electrode activation energy for cracking rate [J.mol-1]",
        "Positive electrode critical stress [Pa]",
        "Positive electrode Young's modulus [Pa]",
        "Positive electrode Poisson's ratio",
        "Positive electrode Paris' law constant b",
        "Positive electrode Paris' law constant m",
        "Initial SEI on cracks thickness [m]",
    ]
    try:
        pv_ai = _pb.ParameterValues("Ai2020")
        for k in _CRACK_BORROW_KEYS:
            try:
                _ = pv[k]
            except (KeyError, Exception):
                try:
                    pv[k] = pv_ai[k]
                except Exception:
                    pass
        # Not in any standard set; use known physical value
        try:
            _ = pv["Lithium metal partial molar volume [m3.mol-1]"]
        except (KeyError, Exception):
            pv["Lithium metal partial molar volume [m3.mol-1]"] = 1.3e-5
    except Exception:
        pass   # Ai2020 unavailable — particle mechanics parameters stay as-is

    for key, (op, val) in _override.items():
        try:
            if op == "multiply":
                pv[key] = float(pv[key]) * val
            else:
                pv[key] = val
        except Exception:
            pass   # key absent — skip silently

    return opts, pv


# ---------------------------------------------------------------------------
# PyBaMMOracle
# ---------------------------------------------------------------------------

class PyBaMMOracle:
    """Battery oracle using PyBaMM SPMe simulations + EIS + ECM fitting.

    The oracle is **stateful**: each call runs ``n_cycles`` charge/discharge
    cycles starting from the cell state left by the previous call, so
    degradation accumulates across active learning iterations.  Call
    :meth:`reset` between experiments to start with a fresh cell.

    If the PyBaMM simulation fails (even after an emergency retry with tighter
    solver settings), :class:`OracleFailure` is raised.  The caller should
    treat this as battery end-of-life and end the active learning loop; the
    oracle state is **not** reset automatically.

    For each queried protocol the oracle:

    1. Builds a PyBaMM ``Experiment`` from the 6-D protocol vector
       ``[C_rate_1_mA, C_rate_2_mA, dur_1_h, dur_2_h, D_rate_mA, dur_d_h]``.
    2. Runs ``n_cycles`` SPMe charge/discharge cycles, continuing from the
       cell state accumulated so far (``starting_solution``).
    3. Reads post-charge and post-discharge SOC from the cycling solution.
    4. Simulates EIS at each SOC using a separate SPMe model with
       ``surface form = differential``, then applies post-hoc ohmic corrections
       for SEI, crack-SEI, dead-Li, and LAM degradation.
    5. Calls ``ecm_model_fn(frequencies, Z_real, Z_imag)`` independently on
       the charge and discharge spectra and concatenates the two 9-D results
       into the 18-D ECM state vector.

    Parameters
    ----------
    ecm_model_fn : callable, optional
        Maps ``(frequencies, Z_real, Z_imag)`` to a shape-``(18,)`` array of
        ECM parameter means.  Defaults to :func:`_autoeis_ecm`; pass
        :func:`_randles_stub_ecm` for a fast no-AutoEIS fallback.
    n_cycles : int
        Number of charge/discharge cycles to simulate per oracle call.
    frequencies : np.ndarray, optional
        EIS frequencies in Hz (default: 60 log-spaced from 0.01 to 10 000 Hz).
    parameter_values : pybamm.ParameterValues, optional
        PyBaMM parameter set (default: Chen2020).
    model : str
        Reduced-order PyBaMM model — ``"SPMe"`` (default, single particle with
        electrolyte), ``"SPM"`` (single particle, no electrolyte overpotential),
        or ``"DFN"`` (full Doyle–Fuller–Newman: most accurate but much slower and
        numerically stiffer — see the numerical-stability docs).  Drives both the
        cycling model and the internal EIS model.  ``"SPM"``/``"DFN"`` are
        primarily for the model-comparison demonstration: the degradation presets
        are calibrated against SPMe, so a given preset may reach end-of-life on a
        different cycle count under SPM or DFN.
    degradation_preset : str
        Degradation physics preset — ``"nominal"``, ``"accelerated"`` (default),
        or ``"severe"``.  See :func:`_build_degradation_config`.
    eol_capacity_fraction : float
        SOH threshold below which :class:`OracleFailure` is raised (default 0.80).
    capacity_check : bool
        If ``True``, append a C/20 reference discharge after each cycle to
        measure usable capacity directly.  Slower but more accurate.
    temperature_K : float
        Ambient temperature in Kelvin (default 298.15 K = 25 °C).  Sets the
        PyBaMM ``"Ambient temperature [K]"`` parameter.
    """

    # Steps appended when capacity_check=True (must match _cap_check_steps()).
    _N_CAP_CHECK_STEPS: int = 4

    def __init__(
        self,
        ecm_model_fn=None,
        n_cycles: int = 1,
        frequencies: np.ndarray | None = None,
        parameter_values=None,
        model: str = "SPMe",
        degradation_preset: str = "accelerated",
        eol_capacity_fraction: float = 0.80,
        capacity_check: bool = False,
        temperature_K: float = 298.15,
        eis_noise_level: float = 0.02,
        eis_noise_model: str = "combined",
        eis_drift_scale: float = 0.0,
        eis_drift_tau_s: float = 600.0,
        eis_drift_n_periods: float = 4.0,
        real_cell_capacity_mah: float = 200.0,
        kinetics_scale: float = 1.0,
        sei_rate_scale: float = 1.0,
        dead_li_decay_scale: float = 1.0,
        plating_rate_scale: float = 1.0,
        c2_stress_scale: float = 0.0,
        c2_stress_slope_mah_per_ma: float = 0.0794,
        c2_stress_ref_ma: float = 75.27,
        dod_lam_scale: float = 0.0,
        rest_s: float = 1200.0,
        circuit: str | None = None,
        action_names: list[str] | None = None,
    ) -> None:
        self._pb = _pb
        # ECM circuit fitted to every spectrum + the protocol/action feature
        # names. Default to the package canonical circuit (battery_oracle._circuit
        # DEFAULT_CIRCUIT); a study pipeline passes its own so featurized records
        # line up. ecm_param_names are derived from the circuit (AutoEIS labels).
        self._circuit = circuit or _PROJECT_CIRCUIT
        self._ecm_param_names = _param_labels_from_circuit(self._circuit)
        self._action_names = list(action_names) if action_names else list(_ACTION_NAMES)
        # OCV rest duration (s) on each side of a cycle. Default 1200 s (20 min)
        # per Jones, Stimming & Lee 2022 (Nat. Commun. 13:4806): each cycle rests
        # 20 min at OCV in both the discharged and charged states before EIS. (The
        # CR2032 config's 10 s charged-state value is potentiostat switching
        # overhead, not a protocol rest.) See _protocol_to_experiment.
        self._rest_s = float(rest_s)
        self.ecm_model_fn = ecm_model_fn or _autoeis_ecm
        self.n_cycles = n_cycles
        # Real-cell-specific protocol scaling: jones2022 cells vary in actual
        # measured capacity (e.g. PJ121 ~42 mAh, not the ~200 mAh assumed by the
        # old hardcoded _CAP_SCALE), so this is exposed per-instance rather than
        # being a fixed class constant. Same-C-rate-fraction scaling onto the
        # Chen2020 5 Ah cell: see _protocol_to_experiment.
        self._cap_scale = 5_000.0 / float(real_cell_capacity_mah)
        self._real_cell_capacity_mah = float(real_cell_capacity_mah)

        # C_rate_2 stress term: an explicit, additive capacity-loss increment,
        # separate from the physics-based SEI/plating pathways. DISABLED BY
        # DEFAULT (c2_stress_scale=0.0) -- see below.
        #
        # The original motivation: a 4D BO search over (kinetics_scale,
        # sei_rate_scale, dead_li_decay_scale, plating_rate_scale) never got
        # the oracle's own d(fade)/d(C_rate_2) within 100x of a real-data
        # slope (0.0794 mAh/mA) fit by OLS on jones2022 variable-discharge
        # cells. That OLS slope was then found to be an artifact: the
        # per-cycle real capacity-fade values it was fit on have mean 0.097
        # mAh but std 11.0 mAh (swings to +-30 mAh on a ~42 mAh cell --
        # physically impossible as true fade, consistent with the dataset's
        # own "computed capacity does not match integral of C-rate" loader
        # warnings). Re-fit with Theil-Sen and with outlier filtering at every
        # threshold from 0.5-5 mAh of single-cycle fade (the physically
        # plausible range for this cell size), the slope is small and not
        # significant (p in [0.45, 0.77], CI spans zero). Only including the
        # implausible >10 mAh swings recovers a "significant" slope -- i.e.
        # the original C_rate_2 finding (project_crate_real_data_finding
        # memory) was driven by measurement noise, not a real relationship.
        #
        # The mechanism below is kept (it is correctly implemented: additive,
        # zero at c2_stress_ref_ma so it doesn't perturb PJ121's own
        # sei_rate_scale/arc_ratio calibration, floored at the cumulative
        # level so it cannot net-heal the battery) in case a future, properly
        # validated C-rate-dependent target is found -- e.g. from nasa/ngen,
        # or from jones2022 data cleaned of the capacity-computation artifact
        # rather than just outlier-filtered. c2_stress_scale=0.0 means this
        # term contributes nothing; set to 1.0 to use c2_stress_slope_mah_per_ma
        # as-is once such a target exists.
        self._c2_stress_scale = float(c2_stress_scale)
        self._c2_stress_slope_mah_per_ma = float(c2_stress_slope_mah_per_ma)
        self._c2_stress_ref_ma = float(c2_stress_ref_ma)
        self._cumulative_c2_stress_ah = 0.0

        # Depth-of-discharge / charge-induced-stress LAM term: an explicit,
        # additive capacity-loss pathway that makes degradation respond to DoD
        # and the (charge-rate x DoD) coupling, the lever the dynamic-cycling
        # literature identifies as dominant for NMC/graphite cells. DISABLED BY
        # DEFAULT (dod_lam_scale=0.0).
        #
        # Motivation. Li et al. 2024 (Cell Reports Physical Science 5:101891,
        # 225 NMC/Gr cells) and Smith et al. find that cycling capacity fade is
        # driven by DoD and a charge-induced stress Stress = (C_chg*DoD)^0.5
        # (a diffusion-induced-stress proxy), accumulating as sqrt of Ah
        # throughput -- NOT by C-rate alone, which is null on its own. A
        # cell-level regression on the real jones2022 variable-discharge cells
        # (AutoEIS R_ct = R3+R5 as the protocol-independent health signal;
        # discharge capacity itself is unusable because it equals the commanded
        # DoD) reproduces the CONTRAST -- DoD/stress predict R_ct growth where
        # C-rate is null -- but only modestly (rho~0.4 among the cells with a
        # clean degradation signal) because every jones2022 variable-discharge
        # cell is shallow (4-17% DoD): there is no deep-cycling arm to size the
        # magnitude from. See project_dod_lam_lever_finding memory.
        #
        # Mechanism (Smith-style sqrt-throughput law, integrated incrementally):
        #   stress_i      = sqrt( (C_rate_1/Q) * (DoD_ah_i/Q) )   [dimensionless]
        #   T            += DoD_ah_i                              [cumulative Ah]
        #   dLoss_i       = dod_lam_scale * stress_i * DoD_ah_i / (2*sqrt(T))
        # which integrates to ~ dod_lam_scale * <stress> * sqrt(T): irreversible
        # active-material loss that grows with depth and throughput and is
        # additive to delta_lli/LAM. The cumulative loss fraction is fed into the
        # EIS (it reduces the negative-electrode active-material volume fraction,
        # growing the R3+R5 charge-transfer arc) so this degradation appears in
        # the observable impedance/state vector the surrogate sees -- the way
        # real jones2022 cells age via R_ct growth -- not only in the hidden SOH.
        # This stands in for the physical particle-cracking/LAM submodel that the
        # PyBaMM 26.6.2.0 OCP regression forces off (see _build_degradation_config).
        # dod_lam_scale must be calibrated
        # to a literature magnitude or a deep-DoD dataset BEFORE enabling --
        # jones2022's shallow range cannot fix it, and fitting to its noisy
        # slope would repeat the c2_stress mistake. 0.0 = contributes nothing.
        self._dod_lam_scale = float(dod_lam_scale)
        self._cumulative_dod_lam_frac = 0.0   # dimensionless cumulative loss fraction
        self._cumulative_dod_lam_ah = 0.0     # = frac * nominal_cap (eol_loss budget)
        self._cumulative_dod_throughput_ah = 0.0
        self.frequencies = (
            frequencies if frequencies is not None else np.logspace(-2, 4, 60)
        )
        self._pv = parameter_values or _pb.ParameterValues("Chen2020")
        self.eol_capacity_fraction = eol_capacity_fraction
        self.capacity_check = capacity_check
        self._temperature_K = float(temperature_K)
        if self._temperature_K != 298.15:
            self._pv["Ambient temperature [K]"] = self._temperature_K
        self._eis_noise_level = float(eis_noise_level)
        self._eis_noise_model = eis_noise_model
        self._eis_drift_scale = float(eis_drift_scale)
        self._eis_drift_tau_s = float(eis_drift_tau_s)
        self._eis_drift_n_periods = float(eis_drift_n_periods)
        try:
            self._c20_A = float(self._pv["Nominal cell capacity [A.h]"]) / 20.0
        except Exception:
            self._c20_A = 5.0 / 20.0  # Chen2020 nominal 5 Ah → 0.25 A

        # Reduced-order model selection (SPMe default; SPM / DFN for comparison).
        # Store both the name (provenance/history) and the resolved class (applied
        # at the cycling and EIS instantiation sites). _build_degradation_config is
        # model-agnostic — the class is applied only where the model is built.
        self._model = str(model)
        if self._model not in _MODEL_CLASSES:
            raise ValueError(
                f"Unknown model {model!r}; choose 'SPMe', 'SPM', or 'DFN'."
            )
        self._model_cls = _MODEL_CLASSES[self._model]

        # Build degradation model options and (optionally) override parameters.
        # "ec reaction limited" is always the SEI base; accelerated/severe add
        # lithium plating and particle cracking on top of it.
        self._deg_opts, self._pv = _build_degradation_config(
            degradation_preset, self._pv,
            kinetics_scale=kinetics_scale, sei_rate_scale=sei_rate_scale,
            dead_li_decay_scale=dead_li_decay_scale,
            plating_rate_scale=plating_rate_scale,
        )

        # EIS model is built fresh each __call__ so degraded ParameterValues can
        # be injected for LAM before each simulation (see §4.4 ORACLE.md).

        self._build_native_state()

        self._prev_solution = None
        self._lock = threading.Lock()

        # Per-call history for diagnostic plots and EOL tracking (cleared by reset()).
        self._history: list[dict] = []
        self._last_Z: np.ndarray | None = None
        self._initial_capacity_ah: float | None = None
        self._initial_lli_ah: float | None = None  # formation-cycle LLI baseline

        # Pre-compute SEI-thickness → ohmic resistance factor (Ohm / m).
        # Specific surface area of spherical particles: a_s = 3 * eps_s / r_p
        # [Doyle, Fuller & Newman, J. Electrochem. Soc. 140(6):1526, 1993, §2.1]
        # SEI film resistance: R_SEI = delta / (sigma_SEI * a_s * L * A_cc)
        # where r_f = delta / sigma_SEI is resistance per unit interfacial area (Ohm.m2)
        # [Christensen & Newman, J. Electrochem. Soc. 151(11):A1977, 2004, eq. (5)]
        #
        # Chen2020 uses "SEI resistivity [Ohm.m]" (= 200 kΩ·m) rather than the
        # "SEI conductivity [S.m-1]" key, which does not exist in that parameter
        # set.  The previous attempt to read the conductivity always raised KeyError
        # and silently fell back to _sei_to_R = 0, disabling the EIS correction
        # entirely.  We now try the resistivity key first.
        try:
            resistivity_sei = float(self._pv["SEI resistivity [Ohm.m]"])
            sigma_sei = 1.0 / resistivity_sei
        except (KeyError, Exception):
            try:
                sigma_sei = float(self._pv["SEI conductivity [S.m-1]"])
            except Exception:
                sigma_sei = 0.0
        try:
            eps_s = float(self._pv["Negative electrode active material volume fraction"])
            r_p   = float(self._pv["Negative particle radius [m]"])
            L_neg = float(self._pv["Negative electrode thickness [m]"])
            W     = float(self._pv["Electrode width [m]"])
            H     = float(self._pv["Electrode height [m]"])
            a_s   = 3.0 * eps_s / r_p          # m⁻¹  [Doyle, Fuller & Newman 1993]
            V_el  = L_neg * W * H               # m³

            self._eps_s_nominal = eps_s         # saved for LAM a-priori injection
            self._sei_to_R      = 1.0 / (sigma_sei * a_s * V_el) if sigma_sei > 0 else 0.0
            self._dead_li_to_R  = 5e-2 / (a_s * V_el)  # ρ_dead≈0.05 Ω·m [OKane2022]

            try:
                N_cr = float(self._pv["Negative electrode number of cracks per unit area [m-2]"])
                self._crack_sei_R_base = (
                    1.0 / (sigma_sei * 2.0 * N_cr * V_el) if sigma_sei > 0 else 0.0
                )
            except Exception:
                self._crack_sei_R_base = 0.0   # Ai2020 crack density not in preset

        except Exception:
            self._eps_s_nominal    = None
            self._sei_to_R         = 0.0
            self._crack_sei_R_base = 0.0
            self._dead_li_to_R     = 0.0

    def _build_native_state(self) -> None:
        """(Re)create the cycling model and the primary/emergency IDAKLUSolvers.

        Every ``__call__`` builds a fresh ``pb.Simulation`` and rediscretizes it
        against a new ``experiment``, but historically bound this to the *same*
        ``self._cycling_model`` and solver objects for the whole oracle lifetime
        (which spans every seed and every policy in one process — hundreds of
        solves).  PyBaMM's discretization step is known to attach processed/cached
        symbols onto the model object across repeated builds, and the IDAKLU C++
        extension's internal workspace does not appear to fully release the
        previous model's compiled CasADi function handles when rebound.  Both
        accumulate over enough solves and eventually corrupt memory, segfaulting
        inside ``idaklu_solver.py:_integrate`` with no Python-catchable exception.
        Rebuilding the model + solvers periodically (call this from ``reset()``,
        i.e. once per policy) keeps native object lifetimes bounded to one
        policy's iterations instead of the whole multi-seed experiment.
        """
        self._cycling_model = self._model_cls(options=self._deg_opts)
        try:
            self._solver = self._pb.IDAKLUSolver(rtol=1e-3, atol=1e-6)
        except Exception:
            self._solver = self._pb.CasadiSolver(
                mode="safe",
                dt_max=60.0,
                rtol=1e-3,
                atol=1e-6,
            )

        # Emergency solver: deliberately a *different solver family*, not just
        # looser IDAKLU tolerances.  The CC->CV switch (charge step -> "Hold at
        # 4.1 V until C/20") reproducibly trips IDA_ERR_FAIL in IDAKLUSolver
        # regardless of rtol/atol (verified at rtol 1e-3, 1e-2, and 1e-6) — the
        # algebraic constraint at that control-mode boundary is the problem,
        # not solver precision.  CasadiSolver(mode="safe") integrates through
        # it reliably, so it is always the emergency solver here.
        self._solver_emerg = self._pb.CasadiSolver(
            mode="safe",
            dt_max=10.0,
            rtol=1e-2,
            atol=1e-5,
        )

    def reset(self) -> None:
        """Reset accumulated cell state so the next call starts with a fresh cell."""
        self._prev_solution = None
        self._history = []
        self._last_Z  = None
        self._initial_capacity_ah = None
        self._initial_lli_ah = None
        self._cumulative_c2_stress_ah = 0.0
        self._cumulative_dod_lam_frac = 0.0
        self._cumulative_dod_lam_ah = 0.0
        self._cumulative_dod_throughput_ah = 0.0
        self._build_native_state()

    # Physical bounds for the Chen2020 SPMe (nominal capacity ~5 Ah).
    # Upper C-rate: 1C = 5 000 mA; SPMe accuracy degrades above this and the
    # DAE solver diverges above ~2–3C.  Upper duration: 8 h is generous for any
    # step that also has a voltage-cutoff termination condition.
    _C_MIN_mA  = 50.0;    _C_MAX_mA  = 10_000.0  # charge step 1 / discharge (2C on 5 Ah cell)
    # charge step 2 (taper stage): real jones2022 variable-discharge cells
    # (see project_crate_real_data_finding memory) run C_rate_2 at 35-118 mA
    # raw, which scales (via _cap_scale, ~118x for these ~42 mAh cells) to
    # 4_140-13_954 mA -- within _C_MAX_mA's validated-stable 2C envelope, so
    # C_rate_2 shares that ceiling rather than an independent, lower one.
    # C_rate_2 is the one protocol dimension real data shows correlates with
    # degradation (rho=0.111, p=8e-05).
    _C2_MIN_mA = 20.0;    _C2_MAX_mA = 10_000.0  # charge step 2 (taper stage)
    _DUR_MIN_s = 60.0;    _DUR_MAX_s = 28_800.0  # 1 min – 8 h
    # Protocol cutoffs from Jones, Stimming & Lee 2022 (Nat. Commun. 13:4806),
    # Methods "Battery cycling": two-stage CC charge (<=15 min/stage) stopping at
    # 4.3 V (no CV hold), single-stage CC discharge until 3.0 V. These replace the
    # earlier 4.1 V/3.3 V + CV-hold approximation.
    _V_CHARGE_MAX   = 4.3   # charge safety cutoff (V)
    _V_DISCHARGE_MIN = 3.0  # discharge cutoff (V)
    _CHARGE_STAGE_MAX_s = 900.0  # 15 min per CC charge stage (paper limit)
    # jones2022 CR2032 coin cells vs Chen2020 SPMe (5 Ah): protocol currents are
    # scaled so the same C-rate fraction is applied to the larger cell. Default
    # assumes a ~200 mAh cell; pass real_cell_capacity_mah to __init__ for an
    # accurate per-cell scale (real measured capacity varies, e.g. PJ121 ~42 mAh).
    _CAP_SCALE = 5_000.0 / 200.0  # legacy default, kept for reference/back-compat

    def _sanitise_current(self, val_mA: float, default_mA: float,
                          lo: float, hi: float) -> float:
        """Return a finite, physically bounded current in Amperes."""
        v = float(default_mA if not np.isfinite(val_mA) else val_mA)
        return float(np.clip(v, lo, hi)) / 1000.0

    def _sanitise_duration(self, val_h: float, default_h: float) -> float:
        """Return a finite, physically bounded duration in seconds."""
        v = float(default_h if not np.isfinite(val_h) else val_h)
        return float(np.clip(v * 3600.0, self._DUR_MIN_s, self._DUR_MAX_s))

    def _cap_check_steps(self) -> tuple[str, ...]:
        """Four-step C/20 capacity check appended after regular cycling.

        Fully charges the cell at C/20 (CV hold to C/20 taper), rests, then
        discharges at C/20 to the lower voltage cutoff.  The final discharge
        capacity is the usable capacity at that point in the cell's life.
        The 25-hour (90 000 s) time limit is a safe upper bound at C/20.
        """
        c = self._c20_A
        vc, vd = self._V_CHARGE_MAX, self._V_DISCHARGE_MIN
        return (
            f"Charge at {c:.4f} A for 90000 seconds or until {vc} V",
            "Hold at {} V until C/50".format(vc),
            "Rest for 300 seconds",
            f"Discharge at {c:.4f} A for 90000 seconds or until {vd} V",
        )

    def _protocol_to_experiment(self, protocol: np.ndarray):
        pb = self._pb
        C1_mA, C2_mA, dur1_h, dur2_h, D_mA, dur_d_h = protocol
        s = self._cap_scale
        C1    = self._sanitise_current(C1_mA * s, 500.0, self._C_MIN_mA,  self._C_MAX_mA)
        C2    = self._sanitise_current(C2_mA * s, 250.0, self._C2_MIN_mA, self._C2_MAX_mA)
        D     = self._sanitise_current(D_mA  * s, 500.0, self._C_MIN_mA,  self._C_MAX_mA)
        # Discharge is governed by the 3.0 V cutoff (Jones 2022): use a generous
        # timeout (>= time to reach 3.0 V even at ~1C) rather than the real
        # duration, so the cell actually fully discharges. No 2 h floor.
        dur_d = max(self._sanitise_duration(dur_d_h, 1.0), 3_600.0)
        # Two-stage CC charge: each stage capped at 15 min (paper limit).
        dur1  = min(self._sanitise_duration(dur1_h, 0.25), self._CHARGE_STAGE_MAX_s)
        dur2  = min(self._sanitise_duration(dur2_h, 0.25), self._CHARGE_STAGE_MAX_s)
        vc, vd, rest = self._V_CHARGE_MAX, self._V_DISCHARGE_MIN, self._rest_s
        # Faithful jones2022 cycle (Jones, Stimming & Lee 2022, Nat. Commun.
        # 13:4806, Methods). Discharge-first order (cyclically equivalent to the
        # paper's discharged-start; preserves the initial_soc=0.8 first-call
        # logic). EIS is read at the relaxed post-rest states (steps[1], steps[4]).
        steps = (
            f"Discharge at {D:.4f} A for {dur_d:.0f} seconds or until {vd} V",   # [0]
            f"Rest for {rest:.0f} seconds",                                       # [1] discharged-state OCV rest -> EIS@discharged
            f"Charge at {C1:.4f} A for {dur1:.0f} seconds or until {vc} V",       # [2] CC stage 1
            f"Charge at {C2:.4f} A for {dur2:.0f} seconds or until {vc} V",       # [3] CC stage 2
            f"Rest for {rest:.0f} seconds",                                       # [4] charged-state OCV rest -> EIS@charged
        )
        # Each oracle call is ONE cycle from PyBaMM's perspective.
        # Wrapping all steps in a tuple tells PyBaMM to treat them as a single
        # cycle; a flat list would make each step a separate "cycle" and cause
        # LLI summary variables to accumulate once per step, inflating totals.
        cycles = [steps] * self.n_cycles
        if self.capacity_check:
            cycles = [steps + self._cap_check_steps()] * self.n_cycles
        # Log the sanitized step strings (not just the raw protocol) so a crash
        # in the native solver — which leaves no Python traceback through this
        # function — still shows exactly what currents/durations were fed in.
        # log.info is suppressed here: oracle.py has no explicit handler/level,
        # so its effective level falls back to logging.lastResort (WARNING+).
        # Use warning so this diagnostic actually reaches stderr before any crash.
        log.warning("[PyBaMMOracle] call %d: raw_protocol=%s steps=%s",
                     len(self._history), np.array2string(np.asarray(protocol), precision=4),
                     steps)
        return pb.Experiment(cycles)

    def __call__(self, protocol: np.ndarray) -> np.ndarray:
        pb = self._pb
        experiment = self._protocol_to_experiment(protocol)
        sim = pb.Simulation(
            self._cycling_model,
            experiment=experiment,
            parameter_values=self._pv,
            solver=self._solver,
        )
        # Lock protects _prev_solution so sequential state accumulation is safe
        # even if callers inadvertently share this oracle across threads.
        # On the first call (no prior solution), start at 80 % SOC so the initial
        # OCV (~3.9–4.0 V) sits safely inside the 4.1 V experiment voltage window.
        # Chen2020's default initial stoichiometry x_neg = 0.901 gives OCV ≈ 4.10 V,
        # which immediately triggers the Maximum-voltage infeasibility event.
        _first_call = self._prev_solution is None
        _solve_kw: dict = {"initial_soc": 0.8} if _first_call else {}
        # Expected step count for the cycle(s) this call adds — used below to
        # detect a *silent* truncation.  PyBaMM's Experiment runner catches
        # internal solver errors (e.g. IDA_ERR_FAIL at the CC->CV switch) on
        # its own callback path and returns whatever was integrated up to the
        # failure point WITHOUT raising a Python exception, so a truncated
        # cycle (e.g. the CV hold + final rest silently dropped) looks
        # identical to success from this function's perspective unless the
        # step count is checked explicitly.
        _n_cycles_added = self.n_cycles
        _expected_steps = experiment.cycle_lengths[-1]

        def _is_truncated(s) -> bool:
            if not s.cycles:
                return False
            added = s.cycles[-_n_cycles_added:] if _n_cycles_added <= len(s.cycles) else s.cycles
            return any(len(c.steps) < _expected_steps for c in added)

        with self._lock:
            try:
                sol = sim.solve(starting_solution=self._prev_solution, **_solve_kw)
                if _is_truncated(sol):
                    raise OracleFailure(
                        "Primary solver silently truncated the cycle "
                        "(fewer steps completed than requested — likely an "
                        "internal IDA_ERR_FAIL at a control-mode switch that "
                        "PyBaMM swallowed without raising)",
                        protocol=np.asarray(protocol).copy(),
                    )
            except Exception as exc1:
                log.warning(
                    "[PyBaMMOracle] primary solver failed (%s: %s); "
                    "retrying with emergency solver (same cell state)",
                    type(exc1).__name__, exc1,
                )
                # Built lazily, only on retry: every call paid for this
                # Simulation's discretization unconditionally even though the
                # vast majority of calls never need it, doubling per-call
                # memory/CPU churn for no benefit on the common path.
                sim_emerg = pb.Simulation(
                    self._cycling_model,
                    experiment=experiment,
                    parameter_values=self._pv,
                    solver=self._solver_emerg,
                )
                try:
                    sol = sim_emerg.solve(starting_solution=self._prev_solution, **_solve_kw)
                except Exception as exc2:
                    raise OracleFailure(
                        f"Battery simulation failed after two attempts: {exc2}",
                        protocol=np.asarray(protocol).copy(),
                    ) from exc2
                if _is_truncated(sol):
                    raise OracleFailure(
                        "Emergency solver also silently truncated the cycle",
                        protocol=np.asarray(protocol).copy(),
                    )
            self._prev_solution = sol

        # Following the jones2022 protocol, EIS is collected at two points per
        # cycle: (i) after the two-step charge/CV-hold (step 5) and (ii) after
        # the discharge (step 1).  Step indices are stable regardless of whether
        # capacity_check is enabled — the cap-check steps are appended at the end.
        x100 = (
            float(self._pv["Initial concentration in negative electrode [mol.m-3]"])
            / float(self._pv["Maximum concentration in negative electrode [mol.m-3]"])
        )
        # With tuple-based experiment, steps live in sol.cycles[-1].steps[i].
        # Faithful 5-step cycle (see _protocol_to_experiment):
        #   [0] discharge  [1] discharged-state rest  [2] charge CC1
        #   [3] charge CC2  [4] charged-state rest
        # EIS is read at the relaxed post-rest states: step 4 (charged), step 1
        # (discharged), matching the paper's "rest 20 min then EIS".
        last_cycle = sol.cycles[-1] if sol.cycles else None
        try:
            x_neg_chg = float(
                last_cycle.steps[4]["Average negative particle stoichiometry"].entries[-1]
            )
        except Exception:
            x_neg_chg = float(sol["Average negative particle stoichiometry"].entries[-1])
        soc_charge = float(np.clip(x_neg_chg / x100, 0.05, 0.99))

        try:
            x_neg_dis = float(
                last_cycle.steps[1]["Average negative particle stoichiometry"].entries[-1]
            )
        except Exception:
            x_neg_dis = x_neg_chg
        soc_discharge = float(np.clip(x_neg_dis / x100, 0.05, 0.99))

        # --- DoD / charge-induced-stress LAM increment (see __init__) -------
        # Computed HERE, before the EIS linearisation, so depth-of-discharge
        # damage reduces ε_s below and therefore shows up in the impedance as a
        # grown charge-transfer arc (R3+R5) — the way real jones2022 cells age —
        # rather than only in the hidden SOH. Accumulated as a dimensionless
        # cell-fraction; mapped onto nominal_cap for the eol_loss budget later.
        if self._dod_lam_scale > 0.0:
            _p = np.asarray(protocol)
            _dod_frac = max(0.0, float(_p[4]) * float(_p[5]) / self._real_cell_capacity_mah)
            _stress = float(np.sqrt(
                max(0.0, float(_p[0]) / self._real_cell_capacity_mah) * _dod_frac
            ))
            self._cumulative_dod_throughput_ah += _dod_frac
            self._cumulative_dod_lam_frac += (
                self._dod_lam_scale * _stress * _dod_frac
                / (2.0 * np.sqrt(max(self._cumulative_dod_throughput_ah, 1e-9)))
            )

        # --- LAM a priori: reduce ε_s so the EIS linearisation sees the ---
        # --- degraded electrode microstructure. [Doyle et al., 1993]    ---
        eis_pv = self._pv.copy()
        lam_frac = 0.0
        if self._eps_s_nominal is not None:
            try:
                lam_pct = float(sol["Loss of active material in negative electrode [%]"].entries[-1])
                lam_frac = float(np.clip(lam_pct / 100.0, 0.0, 0.95))
            except Exception:
                lam_frac = 0.0
            try:
                # Total active-material loss seen by the spectrum = physical
                # cracking (lam_frac, ~0 while particle mechanics is disabled)
                # plus the DoD/charge-induced-stress LAM. lam_frac alone still
                # feeds lam_cap_loss below; the DoD part is added to eol_loss
                # separately, so combining them here does not double-count.
                eff_lam = float(np.clip(lam_frac + self._cumulative_dod_lam_frac, 0.0, 0.95))
                if eff_lam > 1e-4:
                    eis_pv["Negative electrode active material volume fraction"] = (
                        self._eps_s_nominal * (1.0 - eff_lam)
                    )
            except Exception:
                pass

        # --- Pre-read state-variable degradation quantities (post-hoc shifts) ---
        # SEI film is pure ohmic [Christensen & Newman, J. Electrochem. Soc. 151:A1977, 2004]
        sei_thick       = 0.0
        crack_sei_thick = 0.0
        l_crack         = 0.0
        dead_li_thick   = 0.0

        if self._sei_to_R > 0.0:
            try:
                sei_thick = float(sol["X-averaged negative SEI thickness [m]"].entries[-1])
            except Exception:
                pass

        if self._crack_sei_R_base > 0.0:
            try:
                crack_sei_thick = float(sol["X-averaged negative SEI on cracks thickness [m]"].entries[-1])
                l_crack         = float(sol["X-averaged negative particle crack length [m]"].entries[-1])
            except Exception:
                pass

        if self._dead_li_to_R > 0.0:
            try:
                dead_li_thick = float(sol["X-averaged negative dead lithium thickness [m]"].entries[-1])
            except Exception:
                pass

        delta_R_ohmic = sei_thick * self._sei_to_R
        if l_crack > 0.0:
            delta_R_ohmic += crack_sei_thick * self._crack_sei_R_base / l_crack
        delta_R_ohmic += dead_li_thick * self._dead_li_to_R

        def _eis_and_correct(soc: float) -> np.ndarray:
            sim = _pb.EISSimulation(
                self._model_cls(options={"surface form": "differential"}),
                parameter_values=eis_pv,
            )
            Z = np.array(sim.solve(self.frequencies, initial_soc=soc)["Impedance [Ohm]"])
            if delta_R_ohmic > 0.0:
                Z = (Z.real + delta_R_ohmic) + 1j * Z.imag
            if self._eis_noise_level > 0.0 and self._eis_noise_model != "none":
                nl = self._eis_noise_level
                if self._eis_noise_model == "white":
                    Z = _add_white_noise(Z, nl)
                elif self._eis_noise_model == "flicker":
                    Z = _add_flicker_noise(self.frequencies, Z, nl)
                else:  # "combined": split budget across flicker (0.3) and white (0.1), renormalised
                    Z = _add_flicker_noise(self.frequencies, Z, noise_level=nl * 0.75)
                    Z = _add_white_noise(Z, noise_level=nl * 0.25)
            # Non-stationarity drift: EIS measured while the OCP still relaxes
            # (Hallemans et al. 2023, Eqs 40/43). Coupled to self._rest_s, so it
            # vanishes for well-rested cells; disabled at scale 0.0.
            if self._eis_drift_scale > 0.0:
                Z = _add_relaxation_drift(
                    self.frequencies, Z, self._rest_s, self._eis_drift_scale,
                    tau_relax_s=self._eis_drift_tau_s,
                    n_periods=self._eis_drift_n_periods,
                )
            return Z

        Z_charge    = _eis_and_correct(soc_charge)
        Z_discharge = _eis_and_correct(soc_discharge)

        # cap_ah: when capacity_check is enabled, read from the dedicated C/20
        # discharge step (last step) — this is sensitive to LLI and protocol.
        # When disabled, fall back to the voltage-limited discharge capacity
        # from the regular cycling step (less sensitive, but ~20× faster).
        if self.capacity_check:
            try:
                cap_ah = float(
                    last_cycle.steps[-1]["Discharge capacity [A.h]"].entries[-1]
                )
            except Exception:
                try:
                    cap_ah = float(np.max(sol["Discharge capacity [A.h]"].entries))
                except Exception:
                    cap_ah = float("nan")
        else:
            try:
                cap_ah = float(np.max(sol["Discharge capacity [A.h]"].entries))
            except Exception:
                cap_ah = float("nan")

        if self._initial_capacity_ah is None:
            self._initial_capacity_ah = cap_ah

        # EOL criterion: total irreversible lithium inventory loss from all active
        # mechanisms.  Each variable is a cumulative integrator that carries
        # forward via _prev_solution, so it reflects total loss since battery birth.
        # SEI on cracks and dead-Li plating are protocol-sensitive (higher C-rate
        # → more cracking → faster crack-SEI; higher C-rate → more plating → more
        # dead Li), so including them makes EOL respond to the chosen protocol.
        def _try_get(key: str) -> float:
            try:
                return float(sol[key].entries[-1])
            except Exception:
                return 0.0

        # "Loss of capacity to ... plating [A.h]" integrates the NET plating current
        # (plating minus stripping).  In the "partially reversible" model this can be
        # negative when stripping dominates (e.g. low C-rate cycling strips reversible
        # plated Li faster than new Li plates).  Clamp to ≥ 0: stripping returns Li to
        # the active pool — it is not a capacity gain.  The true irreversible loss from
        # plating is the dead-Li component, which trends positive over time.
        cumulative_sei_loss = (
            _try_get("Loss of capacity to negative SEI [A.h]")
            + _try_get("Loss of capacity to negative SEI on cracks [A.h]")
            + max(0.0, _try_get("Loss of capacity to negative lithium plating [A.h]"))
        )
        if cumulative_sei_loss == 0.0:
            cumulative_sei_loss = float("nan")

        # Use the parameter-set nominal capacity (5 Ah for Chen2020) rather than the
        # first-call measured capacity.  The first-call discharge may be truncated if
        # the initial SOC causes early voltage-event termination, making cap_ah ~1 Ah
        # and inflating all subsequent SOH values by ~4×.
        try:
            nominal_cap = float(self._pv["Nominal cell capacity [A.h]"])
        except Exception:
            nominal_cap = 5.0  # Chen2020 fallback

        # Zero-reference the LLI at call 1 to exclude formation-cycle loss.  PyBAMM
        # may accumulate non-zero LLI from the initial simulation even before any
        # cycling damage; subtracting this baseline keeps SOH = 1.0 for a fresh cell.
        if self._initial_lli_ah is None:
            self._initial_lli_ah = (
                cumulative_sei_loss if np.isfinite(cumulative_sei_loss) else 0.0
            )

        delta_lli = (
            max(cumulative_sei_loss - self._initial_lli_ah, 0.0)
            if np.isfinite(cumulative_sei_loss) else 0.0
        )

        # LAM (particle cracking) reduces the electrode's active surface area and
        # capacity independently of LLI.  Add the estimated capacity loss due to LAM
        # so that cracking-dominated degradation also drives EOL.
        # Conservative upper bound: full lam_frac × Q_nominal (graphite-limited cell).
        lam_cap_loss = lam_frac * nominal_cap

        # C_rate_2 stress term (see __init__ docstring comment): an explicit,
        # real-data-calibrated capacity-loss increment, additive to and
        # independent of the SEI/plating pathways above. Computed in the REAL
        # cell's own mAh/SOH-fraction terms, then mapped onto the oracle's
        # internal nominal_cap so it represents the same fractional SOH loss
        # regardless of which reference-cell scale the simulation runs at.
        #
        # The per-cycle increment is signed (negative when C_rate_2 is below
        # the reference, matching the real regression line's slope on both
        # sides) -- clamping it to >= 0 would silently zero out the entire
        # term for any C_rate_2 below the reference, which is half of the
        # real jones2022 variable-discharge range. Only the CUMULATIVE total
        # is floored at 0, so this pathway's contribution can shrink back
        # toward (but never below) "no added stress" -- it cannot make the
        # battery net-heal relative to its formation-cycle baseline.
        c2_mA_raw = float(np.asarray(protocol)[1])
        c2_stress_fraction_this_cycle = (self._c2_stress_scale
            * self._c2_stress_slope_mah_per_ma * (c2_mA_raw - self._c2_stress_ref_ma)
            / self._real_cell_capacity_mah)
        self._cumulative_c2_stress_ah = max(
            0.0, self._cumulative_c2_stress_ah + c2_stress_fraction_this_cycle * nominal_cap
        )

        # DoD / charge-induced-stress LAM term: the per-cycle increment and its
        # cumulative loss FRACTION are computed earlier (before the EIS step, so
        # it also reduces ε_s and grows R3+R5). Here we just map that dimensionless
        # fraction onto the oracle's nominal_cap for the eol_loss budget, keeping
        # it scale-consistent with delta_lli. Smith-style sqrt(throughput) law;
        # the running total only grows -> irreversible active-material loss.
        self._cumulative_dod_lam_ah = self._cumulative_dod_lam_frac * nominal_cap

        eol_loss = (
            delta_lli + lam_cap_loss
            + self._cumulative_c2_stress_ah + self._cumulative_dod_lam_ah
        )
        soh = 1.0 - eol_loss / nominal_cap

        if soh < self.eol_capacity_fraction:
            raise OracleFailure(
                f"Battery reached end-of-life: SOH {soh:.1%} < "
                f"{self.eol_capacity_fraction:.0%} threshold "
                f"(LLI {delta_lli:.3f} Ah + LAM {lam_cap_loss:.3f} Ah "
                f"+ C2-stress {self._cumulative_c2_stress_ah:.3f} Ah "
                f"+ DoD-LAM {self._cumulative_dod_lam_ah:.3f} Ah "
                f"= {eol_loss:.3f} Ah = {eol_loss / nominal_cap:.1%} of nominal)",
                protocol=np.asarray(protocol).copy(),
            )

        self._history.append({
            "call_idx":           len(self._history),
            "model":              self._model,
            "protocol":           np.asarray(protocol).copy(),
            "sei_thickness_nm":   sei_thick * 1e9 if self._sei_to_R > 0.0 else float("nan"),
            "crack_sei_nm":       crack_sei_thick * 1e9,
            "dead_li_nm":         dead_li_thick * 1e9,
            "l_crack_nm":         l_crack * 1e9,
            "lam_pct":            lam_frac * 100.0,
            "lam_cap_loss_ah":    lam_cap_loss,
            "end_soc_charge":     soc_charge,
            "end_soc_discharge":  soc_discharge,
            "end_soh":            soh,
            "capacity_ah":              cap_ah,
            "cumulative_sei_loss_ah":   _try_get("Loss of capacity to negative SEI [A.h]"),
            "cumulative_crack_sei_ah":  _try_get("Loss of capacity to negative SEI on cracks [A.h]"),
            "cumulative_plating_ah":    _try_get("Loss of capacity to negative lithium plating [A.h]"),
            "cumulative_lli_total_ah":  cumulative_sei_loss,
            "delta_lli_ah":             delta_lli,
            "cumulative_c2_stress_ah":  self._cumulative_c2_stress_ah,
            "cumulative_dod_lam_ah":    self._cumulative_dod_lam_ah,
            "cumulative_dod_throughput_ah": self._cumulative_dod_throughput_ah,
            # EIS spectra at this call, jones2022 raw-data convention
            # (freq/Hz, Re(Z)/Ohm, -Im(Z)/Ohm) so a Nyquist plot is just
            # ax.plot(Z_charge_real, Z_charge_neg_imag).
            "Z_charge_real":        Z_charge.real.copy(),
            "Z_charge_neg_imag":    -Z_charge.imag.copy(),
            "Z_discharge_real":     Z_discharge.real.copy(),
            "Z_discharge_neg_imag": -Z_discharge.imag.copy(),
        })
        self._last_Z = Z_charge.copy()

        # ── linKK and DRT use the post-charge spectrum (primary measurement) ────
        linkk_rmse = _linkk_rmse(self.frequencies, Z_charge)

        drt_peaks = np.array([], dtype=float)
        try:
            from battery_oracle._eis.drt import get_drt_impedance
            drt_out, _, _, _ = get_drt_impedance(
                (self.frequencies,), (Z_charge,), (None,), trim=False
            )
            _tau   = drt_out[0][:, 0]
            _gamma = drt_out[0][:, 1]
            _idx, _ = _find_peaks(_gamma)
            if len(_idx):
                drt_peaks = _tau[_idx]
        except Exception:
            pass

        # ── ECM fit: charge and discharge spectra fitted independently ────────
        # ecm_model_fn(frequencies, Z_real, Z_imag) → 18-D (legacy full-vector);
        # take first 9 elements from each call and concatenate.
        _n_params = len(self._ecm_param_names)

        def _fit_half(Z: np.ndarray, _diag: dict) -> np.ndarray:
            if self.ecm_model_fn is _autoeis_ecm:
                full, raw_samples, raw_variables = _autoeis_ecm(
                    self.frequencies, Z.real, Z.imag,
                    circuit=self._circuit, _diag=_diag, return_samples=True,
                )
                _diag["_raw_samples"] = raw_samples
                _diag["_raw_variables"] = raw_variables
            else:
                full = self.ecm_model_fn(self.frequencies, Z.real, Z.imag)
            return full[:_n_params]

        _ecm_diag_chg: dict = {}
        _ecm_diag_dis: dict = {}
        params_charge    = _fit_half(Z_charge,    _ecm_diag_chg)
        params_discharge = _fit_half(Z_discharge, _ecm_diag_dis)
        state = np.concatenate([params_charge, params_discharge])

        self._history[-1].update({
            "linkk_rmse":          linkk_rmse,
            "max_cv":              _ecm_diag_chg.get("max_cv",    float("nan")),
            "converged":           _ecm_diag_chg.get("converged", True),
            "ecm_params":          _ecm_diag_chg.get("ecm_params", None),
            "drt_peaks":           drt_peaks,
            "ecm_params_charge":   params_charge.copy(),
            "ecm_params_discharge": params_discharge.copy(),
            # Raw AutoEIS MCMC posterior samples behind ecm_params_{charge,discharge}
            # above (dict: AutoEIS variable name -> (num_samples,) array), so callers
            # comparing oracle vs. real posterior distributions (e.g.
            # jones_oracle_study.py) don't need to refit — None if AutoEIS fell back
            # to the Randles stub for that state.
            "ecm_samples_charge":      _ecm_diag_chg.get("_raw_samples"),
            "ecm_samples_discharge":   _ecm_diag_dis.get("_raw_samples"),
            "ecm_variables_charge":    _ecm_diag_chg.get("_raw_variables", []),
            "ecm_variables_discharge": _ecm_diag_dis.get("_raw_variables", []),
        })
        return state

    # ── CSV export ───────────────────────────────────────────────────────────

    @staticmethod
    def save_to_csv(
        history: list[dict],
        out_path: str | Path,
        cell_id: str = "oracle_sim",
    ) -> Path:
        """Save oracle history to CSV matching the jones2022 featurized record format.

        Columns: cell_id, cycle, circuit,
        {param}_{state}_{moment} × 72 (9 params × 4 moments × 2 states),
        {action} × 6.

        Variance, kurtosis, and skew are filled with 0.0 because the oracle
        provides only posterior means from the ECM fit.

        Parameters
        ----------
        history :
            Oracle call history (``oracle._history`` or a per-policy copy).
        out_path :
            Destination CSV path.  Parent directories are created if absent.
        cell_id :
            Value written to the ``cell_id`` column.

        Returns
        -------
        Path
            The path that was written.
        """
        # Sequence-level symmetric-arc flip correction, matching the tsdatagen
        # featurization. The per-call ecm_params means cannot be corrected at
        # call time (the swap is only resolvable across the whole cycle list), so
        # recompute corrected means here from the retained MCMC samples. Falls
        # back to the raw ecm_params when samples were not retained (e.g. exp2
        # strips them for memory).
        from battery_oracle._circuit import fix_parameter_flips_dicts
        corrected: dict[tuple[int, str], np.ndarray] = {}
        for state_label, samples_key, vars_key in (
            ("charge",    "ecm_samples_charge",    "ecm_variables_charge"),
            ("discharge", "ecm_samples_discharge", "ecm_variables_discharge"),
        ):
            dicts, idxs, var_lists = [], [], []
            for j, h in enumerate(history):
                s = h.get(samples_key)
                if s:
                    dicts.append({k: np.asarray(v).copy() for k, v in s.items()})
                    idxs.append(j)
                    var_lists.append(h.get(vars_key) or list(s.keys()))
            if len(dicts) >= 2:
                fix_parameter_flips_dicts(dicts, _PROJECT_CIRCUIT)
                for d, j, variables in zip(dicts, idxs, var_lists):
                    corrected[(j, state_label)] = np.array(
                        [float(np.median(d[v])) for v in variables]
                    )

        rows = []
        for hj, h in enumerate(history):
            row: dict = {
                "cell_id": cell_id,
                "cycle":   h["call_idx"],
                "circuit": _PROJECT_CIRCUIT,
            }
            for state_label, key in (
                ("charge",    "ecm_params_charge"),
                ("discharge", "ecm_params_discharge"),
            ):
                params = corrected.get((hj, state_label))
                if params is None:
                    params = h.get(key, np.zeros(len(_ECM_PARAM_NAMES)))
                for i, pname in enumerate(_ECM_PARAM_NAMES):
                    v = float(params[i]) if i < len(params) else 0.0
                    row[f"{pname}_{state_label}_mean"]     = v
                    row[f"{pname}_{state_label}_var"]      = 0.0
                    row[f"{pname}_{state_label}_kurtosis"] = 0.0
                    row[f"{pname}_{state_label}_skew"]     = 0.0
            proto = h.get("protocol", np.zeros(6))
            for i, col in enumerate(_ACTION_NAMES):
                row[col] = float(proto[i]) if i < len(proto) else 0.0
            rows.append(row)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        return out_path

    # ── Diagnostic plots ─────────────────────────────────────────────────────

    def plot_diagnostics(
        self,
        output_dir: str | Path,
        label: str = "",
        real_soh: np.ndarray | None = None,
    ) -> None:
        """Save diagnostic figures for the current oracle run.

        * ``last_cycle.png``   — terminal voltage + current, one trace per cycle
        * ``capacity_sei.png`` — stacked area of cumulative capacity loss by mechanism
        * ``soh.png``          — state of health vs oracle call (oracle only, unless
          ``real_soh`` is given)

        When ``label`` is non-empty, filenames are prefixed with ``{label}_``.


        Parameters
        ----------
        output_dir : str or Path
            Directory in which to save the figures (created if absent).
        label : str
            Prefix for all figure filenames.
        real_soh : np.ndarray, optional
            Real-battery SOH per call, same length/order as ``self._history``
            (one value per oracle call so far). When given, overlaid on
            ``soh.png`` alongside the oracle's own SOH trajectory. Callers
            comparing against real cell data (e.g. jones_oracle_study.py) can
            pass this; other oracle consumers have no real-cell SOH and should
            leave it as ``None``.
        """
        plt.rcParams.update({
            "font.family":       "serif",
            "font.size":         10,
            "axes.titlesize":    11,
            "axes.labelsize":    10,
            "xtick.labelsize":   9,
            "ytick.labelsize":   9,
            "legend.fontsize":   9,
            "lines.linewidth":   1.5,
            "axes.linewidth":    0.8,
            "axes.grid":         False,
            "figure.dpi":        300,
            "axes.spines.top":   True,
            "axes.spines.right": True,
        })

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefix = f"{label}_" if label else ""
        colors = SLIPSTREAM_COLORS

        # ── Raw diagnostics CSV ───────────────────────────────────────────────
        if self._history:
            try:
                rows = []
                for h in self._history:
                    row = {
                        k: v for k, v in h.items()
                        if k not in (
                            "protocol", "Z_charge_real", "Z_charge_neg_imag",
                            "ecm_samples_charge", "ecm_samples_discharge",
                            "ecm_variables_charge", "ecm_variables_discharge",
                        )
                    }
                    proto = np.asarray(h.get("protocol", []))
                    for i, col in enumerate(self._action_names):
                        row[col] = float(proto[i]) if i < len(proto) else float("nan")
                    rows.append(row)
                pd.DataFrame(rows).to_csv(
                    out / f"{prefix}diagnostics.csv", index=False, float_format="%.6g"
                )
            except Exception as exc:
                log.warning("[plot_diagnostics] diagnostics CSV failed: %s", exc)

        # ── Plot 1: voltage and current — one trace per cycle, waterfall offset ─
        if self._prev_solution is not None:
            try:
                sol    = self._prev_solution
                cycles = sol.cycles if sol.cycles else [sol]
                n_cyc  = len(cycles)
                cmap   = slipstream

                # Compute offset from the full voltage range so it scales
                # with whatever protocol was run.
                V_all = sol["Terminal voltage [V]"].entries
                I_all = sol["Current [A]"].entries
                v_step = 0.15 * (V_all.max() - V_all.min()) if n_cyc > 1 else 0.0
                i_step = 0.15 * (I_all.max() - I_all.min()) if n_cyc > 1 else 0.0

                fig1, ax1 = plt.subplots(1, 1, figsize=(3.5, 3.5))
                fig2, ax2 = plt.subplots(1, 1, figsize=(3.5, 3.5))

                for i, cyc in enumerate(cycles):
                    t = cyc["Time [s]"].entries
                    V = cyc["Terminal voltage [V]"].entries
                    I = cyc["Current [A]"].entries
                    t_rel = (t - t[0]) / 3600.0          # hours, reset per cycle
                    color = cmap(i / max(n_cyc - 1, 1))
                    alpha = max(0.4, 1.0 - 0.06 * (n_cyc - 1 - i))
                    ax1.plot(t_rel, V + i * v_step, color=color, alpha=alpha)
                    ax2.plot(t_rel, I + i * i_step, color=color, alpha=alpha)

                ax1.set_box_aspect(1)
                ax1.set_ylabel("Terminal voltage (V)")
                ax1.set_xlabel("Time (h)")

                ax2.set_box_aspect(1)
                ax2.set_ylabel("Current (A)")
                ax2.set_xlabel("Time (h)")

                if n_cyc > 1:
                    sm = plt.cm.ScalarMappable(
                        cmap=cmap,
                        norm=plt.Normalize(vmin=0, vmax=n_cyc - 1),
                    )
                    fig1.colorbar(sm, cax=ax1.inset_axes([1.04, 0, 0.05, 1.0]), label="Cycle")
                    fig2.colorbar(sm, cax=ax2.inset_axes([1.04, 0, 0.05, 1.0]), label="Cycle")

                fig1.tight_layout()
                fig1.savefig(out / f"{prefix}last_cycle_V.png",
                            dpi=300, bbox_inches="tight", transparent=True)
                plt.close(fig1)

                fig2.tight_layout()
                fig2.savefig(out / f"{prefix}last_cycle_I.png",
                            dpi=300, bbox_inches="tight", transparent=True)
                plt.close(fig2)
            except Exception as exc:
                log.warning("[plot_diagnostics] last_cycle plot failed: %s", exc)

        # ── Plot 2: cumulative capacity loss by degradation mechanism ────────
        if self._history:
            try:
                calls = [h["call_idx"] for h in self._history]

                def _arr(key: str) -> np.ndarray:
                    a = np.array(
                        [h.get(key, float("nan")) for h in self._history],
                        dtype=float,
                    )
                    np.nan_to_num(a, nan=0.0, copy=False)
                    return a

                sei_ah      = _arr("cumulative_sei_loss_ah")
                crack_ah    = _arr("cumulative_crack_sei_ah")
                plating_ah  = _arr("cumulative_plating_ah")
                lam_ah      = _arr("lam_cap_loss_ah")

                # Only show mechanisms that are active in this oracle's degradation
                # config — e.g. SEI on cracks is absent for nominal/accelerated presets.
                _deg = self._deg_opts
                _all = [
                    (sei_ah,     "SEI",           True),
                    (crack_ah,   "SEI on cracks", "SEI on cracks" in _deg),
                    (plating_ah, "Li plating",    "lithium plating" in _deg),
                    (lam_ah,     "LAM",           "particle mechanics" in _deg),
                ]
                active_arrs   = [a for a, _, flag in _all if flag]
                active_labels = [l for _, l, flag in _all if flag]
                n_active = len(active_arrs)
                stack_colors = [slipstream(v) for v in np.linspace(0.1, 0.9, max(n_active, 1))]

                fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
                ax.stackplot(
                    calls,
                    *active_arrs,
                    labels=active_labels,
                    colors=stack_colors,
                    alpha=0.85,
                )
                ax.set_box_aspect(1)
                ax.set_ylabel("Cumulative capacity loss (A·h)")
                ax.set_xlabel("Oracle call")
                ax.legend(frameon=False, loc="upper left")

                fig.tight_layout()
                fig.savefig(out / f"{prefix}capacity_sei.png",
                            dpi=300, bbox_inches="tight", transparent=True)
                plt.close(fig)
            except Exception as exc:
                log.warning("[plot_diagnostics] capacity_sei plot failed: %s", exc)

            try:
                soh_history = np.array(
                    [h.get("end_soh", float("nan")) for h in self._history],
                    dtype=float,
                )
                fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
                ax.plot(calls, soh_history, color=colors[0], marker="o", ms=4,
                        fillstyle="none", label="Oracle" if real_soh is not None else None)
                nan_mask = ~np.isfinite(soh_history)
                if nan_mask.any():
                    ax.scatter(np.array(calls)[nan_mask], np.zeros(nan_mask.sum()),
                               marker="x", color=SLIPSTREAM_COLORS[5], zorder=5, s=40)
                if real_soh is not None:
                    real_soh_arr = np.asarray(real_soh, dtype=float)
                    ax.plot(
                        calls[:len(real_soh_arr)], real_soh_arr, color=SLIPSTREAM_COLORS[5],
                        marker="s", ms=4, fillstyle="none", linestyle="--", label="Real",
                    )
                ax.axhline(self.eol_capacity_fraction, color="grey", lw=0.8,
                           linestyle="--", label="EOL threshold")
                ax.set_box_aspect(1)
                ax.set_ylabel("State of Health")
                ax.set_xlabel("Oracle call")
                ax.legend(frameon=False)

                fig.tight_layout()
                fig.savefig(out / f"{prefix}soh.png", dpi=300, bbox_inches="tight", transparent=True)
                plt.close(fig)
            except Exception as exc:
                log.warning("[plot_diagnostics] soh plot failed: %s", exc)

        # ── Plot 4: EIS Nyquist spectra over time, jones2022 raw-data style ───
        # (Re(Z), -Im(Z)) per oracle call, colored by call index — same
        # waterfall/colorbar convention as the last_cycle V/I plots above.
        if self._history and "Z_charge_real" in self._history[0]:
            try:
                n_calls = len(self._history)
                cmap = slipstream
                fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
                for h in self._history:
                    color = cmap(h["call_idx"] / max(n_calls - 1, 1))
                    ax.plot(
                        h["Z_charge_real"], h["Z_charge_neg_imag"],
                        color=color, marker="o", ms=3, fillstyle="none", lw=1.0,
                    )
                ax.set_xlabel("Re(Z) (Ω)")
                ax.set_ylabel("−Im(Z) (Ω)")
                ax.set_aspect("equal", adjustable="datalim")

                if n_calls > 1:
                    sm = plt.cm.ScalarMappable(
                        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_calls - 1),
                    )
                    fig.colorbar(sm, cax=ax.inset_axes([1.04, 0, 0.05, 1.0]), label="Oracle call")

                fig.tight_layout()
                fig.savefig(out / f"{prefix}eis_nyquist.png",
                            dpi=300, bbox_inches="tight", transparent=True)
                plt.close(fig)
            except Exception as exc:
                log.warning("[plot_diagnostics] eis_nyquist plot failed: %s", exc)


# ---------------------------------------------------------------------------
# Candidate grid
# ---------------------------------------------------------------------------

def make_pybamm_candidates(
    c_rate_min_mA: float = 68.0,
    c_rate_max_mA: float = 140.0,
    n_candidates: int = 15,
    d_rate_mA: float = 100.0,
    d_dur_h: float = 1.0,
    dur_h: float = 1.0,
) -> list[np.ndarray]:
    """Build a 6-D protocol candidate grid varying only the first charge current.

    Protocol layout matches :class:`PyBaMMOracle`::

        [C_rate_1_mA, C_rate_2_mA, dur_1_h, dur_2_h, D_rate_mA, dur_d_h]

    The second charge stage is set to half the first (two-step taper); the
    discharge stage uses ``d_rate_mA`` and ``d_dur_h``.

    Parameters
    ----------
    c_rate_min_mA, c_rate_max_mA : float
        Range of first-stage charge current to sweep (mA).
    n_candidates : int
        Number of points in the grid.
    d_rate_mA : float
        Fixed discharge current (mA).
    d_dur_h : float
        Fixed discharge duration (h).
    dur_h : float
        Fixed first-stage charge duration (h); second stage gets ``dur_h / 2``.
    """
    c_rates = np.linspace(c_rate_min_mA, c_rate_max_mA, n_candidates)
    return [
        np.array([c, c * 0.5, dur_h, dur_h * 0.5, d_rate_mA, d_dur_h], dtype=np.float64)
        for c in c_rates
    ]
