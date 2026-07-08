"""Kramers-Kronig (KK) transform pipeline for EIS data validation.

Vendored from ``battery_forecast/eis/kk.py``. The only function the oracle calls
is :func:`linkk_rmse` (uses ``impedance.py``, a core dependency). The
``autoeis``-dependent :func:`ecm_kk_validate` is retained for completeness but
imports ``autoeis`` lazily so this module loads without the ``[autoeis]`` extra.

Implements the three-step procedure from Luo et al. (2021):
  1. kk_interpolate  - cubic spline onto a uniform log-spaced grid
  2. kk_extrapolate  - ECM-based extension beyond the measured range
  3. kk_transform    - numerical KK integral with singularity regularisation
  4. kk_residuals    - normalised residuals at the original measurement grid
"""

import numpy as np
from scipy.interpolate import CubicSpline as _CubicSpline


def kk_interpolate(freq, Z, n_per_decade=20):
    """Cubic spline onto a uniform log-spaced grid (Luo et al. 2021, Step 1).

    Dense, evenly-spaced sampling improves the numerical accuracy of the KK
    quadrature by eliminating irregular spacing in the original measurement grid.
    """
    freq = np.asarray(freq, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    idx = np.argsort(freq)
    freq, Z = freq[idx], Z[idx]

    log_f = np.log10(freq)
    n_pts = max(2, int(round((log_f[-1] - log_f[0]) * n_per_decade)) + 1)
    log_f_i = np.linspace(log_f[0], log_f[-1], n_pts)

    Z_i = _CubicSpline(log_f, Z.real)(log_f_i) + 1j * _CubicSpline(log_f, Z.imag)(log_f_i)
    return 10.0 ** log_f_i, Z_i


def kk_extrapolate(freq_interp, Z_interp, circuit_fn, params, decades=4, n_per_decade=20):
    """ECM-based extension beyond the measured range (Luo et al. 2021, Step 2).

    Truncation of the KK integral at finite frequency limits introduces systematic
    error. Evaluating the fitted ECM outside the measurement window gives a
    physically consistent extrapolation that makes the error negligible.
    4 decades each side is needed for series-CPE circuits where Z_re diverges at DC.
    """
    freq_interp = np.asarray(freq_interp, dtype=float)
    f_min, f_max = freq_interp[0], freq_interp[-1]
    n_ext = int(decades * n_per_decade)

    f_lo = np.logspace(np.log10(f_min) - decades, np.log10(f_min), n_ext + 1)[:-1]
    f_hi = np.logspace(np.log10(f_max), np.log10(f_max) + decades, n_ext + 1)[1:]

    Z_lo = np.asarray(circuit_fn(f_lo, params), dtype=complex)
    Z_hi = np.asarray(circuit_fn(f_hi, params), dtype=complex)

    freq_full = np.concatenate([f_lo, freq_interp, f_hi])
    Z_full = np.concatenate([Z_lo, Z_interp, Z_hi])
    return freq_full, Z_full


def kk_transform(freq, Z, a=1e-4):
    """Numerical KK transform with ai singularity regularisation (Luo et al. 2021, Step 3).

    Sign convention (engineering, Im(Z) < 0 for capacitive):
        Z_re(ω) - Z_re(∞) = -(2/π) Re[Σ (ω_j Z_im(ω_j) - ω Z_im(ω)) / ((ω_j+ai)²-ω²) Δω_j]
        Z_im(ω)            = +(2/π) Re[Σ ω·(Z_re(ω_j) - Z_re(ω))  / ((ω_j+ai)²-ω²) Δω_j]

    The ai shift moves the Cauchy pole off the real axis; the subtracted-numerator
    form removes the integrable singularity so no principal-value treatment is needed.
    Integration weights: trapezoidal on log(ω), Δω_j = ω_j · Δ(ln ω_j).
    """
    freq = np.asarray(freq, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    idx = np.argsort(freq)
    freq, Z = freq[idx], Z[idx]

    omega = 2.0 * np.pi * freq
    Z_re, Z_im = Z.real, Z.imag
    Z_inf = Z_re[-1]  # high-frequency limit ≈ ohmic resistance

    dw = np.gradient(np.log(omega)) * omega  # Δω_j = ω_j Δ(ln ω_j)

    om_q = omega[:, None]
    om_s = omega[None, :]
    dw_s = dw[None, :]
    zre_s, zre_q = Z_re[None, :], Z_re[:, None]
    zim_s, zim_q = Z_im[None, :], Z_im[:, None]

    denom = (om_s + a * 1j) ** 2 - om_q ** 2

    Z_re_kk = Z_inf - (2.0 / np.pi) * np.sum(
        np.real((om_s * zim_s - om_q * zim_q) / denom) * dw_s, axis=1
    )
    Z_im_kk = (2.0 / np.pi) * np.sum(
        np.real(om_q * (zre_s - zre_q) / denom) * dw_s, axis=1
    )
    return Z_re_kk + 1j * Z_im_kk


def kk_residuals(freq_meas, Z_meas, freq_full, Z_kk_full):
    """Normalised KK residuals at the measurement frequencies (Luo et al. 2021, eqs. 14-16).

    Returns a dict with scalar E_re, E_im, E and per-frequency arrays res_re, res_im,
    plus Z_kk_at_meas (KK prediction interpolated back to measurement grid).
    """
    freq_meas = np.asarray(freq_meas, dtype=float)
    Z_meas = np.asarray(Z_meas, dtype=complex)

    idx = np.argsort(freq_full)
    log_f_full = np.log10(np.asarray(freq_full, dtype=float)[idx])
    Z_kk_s = np.asarray(Z_kk_full, dtype=complex)[idx]

    log_f_meas = np.log10(freq_meas)
    Z_kk_meas = (
        _CubicSpline(log_f_full, Z_kk_s.real)(log_f_meas)
        + 1j * _CubicSpline(log_f_full, Z_kk_s.imag)(log_f_meas)
    )

    res_re = np.abs((Z_kk_meas.real - Z_meas.real) / Z_meas.real)
    res_im = np.abs((Z_kk_meas.imag - Z_meas.imag) / Z_meas.imag)

    E_re = float(np.sqrt(np.mean(res_re ** 2)))
    E_im = float(np.sqrt(np.mean(res_im ** 2)))

    return {
        "E_re": E_re,
        "E_im": E_im,
        "E": float(np.sqrt(E_re ** 2 + E_im ** 2)),
        "Z_kk_at_meas": Z_kk_meas,
        "res_re": res_re,
        "res_im": res_im,
    }


def kk_pipeline(freq, Z, circuit_fn, params, decades=20, n_per_decade=50, a=1e-6):
    """Run the full KK pipeline: interpolate → ECM-extrapolate → transform → residuals.

    Convenience wrapper around the four individual steps. Returns the kk_residuals dict
    (keys: E_re, E_im, E, Z_kk_at_meas, res_re, res_im).
    """
    freq_i, Z_i = kk_interpolate(freq, Z, n_per_decade=n_per_decade)
    freq_full, Z_full = kk_extrapolate(freq_i, Z_i, circuit_fn, params, decades=decades)
    Z_kk_full = kk_transform(freq_full, Z_full, a=a)
    return kk_residuals(freq, Z, freq_full, Z_kk_full)


def linkk_residuals(freq, Z, c: float = 0.85, max_M: int = 50):
    """Run impedance.py's linKK on one spectrum; return ``(freq, res_real, res_imag)``.

    The shared core under :func:`linkk_rmse` and the tune engine's low/high-ratio
    drift metric: sanitizes inputs (finite, positive frequency, sorted ascending),
    applies the NumPy-2.x eval-builder fix (NumPy 2.x changed scalar repr to
    "np.float64(...)", which breaks impedance.py's eval()-based circuit builder),
    and suppresses linKK's stdout chatter. Uses impedance.py (numpy-based) so it
    is safe to call from multiple threads alongside a JAX-based inference run.

    Returns ``None`` on failure or when fewer than 8 usable points remain.
    """
    import contextlib
    import io

    import impedance.models.circuits.elements as _ce
    from impedance.validation import linKK
    _ce.circuit_elements.setdefault("np", np)

    f = np.asarray(freq, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    m = np.isfinite(f) & np.isfinite(Z.real) & np.isfinite(Z.imag) & (f > 0)
    f, Z = f[m], Z[m]
    if len(f) < 8:
        return None
    o = np.argsort(f)
    f, Z = f[o], Z[o]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, _Z_fit, res_real, res_imag = linKK(f, Z, c=c, max_M=max_M)
    except Exception:
        return None
    return f, np.asarray(res_real), np.asarray(res_imag)


def linkk_rmse(freq, Z, c: float = 0.85, max_M: int = 50):
    """Compute linKK reconstruction RMSE for one spectrum.

    ``c``/``max_M`` default to the values documented in
    config_oracle_defaults.yml's ``eis.linkk`` section; ``PyBaMMOracle``
    passes its own (YAML-overridable) ``linkk_c``/``linkk_max_M`` here.

    Returns the RMS of the relative residuals as a plain float, or nan on
    any failure.
    """
    out = linkk_residuals(freq, Z, c=c, max_M=max_M)
    if out is None:
        return float("nan")
    _, res_real, res_imag = out
    return float(np.sqrt(np.mean(res_real ** 2 + res_imag ** 2)))


def ecm_kk_validate(
    mcmc,
    circuit,
    freq_meas,
    Z_meas,
    decades=20,
    n_per_decade=50,
    kk_a=1e-6,
):
    """KK self-consistency check and measurement validation using a fitted ECM.

    Extracts posterior median parameters from MCMC samples, then runs a two-stage
    KK pipeline:
      1. ECM self-consistency: applies KK to the ECM over a wide frequency grid to
         establish the numerical floor. A series-CPE circuit causes Z_re to diverge
         at DC, so err_im is inherently 5-10% even with 20 decades.
      2. Measurement validation: interpolates and ECM-extrapolates the measured data,
         applies KK, and returns normalised residuals.

    Requires the ``[autoeis]`` extra (imported lazily here).
    """
    import autoeis as ae

    # Extract median params from MCMC posterior
    if hasattr(mcmc, "get_samples"):
        samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()
                   if not k.startswith("sigma")}
    else:
        samples = {k: np.asarray(v) for k, v in mcmc.items()
                   if not k.startswith("sigma")}
    median_params = {k: float(np.median(v)) for k, v in samples.items()}
    labels = ae.parser.get_parameter_labels(circuit)
    params = np.array([float(median_params[lbl]) for lbl in labels])
    circuit_fn = ae.utils.generate_circuit_fn(circuit)

    freq_meas = np.asarray(freq_meas, dtype=float)
    Z_meas = np.asarray(Z_meas, dtype=complex)
    lf = np.sort(freq_meas)

    # Stage 1: ECM self-consistency over a wide symmetric grid
    f_wide = np.logspace(np.log10(lf[0]) - decades, np.log10(lf[-1]) + decades, 500)
    Z_ecm_wide = np.asarray(circuit_fn(f_wide, params), dtype=complex)
    Z_ecm_wide_kk = kk_transform(f_wide, Z_ecm_wide, a=kk_a)
    Z_ecm_orig = np.asarray(circuit_fn(lf, params), dtype=complex)
    Z_ecm_kk_at_meas = (
        _CubicSpline(np.log10(f_wide), Z_ecm_wide_kk.real)(np.log10(lf))
        + 1j * _CubicSpline(np.log10(f_wide), Z_ecm_wide_kk.imag)(np.log10(lf))
    )
    fl_re = np.max(np.abs(Z_ecm_orig.real))
    fl_im = np.max(np.abs(Z_ecm_orig.imag))
    ecm_err_re = float(np.max(np.abs(Z_ecm_kk_at_meas.real - Z_ecm_orig.real)) / fl_re)
    ecm_err_im = float(np.max(np.abs(Z_ecm_kk_at_meas.imag - Z_ecm_orig.imag)) / fl_im)

    # Stage 2: interpolate + ECM-extrapolate measured data, then KK transform
    freq_i, Z_i = kk_interpolate(freq_meas, Z_meas, n_per_decade=n_per_decade)
    freq_full, Z_full = kk_extrapolate(freq_i, Z_i, circuit_fn, params, decades=decades)
    Z_kk_full = kk_transform(freq_full, Z_full, a=kk_a)
    kk_val = kk_residuals(freq_meas, Z_meas, freq_full, Z_kk_full)

    return {
        "kk_val": kk_val,
        "ecm_err_re": ecm_err_re,
        "ecm_err_im": ecm_err_im,
        "freq_full": freq_full,
        "Z_full": Z_full,
        "circuit_fn": circuit_fn,
        "params": params,
        "mcmc_samples": samples,
        "median_params": median_params,
    }
