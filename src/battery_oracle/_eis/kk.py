"""linKK (Schönleber et al. 2014) residuals of an EIS spectrum: the oracle's per-cycle
Kramers-Kronig consistency check (:func:`linkk_rmse`). Uses ``impedance.py``, a core
dependency. The full KK transform pipeline lives in battery-forecast
(``battery_forecast.eis.kk``).
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


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
    except Exception as exc:                  # noqa: BLE001 - a diagnostic, not the state
        log.warning("linKK failed (%s: %s)", type(exc).__name__, str(exc)[:200])
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
