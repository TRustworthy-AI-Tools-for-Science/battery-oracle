"""EIS measurement-noise models used by the oracle.

Vendored from ``battery_forecast/analysis/noise.py`` — only the three additive
noise terms the oracle applies (white, 1/f flicker, and rest-coupled relaxation
drift). Deliberately does **not** pull in ``autoeis`` (the parametric
``add_drift_noise``/``add_combined_noise`` path in the original module needs
``autoeis.parser``; the oracle composes its own "combined" model inline from
flicker + white, so it is not needed here).

Original attribution: noise functions from Q. Shi and R. Zhang.
"""
from __future__ import annotations

import numpy as np


def add_white_noise(signals, noise_level):
    mag = noise_level * np.abs(signals)
    return signals + np.random.normal(0, mag) + 1j * np.random.normal(0, mag)


def add_flicker_noise(frequencies, signals, noise_level=0.05):
    amp = noise_level * np.abs(signals) / np.sqrt(np.asarray(frequencies))
    return signals + np.random.normal(scale=amp) + 1j * np.random.normal(scale=amp)


def add_relaxation_drift(frequencies, signals, rest_s, drift_scale,
                         tau_relax_s=600.0, n_periods=4, i_amp=1.0):
    """Non-stationarity drift from measuring EIS while the OCP is still relaxing.

    Direct discrete realisation of Hallemans, Howey, Widanage et al. (2023),
    "Electrochemical impedance spectroscopy beyond linearity and stationarity —
    a critical review" (arXiv:2304.08126), Eqs (40) & (43). The measured voltage
    carries a drift signal ``v0(t)`` (the open-circuit potential still relaxing,
    Eq 40); the single-sine impedance extraction ``Z = Vm/Im`` (Eq 18/19) then
    picks up that drift, so the measured spectrum is the fully-relaxed ``Z(w)``
    plus ``dZ(wm) = (1/Im)*(2/Tm) * integral( v0(t) exp(-j wm t) dt )`` over each
    frequency's own measurement window ``Tm``.

    Because the sweep runs sequentially high -> low frequency (their §3.2.1) and
    a slow ``v0(t)`` averages to ~0 against a fast high-frequency sinusoid but
    survives against a slow low-frequency one, the contamination is concentrated
    at low frequency and coherent (one monotonic relaxation trajectory) — i.e.
    Kramers-Kronig-violating, unlike the i.i.d. white/flicker terms. The amplitude
    is rest-coupled via ``exp(-rest_s / tau_relax_s)`` and vanishes as the cell
    becomes well-rested (``rest_s >> tau_relax_s``) or ``drift_scale = 0``.

    Known limitation (why this ships default-OFF). This is only the ``v0(t)``
    drift-leakage part of Eq (43); the integer-period single-sine extraction it
    models inherently *rejects* a slow additive drift (a near-constant ``v0``
    integrated against ``exp(-j wm t)`` over whole periods is ~0 — exactly why
    lock-in measurement is used), so ``v0`` leaks only through its small
    within-window change (~window/tau). This term alone therefore produces
    negligible low-frequency KK violation and cannot reproduce the real cells'
    low/high linKK ratio at any ``drift_scale``/``tau``. The dominant real effect
    is Eq (43)'s *second* term — the impedance ``Z(wm,t)`` itself changing as SOC
    drifts during the sweep (Eq 42) — which needs the oracle's Z-vs-SOC
    sensitivity and is not implemented here.

    Parameters
    ----------
    frequencies : array-like
        EIS frequencies in Hz.
    signals : np.ndarray
        Complex impedance ``Z(w)`` (the fully-relaxed spectrum).
    rest_s : float
        OCV rest time before the sweep (seconds). Larger -> less residual drift.
    drift_scale : float
        Drift magnitude knob; sets the relaxation depth ``dV``. 0 disables.
    tau_relax_s : float
        OCP relaxation time constant (seconds).
    n_periods : int
        Number of periods measured per frequency (sets each window ``Tm``).
    i_amp : float
        Excitation current amplitude ``Im`` used in the ``Z = Vm/Im`` extraction.
    """
    f = np.asarray(frequencies, dtype=float)
    signals = np.asarray(signals, dtype=complex)
    if drift_scale == 0.0 or np.exp(-rest_s / tau_relax_s) < 1e-12:
        return signals
    order = np.argsort(f)[::-1]                       # sequential high -> low sweep (§3.2.1)
    T = n_periods / f                                 # measurement window per frequency
    t_start = np.zeros_like(f)
    t_start[order] = np.concatenate([[0.0], np.cumsum(T[order])[:-1]])  # sweep-time of each freq
    dV = drift_scale * np.random.normal()             # OCP relaxation depth (varies cycle-to-cycle)
    dZ = np.zeros_like(signals, dtype=complex)
    for k in range(len(f)):
        tt = np.linspace(t_start[k], t_start[k] + T[k], 256)
        v0 = dV * np.exp(-(rest_s + tt) / tau_relax_s)                   # v0(t): OCP drift (Eq 40)
        w = 2.0 * np.pi * f[k]
        dZ[k] = (2.0 / (i_amp * T[k])) * np.trapezoid(v0 * np.exp(-1j * w * tt), tt)  # Z=V/I (Eq 18/19)
    return signals + dZ
