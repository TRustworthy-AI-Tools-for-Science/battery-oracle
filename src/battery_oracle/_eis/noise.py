"""EIS measurement-noise models used by the oracle.

Vendored from ``battery_forecast/analysis/noise.py`` — only the two additive
noise terms the oracle applies (white and 1/f flicker). Deliberately does **not** pull in ``autoeis`` (the parametric
``add_drift_noise``/``add_combined_noise`` path in the original module needs
``autoeis.parser``; the oracle composes its own "combined" model inline from
flicker + white, so it is not needed here).

Original attribution: noise functions from Q. Shi and R. Zhang.
"""
from __future__ import annotations

import numpy as np


def add_white_noise(signals, noise_level, rng: np.random.Generator | None = None):
    """Add i.i.d. Gaussian noise scaled by ``noise_level * |signals|``.

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Source of randomness. Defaults to a fresh ``np.random.default_rng()``
        (unseeded) if not given — pass a seeded generator for reproducible
        output.
    """
    rng = rng if rng is not None else np.random.default_rng()
    mag = noise_level * np.abs(signals)
    return signals + rng.normal(0, mag) + 1j * rng.normal(0, mag)


def add_flicker_noise(
    frequencies, signals, noise_level=0.05, rng: np.random.Generator | None = None
):
    """Add 1/f-scaled Gaussian noise to the impedance spectrum.

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Source of randomness. Defaults to a fresh ``np.random.default_rng()``
        (unseeded) if not given — pass a seeded generator for reproducible
        output.
    """
    rng = rng if rng is not None else np.random.default_rng()
    amp = noise_level * np.abs(signals) / np.sqrt(np.asarray(frequencies))
    return signals + rng.normal(scale=amp) + 1j * rng.normal(scale=amp)
