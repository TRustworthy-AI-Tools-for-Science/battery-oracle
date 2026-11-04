"""Stage-A fitting: the beginning-of-life chain fit and its checks (the ``[fit]``
extra; needs PyBOP).

* :mod:`battery_oracle.fit.chain`: the cell's whole measured history as one
  integration, fitted with PyBOP (:func:`~battery_oracle.fit.chain.fit_cells`).
* :mod:`battery_oracle.fit.config`: the fit's settings, recorded in each report.
* :mod:`battery_oracle.fit.pybop_ext`: patches and the warm restart on top of PyBOP.
"""
try:
    import pybop  # noqa: F401
except ImportError as exc:                                   # pragma: no cover
    raise ImportError("battery_oracle.fit needs PyBOP: install battery-oracle[fit]") from exc
