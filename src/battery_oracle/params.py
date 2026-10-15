"""Beginning-of-life parameter adjustments shared by the oracle and external fitters.

A *scale* multiplies an existing PyBaMM parameter, whether it is a number or a
function (e.g. ``"Negative electrode exchange-current density [A.m-2]"`` is a
function of concentrations and temperature in Chen2020). Expressing fitted
beginning-of-life (BOL) values as multipliers on the base parameter set keeps the
oracle's own structure (degradation presets, ``kinetics_scale``) intact and lets a
fitter such as PyBOP optimise plain scalars:

* :func:`add_scale_parameters` rewrites each named entry as
  ``original * pybamm.Parameter("<name> scale")`` (default 1.0), so a fitter can
  treat ``"<name> scale"`` as an ordinary scalar parameter.
* :func:`apply_parameter_scales` applies fitted numeric multipliers directly.

Both produce the same expression for the same multiplier, so a scale fitted
through the first route reproduces exactly when the oracle applies it through the
second.
"""
from __future__ import annotations

from collections.abc import Mapping

SCALE_SUFFIX = " scale"


def scale_parameter_name(name: str) -> str:
    """Name of the scalar multiplier parameter created for ``name``."""
    return f"{name}{SCALE_SUFFIX}"


def _scaled(original, factor):
    """Return ``original * factor`` for a numeric or callable parameter value."""
    if callable(original):
        return (lambda fn, s: lambda *a, **kw: fn(*a, **kw) * s)(original, factor)
    return original * factor


def add_scale_parameters(parameter_values, names) -> dict[str, str]:
    """Wrap each parameter in ``names`` as ``original * Parameter("<name> scale")``.

    Mutates ``parameter_values`` in place and returns ``{name: scale_name}``. The
    scale parameters default to 1.0, so the model is unchanged until a fitter
    overrides them.
    """
    import pybamm

    created: dict[str, str] = {}
    for name in names:
        scale_name = scale_parameter_name(name)
        parameter_values[name] = _scaled(parameter_values[name], pybamm.Parameter(scale_name))
        parameter_values.update({scale_name: 1.0}, check_already_exists=False)
        created[name] = scale_name
    return created


def apply_parameter_scales(parameter_values, scales: Mapping[str, float] | None):
    """Multiply each named parameter by its factor. Mutates and returns ``parameter_values``."""
    for name, factor in (scales or {}).items():
        factor = float(factor)
        if factor != 1.0:
            parameter_values[name] = _scaled(parameter_values[name], factor)
    return parameter_values


def apply_parameter_overrides(parameter_values, overrides: Mapping[str, float] | None):
    """Set named parameters to absolute values (new names allowed). Mutates and returns."""
    if overrides:
        parameter_values.update(dict(overrides), check_already_exists=False)
    return parameter_values
