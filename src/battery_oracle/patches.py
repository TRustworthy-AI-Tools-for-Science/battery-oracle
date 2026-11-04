"""PyBaMM patch of the oracle's discretisation (PyBaMM only; the fit's PyBOP patches are
in :mod:`battery_oracle.fit.pybop_ext`).

``harmonic_flux_coefficients``
    PyBaMM averages a coefficient onto cell edges harmonically only when it
    multiplies a bare gradient; the DFN electrolyte current multiplies its
    effective conductivity by a *combination* of gradients, so at the
    electrode|separator interfaces (where eps^b jumps several-fold) it is averaged
    arithmetically. Patched to the harmonic (series) mean for any flux expression
    (upstream PR #5856; drop this patch when PyBaMM ships it). It fires on the
    oracle's electrolyte current; the chain fit's model (differential surface form)
    has no such term.

Safety. The patch is installed only if the installed PyBaMM matches the tested
version and the patched target has the expected signature; otherwise :func:`apply`
raises (``strict``) or skips it with a warning. It can be reverted (:func:`revert`).
"""
from __future__ import annotations

import inspect
import logging
import warnings
from typing import Callable

log = logging.getLogger(__name__)

#: PyBaMM version the patch was written and tested against (26.10 is the only supported
#: version: pyproject pins >= 26.10).
TESTED_PYBAMM: tuple[str, ...] | str = ("26.10.1.0",)

_ORIGINALS: dict[str, Callable] = {}
_APPLIED: set[str] = set()


class PatchError(RuntimeError):
    """A patch target is missing or has an unexpected signature/version."""


def check_pybamm_version() -> None:
    import pybamm

    tested = (TESTED_PYBAMM,) if isinstance(TESTED_PYBAMM, str) else tuple(TESTED_PYBAMM)
    if pybamm.__version__ not in tested:
        raise PatchError(f"pybamm {pybamm.__version__} not in tested {tested}")


def require_signature(fn, names: list[str], where: str) -> None:
    params = list(inspect.signature(fn).parameters)
    if params[: len(names)] != names:
        raise PatchError(f"{where} signature {params} != expected prefix {names}")



# ---------------------------------------------------------------------------
# Harmonic averaging of coefficients that multiply a flux expression
# ---------------------------------------------------------------------------

def _is_flux_term(sym) -> bool:
    """A gradient, possibly negated or multiplied by a node coefficient."""
    import pybamm

    if isinstance(sym, pybamm.Gradient):
        return True
    if isinstance(sym, pybamm.Negate):
        return _is_flux_term(sym.children[0])
    if isinstance(sym, pybamm.Multiplication):
        a, b = sym.children
        return ((isinstance(b, pybamm.Gradient) and not a.evaluates_on_edges("primary"))
                or (isinstance(a, pybamm.Gradient) and not b.evaluates_on_edges("primary")))
    return False


def _is_flux(sym) -> bool:
    """True for a COMBINATION of gradient terms (a flux driving force such as
    ``chi grad(c) - grad(phi)``): an Addition/Subtraction whose terms are all
    gradients (optionally scaled). A bare gradient is left to stock PyBaMM (already
    harmonic); nested products are left alone (a coefficient that is zero on both
    sides of an edge has no harmonic mean)."""
    import pybamm

    if isinstance(sym, (pybamm.Addition, pybamm.Subtraction)):
        return all(_is_flux_term(c) or _is_flux(c) for c in sym.children)
    return False


def _install_harmonic_flux_coefficients() -> None:
    """PyBaMM's finite volumes map a node coefficient onto cell edges with the
    harmonic mean only when it multiplies a bare ``Gradient``; for any other flux
    expression they use the arithmetic mean. The DFN electrolyte current is written
    ``(kappa tor) * (chi grad(c_e) - grad(phi_e))``, so its effective conductivity
    is ARITHMETICALLY averaged at the electrode|separator interfaces, where the
    transport efficiency eps^b jumps (Ai2020: 0.040 | 0.354 | 0.124): the interface
    conductance is then overestimated by (a+b)^2/(4ab) (~2.8x for a 9x jump)
    against the series (harmonic) value of the control-volume method (Patankar 1980;
    the Recktenwald note PyBaMM cites). Patched: a coefficient multiplying any flux
    expression (see ``_is_flux``) gets the harmonic mean, so every DFN flux --
    electrolyte salt and current, solid current, heat -- treats discontinuous
    coefficients at interfaces the same, consistent way."""
    import pybamm

    fv = pybamm.FiniteVolume
    require_signature(fv.process_binary_operators,
                       ["self", "bin_op", "left", "right", "disc_left", "disc_right"],
                       "FiniteVolume.process_binary_operators")
    original = fv.process_binary_operators
    _ORIGINALS["FiniteVolume.process_binary_operators"] = original

    def process_binary_operators(self, bin_op, left, right, disc_left, disc_right):
        le = left.evaluates_on_edges("primary")
        re_ = right.evaluates_on_edges("primary")
        # only a coefficient MULTIPLYING a flux gets the harmonic mean (as upstream PR #5856):
        # an addend (e.g. a node term added to a flux in the EIS-wrapped model) has no series
        # meaning, and is zero at rest, where its harmonic mean 1/(b/0 + (1-b)/0) is undefined
        # (non-finite Jacobian at t = 0 with a stoichiometry-dependent conductivity);
        # division stays arithmetic (PyBaMM#5863)
        if isinstance(bin_op, pybamm.Multiplication) and le != re_:
            if le and not isinstance(left, pybamm.Gradient) and _is_flux(left):
                disc_right = self.node_to_edge(disc_right, method="harmonic")
                return pybamm.simplify_if_constant(bin_op.create_copy([disc_left, disc_right]))
            if re_ and not isinstance(right, pybamm.Gradient) and _is_flux(right):
                disc_left = self.node_to_edge(disc_left, method="harmonic")
                return pybamm.simplify_if_constant(bin_op.create_copy([disc_left, disc_right]))
        return original(self, bin_op, left, right, disc_left, disc_right)

    fv.process_binary_operators = process_binary_operators


def _remove_harmonic_flux_coefficients() -> None:
    import pybamm

    pybamm.FiniteVolume.process_binary_operators = _ORIGINALS.pop("FiniteVolume.process_binary_operators")


# ---------------------------------------------------------------------------

_PATCHES = {
    "harmonic_flux_coefficients": (_install_harmonic_flux_coefficients, _remove_harmonic_flux_coefficients),
}


def apply(names: list[str] | None = None, *, strict: bool = True) -> list[str]:
    """Install the named patches (default all); returns the names now applied.

    ``strict``: raise :class:`PatchError` if versions/signatures do not match;
    otherwise skip that patch with a warning.
    """
    applied = []
    for name in names or list(_PATCHES):
        if name in _APPLIED:
            applied.append(name)
            continue
        try:
            check_pybamm_version()
            _PATCHES[name][0]()
        except Exception as exc:
            if strict:
                raise PatchError(f"patch {name!r} not applied: {exc}") from exc
            warnings.warn(f"patch {name!r} skipped: {exc}", stacklevel=2)
            continue
        _APPLIED.add(name)
        applied.append(name)
        log.info("applied patch %s", name)
    return applied


def revert(names: list[str] | None = None) -> None:
    """Remove the named patches (default all applied), restoring stock behaviour."""
    for name in names or list(_APPLIED):
        if name in _APPLIED:
            _PATCHES[name][1]()
            _APPLIED.discard(name)


def applied() -> set[str]:
    return set(_APPLIED)
