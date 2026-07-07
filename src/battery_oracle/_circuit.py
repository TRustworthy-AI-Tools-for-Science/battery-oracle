"""ECM circuit parsing, arc-flip correction, and canonical parameter names.

Vendored from ``battery_forecast`` (``battmap.py`` + ``utils.py``) so the oracle
package is self-contained. Pure ``re``/``numpy`` — no heavy dependencies.

AutoEIS parameter labels: ``Rx`` = resistor, ``Cx`` = capacitor, ``Pxw`` = CPE
admittance Q, ``Pxn`` = CPE exponent n.
"""
from __future__ import annotations

import re
from itertools import permutations

import numpy as np

# ---------------------------------------------------------------------------
# Canonical circuit + parameter/action names (single source of truth for the
# standalone package; a study can override per-instance via PyBaMMOracle kwargs).
# Values mirror battery_forecast's config/datasets.yaml default_circuit +
# action_names so featurized records line up when used inside that pipeline.
# ---------------------------------------------------------------------------
DEFAULT_CIRCUIT = "R1-[R2,P3]-[R4,P5]"

ACTION_FEATURE_NAMES = [
    "C_rate_1", "C_rate_2", "duration_1", "duration_2", "D_rate", "duration_d",
]
"""Names for the 6-D action / protocol feature appended to the state vector."""


def _param_labels_from_circuit(circuit: str) -> list[str]:
    """AutoEIS parameter labels for a circuit string (no autoeis import needed).

    Walks elements left-to-right: ``R{i}`` -> ``['R{i}']``, ``C{i}`` -> ``['C{i}']``,
    ``P{i}`` -> ``['P{i}w', 'P{i}n']`` (CPE admittance Q + exponent n). Matches
    ``autoeis.parser.get_parameter_labels`` for R/C/P Randles-type circuits.
    """
    labels: list[str] = []
    for kind, idx in re.findall(r"([RCP])(\d+)", circuit):
        if kind == "P":
            labels += [f"P{idx}w", f"P{idx}n"]
        else:
            labels.append(f"{kind}{idx}")
    return labels


ECM_PARAM_NAMES = _param_labels_from_circuit(DEFAULT_CIRCUIT)
"""Canonical ECM parameter names derived from ``DEFAULT_CIRCUIT``."""


# ---------------------------------------------------------------------------
# Symmetric-Randles arc-flip correction (vendored from utils.py)
# ---------------------------------------------------------------------------
_LEGACY_RANDLES_PAIRS = [("R3", "P4"), ("R5", "P6")]


def randles_pairs_from_circuit(circuit: str) -> list[tuple[str, str]]:
    """Extract symmetric Randles ``[R, CPE]`` arc pairs from an ECM circuit string.

    Parses each bracketed parallel branch; within a branch the element whose
    name starts with ``R`` is the resistor and the one starting with ``P`` is
    the CPE.  Circuit-agnostic, so the same code handles the project circuit
    ``R1-P2-[R3,P4]-[R5,P6]`` -> ``[("R3","P4"), ("R5","P6")]`` and the oracle's
    internal ``R0-P0-[R1,P1]-[R2,P2]`` -> ``[("R1","P1"), ("R2","P2")]``.
    """
    pairs: list[tuple[str, str]] = []
    for branch in re.findall(r"\[([^\[\]]+)\]", circuit):
        elems = [e.strip() for e in branch.split(",")]
        r = next((e for e in elems if e.startswith("R")), None)
        p = next((e for e in elems if e.startswith("P")), None)
        if r is not None and p is not None:
            pairs.append((r, p))
    return pairs


def _apply_arc_permutation(s: dict, perm, pairs) -> None:
    """In-place: relabel arcs so slot ``i`` receives the (R, Pw, Pn) samples
    currently held by arc ``perm[i]``.  ``perm`` is a permutation of ``range(N)``."""
    if tuple(perm) == tuple(range(len(pairs))):
        return  # identity — nothing to do
    orig = {}
    for r, p in pairs:
        orig[r] = s[r]
        orig[p + "w"] = s[p + "w"]
        orig[p + "n"] = s[p + "n"]
    for new_i, old_i in enumerate(perm):
        r_new, p_new = pairs[new_i]
        r_old, p_old = pairs[old_i]
        s[r_new] = orig[r_old]
        s[p_new + "w"] = orig[p_old + "w"]
        s[p_new + "n"] = orig[p_old + "n"]


def _arc_perm_dist(prev: dict, curr: dict, perm, pairs) -> float:
    """Squared log (R, Pw) + linear (Pn) continuity distance between ``prev`` and
    ``curr`` when ``curr``'s arcs are relabelled by ``perm`` (slot ``i`` <- ``perm[i]``)."""
    d = 0.0
    for new_i, old_i in enumerate(perm):
        r_new, p_new = pairs[new_i]   # prev's slot
        r_old, p_old = pairs[old_i]   # curr data moving into that slot
        d += (np.log(np.median(curr[r_old])) - np.log(np.median(prev[r_new]))) ** 2
        d += (np.log(np.median(curr[p_old + "w"])) - np.log(np.median(prev[p_new + "w"]))) ** 2
        d += (np.median(curr[p_old + "n"]) - np.median(prev[p_new + "n"])) ** 2
    return d


def _fix_arc_flips(sample_dicts: list[dict], pairs) -> None:
    """Generic N-arc, sequence-aware flip correction (in-place core).

    AutoEIS cannot distinguish symmetric Randles arcs, so their labels swap
    between cycles.  Anchor cycle 0 by ordering arcs by CPE exponent (n)
    descending, then for every subsequent cycle pick the arc permutation (of the
    ``N!`` possibilities) that minimises the continuity distance to the
    already-corrected previous cycle.  No-op for fewer than two arcs (no
    ambiguity) and for N >= 2 reduces to keep-vs-swap when N == 2.
    """
    if len(pairs) < 2 or not sample_dicts:
        return
    idx = list(range(len(pairs)))
    # Anchor cycle 0: highest-exponent arc -> slot 0, next -> slot 1, ...
    s0 = sample_dicts[0]
    order = tuple(sorted(idx, key=lambda i: -np.median(s0[pairs[i][1] + "n"])))
    _apply_arc_permutation(s0, order, pairs)
    # Subsequent cycles: minimise distance to the previous (corrected) cycle.
    for k in range(1, len(sample_dicts)):
        prev, curr = sample_dicts[k - 1], sample_dicts[k]
        best = min(permutations(idx), key=lambda pm: _arc_perm_dist(prev, curr, pm, pairs))
        _apply_arc_permutation(curr, best, pairs)


def fix_parameter_flips_dicts(sample_dicts, circuit: str | None = None) -> None:
    """Flip-correct a list of per-cycle posterior-sample dicts in place.

    Lowest-level public entry point.  Arc pairs are derived from ``circuit``
    (legacy project pairs when None).
    """
    pairs = randles_pairs_from_circuit(circuit) if circuit else _LEGACY_RANDLES_PAIRS
    _fix_arc_flips(list(sample_dicts), pairs)
