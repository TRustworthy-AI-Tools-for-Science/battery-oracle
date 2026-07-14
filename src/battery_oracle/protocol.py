"""``Protocol``: a named, unit-explicit charge/discharge protocol vector.

Mirrors the discipline already used for the oracle's *output* (see
``oracle.state_vector_schema``) on its *input*: :class:`PyBaMMOracle.__call__`
and :meth:`PyBaMMOracle.run_cycle` accept a bare 6-D (or 7-D) ``np.ndarray``
for backward compatibility, but the slot order/meaning was previously only
documented in comments (``_circuit.ACTION_FEATURE_NAMES``). ``Protocol``
gives that vector a name and a unit for each slot.

Deliberately import-light (``dataclasses``, ``numpy``, and
``battery_oracle._circuit`` only) so it has no PyBaMM/AutoEIS dependency and
cannot participate in an import cycle with ``battery_oracle.oracle``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from battery_oracle._circuit import ACTION_FEATURE_NAMES

# ---------------------------------------------------------------------------
# Single canonical slot-order mapping: dataclass field name <-> the existing
# ACTION_FEATURE_NAMES entry it corresponds to. ACTION_FEATURE_NAMES (defined
# in _circuit.py) remains the one source of truth for slot ORDER; this tuple
# just attaches unit-explicit dataclass field names to it, and the assertions
# below make any future reordering of ACTION_FEATURE_NAMES fail loudly here
# instead of silently desyncing Protocol from the rest of the package (see
# the project's no-hardcoded-vector-lengths rule).
# ---------------------------------------------------------------------------
_PROTOCOL_SLOTS: tuple[tuple[str, str], ...] = (
    ("charge_current_1_mA", "C_rate_1"),
    ("charge_current_2_mA", "C_rate_2"),
    ("charge_duration_1_h", "duration_1"),
    ("charge_duration_2_h", "duration_2"),
    ("discharge_current_mA", "D_rate"),
    ("discharge_duration_h", "duration_d"),
)
assert len(_PROTOCOL_SLOTS) == len(ACTION_FEATURE_NAMES), (
    "Protocol._PROTOCOL_SLOTS must have exactly one entry per "
    "_circuit.ACTION_FEATURE_NAMES slot."
)
assert tuple(action_name for _, action_name in _PROTOCOL_SLOTS) == tuple(ACTION_FEATURE_NAMES), (
    "Protocol._PROTOCOL_SLOTS order must match _circuit.ACTION_FEATURE_NAMES exactly "
    "-- this is the single canonical slot order every protocol representation "
    "(Protocol, the experiment YAML's protocol fields, the raw ndarray) derives from."
)

PROTOCOL_FIELD_NAMES: tuple[str, ...] = tuple(field_name for field_name, _ in _PROTOCOL_SLOTS)
"""``Protocol`` dataclass field names (excluding ``T_ambient_K``) in canonical
slot order -- derived from :data:`battery_oracle._circuit.ACTION_FEATURE_NAMES`,
never redefined independently. Other modules that need the protocol field
order (e.g. ``experiment._PROTOCOL_FIELDS``) should derive from this tuple
rather than hardcoding their own copy of the order."""


@dataclass(frozen=True)
class Protocol:
    """A named, unit-explicit charge/discharge protocol.

    The oracle simulates one two-stage-CC-charge / single-stage-discharge
    cycle per call (see ``PyBaMMOracle._protocol_to_experiment``); this
    dataclass names the 6 (or 7, with an ambient-temperature override) values
    that fully describe that cycle, in the same slot order as the bare
    ``np.ndarray`` the oracle has always accepted
    (:data:`battery_oracle.ACTION_FEATURE_NAMES`). Despite the historical
    ``C_rate``/``D_rate`` naming in ``ACTION_FEATURE_NAMES``, these slots are
    **currents in mA**, not C-rates -- the field names here spell that out.

    Parameters
    ----------
    charge_current_1_mA : float
        First-stage constant-current charge magnitude, in mA (real-cell
        scale; the oracle rescales internally via ``real_cell_capacity_mah``
        onto its PyBaMM cell).
    charge_current_2_mA : float
        Second-stage (taper) constant-current charge magnitude, in mA.
    charge_duration_1_h : float
        Maximum duration of the first charge stage, in hours (the stage also
        ends early on the ``v_charge_max`` cutoff).
    charge_duration_2_h : float
        Maximum duration of the second charge stage, in hours.
    discharge_current_mA : float
        Constant-current discharge magnitude, in mA.
    discharge_duration_h : float
        Requested discharge duration, in hours. The oracle floors the
        *applied* duration at 3600 s (1 h) regardless of this value, so the
        cell actually reaches the ``v_discharge_min`` cutoff instead of being
        cut off early -- see ``PyBaMMOracle``'s class docstring and
        :attr:`~PyBaMMOracle.run_cycle`'s ``CycleResult.protocol_applied``.
    T_ambient_K : float or None, optional
        Per-cycle ambient temperature override, in Kelvin (slot 6). Only
        meaningful -- and only emitted by :meth:`to_array` -- when the oracle
        was constructed with ``thermal="lumped", use_temperature_protocol=True``.
        ``None`` (default) means "use the oracle's configured ambient
        temperature," producing the standard 6-D vector.

    Examples
    --------
    >>> p = Protocol(
    ...     charge_current_1_mA=1000.0, charge_current_2_mA=500.0,
    ...     charge_duration_1_h=0.25, charge_duration_2_h=0.25,
    ...     discharge_current_mA=1000.0, discharge_duration_h=1.0,
    ... )
    >>> p.to_array().shape
    (6,)
    >>> Protocol.from_array(p.to_array()) == p
    True
    """

    charge_current_1_mA: float
    charge_current_2_mA: float
    charge_duration_1_h: float
    charge_duration_2_h: float
    discharge_current_mA: float
    discharge_duration_h: float
    T_ambient_K: float | None = None

    def to_array(self) -> np.ndarray:
        """Return the canonical-order ``np.ndarray`` the oracle consumes.

        Returns
        -------
        np.ndarray
            Shape ``(6,)``, or ``(7,)`` when :attr:`T_ambient_K` is not
            ``None`` (slot 6 = ``T_ambient_K``).
        """
        values = [float(getattr(self, name)) for name in PROTOCOL_FIELD_NAMES]
        if self.T_ambient_K is not None:
            values.append(float(self.T_ambient_K))
        return np.asarray(values, dtype=np.float64)

    @classmethod
    def from_array(cls, u: "np.ndarray | Sequence[float]") -> "Protocol":
        """Build a :class:`Protocol` from a length-6 or length-7 array-like.

        Parameters
        ----------
        u : array-like
            Length 6 (core protocol) or 7 (core protocol + ``T_ambient_K`` in
            slot 6), in the canonical slot order.

        Returns
        -------
        Protocol

        Raises
        ------
        ValueError
            If ``u`` is not length 6 or 7.
        """
        arr = np.asarray(u, dtype=np.float64)
        n_core = len(PROTOCOL_FIELD_NAMES)
        if arr.shape[-1] not in (n_core, n_core + 1):
            raise ValueError(
                f"Protocol.from_array expects length {n_core} (core protocol) or "
                f"{n_core + 1} (core protocol + T_ambient_K); got length {arr.shape[-1]}."
            )
        kwargs = dict(zip(PROTOCOL_FIELD_NAMES, (float(x) for x in arr[:n_core])))
        if arr.shape[-1] > n_core:
            kwargs["T_ambient_K"] = float(arr[n_core])
        return cls(**kwargs)
