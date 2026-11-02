"""The oracle's per-cycle output: the named state-vector layout and :class:`CycleResult`."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from battery_oracle._circuit import DEFAULT_CIRCUIT as _PROJECT_CIRCUIT
from battery_oracle._circuit import _param_labels_from_circuit
from battery_oracle.failures import FailureKind
from battery_oracle.protocol import Protocol


# ---------------------------------------------------------------------------
# State-vector layout
# ---------------------------------------------------------------------------

def state_vector_schema(
    n_params: int,
    *,
    has_std: bool = True,
    has_t_cell: bool = False,
) -> dict[str, tuple[int, int]]:
    """Ordered ``{block_name: (lo, hi)}`` map describing the returned state vector.

    The layout is *derived*, never hardcoded (see the project's
    no-hardcoded-vector-lengths rule): per-half width is ``n_params`` (=
    ``len(_param_labels_from_circuit(circuit))``), and which blocks exist — and
    therefore the total length — is a function of the active feature flags:

        [means_charge | means_discharge]                      always
        [std_charge   | std_discharge]                        when ``has_std``
        [T_cell_K]                                            when ``has_t_cell`` (single scalar, tail)

    Downstream code (DMDc, audit hook) should slice by name via this map rather
    than by magic index. ``has_t_cell`` is populated by the Phase-2 thermal work.
    """
    blocks = ["means_charge", "means_discharge"]
    if has_std:
        blocks += ["std_charge", "std_discharge"]
    schema: dict[str, tuple[int, int]] = {}
    off = 0
    for b in blocks:
        schema[b] = (off, off + n_params)
        off += n_params
    if has_t_cell:
        schema["T_cell_K"] = (off, off + 1)
    return schema


# Default-circuit instance, for documentation / downstream that only needs the
# shipped layout. Per-oracle layout lives on ``self.state_vector_schema``.
STATE_VECTOR_SCHEMA = state_vector_schema(len(_param_labels_from_circuit(_PROJECT_CIRCUIT)))


# ---------------------------------------------------------------------------
# CycleResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CycleResult:
    """Named, schema-aware result of one :meth:`PyBaMMOracle.run_cycle` call.

    ``PyBaMMOracle.__call__`` (the hard array-in/array-out compatibility
    contract consumed by the downstream AL/DMDc loop) returns only
    :attr:`state`; ``run_cycle`` returns the full ``CycleResult`` so a novice
    caller doesn't have to reach into the private ``oracle._history`` list
    (see :attr:`PyBaMMOracle.history`) to get the SOH, SOC, EIS spectra, or
    fitted ECM parameters behind a given state vector.

    Parameters
    ----------
    state : np.ndarray
        The same vector ``PyBaMMOracle.__call__`` returns for this cycle
        (``[means_charge | means_discharge | std_charge | std_discharge |
        T_cell_K?]`` -- see :attr:`schema`).
    schema : dict
        The oracle's ``state_vector_schema`` at the time of this call (``{block
        name: (lo, hi)}``); pass a block name to :meth:`block` to slice ``state``.
    soh : float
        State of health (1.0 = fresh cell) at the end of this cycle.
    soc_charge, soc_discharge : float
        Fractional state of charge read after the post-charge / post-discharge
        rest.
    capacity_ah : float
        Measured or voltage-limited discharge capacity for this cycle, in Ah.
    fidelity : str
        ``"full"`` for every completed cycle (the model is never swapped; an
        emergency-solver retry is recorded in ``failure_kind`` instead).
    failure_kind : FailureKind or None
        Non-fatal degradation classification recorded for this cycle, if any.
        ``None`` for a full-fidelity cycle. (A *fatal* failure raises
        :class:`OracleFailure` instead of returning a ``CycleResult``.)
    protocol : np.ndarray
        The protocol vector exactly as received by ``run_cycle`` (before
        sanitisation).
    protocol_applied : Protocol
        The sanitised protocol actually simulated -- see
        ``PyBaMMOracle._protocol_to_experiment`` and the loud-sanitisation
        warning it emits when a value was clamped.
    eis_charge, eis_discharge : tuple of (np.ndarray, np.ndarray)
        ``(frequencies_hz, Z)`` for the post-charge / post-discharge
        synthesised EIS spectrum; ``Z`` is complex (``Z.real``/``Z.imag`` in Ohm).
    ecm_charge, ecm_discharge : dict of str to float
        Fitted ECM parameter means for this cycle, keyed by the circuit's
        AutoEIS parameter label (e.g. ``"R1"``, ``"P3w"``, ``"P3n"``).
    ecm_std_charge, ecm_std_discharge : dict of str to float
        Per-parameter AutoEIS posterior standard deviation, same keys as
        :attr:`ecm_charge`/:attr:`ecm_discharge`. All-NaN values when AutoEIS
        was unavailable and the Randles stub fitted instead.
    raw : Mapping
        The full per-cycle history row (read-only view; the same dict
        appended to ``oracle.history``), for diagnostics not otherwise
        surfaced above.

    Examples
    --------
    >>> result = oracle.run_cycle(protocol)
    >>> result.soh
    0.97...
    >>> np.asarray(result).shape == (oracle.state_vector_len,)
    True
    >>> result.block("means_charge").shape
    (7,)
    """

    state: np.ndarray
    schema: dict
    soh: float
    soc_charge: float
    soc_discharge: float
    capacity_ah: float
    fidelity: str
    failure_kind: "FailureKind | None"
    protocol: np.ndarray
    protocol_applied: Protocol
    eis_charge: tuple
    eis_discharge: tuple
    ecm_charge: dict
    ecm_discharge: dict
    ecm_std_charge: dict
    ecm_std_discharge: dict
    raw: Mapping = field(default_factory=dict)

    def __array__(self, dtype=None) -> np.ndarray:
        """Allow ``np.asarray(result)`` to yield :attr:`state` directly."""
        return np.asarray(self.state, dtype=dtype)

    def block(self, name: str) -> np.ndarray:
        """Slice :attr:`state` by schema block name (e.g. ``"means_charge"``).

        Parameters
        ----------
        name : str
            A key of :attr:`schema` (``"means_charge"``, ``"means_discharge"``,
            ``"std_charge"``, ``"std_discharge"``, and ``"T_cell_K"`` when the
            oracle uses lumped thermal).

        Returns
        -------
        np.ndarray
            ``state[lo:hi]`` for the block's ``(lo, hi)`` schema range.
        """
        lo, hi = self.schema[name]
        return self.state[lo:hi]
