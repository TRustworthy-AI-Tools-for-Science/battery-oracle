"""How an oracle cycle fails: the machine-readable kind and the exception the caller sees."""
from __future__ import annotations

from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class FailureKind(str, Enum):
    """Machine-readable classification of an oracle failure/degradation.

    A ``str``-Enum so the value serialises directly to CSV/JSON (``fk.value``)
    and compares equal to its string form. Stored on :class:`~battery_oracle.OracleFailure` and
    (for non-fatal degradations) in the per-cycle history dict, so an audit hook
    can correlate failure modes with calibration outcomes without string-matching
    the human-readable message.
    """

    SOLVER_TRUNCATION  = "solver_truncation"   # step-count mismatch detected by _is_truncated
    SOLVER_FAILURE     = "solver_failure"       # solver raised and no recovery succeeded
    VOLTAGE_INFEASIBLE = "voltage_infeasible"   # experiment voltage window violated
    END_OF_LIFE        = "end_of_life"          # SOH < eol_capacity_fraction
    SOLVER_DEGRADED    = "solver_degraded"      # the emergency solver was needed (non-fatal)


class OracleFailure(RuntimeError):
    """Raised when the PyBaMM simulation fails and no recovery is possible.

    The caller should treat this as battery end-of-life: do not reset the
    oracle, do not substitute synthetic data — end the active learning loop.
    The last protocol that triggered the failure is stored in
    ``self.protocol`` for diagnostics, and a machine-readable
    :class:`FailureKind` in ``self.failure_kind``.

    Note: ``FailureKind.SOLVER_DEGRADED`` is a *non-fatal* status recorded in the
    history dict, never raised — raising it would end the AL loop.
    """
    def __init__(
        self,
        message: str,
        protocol: np.ndarray | None = None,
        *,
        failure_kind: FailureKind,
    ) -> None:
        super().__init__(message)
        self.protocol = protocol
        self.failure_kind = failure_kind


class _SilentTruncation(Exception):
    """Internal signal: PyBaMM returned a solution with fewer steps than requested
    (a swallowed IDA_ERR_FAIL). Used inside the solver-fallback ladder to route a
    truncated solve to the next tier; never surfaces to callers."""
