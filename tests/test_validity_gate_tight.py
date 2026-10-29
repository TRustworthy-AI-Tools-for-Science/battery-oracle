"""evaluate_chain's validity gate: an exact-law residual above its limit triggers solves
10x and 100x tighter, and passes only if it converges with the tolerance (a tolerance error)."""
import numpy as np
import pytest

from battery_oracle import bol_chain as bc
from battery_oracle import validity


class FakeChain:
    """A chain whose integral charge-balance residual is ``k * rtol ** p`` (p = 1 or a
    sub-linear p = 0.44: a tolerance error; p = 0: a real violation independent of it)."""

    def __init__(self, power):
        self.power, self.rtol, self.atol = power, 1e-6, 1e-8
        self.fixed, self.keep_solution, self.x_factor, self.tols = {}, False, 1, []

    def solve(self, inputs):
        return {"diag": {}, "records": [[3.9]], "spectra": [], "offsets": [0.0]}

    def _solve(self, inputs):
        self.tols.append(self.rtol)
        return {"diag": {}, "records": [[3.9]], "spectra": [], "offsets": [0.0], "solution": self.rtol,
                "parameter_values": None}


@pytest.fixture
def fake_measures(monkeypatch):
    def measure(rtol, pv):
        v = 1e-3 * (rtol / 1e-6) ** chain_power[0]
        return {"charge": 0.0, "charge_integral": v, "lithium": 0.0, "side_loss_rel": 0.0}
    chain_power = [1.0]
    monkeypatch.setattr(validity, "measure", measure)
    monkeypatch.setattr(validity, "relevance", lambda *a, **k: [])
    monkeypatch.setattr(bc, "_mesh_agreement", lambda *a, **k: {"name": "mesh", "status": "pass"})
    return chain_power


@pytest.mark.parametrize("power, status", [(1.0, "pass"), (0.44, "pass"), (0.0, "fail")])
def test_exact_law_above_limit_is_decided_by_tighter_solves(fake_measures, power, status):
    fake_measures[0] = power
    chain = FakeChain(power)
    data = {"cell_id": "c", "records": [{"kind": "cc", "label": "a", "V": [3.9], "t": [0.0]}], "spectra": []}
    out = bc.evaluate_chain(data, None, {}, chain=chain)
    rec = next(r for r in out["diagnostics"]["validity"] if "integral charge balance" in r["name"])
    assert rec["status"] == status
    assert sorted(set(chain.tols))[:2] == pytest.approx([1e-8, 1e-7])   # the 10x and 100x tighter solves ran
    assert (chain.rtol, chain.atol) == (1e-6, 1e-8)             # and the tolerances were restored


def test_no_tighter_solve_when_the_exact_laws_pass(fake_measures, monkeypatch):
    monkeypatch.setattr(validity, "measure",
                        lambda rtol, pv: {"charge": 0.0, "charge_integral": 0.0, "lithium": 0.0, "side_loss_rel": 0.0})
    chain = FakeChain(1.0)
    data = {"cell_id": "c", "records": [{"kind": "cc", "label": "a", "V": [3.9], "t": [0.0]}], "spectra": []}
    bc.evaluate_chain(data, None, {}, chain=chain)
    assert np.allclose(chain.tols, 1e-6)                        # only the two mesh solves, no tighter ones
