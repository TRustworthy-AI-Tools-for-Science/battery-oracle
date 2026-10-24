"""Envelope probe (bol_chain.with_envelope_probe / envelope_setting): single- and two-stage
charge, parsing of ORACLE_FIT_ENVELOPE_MA (synthetic data only)."""
import numpy as np
import pytest

from battery_oracle import bol_chain as bc

DATA = {"groups": [{"name": "g0", "duration_s": 100.0}], "current": [{"group": 0, "t": np.array([0.0, 100.0]),
                                                                      "I": np.array([0.0, 0.0])}]}


def _charge(probe):
    t, i = probe["t"], probe["I"]
    return float(-np.trapezoid(np.minimum(i, 0.0), t)), t, i


def test_single_stage_probe_unchanged():
    d = bc.with_envelope_probe(DATA, 150.0, 120.0)
    p = d["current"][-1]
    assert d["groups"][-1]["probe"] and d["groups"][-1]["charge2_ma"] is None
    assert set(np.round(np.unique(p["I"]), 6)) == {-0.15, 0.0, 0.12}
    q, _, _ = _charge(p)
    assert q == pytest.approx(0.15 * bc.PROBE_STEP_S, rel=1e-3)


def test_two_stage_probe_charges_c1_then_c2_back_to_back():
    d = bc.with_envelope_probe(DATA, 160.0, 150.0, 84.0)
    p = d["current"][-1]
    t, i = p["t"], p["I"]
    charge = i < 0
    assert set(np.round(np.unique(i[charge]), 6)) == {-0.16, -0.084}
    # stage 1 then stage 2 with no rest between them, each PROBE_STAGE_S long
    k1 = np.flatnonzero(np.isclose(i, -0.16))
    k2 = np.flatnonzero(np.isclose(i, -0.084))
    assert k2[0] == k1[-1] + 1
    assert t[k1[-1]] - t[k1[0]] == pytest.approx(bc.PROBE_STAGE_S - bc.EDGE_RAMP_S, abs=1e-9)
    assert t[k2[-1]] - t[k2[0]] == pytest.approx(bc.PROBE_STAGE_S - bc.EDGE_RAMP_S, abs=1e-9)
    assert np.isclose(i, 0.15).any() and d["groups"][-1]["duration_s"] == pytest.approx(t[-1] + bc.EDGE_RAMP_S)


@pytest.mark.parametrize("text, expect", [("159.4;159.4", (159.4, 159.4)), ("160;84;150", (160.0, 150.0, 84.0)),
                                          ("160,84,150", (160.0, 150.0, 84.0))])
def test_envelope_setting_parses_two_or_three_values(monkeypatch, text, expect):
    monkeypatch.setattr(bc, "ENVELOPE_MA", text)
    assert bc.envelope_setting() == expect


def test_envelope_setting_rejects_other_lengths(monkeypatch):
    monkeypatch.setattr(bc, "ENVELOPE_MA", "1;2;3;4")
    with pytest.raises(ValueError):
        bc.envelope_setting()
