"""battery_oracle.validity helpers on synthetic fields."""
import numpy as np

from battery_oracle import validity


def test_one_sided_extrapolation_sees_a_jump_and_not_a_smooth_field():
    pos = np.linspace(0.5, 9.5, 10)                  # nodes, interface at 5.0
    smooth = (2.0 * pos)[:, None]
    a, b = validity._one_sided(smooth, pos, 5.0)
    assert np.allclose(a, b)
    jump = np.where(pos < 5.0, 1.0, 3.0)[:, None]
    a, b = validity._one_sided(jump, pos, 5.0)
    assert np.allclose(np.abs(a - b), 2.0)


def test_alternating_run_detects_odd_even_mode():
    x = np.linspace(0, 1, 20)
    smooth = np.sin(x)[:, None]
    wiggle = (np.sin(x) + 0.01 * (-1.0) ** np.arange(20))[:, None]
    assert validity._max_alternating_run(smooth, 1e-9) < validity.LIMITS["oscillation run"][0]
    assert validity._max_alternating_run(wiggle, 1e-9) >= validity.LIMITS["oscillation run"][1]


def _m(**kw):
    base = {"i_e": 0.01, "N_e": 0.01, "c_e": 0.004, "phi_e": 1e-9, "collector": 0.003, "charge": 1e-15,
            "lithium": 1e-7, "side_loss_rel": 3e-3, "oscillation": 2}
    return {**base, **kw}


def test_judge_passes_converging_jumps_and_fails_persistent_ones():
    coarse = _m(N_e=0.069)
    fine = _m(i_e=0.0025, N_e=0.018, c_e=0.001, collector=0.0008)   # all shrink ~x4: resolution
    assert all(c["status"] != "fail" for c in validity.judge(coarse, fine))
    stuck = validity.judge(coarse, {**fine, "N_e": 0.060})          # N_e barely shrinks: real jump
    assert any(c["status"] == "fail" and "N_e" in c["name"] for c in stuck)


def test_judge_exact_laws_use_solver_tolerance_and_side_reactions():
    bad_charge = validity.judge(_m(charge=1e-3), _m(charge=1e-3), rtol=1e-6)
    assert any(c["status"] == "fail" and "charge" in c["name"] for c in bad_charge)
    # lithium drift limit = 1 % of the lithium consumed by side reactions
    ok = validity.judge(_m(lithium=2e-5), _m(lithium=2e-5))
    bad = validity.judge(_m(lithium=1e-4), _m(lithium=1e-4))
    assert not any(c["status"] == "fail" and "lithium" in c["name"] for c in ok)
    assert any(c["status"] == "fail" and "lithium" in c["name"] for c in bad)


def test_exact_law_residual_passes_only_if_it_is_a_tolerance_error():
    coarse = fine = _m(charge=5e-4)
    shrinks = validity.judge(coarse, fine, tight={"charge": 5e-6})
    stuck = validity.judge(coarse, fine, tight={"charge": 4e-4})
    untested = validity.judge(coarse, fine)
    status = lambda recs: [c["status"] for c in recs if "charge" in c["name"]][0]   # noqa: E731
    assert status(shrinks) == "pass" and status(stuck) == "fail" and status(untested) == "fail"


def test_measure_maps_each_continuity_record_to_its_own_field(monkeypatch):
    names = {"i_e": 0.017, "N_e": 0.068, "c_e": 0.005, "phi_e": 2e-12}
    recs = [{"name": f"validity[PJ248] {k} continuous across interfaces", "value": v, "note": ""}
            for k, v in names.items()]
    monkeypatch.setattr(validity, "check_solution", lambda sol, pv, cell="": recs)
    m = validity.measure(object(), None)
    assert {k: m[k] for k in names} == names        # phi_e must not overwrite i_e
