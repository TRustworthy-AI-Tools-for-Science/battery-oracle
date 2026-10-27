"""battery_oracle.validity helpers on synthetic fields."""
import numpy as np
import pytest

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


def test_time_derivative_is_exact_within_segments_and_never_spans_a_join():
    """Two segments joined at a repeated time (as the segmented solve produces): the
    derivative is exact inside each, and a jump across the join gives no spike."""
    import pytest

    t1 = np.linspace(0.0, 10.0, 11)
    t2 = np.linspace(10.0 + 1e-13, 20.0, 11)
    t = np.r_[t1, t2]
    y = np.r_[2.0 * t1, 5.0 + 3.0 * t2]          # slope 2 then 3, value jumps by 5 at the join
    d = validity.time_derivative(y, t)
    assert np.allclose(d[:11], 2.0) and np.allclose(d[11:], 3.0)
    assert np.allclose(validity.time_derivative(y, t, stride=2), d)      # linear: coarse estimate agrees
    joins = validity._segment_joins(t)
    assert joins.sum() == 2 and joins[10] and joins[11]
    assert np.max(np.abs(d)) == pytest.approx(3.0)


def test_step_mask_excludes_the_pre_step_sample_and_smoothed_onsets():
    """A sharp step and a ramped onset (sub-threshold increments, far faster than cycling):
    both are steps; the sample just before each and STEP_WINDOW_S after each are not judged,
    slow cycling-rate changes are judged."""
    i_1c = 1.0
    t = np.r_[np.linspace(0.0, 10.0, 11), 10.0 + 1e-4 * np.arange(1, 6), np.linspace(11.0, 30.0, 20),
              np.linspace(31.0, 60.0, 30)]
    i = np.zeros(t.size)
    ramp = (t > 10.0) & (t < 10.001)
    i[ramp] = 0.005 * np.arange(1, ramp.sum() + 1)            # 0.005 i_1C per 1e-4 s: below 1 % per sample
    i[(t >= 11.0) & (t < 31.0)] = 0.025
    i[t >= 31.0] = 2.0 + 0.001 * (t[t >= 31.0] - 31.0)         # a sharp step, then a slow drift
    steps, judged = validity._step_mask(t, i, i_1c)
    k0 = 10                                                    # last rest sample before the ramped onset
    assert not judged[k0] and judged[k0 - 1]
    assert steps.min() == pytest.approx(t[k0 + 1])
    k1 = int(np.flatnonzero(t == 31.0)[0])
    assert not judged[k1 - 1] and not judged[k1]
    assert judged[(t > 31.0 + validity.STEP_WINDOW_S)].all()   # slow drift is cycling, judged
    assert judged[(t > 10.001 + validity.STEP_WINDOW_S) & (t < 30.0 - 1e-9)].all()


def test_corrected_integral_charge_balance_holds_for_a_differential_dfn():
    """A DFN with the double layer: the integral balance including a C_dl d(delta phi)/dt holds
    to solver tolerance, through current steps and rests."""
    import pybamm

    pv = pybamm.ParameterValues("Marquis2019")
    pv.update({"Negative electrode double-layer capacity [F.m-2]": 2.0,
               "Positive electrode double-layer capacity [F.m-2]": 2.0})
    model = pybamm.lithium_ion.DFN(options={"surface form": "differential"})
    exp = pybamm.Experiment(["Discharge at 1C for 300 seconds", "Rest for 120 seconds",
                             "Charge at 0.5C for 300 seconds"], period="2 seconds")
    sol = pybamm.Simulation(model, parameter_values=pv, experiment=exp,
                            solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)).solve()
    checks = {c["name"]: c for c in validity.check_solution(sol, pv)}
    integral = next(c for n, c in checks.items() if "integral charge balance" in n)
    assert integral["value"] is not None and integral["value"] < 1e-5, integral
    assert "double layer" in integral["note"]
