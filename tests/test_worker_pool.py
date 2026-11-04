"""Worker pool of the chain fits (pybop_ext.run_candidates): a worker killed from outside
(out of memory) must not be scored as a failed simulation."""
import os
import signal
import types

import numpy as np
import pytest

from battery_oracle.fit import pybop_ext


@pytest.fixture
def fake_problem(monkeypatch):
    """Stock evaluation replaced by a function of the candidate dict."""
    def use(fn):
        def original(meta, inputs, calculate_sensitivities):
            return types.SimpleNamespace(values=np.array([fn(inputs[0])]))
        monkeypatch.setitem(pybop_ext._ORIGINALS, "MetaProblem.evaluate_batch", original)
        monkeypatch.setattr(pybop_ext, "POOL_STATS", {"killed_workers": 0, "workers": None})
        return types.SimpleNamespace(_n_workers=3)
    return use


def test_killed_worker_is_rerun_not_scored(fake_problem, tmp_path):
    marker = tmp_path / "killed_once"

    def cost(x):
        if x["kill"] and not marker.exists():
            marker.write_text("x")
            os.kill(os.getpid(), signal.SIGKILL)       # what the OOM killer does
        return x["value"]

    meta = fake_problem(cost)
    inputs = [{"kill": False, "value": 1.0}, {"kill": True, "value": 2.0}, {"kill": False, "value": 3.0}]
    costs = pybop_ext.run_candidates(meta, inputs, 3, timeout_s=60)
    assert list(costs) == [1.0, 2.0, 3.0]
    assert pybop_ext.POOL_STATS["killed_workers"] == 1
    assert meta._n_workers < 3                           # backs off for later batches


def test_worker_killed_while_alone_raises(fake_problem):
    def cost(x):
        os.kill(os.getpid(), signal.SIGKILL)

    meta = fake_problem(cost)
    with pytest.raises(RuntimeError, match="too little memory"):
        pybop_ext.run_candidates(meta, [{"v": 1}, {"v": 2}], 2, timeout_s=60)


def test_candidate_exception_is_still_scored_inf(fake_problem):
    def cost(x):
        if x["bad"]:
            raise ValueError("solver failed")
        return 5.0

    meta = fake_problem(cost)
    costs = pybop_ext.run_candidates(meta, [{"bad": True}, {"bad": False}], 2, timeout_s=60)
    assert np.isinf(costs[0]) and costs[1] == 5.0
    assert pybop_ext.POOL_STATS["killed_workers"] == 0


def test_fit_checkpoint_keeps_the_best_candidate(tmp_path):
    """Each fit problem keeps its own best-so-far candidate (FitConfig.checkpoint), so a
    fit killed at the job time limit still leaves its parameters, and two fits in one
    process do not share a best."""
    import json

    path = tmp_path / "cell.checkpoint.json"
    best, other = {"cost": np.inf}, {"cost": np.inf}
    pybop_ext._checkpoint_best(best, str(path), [{"a": 1.0}, {"a": np.array([2.0])}], np.array([3.0, 1.0]))
    first = json.loads(path.read_text())
    assert first["best_cost"] == 1.0 and first["best_inputs"] == {"a": 2.0}
    pybop_ext._checkpoint_best(best, str(path), [{"a": 5.0}], np.array([4.0]))            # worse: unchanged
    assert json.loads(path.read_text())["best_inputs"] == {"a": 2.0}
    pybop_ext._checkpoint_best(best, str(path), [{"a": 7.0}], np.array([np.inf]))         # failed: unchanged
    assert json.loads(path.read_text())["best_cost"] == 1.0
    pybop_ext._checkpoint_best(other, None, [{"a": 9.0}], np.array([2.0]))                # another fit
    assert other["cost"] == 2.0 and best["cost"] == 1.0
