"""Worker pool of the chain fits (patches.run_candidates): a worker killed from outside
(out of memory) must not be scored as a failed simulation."""
import os
import signal
import types

import numpy as np
import pytest

from battery_oracle import patches


@pytest.fixture
def fake_problem(monkeypatch):
    """Stock evaluation replaced by a function of the candidate dict."""
    def use(fn):
        def original(meta, inputs, calculate_sensitivities):
            return types.SimpleNamespace(values=np.array([fn(inputs[0])]))
        monkeypatch.setitem(patches._ORIGINALS, "MetaProblem.evaluate_batch", original)
        monkeypatch.setattr(patches, "POOL_STATS", {"killed_workers": 0, "workers": None})
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
    costs = patches.run_candidates(meta, inputs, 3, timeout_s=60)
    assert list(costs) == [1.0, 2.0, 3.0]
    assert patches.POOL_STATS["killed_workers"] == 1
    assert meta._n_workers < 3                           # backs off for later batches


def test_worker_killed_while_alone_raises(fake_problem):
    def cost(x):
        os.kill(os.getpid(), signal.SIGKILL)

    meta = fake_problem(cost)
    with pytest.raises(RuntimeError, match="too little memory"):
        patches.run_candidates(meta, [{"v": 1}, {"v": 2}], 2, timeout_s=60)


def test_candidate_exception_is_still_scored_inf(fake_problem):
    def cost(x):
        if x["bad"]:
            raise ValueError("solver failed")
        return 5.0

    meta = fake_problem(cost)
    costs = patches.run_candidates(meta, [{"bad": True}, {"bad": False}], 2, timeout_s=60)
    assert np.isinf(costs[0]) and costs[1] == 5.0
    assert patches.POOL_STATS["killed_workers"] == 0
