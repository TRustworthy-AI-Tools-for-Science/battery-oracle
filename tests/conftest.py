import numpy as np
import pytest

#: One 6-D protocol [C_rate_1, C_rate_2 mA, dur_1, dur_2 h, D_rate mA, dur_d h]
#: (a moderate two-stage charge and a 1 h discharge).
PROTOCOL = np.array([68.0, 34.0, 1.0, 0.5, 100.0, 1.0])


@pytest.fixture(scope="session")
def one_cycle():
    """A fresh default oracle (Randles-stub ECM, seed 0) after one DFN cycle of PROTOCOL:
    (oracle, CycleResult). Shared by the one-cycle smoke tests; do not mutate."""
    from battery_oracle import PyBaMMOracle, randles_stub_ecm

    oracle = PyBaMMOracle(ecm_model_fn=randles_stub_ecm, seed=0)
    return oracle, oracle.run_cycle(PROTOCOL)
