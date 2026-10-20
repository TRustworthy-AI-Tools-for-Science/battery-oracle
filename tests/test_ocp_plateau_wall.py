"""PlateauWall LiCoO2 OCP: Ai2020 below the wall region, monotone, a wall of the set drop at
the parameterised position, the plateau untouched away from it (algorithmic checks)."""
import numpy as np
import pybamm
import pytest

import battery_oracle.parameter_sets as ps


def _u(y, y_w):
    pv = pybamm.ParameterValues({ps.LICO2_WALL_Y: y_w})
    return float(np.ravel(pv.process_symbol(ps.lico2_ocp_plateau_wall(pybamm.Scalar(y))).evaluate())[0])


def _ai(y):
    x, u = ps._lico2_monotone_table()
    o = np.argsort(x)
    return float(np.interp(y, np.asarray(x)[o], np.asarray(u)[o]))


@pytest.mark.parametrize("y", [0.45, 0.6, 0.75, 0.85, 0.9])
def test_equals_ai2020_below_the_wall_region(y):
    assert _u(y, 0.96) == pytest.approx(_ai(y), abs=2e-3)       # cubic vs linear interpolation of one table


@pytest.mark.parametrize("y_w", [0.93, 0.96, 0.99])
def test_monotone_and_wall_drop(y_w):
    ys = np.linspace(0.4, 1.0, 1201)
    u = np.array([_u(y, y_w) for y in ys])
    # monotone up to the Ai2020 table's own noise on the plateau (rises <= 0.022 mV at y ~ 0.79-0.83,
    # present in lico2_ocp_monotone too); the wall adds none
    assert np.all(np.diff(u) < 5e-5)
    before, after = _u(y_w - 6 * ps.LICO2_WALL_WIDTH, y_w), _u(y_w + 6 * ps.LICO2_WALL_WIDTH, y_w)
    plateau_change = abs(_u(y_w + 6 * ps.LICO2_WALL_WIDTH, 2.0) - _u(y_w - 6 * ps.LICO2_WALL_WIDTH, 2.0))
    assert before - after == pytest.approx(ps.LICO2_WALL_DROP_V, abs=plateau_change + 1e-3)


def test_wall_moves_with_its_parameter_and_leaves_the_plateau_alone():
    assert _u(0.95, 0.93) < _u(0.95, 0.97) - 0.5 * ps.LICO2_WALL_DROP_V     # past the wall vs before it
    for y_w in (0.93, 0.97):
        assert _u(y_w - 0.03, y_w) == pytest.approx(_u(y_w - 0.03, 2.0), abs=1e-4)


def test_chemistry_installs_the_wall():
    pv = ps.lir2032(graphite_ocp="Mohtat2020", lico2_ocp="PlateauWall", lico2_ds="vacancy")
    assert pv["Positive electrode OCP [V]"] is ps.lico2_ocp_plateau_wall
    assert ps.LICO2_WALL_Y in pv.keys()
