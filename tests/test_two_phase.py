"""Algorithmic tests of battery_oracle.two_phase (MSMR phase-coexistence positive electrode).

Synthetic reactions and a generic PyBaMM set (Marquis2019) only; invariants and limits:
inverse consistency, derivative, the ideal-lattice-gas limit (Theta = 1, and MSMR reducing
to the single-phase Fickian particle), the two-phase limit (Theta -> 0 and a flat plateau as
w -> 0), cyclable-lithium conservation of the state shift, and particle lithium conservation
of a solved MSMR model on the mesh.
"""
from __future__ import annotations

import numpy as np
import pytest

from battery_oracle import two_phase as tp

RT_F = tp.R_GAS * tp.T_REF_K / tp.F
SYNTH = [tp.Reaction(4.2, 0.3, 1.2), tp.Reaction(3.95, 0.4, 0.4), tp.Reaction(3.7, 0.3, 0.8)]


# ---------------------------------------------------------------------------
# thermodynamics
# ---------------------------------------------------------------------------

def test_occupancy_bounds_and_monotone():
    U = np.linspace(2.0, 6.0, 4001)
    x = tp.msmr_x(U, SYNTH)
    assert np.all(np.diff(x) <= 0)                       # saturates to round-off at the ends
    assert np.all(tp.msmr_dxdU(np.linspace(3.0, 5.0, 2001), SYNTH) < 0)   # strictly, analytically
    assert x[0] == pytest.approx(sum(r.X for r in SYNTH), abs=1e-6)
    assert x[-1] == pytest.approx(0.0, abs=1e-6)
    assert np.all((x >= 0) & (x <= sum(r.X for r in SYNTH)))


def test_ocp_inverts_occupancy():
    y = np.linspace(0.02, 0.98, 97)
    U = tp.msmr_ocp(y, SYNTH)
    assert np.max(np.abs(tp.msmr_x(U, SYNTH) - y)) < 1e-10


def test_derivative_matches_finite_difference():
    U = np.linspace(3.5, 4.4, 50)
    h = 1e-6
    fd = (tp.msmr_x(U + h, SYNTH) - tp.msmr_x(U - h, SYNTH)) / (2 * h)
    assert np.allclose(tp.msmr_dxdU(U, SYNTH), fd, rtol=1e-6, atol=1e-9)


def test_ideal_lattice_gas_has_unit_thermodynamic_factor():
    one = [tp.Reaction(4.0, 1.0, 1.0)]
    y = np.linspace(0.05, 0.95, 19)
    assert np.allclose(tp.thermodynamic_factor(y, one), 1.0, rtol=1e-8)
    # and its OCP is the Nernst/Langmuir form U0 + RT/F ln((1 - y) / y)
    assert np.allclose(tp.msmr_ocp(y, one), 4.0 + RT_F * np.log((1 - y) / y), atol=1e-10)


@pytest.mark.parametrize("gap", [(0.3, 0.6), (0.75, 0.93)])
def test_two_phase_limit_flattens_plateau(gap):
    """A narrow reaction of occupancy X = gap width: the OCP slope inside the gap and the
    thermodynamic factor there fall with w (-> the flat, Theta -> 0 two-phase limit)."""
    lo, hi = gap
    slopes, thetas = [], []
    for w in (0.2, 0.05, 0.01):
        rx = [tp.Reaction(4.3, lo, 1.0), tp.Reaction(3.9, hi - lo, w), tp.Reaction(3.5, 1.0 - hi, 1.0)]
        y = np.linspace(lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo), 11)
        U = tp.msmr_ocp(y, rx)
        slopes.append(float(np.max(np.abs(np.diff(U) / np.diff(y)))))
        thetas.append(float(np.max(tp.thermodynamic_factor(y, rx))))
    assert slopes[0] > slopes[1] > slopes[2]
    assert thetas[0] > thetas[1] > thetas[2]
    assert slopes[2] < 0.1 * slopes[0]


def test_fit_recovers_a_synthetic_ocp():
    y = np.linspace(0.05, 0.95, 120)
    U = tp.msmr_ocp(y, SYNTH)
    rx, rep = tp.fit_msmr_reactions(y, U, n=3, n_starts=6, seed=1)
    assert rep["sum_X"] == pytest.approx(1.0, abs=1e-12)
    assert rep["rmse_U_V"] < 2e-3
    assert [r.U0 for r in rx] == sorted([r.U0 for r in rx], reverse=True)


def test_fit_honours_the_plateau_constraint():
    y = np.linspace(0.05, 0.95, 120)
    U = tp.msmr_ocp(y, SYNTH)
    rx, rep = tp.fit_msmr_reactions(y, U, n=4, plateau=(0.4, 0.6, 0.03), n_starts=4)
    narrow = [r for r in rx if r.w == pytest.approx(0.03)]
    assert len(narrow) == 1 and narrow[0].X == pytest.approx(0.2)
    assert rep["sum_X"] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# configuration helpers
# ---------------------------------------------------------------------------

def test_msmr_options_switch_only_the_positive_electrode():
    base = {"open-circuit potential": "single", "particle": "Fickian diffusion", "SEI": "constant"}
    o = tp.msmr_options(base, 4)
    assert o["open-circuit potential"] == ("single", "MSMR")
    assert o["particle"] == ("Fickian diffusion", "MSMR")
    assert o["intercalation kinetics"][1] == "MSMR"
    assert o["number of MSMR reactions"] == ("none", "4")
    assert o["SEI"] == "constant" and base["particle"] == "Fickian diffusion"   # base not mutated
    o2 = tp.msmr_options({"open-circuit potential": ("one-state hysteresis", "single")}, 2)
    assert o2["open-circuit potential"] == ("one-state hysteresis", "MSMR")


def _synthetic_pv():
    return {"Electrode height [m]": 0.1, "Electrode width [m]": 0.2,
            "Number of electrodes connected in parallel to make a cell": 2.0,
            "Negative electrode active material volume fraction": 0.6,
            "Positive electrode active material volume fraction": 0.5,
            "Negative electrode thickness [m]": 80e-6, "Positive electrode thickness [m]": 60e-6,
            "Maximum concentration in negative electrode [mol.m-3]": 30000.0,
            "Maximum concentration in positive electrode [mol.m-3]": 50000.0,
            "Initial concentration in negative electrode [mol.m-3]": 24000.0,
            "Initial concentration in positive electrode [mol.m-3]": 20000.0}


@pytest.mark.parametrize("y_new", [0.3, 0.55, 0.8])
def test_state_shift_conserves_cyclable_lithium(y_new):
    pv = _synthetic_pv()
    n_n, n_p = tp.electrode_capacities_mol(pv)
    x0 = pv["Initial concentration in negative electrode [mol.m-3]"] / pv["Maximum concentration in negative electrode [mol.m-3]"]
    y0 = pv["Initial concentration in positive electrode [mol.m-3]"] / pv["Maximum concentration in positive electrode [mol.m-3]"]
    x1, y1 = tp.shift_initial_state(pv, y_new)
    assert y1 == y_new
    assert x1 * n_n + y1 * n_p == pytest.approx(x0 * n_n + y0 * n_p, rel=1e-12)


class _Var:
    def __init__(self, a):
        self.entries = np.asarray(a, float)


def test_particle_balance_residual():
    q = np.linspace(0.0, 1e-3, 11)
    sol = {tp.FLUX_COUNTER: _Var(q), "Total lithium in positive electrode [mol]": _Var(0.5 + q)}
    assert tp.particle_lithium_balance(sol)["residual_rel"] < 1e-12
    leak = 0.5 + 0.9 * q
    assert tp.particle_lithium_balance(sol, inventory=leak)["residual_rel"] == pytest.approx(0.1, rel=1e-9)


# ---------------------------------------------------------------------------
# solved models (PyBaMM): reduction to the single-phase particle, conservation
# ---------------------------------------------------------------------------

pybamm = pytest.importorskip("pybamm")
U0, Y0, D, J0 = 4.0, 0.4, 2e-14, 2.0


def _ideal_pair():
    """(standard pv, MSMR pv): positive electrode an ideal lattice gas in both forms --
    OCP U0 + RT/F ln((1-y)/y), constant D, j0 = J0 sqrt(c_e/c_e0) sqrt(y(1-y))."""
    base = pybamm.ParameterValues("Marquis2019")
    c_e0 = float(base["Initial concentration in electrolyte [mol.m-3]"])
    c_max = float(base["Maximum concentration in positive electrode [mol.m-3]"])
    base.update({"Positive particle diffusivity [m2.s-1]": D,
                 "Positive electrode OCP entropic change [V.K-1]": 0.0,
                 "Negative electrode OCP entropic change [V.K-1]": 0.0})
    std = base.copy()
    std.update({
        "Positive electrode OCP [V]": lambda sto: U0 + RT_F * pybamm.log((1 - sto) / sto),
        "Positive electrode exchange-current density [A.m-2]":
            lambda c_e, c_s, c_s_max, T: J0 * (c_e / c_e0) ** 0.5 * (c_s / c_s_max * (1 - c_s / c_s_max)) ** 0.5,
        "Initial concentration in positive electrode [mol.m-3]": Y0 * c_max,
    })
    rx = [tp.Reaction(U0, 1.0, 1.0)]
    ms = tp.msmr_parameter_values(std, rx, y_init=Y0)
    return std, ms, rx


def _solve(model, pv, r, t_end=900.0, rate=1.0, n_msmr=None):
    if n_msmr is not None:
        tp.add_msmr_thermal_correction(model, n_msmr)
    tp.add_lithium_flux_counter(model)
    pv = pv.copy()
    pv.update({"Current function [A]": rate * float(pv["Nominal cell capacity [A.h]"])})
    sim = pybamm.Simulation(model, parameter_values=pv,
                            var_pts={"x_n": 8, "x_s": 6, "x_p": 8, "r_n": 10, "r_p": r},
                            solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-10))
    sol = sim.solve([0.0, t_end], t_interp=np.linspace(0.0, t_end, 181))
    return sol, sim.mesh


def test_msmr_j0_mapping_reproduces_the_template_j0():
    std, ms, rx = _ideal_pair()
    for y in (0.2, 0.5, 0.7):
        U = tp.msmr_ocp(y, rx)
        j0_ref = float(ms["Positive electrode host site reference exchange-current density (0) [A.m-2]"])
        assert float(tp.msmr_j0_total(U, rx, j0_ref)[0]) == pytest.approx(tp.template_j0(std, y), rel=1e-6)


def test_ideal_msmr_reduces_to_single_phase_fickian_particle():
    """One ideal reaction (X = 1, w = 1) is exactly Fick's law with the Langmuir OCP. The two
    discretisations differ (FV in U vs in c), so the voltage difference must vanish under mesh
    refinement (>= 2x per halving of the radial step)."""
    std, ms, rx = _ideal_pair()
    diffs = []
    for r in (10, 20, 40):
        s1, _ = _solve(pybamm.lithium_ion.SPM(), std, r)
        s2, _ = _solve(pybamm.lithium_ion.SPM(tp.msmr_options({}, 1)), ms, r)
        diffs.append(float(np.max(np.abs(s1["Voltage [V]"].entries - s2["Voltage [V]"].entries))))
    assert diffs[0] > diffs[1] > diffs[2]
    assert diffs[1] / diffs[2] > 2.0
    assert diffs[2] < 1e-3


@pytest.mark.parametrize("model_cls", ["SPM", "DFN"])
def test_msmr_particle_conserves_lithium_on_the_mesh(model_cls):
    """Inventory (summed on the finite volumes from the solved potential) changes exactly by the
    integrated interfacial flux, for a two-phase-limit (narrow plateau) reaction set; the
    particle surface flux equals the reaction rate; stoichiometry stays in [0, 1]."""
    std, _, _ = _ideal_pair()
    rx = [tp.Reaction(4.2, 0.45, 1.0), tp.Reaction(3.9, 0.3, 0.03), tp.Reaction(3.6, 0.25, 1.0)]
    ms = tp.msmr_parameter_values(std, rx, y_init=0.4)
    model = getattr(pybamm.lithium_ion, model_cls)(tp.msmr_options({}, len(rx)))
    sol, mesh = _solve(model, ms, 20, t_end=1200.0, rate=2.0)
    inv = tp.particle_inventory(sol, mesh, ms, rx)
    bal = tp.particle_lithium_balance(sol, inventory=inv)
    assert bal["delivered_mol"] > 0
    assert bal["residual_rel"] < 1e-5
    if model_cls == "DFN":
        assert tp.surface_flux_consistency(sol) < 1e-6
    lo, hi = tp.stoichiometry_range(sol)["positive"]
    assert 0.0 <= lo and hi <= 1.0


@pytest.mark.parametrize("corrected", [False, True])
def test_msmr_lumped_thermal_lithium_balance(corrected):
    """With a time-varying temperature, x(U, T) moves at fixed U: PyBaMM's MSMR equation (no
    dx/dT term) leaks lithium; add_msmr_thermal_correction closes the balance."""
    std, _, _ = _ideal_pair()
    rx = [tp.Reaction(4.2, 0.45, 1.0), tp.Reaction(3.9, 0.3, 0.03), tp.Reaction(3.6, 0.25, 1.0)]
    ms = tp.msmr_parameter_values(std, rx, y_init=0.4)
    ms.update({"Total heat transfer coefficient [W.m-2.K-1]": 0.1})       # let the cell warm up
    model = pybamm.lithium_ion.SPM({**tp.msmr_options({}, len(rx)), "thermal": "lumped"})
    sol, mesh = _solve(model, ms, 20, t_end=1200.0, rate=2.0, n_msmr=len(rx) if corrected else None)
    T = np.asarray(sol["Volume-averaged cell temperature [K]"].entries)
    assert np.ptp(T) > 0.5
    res = tp.particle_lithium_balance(sol, inventory=tp.particle_inventory(sol, mesh, ms, rx))["residual_rel"]
    if corrected:
        assert res < 1e-5
    else:
        assert res > 1e-4


def test_thermal_correction_is_a_no_op_when_isothermal():
    rx = [tp.Reaction(4.0, 1.0, 1.0)]
    model = pybamm.lithium_ion.SPM(tp.msmr_options({}, 1))
    before = dict(model.rhs)
    tp.add_msmr_thermal_correction(model, 1)
    assert all(model.rhs[k] is v for k, v in before.items())
