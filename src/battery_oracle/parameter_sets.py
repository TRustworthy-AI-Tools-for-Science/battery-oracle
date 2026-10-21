"""Custom PyBaMM parameter sets for the oracle.

``LiR2032`` describes the commercial rechargeable LiCoO2/graphite coin cell used in
the jones2022 dataset (Powerstream LiR2032; 35 mAh nominal, ~42 mAh measured;
Jones, Stimming & Lee 2022, Nat. Commun. 13:4806), at its real size: the oracle
simulates the real cell's currents directly, with no scaling onto a larger cell.

It is built on a published LiCoO2/graphite set (``template``, default Ai2020:
Ai et al. 2020, J. Electrochem. Soc. 167:013512), which supplies everything the
jones2022 data cannot identify -- electrode OCPs, thicknesses, porosities,
active-material fractions, particle radii, Bruggeman exponents, electrolyte
transport, conductivities, thermal, SEI and particle-mechanics parameters. The
choice of template and the comparison with coin-cell literature is recorded in
battery-forecast ``_results/oracle_recal/lir2032_parameter_comparison.md``.

What changes from the template here (all later refined by
:mod:`battery_oracle.bol_fit` against the real data):

* **one electrode pair** (the template's 34-layer pouch becomes a single layer of
  the cell's total electrode area);
* **electrode area** scaled from the template's area-per-capacity to the nominal
  capacity (~15 cm2 for 42 mAh). A commercial LiR2032 cannot be a single 15 mm disc
  -- at these loadings that holds only ~6 mAh -- so the area is an effective,
  fitted quantity (balancing sets it from the fitted electrode capacities);
* **nominal capacity** 42 mAh (measured on the jones2022 cells; datasheet 35 mAh);
* **voltage window** 3.0-4.3 V (jones2022 protocol cut-offs);
* **Ai2020 electrolyte diffusivity unit fix**: PyBaMM's
  ``electrolyte_diffusivity_Ai2020`` (checked in pybamm 26.7.1) is the
  Valoen-Reimers correlation, which yields cm2/s, returned as m2/s without the
  1e-4 conversion -- 3.2e-6 m2/s at 1 M / 25 C, ~1e4 x the measured ~3e-10 m2/s
  (Chen2020 1.8e-10, Ramadass2004 7.5e-10). Uncorrected, the electrolyte never
  limits the rate (the DFN shows a +-0.02% concentration swing at 3.6C). Found by
  the physicality report's electrolyte-depletion check.
"""
from __future__ import annotations

import math

import numpy as np

#: Measured capacity of the jones2022 LiR2032 cells (deepest discharge, ~42 mAh).
LIR2032_NOMINAL_AH = 0.042
#: Voltage window of the jones2022 protocol (Jones et al. 2022 Methods).
LIR2032_V_MIN, LIR2032_V_MAX = 3.0, 4.3
#: LiCoO2/graphite PyBaMM sets usable as the LiR2032 template.
LIR2032_TEMPLATES = ("Ai2020", "Ramadass2004")
#: cm2/s -> m2/s for PyBaMM's Ai2020 electrolyte diffusivity (see module doc).
AI2020_ELECTROLYTE_DIFFUSIVITY_UNIT_FIX = 1e-4
#: LiR2032 can: 20 mm diameter, 3.2 mm high.
COIN_RADIUS_M, COIN_HEIGHT_M = 0.010, 0.0032
COIN_VOLUME_M3 = math.pi * COIN_RADIUS_M ** 2 * COIN_HEIGHT_M
COIN_COOLING_AREA_M2 = 2 * math.pi * COIN_RADIUS_M ** 2 + 2 * math.pi * COIN_RADIUS_M * COIN_HEIGHT_M
#: Whole-cell heat capacity [J/K]: LIR2032 2.5-2.8 g (datasheets) x 0.96-1.04 J/(g K)
#: (LiCoO2 cells, Maleki et al. 1999, J. Electrochem. Soc. 146:947).
COIN_HEAT_CAPACITY_J_K = 2.6
#: Activation energies [J/mol] of a LiCoO2/graphite set (Ramadass et al. 2004,
#: J. Electrochem. Soc. 151:A196, as in PyBaMM's Ramadass2004): Ai2020 inherits
#: Dualfoil's 5 kJ/mol, 5-10x below published charge-transfer and solid-diffusion
#: values (Ecker 2015: 53/44; Chen 2020: 35/18 kJ/mol).
LIR2032_ACTIVATION_J_MOL = {
    "Negative electrode exchange-current density [A.m-2]": 37480.0,
    "Positive electrode exchange-current density [A.m-2]": 39570.0,
    "Negative particle diffusivity [m2.s-1]": 42770.0,
    "Positive particle diffusivity [m2.s-1]": 18550.0,
}
AI2020_ACTIVATION_J_MOL = 5000.0
T_REF_K = 298.15


def _with_activation(fn, e_new: float, e_old: float = AI2020_ACTIVATION_J_MOL):
    """``fn`` (its last argument is the temperature) re-referenced from activation
    energy ``e_old`` to ``e_new``: x exp((e_new - e_old)/R (1/T_ref - 1/T))."""
    import pybamm

    def wrapped(*args):
        T = args[-1]
        return fn(*args) * pybamm.exp((e_new - e_old) / pybamm.constants.R * (1.0 / T_REF_K - 1.0 / T))

    return wrapped
#: Prior centre for the heat-transfer coefficient [W/(m2 K)] (fitted in stage A).
COIN_H_DEFAULT = 10.0

#: Lithiation dependence of the LiCoO2 solid diffusivity. Ai2020 (Dualfoil) uses a
#: constant D_s; measured Li_yCoO2 diffusivity falls steeply as y -> 1: tracer
#: D* from ~2e-12 to 3e-17 cm2/s over 0.4 < y < 1 (thin films; Hasegawa/Kuwata et
#: al. 2021, Phys. Chem. Chem. Phys., doi 10.1039/d0cp04598e), and SOC-dependent
#: chemical D from GITT (Hess et al. 2015, J. Power Sources, doi
#: 10.1016/j.jpowsour.2015.07.080). Form: D_s(y) = D_s * 10**(-drop * s**2),
#: s = max(0, (y - knee) / (1 - knee)) -- constant below the knee, falling by
#: ``drop`` decades at full lithiation. drop = 0 (default) is the constant D_s.
LICO2_DS_KNEE = "Positive particle diffusivity knee stoichiometry"
LICO2_DS_DROP = "Positive particle diffusivity decades drop at full lithiation"
LICO2_DS_DEFAULTS = {LICO2_DS_KNEE: 0.85, LICO2_DS_DROP: 0.0}
#: Same form for graphite, D_s(x) = D_s * 10**(-drop * s**2) above the knee x_k: the
#: GITT/PITT chemical diffusivity of Li_xC6 falls by ~1-2 decades through the
#: stage-2 -> stage-1 region (x ~0.5 -> 1), with minima at the two-phase plateaus (Levi
#: & Aurbach 1997, J. Phys. Chem. B 101, 4641; Persson et al. 2010, J. Phys. Chem.
#: Lett. 1, 1176 -- citations to check). The constant-D_s template has no such slowing
#: of the filling graphite, which limits a high-rate charge. drop = 0 (default) is the
#: constant D_s.
GRAPHITE_DS_KNEE = "Negative particle diffusivity knee stoichiometry"
GRAPHITE_DS_DROP = "Negative particle diffusivity decades drop at full lithiation"
GRAPHITE_DS_DEFAULTS = {GRAPHITE_DS_KNEE: 0.6, GRAPHITE_DS_DROP: 0.0}
#: Stoichiometry dependence of the exchange-current density beyond the standard
#: j0 ~ sqrt(theta (1 - theta)) (theta = surface stoichiometry): j0 x [4 theta (1 - theta)
#: + J0_SHAPE_FLOOR]^gamma / (1 + J0_SHAPE_FLOOR)^gamma, unchanged at theta = 0.5 and
#: falling faster towards a full or empty surface for gamma > 0 (gamma = 0, the default,
#: is the standard form). Evidence: the jones2022 charge-transfer arc of the rested cells
#: grows from 1.02 to 1.86 ohm between 4.18 and 3.58 V OCV (+82%, all 32 cells;
#: scripts/eis_vs_state.py), faster than the fitted standard-form model (slope -1.1 vs
#: -1.5 ohm/V; scripts/model_arc_vs_ocv.py); SOC-dependent, asymmetric Butler-Volmer
#: kinetics of Li_xCoO2 from GITT: Hess, A. et al., J. Power Sources 299, 156 (2015),
#: doi 10.1016/j.jpowsour.2015.07.080.
J0_SHAPE_N = "Negative electrode exchange-current density stoichiometry exponent"
J0_SHAPE_P = "Positive electrode exchange-current density stoichiometry exponent"
J0_SHAPE_DEFAULTS = {J0_SHAPE_N: 0.0, J0_SHAPE_P: 0.0}
J0_SHAPE_FLOOR = 1.0e-3


def _with_j0_shape(fn, exponent_name: str):
    """Exchange-current density ``fn(c_e, c_s_surf, c_s_max, T)`` x the stoichiometry
    factor of J0_SHAPE_N / J0_SHAPE_P (see there)."""
    import pybamm

    def wrapped(c_e, c_s_surf, c_s_max, T):
        gamma = pybamm.Parameter(exponent_name)
        theta = c_s_surf / c_s_max
        shape = (4 * theta * (1 - theta) + J0_SHAPE_FLOOR) / (1 + J0_SHAPE_FLOOR)
        return fn(c_e, c_s_surf, c_s_max, T) * pybamm.maximum(shape, J0_SHAPE_FLOOR) ** gamma

    return wrapped


def _with_lithiation_drop(fn, knee_name: str = LICO2_DS_KNEE, drop_name: str = LICO2_DS_DROP):
    """``fn(sto, T)`` x 10**(-drop s^2) above the knee (see LICO2_DS_KNEE,
    GRAPHITE_DS_KNEE). ``fn`` may also be a constant."""
    import pybamm

    def wrapped(sto, T):
        knee = pybamm.Parameter(knee_name)
        drop = pybamm.Parameter(drop_name)
        s = pybamm.maximum((sto - knee) / (1.0 - knee), 0.0)
        base = fn(sto, T) if callable(fn) else fn
        return base * pybamm.exp(-drop * math.log(10.0) * s ** 2)

    return wrapped


#: Real data range of OCP interpolants that are continued beyond their data, by
#: interpolant name (read by ``bol_fit.ocp_data_range``).
OCP_DATA_RANGES: dict[str, tuple[float, float]] = {}
LICO2_MONOTONE_NAME = "lico2_ocp_Ai2020_monotone"


def _lico2_monotone_table(step: float = 0.01, n_slope: int = 5, step_hi: float = 0.002, y_hi: float = 1.03):
    """Ai2020 LiCoO2 OCP table, continued linearly (monotonic) at both ends: from y = 0
    to the data start, and from the data end (y = 0.9989, 3.41 V, still falling at
    -73 V per unit y) to y_hi. Without the upper continuation PyBaMM extrapolates the
    end cubic, which need not keep falling: with a fine particle mesh the LiCoO2 surface
    reaches the data end during a discharge and the solver fails at 3.003 V, just above
    the 3.0 V stop that a real cell reaches as Li_yCoO2 fills (radial_ref_why.py,
    2026-10-16). The OCP guard still fails candidates that go beyond OCP_GUARD_TOL."""
    import numpy as np
    from pybamm.input.parameters.lithium_ion.Ai2020 import lico2_ocp_Ai2020_data

    _, (x, y) = lico2_ocp_Ai2020_data
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    order = np.argsort(x)
    x, y = x[order], y[order]
    slope = np.polyfit(x[:n_slope], y[:n_slope], 1)[0]          # dU/dy < 0 at the data start
    x_ext = np.arange(0.0, x[0] - step / 2, step)
    y_ext = y[0] + slope * (x_ext - x[0])
    slope_hi = (y[-1] - y[-2]) / (x[-1] - x[-2])                 # dU/dy of the last data segment
    x_up = np.arange(x[-1] + step_hi, y_hi + step_hi / 2, step_hi)
    y_up = y[-1] + slope_hi * (x_up - x[-1])
    OCP_DATA_RANGES[LICO2_MONOTONE_NAME] = (float(x[0]), float(x[-1]))
    return np.r_[x_ext, x, x_up], np.r_[y_ext, y, y_up]


#: LiCoO2 solid diffusivity from the measured OCP (``lico2_ds="thermo"`` of :func:`lir2032`),
#: replacing the hand-chosen knee/drop form. Lattice gas with site exclusion: flux
#: N = -M c grad(mu), vacancy-limited mobility M = D0 (1 - y) / kT (Bazant, Acc. Chem. Res.
#: 46, 1144 (2013); Bazant et al., arXiv 1204.2934), mu = -F U(y) from the OCP, gives Fick's
#: law with D(y) = D0 Theta(y), Theta(y) = -(F/RT) y (1 - y) dU/dy (Theta = 1 for an ideal
#: lattice gas). The shape of D(y) is the measured OCP's; only D0 (the template's D, scaled
#: by the fit) is free. The collapse near full lithiation is the vacancy mechanism measured
#: for Li_xCoO2 (tracer D* falling steeply to a minimum at x = 1; Hasegawa, Kuwata et al.,
#: Phys. Chem. Chem. Phys. 2021, doi 10.1039/d0cp04598e).
LICO2_THETA_NAME = "lico2_thermodynamic_factor"
LICO2_THETA_FLOOR = 1.0e-3            # keeps D > 0 where the OCP is flat (two-phase) or y -> 1
LICO2_THETA_T_REF = 298.15


def lico2_thermodynamic_factor_table(step: float = 5e-4):
    """(y, Theta(y)) on a grid from the monotone Ai2020 OCP table (cubic spline slope)."""
    import numpy as np
    from scipy.interpolate import CubicSpline

    x, u = _lico2_monotone_table()
    order = np.argsort(x)
    x, u = np.asarray(x, float)[order], np.asarray(u, float)[order]
    keep = np.r_[True, np.diff(x) > 1e-9]
    spline = CubicSpline(x[keep], u[keep])
    y = np.arange(0.0, 1.0 + step / 2, step)
    theta = -(96485.33212 / (8.314462618 * LICO2_THETA_T_REF)) * y * (1.0 - y) * spline(y, 1)
    return y, np.maximum(theta, LICO2_THETA_FLOOR)


def _with_thermodynamic_factor(fn):
    """``fn(sto, T)`` x Theta(sto) (see LICO2_THETA_NAME)."""
    import pybamm

    y, theta = lico2_thermodynamic_factor_table()

    def wrapped(sto, T):
        th = pybamm.Interpolant(y, theta, pybamm.minimum(pybamm.maximum(sto, 0.0), 1.0),
                                name=LICO2_THETA_NAME, interpolator="linear")
        base = fn(sto, T) if callable(fn) else fn
        return base * th

    return wrapped


#: Vacancy-limited LiCoO2 diffusivity (``lico2_ds="vacancy"`` of :func:`lir2032`):
#: D(y) = D0 max(1 - y, LICO2_VACANCY_FLOOR). Site-exclusion mobility without the
#: thermodynamic factor (Hasegawa, Kuwata et al. 2021, eq. for D* = D0 (1 - x)); it makes a
#: near-full surface region whose transport collapses, which a coarse radial mesh cannot
#: resolve (use ORACLE_FIT_R_MESH=surface with a strong stretch). Only D0 is free.
LICO2_VACANCY_FLOOR = 1.0e-3


def _with_vacancy_limit(fn):
    """``fn(sto, T)`` x max(1 - sto, LICO2_VACANCY_FLOOR)."""
    import pybamm

    def wrapped(sto, T):
        base = fn(sto, T) if callable(fn) else fn
        return base * pybamm.maximum(1.0 - sto, LICO2_VACANCY_FLOOR)

    return wrapped


def _with_ocp_thermodynamic_factor(fn, ocp):
    """``fn(sto, T)`` x Theta(sto, T) = -(F / R T) sto (1 - sto) dU/dsto, with U the installed
    positive OCP ``ocp`` differentiated symbolically (no table, no copied constants): the
    thermodynamically consistent chemical diffusivity of a vacancy-limited lattice gas,
    D = D0 Theta (see LICO2_THETA_NAME), using the same OCP as the voltage. Floored at
    LICO2_THETA_FLOOR only as a numerical guard (an analytic monotone OCP keeps Theta > 0)."""
    import pybamm

    # differentiate w.r.t. a placeholder, then substitute the real stoichiometry: diff() w.r.t. an
    # expression is unreliable, because PyBaMM folds constants into it (1.062 * (c / c_max) becomes
    # one Multiplication with a new constant, and the original node is gone -> derivative 0)
    y = pybamm.InputParameter("__lico2_ocp_stoichiometry__")
    du_dy = ocp(y).diff(y)

    def wrapped(sto, T):
        theta = -(96485.33212 / (8.314462618 * T)) * sto * (1.0 - sto) * _substitute(du_dy, y, sto)
        base = fn(sto, T) if callable(fn) else fn
        return base * pybamm.maximum(theta, LICO2_THETA_FLOOR)

    return wrapped


def _substitute(sym, old, new):
    """Copy of the PyBaMM expression ``sym`` with every node equal to ``old`` replaced by ``new``."""
    if sym == old:
        return new
    if not sym.children:
        return sym
    return sym.create_copy(new_children=[_substitute(c, old, new) for c in sym.children])


def lico2_ocp_monotone(sto):
    """Ai2020 LiCoO2 OCP, continued linearly (monotonic) beyond both ends of its data."""
    import pybamm

    x, y = _lico2_monotone_table()
    return pybamm.Interpolant(x, y, sto, name=LICO2_MONOTONE_NAME, interpolator="cubic")


#: Literature-shaped LiCoO2 OCP tail (``lico2_ocp="PlateauWall"``, 2026-10-20). Measured
#: Li_xCoO2 potentials stay on the two-phase plateau (3.88-3.92 V) and then fall in a
#: near-vertical wall within the last few percent of lithiation: Reimers & Dahn, J.
#: Electrochem. Soc. 139, 2091 (1992), Fig. 2a (C/50 discharge: ~3.75 V at x = 0.93,
#: ~3.5 V at 0.95-0.96); Ohzuku & Ueda, J. Electrochem. Soc. 141, 2972 (1994), Figs. 7, 9
#: (10-h OCV flat at 3.92 V to x ~ 1, then ~3.1 V for pristine LiCoO2). Ai2020's tail is
#: ~0.04 later and softer (3.82 V at 0.95, 3.66 V at 0.99). Form: Ai2020 up to
#: LICO2_WALL_FROM, its plateau slope continued beyond, minus a smooth wall of
#: LICO2_WALL_DROP_V and width LICO2_WALL_WIDTH centred at the parameter LICO2_WALL_Y.
#: The papers fix the wall's shape but not its position in a given cell's stoichiometry
#: (their x is from charge passed, with 5-7 % irreversibility; Li/Co > 1 shifts the tail,
#: Levasseur et al. 2002), so LICO2_WALL_Y is a fit parameter bounded to 0.93-0.99.
LICO2_WALL_Y = "Positive electrode OCP wall stoichiometry"
LICO2_WALL_DEFAULTS = {LICO2_WALL_Y: 0.965}
LICO2_WALL_BOUNDS = {LICO2_WALL_Y: (0.93, 0.99)}
LICO2_WALL_FROM = 0.90
LICO2_WALL_DROP_V = 0.8
LICO2_WALL_WIDTH = 0.004


def lico2_ocp_plateau_wall(sto):
    """Ai2020 LiCoO2 OCP to LICO2_WALL_FROM, the plateau slope continued, minus the wall."""
    import pybamm

    x, y = _lico2_monotone_table()
    order = np.argsort(x)
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    slope = float((np.interp(LICO2_WALL_FROM, xs, ys) - np.interp(LICO2_WALL_FROM - 0.05, xs, ys)) / 0.05)
    base = pybamm.Interpolant(x, y, pybamm.minimum(sto, LICO2_WALL_FROM), name=LICO2_MONOTONE_NAME,
                              interpolator="cubic")
    plateau = base + slope * pybamm.maximum(sto - LICO2_WALL_FROM, 0.0)
    wall = 0.5 * LICO2_WALL_DROP_V * (1.0 + pybamm.tanh((sto - pybamm.Parameter(LICO2_WALL_Y)) / LICO2_WALL_WIDTH))
    return plateau - wall


#: Stoichiometry-dependent LiCoO2 electrode conductivity (``lico2_sigma="mit"`` of :func:`lir2032`):
#: Li_yCoO2 is metallic when delithiated (~1e0-1e3 S/cm) and semiconducting as LiCoO2 (~1e-3 S/cm),
#: the metal-insulator transition spanning the two-phase region y ~ 0.75-0.94 (Ohzuku & Ueda, J.
#: Electrochem. Soc. 141, 2972 (1994), citing ref. 20; Menetrier et al., J. Mater. Chem. 9, 1135
#: (1999)). sigma(y) = sigma_0 x ratio**w(y), w the insulating-phase fraction by the lever rule
#: over LICO2_MIT_RANGE, PyBaMM's sigma(sto, T) evaluated at the surface stoichiometry. Caveat: the
#: matrix conductivity includes the carbon additive, so this is a surrogate for the particle-level
#: electronic resistance; LICO2_SIGMA_RATIO (lithiated / delithiated) is therefore an effective,
#: bounded quantity, not the bulk LiCoO2 ratio.
LICO2_SIGMA_RATIO = "Positive electrode conductivity ratio lithiated / delithiated"
LICO2_SIGMA_DEFAULTS = {LICO2_SIGMA_RATIO: 1.0}
LICO2_MIT_RANGE = (0.75, 0.94)


def _lico2_mit_conductivity(sigma_0: float):
    import pybamm

    lo, hi = LICO2_MIT_RANGE

    def sigma(*args):
        if len(args) == 1:                       # sigma(T): callers without a stoichiometry
            return pybamm.Scalar(sigma_0)
        sto, _ = args
        w = pybamm.minimum(pybamm.maximum((sto - lo) / (hi - lo), 0.0), 1.0)
        return sigma_0 * pybamm.exp(w * pybamm.log(pybamm.Parameter(LICO2_SIGMA_RATIO)))

    return sigma


def _total_electrode_area_m2(pv) -> float:
    return (float(pv["Electrode height [m]"]) * float(pv["Electrode width [m]"])
            * float(pv["Number of electrodes connected in parallel to make a cell"]))


#: Graphite OCP alternatives for the LiR2032 set (``graphite_ocp`` of :func:`lir2032`).
#: "Ramadass2004": MCMB graphite (Sony 18650 LiCoO2/MCMB, Ramadass et al. 2004), with
#: its Moura2016 entropic term. Chosen by the jones2022 checks: the rest-voltage OCV
#: fit (4.7 vs 7.2 mV, ocp_source_compare) and the formation dV/dQ shape (log-dV/dQ
#: correlation 0.89 vs 0.33, dvdq_analysis). Caveat: its smooth staging may win partly
#: because 0.95C data blur graphite staging; compare with a C/7 reference (Inaba et al.
#: 2026, J. Electrochem. Soc. 173:040515) before treating it as the cell's graphite.
#: "Chen2020": LG M50 graphite measured by GITT (Chen et al. 2020, J. Electrochem.
#: Soc. 167:080534), resolved staging; entropic term zero as in Chen2020. It is the
#: only tested graphite (with Ai2020 LiCoO2 and the literature-limited rest-voltage
#: balance) whose zero-rate dQ/dV has a small ~3.68 V peak below a dominant ~3.77 V
#: one -- the pattern of the same-format LIR2032 reference at C/7 (Inaba et al. 2026,
#: peaks 3.72 / 3.82 V charge, 3.665 / ~3.75 V discharge); rest-voltage RMSE 5.8 mV.
#: "Ecker2015" / "Mohtat2020": pure-graphite GITT curves (Ecker et al. 2015 Kokam;
#: Mohtat et al. 2020), entropic term zero.
GRAPHITE_OCP_OPTIONS = (None, "Ramadass2004", "Chen2020", "Ecker2015", "Mohtat2020")
#: LiCoO2 OCP alternatives (``lico2_ocp``): None = Ai2020 (monotone continuation);
#: "Marquis2019" = Dualfoil LiCoO2 (analytic over 0-1; resolves the ~4.0-4.1 V
#: order-disorder region), entropic term zero.
LICO2_OCP_OPTIONS = (None, "Marquis2019", "PlateauWall")


def _ai2020_electrolyte_diffusivity_in_cm2(pv) -> bool:
    """True if this PyBaMM's Ai2020 electrolyte diffusivity still returns cm2/s (the bug
    AI2020_ELECTROLYTE_DIFFUSIVITY_UNIT_FIX corrects); False once upstream returns m2/s
    (PyBaMM >= 26.10 multiplies by 1e-4 itself). Decided from the value at 1000 mol/m3 and
    298 K: ~3e-6 in cm2/s, ~3e-10 in m2/s. Applying the fix twice makes D_e 1e4 x too small
    (the electrolyte then depletes within minutes at 1C)."""
    import pybamm

    f = pv["Electrolyte diffusivity [m2.s-1]"]
    if not callable(f):
        return float(f) > 1e-8
    value = float(np.asarray(pv.process_symbol(f(pybamm.Scalar(1000.0), pybamm.Scalar(298.15))).evaluate()).ravel()[0])
    return value > 1e-8


def lir2032(template: str = "Ai2020", nominal_ah: float = LIR2032_NOMINAL_AH,
            graphite_ocp: str | None = None, lico2_ocp: str | None = None, lico2_ds: str = "knee",
            lico2_sigma: str | None = None):
    """``pybamm.ParameterValues`` for the jones2022 LiR2032 coin cell (see module doc).
    ``graphite_ocp`` replaces the template's graphite OCP (GRAPHITE_OCP_OPTIONS).
    ``lico2_ds``: "knee" (D x 10**(-drop s^2) above a fitted knee, LICO2_DS_KNEE),
    "thermo" (D0 x the OCP's thermodynamic factor, LICO2_THETA_NAME; no shape parameters) or
    "vacancy" (D0 x max(1 - y, LICO2_VACANCY_FLOOR); no shape parameters)."""
    import pybamm

    if lico2_ds not in ("knee", "thermo", "vacancy"):
        raise ValueError(f"lico2_ds must be 'knee', 'thermo' or 'vacancy', got {lico2_ds!r}")
    if lico2_ds == "thermo" and lico2_ocp not in (None, "Marquis2019"):
        raise ValueError("lico2_ds='thermo' needs the Ai2020 table (lico2_ocp=None) or an analytic OCP "
                         "(lico2_ocp='Marquis2019')")

    if lico2_ocp not in LICO2_OCP_OPTIONS:
        raise ValueError(f"lico2_ocp must be one of {LICO2_OCP_OPTIONS}, got {lico2_ocp!r}")
    if graphite_ocp not in GRAPHITE_OCP_OPTIONS:
        raise ValueError(f"graphite_ocp must be one of {GRAPHITE_OCP_OPTIONS}, got {graphite_ocp!r}")

    if template not in LIR2032_TEMPLATES:
        raise ValueError(f"template must be one of {LIR2032_TEMPLATES}, got {template!r}")
    pv = pybamm.ParameterValues(template)
    area = _total_electrode_area_m2(pv) * nominal_ah / float(pv["Nominal cell capacity [A.h]"])
    side = math.sqrt(area)
    pv.update({
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Electrode height [m]": side,
        "Electrode width [m]": side,
        "Nominal cell capacity [A.h]": nominal_ah,
        "Lower voltage cut-off [V]": LIR2032_V_MIN,
        "Upper voltage cut-off [V]": LIR2032_V_MAX,
        "Contact resistance [Ohm]": 0.0,
        "Cell volume [m3]": COIN_VOLUME_M3,
        "Cell cooling surface area [m2]": COIN_COOLING_AREA_M2,
        "Cell heat capacity [J.K-1.m-3]": COIN_HEAT_CAPACITY_J_K / COIN_VOLUME_M3,
        "Total heat transfer coefficient [W.m-2.K-1]": COIN_H_DEFAULT,
    }, check_already_exists=False)
    if template == "Ai2020":
        # only the unit fix depends on the PyBaMM version; everything below it applies always
        # (until 2026-10-20 the whole block sat under this check, so on PyBaMM >= 26.10 the
        # monotone OCP, activation energies, D_s forms and j0 shape were all silently dropped)
        if _ai2020_electrolyte_diffusivity_in_cm2(pv):
            from battery_oracle.params import apply_parameter_scales
            apply_parameter_scales(pv, {"Electrolyte diffusivity [m2.s-1]":
                                        AI2020_ELECTROLYTE_DIFFUSIVITY_UNIT_FIX})
        pv.update({"Positive electrode OCP [V]": lico2_ocp_monotone})
        pv.update({name: _with_activation(pv[name], e) for name, e in LIR2032_ACTIVATION_J_MOL.items()})
        if lico2_ds == "thermo" and lico2_ocp is None:
            pv.update({"Positive particle diffusivity [m2.s-1]":
                       _with_thermodynamic_factor(pv["Positive particle diffusivity [m2.s-1]"])})
        elif lico2_ds == "thermo":
            pass                                  # Theta from the OCP installed below
        elif lico2_ds == "vacancy":
            pv.update({"Positive particle diffusivity [m2.s-1]":
                       _with_vacancy_limit(pv["Positive particle diffusivity [m2.s-1]"])})
        else:
            pv.update({"Positive particle diffusivity [m2.s-1]":
                       _with_lithiation_drop(pv["Positive particle diffusivity [m2.s-1]"])})
            pv.update(dict(LICO2_DS_DEFAULTS), check_already_exists=False)
        pv.update({"Negative particle diffusivity [m2.s-1]":
                   _with_lithiation_drop(pv["Negative particle diffusivity [m2.s-1]"],
                                         GRAPHITE_DS_KNEE, GRAPHITE_DS_DROP)})
        pv.update(dict(GRAPHITE_DS_DEFAULTS), check_already_exists=False)
        pv.update({"Negative electrode exchange-current density [A.m-2]":
                   _with_j0_shape(pv["Negative electrode exchange-current density [A.m-2]"], J0_SHAPE_N),
                   "Positive electrode exchange-current density [A.m-2]":
                   _with_j0_shape(pv["Positive electrode exchange-current density [A.m-2]"], J0_SHAPE_P)})
        pv.update(dict(J0_SHAPE_DEFAULTS), check_already_exists=False)
        _lico2_monotone_table()          # registers the data range
    if graphite_ocp == "Ramadass2004":
        from pybamm.input.parameters.lithium_ion.Ramadass2004 import (
            graphite_entropic_change_Moura2016, graphite_ocp_Ramadass2004)
        pv.update({"Negative electrode OCP [V]": graphite_ocp_Ramadass2004,
                   "Negative electrode OCP entropic change [V.K-1]": graphite_entropic_change_Moura2016})
    elif graphite_ocp == "Chen2020":
        from pybamm.input.parameters.lithium_ion.Chen2020 import graphite_LGM50_ocp_Chen2020
        pv.update({"Negative electrode OCP [V]": graphite_LGM50_ocp_Chen2020,
                   "Negative electrode OCP entropic change [V.K-1]": 0.0})
    elif graphite_ocp in ("Ecker2015", "Mohtat2020"):
        pv.update({"Negative electrode OCP [V]": pybamm.ParameterValues(graphite_ocp)["Negative electrode OCP [V]"],
                   "Negative electrode OCP entropic change [V.K-1]": 0.0})
    if lico2_ocp == "PlateauWall":
        if template != "Ai2020":
            raise ValueError("lico2_ocp='PlateauWall' continues the Ai2020 LiCoO2 table (template='Ai2020')")
        pv.update({"Positive electrode OCP [V]": lico2_ocp_plateau_wall})
        pv.update(dict(LICO2_WALL_DEFAULTS), check_already_exists=False)
    if lico2_sigma == "mit":
        pv.update({"Positive electrode conductivity [S.m-1]":
                   _lico2_mit_conductivity(float(pv["Positive electrode conductivity [S.m-1]"]))})
        pv.update(dict(LICO2_SIGMA_DEFAULTS), check_already_exists=False)
    elif lico2_sigma is not None:
        raise ValueError(f"lico2_sigma must be None or 'mit', got {lico2_sigma!r}")
    if lico2_ocp == "Marquis2019":
        pv.update({"Positive electrode OCP [V]": pybamm.ParameterValues("Marquis2019")["Positive electrode OCP [V]"],
                   "Positive electrode OCP entropic change [V.K-1]": 0.0})
        if lico2_ds == "thermo":
            pv.update({"Positive particle diffusivity [m2.s-1]":
                       _with_ocp_thermodynamic_factor(pv["Positive particle diffusivity [m2.s-1]"],
                                                      pv["Positive electrode OCP [V]"])})
    return pv


#: Custom sets by name -> builder. Names here are accepted wherever the oracle
#: takes a PyBaMM parameter-set name (``chemistry`` / ``cycling.parameter_set``).
CUSTOM_PARAMETER_SETS = {
    "LiR2032": lir2032,
    "LiR2032_Ramadass2004": lambda: lir2032(template="Ramadass2004"),
    "LiR2032_RamadassGraphite": lambda: lir2032(graphite_ocp="Ramadass2004"),
    "LiR2032_ChenGraphite": lambda: lir2032(graphite_ocp="Chen2020"),
    "LiR2032_EckerGraphite": lambda: lir2032(graphite_ocp="Ecker2015"),
    "LiR2032_MohtatGraphite": lambda: lir2032(graphite_ocp="Mohtat2020"),
    "LiR2032_MohtatGraphite_ThermoD": lambda: lir2032(graphite_ocp="Mohtat2020", lico2_ds="thermo"),
    "LiR2032_MohtatGraphite_VacancyD": lambda: lir2032(graphite_ocp="Mohtat2020", lico2_ds="vacancy"),
    # PyBaMM's own Dualfoil LiCoO2 OCP (Marquis2019; stoichiometry stretched by 1.062): flat 3.91 V
    # plateau ending in a near-vertical wall at y ~ 0.95-0.97, as measured (Reimers & Dahn 1992;
    # Ohzuku & Ueda 1994) -- preferred over the custom PlateauWall below (upstream first)
    "LiR2032_MohtatGraphite_MarquisLCO": lambda: lir2032(graphite_ocp="Mohtat2020", lico2_ocp="Marquis2019"),
    "LiR2032_MohtatGraphite_MarquisLCO_VacancyD": lambda: lir2032(graphite_ocp="Mohtat2020", lico2_ocp="Marquis2019",
                                                                  lico2_ds="vacancy"),
    # thermodynamically consistent: D = D0 Theta(y, T) from the same (analytic) OCP
    "LiR2032_MohtatGraphite_MarquisLCO_ThermoD": lambda: lir2032(graphite_ocp="Mohtat2020", lico2_ocp="Marquis2019",
                                                                 lico2_ds="thermo"),
    # + LiCoO2 metal-insulator transition in the electrode conductivity (a-priori test, 2026-10-21)
    "LiR2032_MohtatGraphite_MarquisLCO_ThermoD_MIT": lambda: lir2032(graphite_ocp="Mohtat2020",
                                                                     lico2_ocp="Marquis2019", lico2_ds="thermo",
                                                                     lico2_sigma="mit"),
    "LiR2032_MohtatGraphite_Wall": lambda: lir2032(graphite_ocp="Mohtat2020", lico2_ocp="PlateauWall"),
    "LiR2032_MohtatGraphite_Wall_VacancyD": lambda: lir2032(graphite_ocp="Mohtat2020", lico2_ocp="PlateauWall",
                                                            lico2_ds="vacancy"),
    "LiR2032_EckerGraphite_MarquisLCO": lambda: lir2032(graphite_ocp="Ecker2015", lico2_ocp="Marquis2019"),
    "LiR2032_ChenGraphite_MarquisLCO": lambda: lir2032(graphite_ocp="Chen2020", lico2_ocp="Marquis2019"),
}


def resolve_parameter_values(name: str):
    """A custom set from :data:`CUSTOM_PARAMETER_SETS`, else ``pybamm.ParameterValues(name)``."""
    import pybamm

    builder = CUSTOM_PARAMETER_SETS.get(name)
    return builder() if builder is not None else pybamm.ParameterValues(name)


def nominal_capacity_mah(name: str) -> float:
    """Nominal capacity [mAh] of a named (custom or PyBaMM) parameter set."""
    return 1000.0 * float(resolve_parameter_values(name)["Nominal cell capacity [A.h]"])
