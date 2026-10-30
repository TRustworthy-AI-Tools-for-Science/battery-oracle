"""Custom PyBaMM parameter sets for the oracle.

``LiR2032`` describes the commercial rechargeable LiCoO2/graphite coin cell used in
the jones2022 dataset (Powerstream LiR2032; 35 mAh nominal, ~42 mAh measured;
Jones, Stimming & Lee 2022, Nat. Commun. 13:4806), at its real size: the oracle
simulates the real cell's currents directly, with no scaling onto a larger cell.

It is built on PyBaMM's Ai2020 LiCoO2/graphite set (Ai et al. 2020, J. Electrochem.
Soc. 167:013512), which supplies everything the jones2022 data cannot identify --
thicknesses, porosities, active-material fractions, particle radii, Bruggeman
exponents, electrolyte transport, conductivities, thermal and SEI parameters. What
changes from it here (the stage-A fit refines the rest against the real data):

* **one electrode pair** (the template's 34-layer pouch becomes a single layer of
  the cell's total electrode area);
* **electrode area** scaled from the template's area-per-capacity to the nominal
  capacity (~15 cm2 for 42 mAh). A commercial LiR2032 cannot be a single 15 mm disc
  -- at these loadings that holds only ~6 mAh -- so the area is an effective, fitted
  quantity (the stage-A balance sets it from the fitted electrode capacities);
* **nominal capacity** 42 mAh (measured on the jones2022 cells; datasheet 35 mAh);
* **voltage window** 3.0-4.3 V (jones2022 protocol cut-offs);
* **coin-cell thermal** volume, cooling area and heat capacity of the LiR2032 can;
* **activation energies** of a LiCoO2/graphite set (Ramadass et al. 2004), replacing
  Dualfoil's 5 kJ/mol;
* **LiCoO2 OCP** of PyBaMM's Marquis2019 (Dualfoil LiCoO2, stoichiometry stretched by
  1.062): a flat ~3.91 V plateau ending in a near-vertical wall at y ~ 0.95-0.97, as
  measured (Reimers & Dahn 1992; Ohzuku & Ueda 1994);
* **LiCoO2 solid diffusivity** D = D0 Theta(y, T), the thermodynamic factor of that
  same OCP (see :func:`_with_ocp_thermodynamic_factor`); only D0 is fitted;
* **graphite OCP** of Mohtat2020 (chosen by the formation dQ/dV peaks; alternatives in
  :data:`GRAPHITE_OCP_OPTIONS`).
"""
from __future__ import annotations

import math

import numpy as np

#: Measured capacity of the jones2022 LiR2032 cells (deepest discharge, ~42 mAh).
LIR2032_NOMINAL_AH = 0.042
#: Voltage window of the jones2022 protocol (Jones et al. 2022 Methods).
LIR2032_V_MIN, LIR2032_V_MAX = 3.0, 4.3
#: LiR2032 can: 20 mm diameter, 3.2 mm high.
COIN_RADIUS_M, COIN_HEIGHT_M = 0.010, 0.0032
COIN_VOLUME_M3 = math.pi * COIN_RADIUS_M ** 2 * COIN_HEIGHT_M
COIN_COOLING_AREA_M2 = 2 * math.pi * COIN_RADIUS_M ** 2 + 2 * math.pi * COIN_RADIUS_M * COIN_HEIGHT_M
#: Whole-cell heat capacity [J/K]: LIR2032 2.5-2.8 g (datasheets) x 0.96-1.04 J/(g K)
#: (LiCoO2 cells, Maleki et al. 1999, J. Electrochem. Soc. 146:947).
COIN_HEAT_CAPACITY_J_K = 2.6
#: Prior centre for the heat-transfer coefficient [W/(m2 K)] (fitted in stage A).
COIN_H_DEFAULT = 10.0
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
#: Floor on the LiCoO2 thermodynamic factor: a numerical guard only (the analytic,
#: monotone Marquis2019 OCP keeps Theta > 0).
LICO2_THETA_FLOOR = 1.0e-3
#: Graphite OCP options of :func:`lir2032`: "Mohtat2020" (default), "Ecker2015" (pure-graphite
#: GITT curves; entropic term zero), "Chen2020" (LG M50 graphite by GITT, resolved staging;
#: Chen et al. 2020, J. Electrochem. Soc. 167:080534), "Ramadass2004" (MCMB graphite with its
#: Moura2016 entropic term), "Ai2020" (the template's own).
GRAPHITE_OCP_OPTIONS = ("Mohtat2020", "Ecker2015", "Chen2020", "Ramadass2004", "Ai2020")


def _with_activation(fn, e_new: float, e_old: float = AI2020_ACTIVATION_J_MOL):
    """``fn`` (its last argument is the temperature) re-referenced from activation
    energy ``e_old`` to ``e_new``: x exp((e_new - e_old)/R (1/T_ref - 1/T))."""
    import pybamm

    def wrapped(*args):
        T = args[-1]
        return fn(*args) * pybamm.exp((e_new - e_old) / pybamm.constants.R * (1.0 / T_REF_K - 1.0 / T))

    return wrapped


def _with_ocp_thermodynamic_factor(fn, ocp):
    """``fn(sto, T)`` x Theta(sto, T) = -(F / R T) sto (1 - sto) dU/dsto, with U the positive OCP
    ``ocp`` differentiated symbolically: the chemical diffusivity of a lattice gas with site
    exclusion, D = D0 Theta. Flux N = -M c grad(mu), vacancy-limited mobility M = D0 (1 - y)/kT
    (Bazant, Acc. Chem. Res. 46, 1144 (2013)), mu = -F U(y), gives Fick's law with this D; Theta
    = 1 for an ideal lattice gas. The shape of D(y) is the OCP's, so only D0 is free. The
    collapse near full lithiation is the vacancy mechanism measured for Li_xCoO2 (Hasegawa,
    Kuwata et al., Phys. Chem. Chem. Phys. 2021, doi 10.1039/d0cp04598e)."""
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


def _total_electrode_area_m2(pv) -> float:
    return (float(pv["Electrode height [m]"]) * float(pv["Electrode width [m]"])
            * float(pv["Number of electrodes connected in parallel to make a cell"]))


def _graphite_ocp(name: str) -> dict:
    """Negative-electrode OCP (and its entropic term) of a graphite option."""
    import pybamm

    if name == "Ramadass2004":
        from pybamm.input.parameters.lithium_ion.Ramadass2004 import (
            graphite_entropic_change_Moura2016, graphite_ocp_Ramadass2004)
        return {"Negative electrode OCP [V]": graphite_ocp_Ramadass2004,
                "Negative electrode OCP entropic change [V.K-1]": graphite_entropic_change_Moura2016}
    if name == "Chen2020":
        from pybamm.input.parameters.lithium_ion.Chen2020 import graphite_LGM50_ocp_Chen2020
        return {"Negative electrode OCP [V]": graphite_LGM50_ocp_Chen2020,
                "Negative electrode OCP entropic change [V.K-1]": 0.0}
    return {"Negative electrode OCP [V]": pybamm.ParameterValues(name)["Negative electrode OCP [V]"],
            "Negative electrode OCP entropic change [V.K-1]": 0.0}


def lir2032(nominal_ah: float = LIR2032_NOMINAL_AH, graphite_ocp: str = "Mohtat2020"):
    """``pybamm.ParameterValues`` for the jones2022 LiR2032 coin cell (see module doc);
    ``graphite_ocp`` from :data:`GRAPHITE_OCP_OPTIONS`."""
    import pybamm

    if graphite_ocp not in GRAPHITE_OCP_OPTIONS:
        raise ValueError(f"graphite_ocp must be one of {GRAPHITE_OCP_OPTIONS}, got {graphite_ocp!r}")
    pv = pybamm.ParameterValues("Ai2020")
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
    pv.update({name: _with_activation(pv[name], e) for name, e in LIR2032_ACTIVATION_J_MOL.items()})
    if graphite_ocp != "Ai2020":
        pv.update(_graphite_ocp(graphite_ocp))
    pv.update({"Positive electrode OCP [V]": pybamm.ParameterValues("Marquis2019")["Positive electrode OCP [V]"],
               "Positive electrode OCP entropic change [V.K-1]": 0.0})
    pv.update({"Positive particle diffusivity [m2.s-1]":
               _with_ocp_thermodynamic_factor(pv["Positive particle diffusivity [m2.s-1]"],
                                              pv["Positive electrode OCP [V]"])})
    return pv


#: Custom sets by name -> builder. Names here are accepted wherever the oracle
#: takes a PyBaMM parameter-set name (``chemistry`` / ``cycling.parameter_set``).
#: "LiR2032_MohtatGraphite_MarquisLCO_ThermoD" is the name the stage-A fits were run under.
CUSTOM_PARAMETER_SETS = {
    "LiR2032": lir2032,
    "LiR2032_MohtatGraphite_MarquisLCO_ThermoD": lir2032,
}


def resolve_parameter_values(name: str):
    """A custom set from :data:`CUSTOM_PARAMETER_SETS`, else ``pybamm.ParameterValues(name)``."""
    import pybamm

    builder = CUSTOM_PARAMETER_SETS.get(name)
    return builder() if builder is not None else pybamm.ParameterValues(name)


def nominal_capacity_mah(name: str) -> float:
    """Nominal capacity [mAh] of a named (custom or PyBaMM) parameter set."""
    return 1000.0 * float(resolve_parameter_values(name)["Nominal cell capacity [A.h]"])


def ocp_data_range(pv, domain: str) -> tuple[float, float]:
    """Stoichiometry range covered by an electrode OCP's data (its interpolant's x range);
    ``(0, 1)`` for an analytic OCP. Outside it PyBaMM extrapolates."""
    import pybamm

    f = pv[f"{domain.capitalize()} electrode OCP [V]"]
    sym = f(pybamm.Scalar(0.5)) if callable(f) else f
    if isinstance(sym, pybamm.Symbol):
        for n in sym.pre_order():
            if isinstance(n, pybamm.Interpolant):
                x = np.asarray(n.x[0])
                return float(np.min(x)), float(np.max(x))
    return 0.0, 1.0
