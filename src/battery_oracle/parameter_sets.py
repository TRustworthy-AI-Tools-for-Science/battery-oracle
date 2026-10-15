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


def _lico2_monotone_table(step: float = 0.01, n_slope: int = 5):
    """Ai2020 LiCoO2 OCP table + a linear continuation from y = 0 to the data start."""
    import numpy as np
    from pybamm.input.parameters.lithium_ion.Ai2020 import lico2_ocp_Ai2020_data

    _, (x, y) = lico2_ocp_Ai2020_data
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    order = np.argsort(x)
    x, y = x[order], y[order]
    slope = np.polyfit(x[:n_slope], y[:n_slope], 1)[0]          # dU/dy < 0 at the data start
    x_ext = np.arange(0.0, x[0] - step / 2, step)
    y_ext = y[0] + slope * (x_ext - x[0])
    OCP_DATA_RANGES[LICO2_MONOTONE_NAME] = (float(x[0]), float(x[-1]))
    return np.r_[x_ext, x], np.r_[y_ext, y]


def lico2_ocp_monotone(sto):
    """Ai2020 LiCoO2 OCP, continued linearly (monotonic) below its data (see module doc)."""
    import pybamm

    x, y = _lico2_monotone_table()
    return pybamm.Interpolant(x, y, sto, name=LICO2_MONOTONE_NAME, interpolator="cubic")


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
LICO2_OCP_OPTIONS = (None, "Marquis2019")


def lir2032(template: str = "Ai2020", nominal_ah: float = LIR2032_NOMINAL_AH,
            graphite_ocp: str | None = None, lico2_ocp: str | None = None):
    """``pybamm.ParameterValues`` for the jones2022 LiR2032 coin cell (see module doc).
    ``graphite_ocp`` replaces the template's graphite OCP (GRAPHITE_OCP_OPTIONS)."""
    import pybamm

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
        from battery_oracle.params import apply_parameter_scales
        apply_parameter_scales(pv, {"Electrolyte diffusivity [m2.s-1]":
                                    AI2020_ELECTROLYTE_DIFFUSIVITY_UNIT_FIX})
        pv.update({"Positive electrode OCP [V]": lico2_ocp_monotone})
        pv.update({name: _with_activation(pv[name], e) for name, e in LIR2032_ACTIVATION_J_MOL.items()})
        pv.update({"Positive particle diffusivity [m2.s-1]":
                   _with_lithiation_drop(pv["Positive particle diffusivity [m2.s-1]"])})
        pv.update(dict(LICO2_DS_DEFAULTS), check_already_exists=False)
        pv.update({"Negative particle diffusivity [m2.s-1]":
                   _with_lithiation_drop(pv["Negative particle diffusivity [m2.s-1]"],
                                         GRAPHITE_DS_KNEE, GRAPHITE_DS_DROP)})
        pv.update(dict(GRAPHITE_DS_DEFAULTS), check_already_exists=False)
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
    if lico2_ocp == "Marquis2019":
        pv.update({"Positive electrode OCP [V]": pybamm.ParameterValues("Marquis2019")["Positive electrode OCP [V]"],
                   "Positive electrode OCP entropic change [V.K-1]": 0.0})
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
