"""Stage A, one continuous measured history per cell (the measurement as performed).

Replaces :mod:`battery_oracle.bol_joint`, which simulated every record on its own,
starting from a relaxed cell at the record's first voltage. The jones2022 records
run back to back (a cycling record starts ~5 min after the previous record's
20 min rest and EIS sweep, while the voltage still drifts ~10 uV/s), so that
initial state threw away the gradients the non-stationary EIS patch exists to
model, and re-anchored the state of charge at every record.

Here, per cell and per candidate parameter set, ONE model is integrated ONCE
through the cell's whole measured current history -- formation, every cycling
record, every rest and every EIS sweep, in order -- starting from the as-received
cell, the only truly relaxed state. From that one solution:

* each GCPL record's simulated voltage is compared with its measurement;
* each EIS spectrum is the model linearised at the simulated state where each
  frequency was measured (window = between consecutive measured time stamps on
  the cycling clock), plus the lock-in drift leakage from the simulated dV/dt
  (:mod:`battery_oracle.eis_nonstationary`).

Model (``CHAIN_OPTIONS``): DFN with the EIS voltage/current variables (PyBOP's
``EISSimulator.set_up_for_eis``); double layer (``surface form: differential``);
contact resistance; lumped thermal with the coin cell's whole-cell heat capacity
(:mod:`battery_oracle.parameter_sets`); SEI ``constant`` -- the SEI film
resistance the oracle's SEI model has at beginning of life, without growth.

Fitted (PyBOP ``Parameter``; bounds centred on the warm start because PyBOP's
population optimisers start at the prior mean): the seven kinetic/transport/
double-layer multipliers and electrolyte diffusivity, contact resistance, electrode
area (parallel-layer count), negative active fraction (N:P; the porosity follows so
the volume fractions sum to 1), electrode thickness (both electrodes),
cyclable lithium (independent of N:P), heat-transfer coefficient, and the length
of each unrecorded rest between formation and cycling sequences.

Initial state: the cell's first voltage (as received, after storage) is an open-
circuit voltage; with the cyclable lithium it fixes both electrode stoichiometries
(``initial_stoichiometries``). Electrode OCPs are guarded: a simulated surface
stoichiometry beyond the OCP data range by more than ``OCP_GUARD_TOL`` fails the
candidate; smaller excursions are counted and reported.

Each view (``ChainView``) is a PyBOP simulator over one record or spectrum; all
views of a cell share one ``ChainModel``, which caches solutions, so a candidate
is integrated once however many views it is scored on.
"""
from __future__ import annotations

import logging
from pathlib import Path
import os
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from battery_oracle.bol_fit import EIS_SCALES, ocp_data_range

#: Multiplied parameters of the chain fit: the EIS set plus electrolyte diffusivity
#: (the 1-4C coin-cell steps are electrolyte-transport limited; without it the fit
#: drives the electrolyte to ~2% of its initial concentration on 3C discharges).
CHAIN_SCALES = tuple(EIS_SCALES) + ("Electrolyte diffusivity [m2.s-1]",)
from battery_oracle.bol_joint import (CONTACT_R, EPS_N, N_PARALLEL, OCV0, OCV100, SIGMA_V, SIGMA_Z_REL,
                                      _centred, widened)
from battery_oracle.params import apply_parameter_overrides, apply_parameter_scales, scale_parameter_name

log = logging.getLogger(__name__)

CHAIN_OPTIONS = {
    "surface form": "differential",
    "contact resistance": "true",
    "thermal": "lumped",
    "use lumped thermal capacity": "true",
    "SEI": "constant",
}
#: SEI submodels for stage A. "constant": beginning-of-life film resistance, no
#: growth. "ec reaction limited": the oracle's growth model (oracle.py
#: _build_degradation_config), so the fitted balance carries the lithium the SEI
#: consumes -- with "constant" the fit can leave no lithium margin and the oracle's
#: SEI growth then empties the graphite at the end of a discharge (R1 finding).
SEI_MODES = ("constant", "ec reaction limited")
#: default electrolyte floor (fraction of c0) for the depletion stop; env ORACLE_FIT_CE_FLOOR
CE_FLOOR_DEFAULT = 0.0
#: oracle EC-diffusivity factor = 0.25 (commercial-electrolyte correction) x sei_rate_scale
EC_DIFFUSIVITY = "EC diffusivity [m2.s-1]"


#: Graphite OCP hysteresis (PyBaMM one-state model, Axen et al. 2022): lithiation /
#: delithiation branches U -/+ HYST_DELTA; the state h relaxes over a change in
#: surface stoichiometry of ~2/HYST_DECAY (dh/dsto = gamma (1 - sign h)/2), i.e. ~4%
#: of the graphite range at gamma 50, and persists through rests. Graphite staging
#: shows ~10-20 mV between lithiation and delithiation (Mercer et al. 2021, J. Mater.
#: Chem. A, doi 10.1039/D0TA10403E; Li et al. 2022 Electrochemistry, 0.05C) -> half-gap
#: bounds 0-12.5 mV.
HYST_DELTA = "Negative electrode OCP hysteresis half-gap [V]"
HYST_BOUNDS = (0.0, 0.0125)
HYST_DECAY = 50.0
#: Particle-size distribution (PyBaMM "particle size": "distribution", lognormal,
#: area-weighted; get_size_distribution_parameters): relative width sigma_R / R_av of
#: each electrode. Commercial graphite and LiCoO2 powders: ~0.3-0.5 (D10-D90 spread);
#: bounds 0.15-0.6. Mesh of PSD_POINTS radii per electrode keeps the cost bounded.
PSD_SD_N = "Negative particle radius relative width"
PSD_SD_P = "Positive particle radius relative width"
PSD_BOUNDS = {PSD_SD_N: (0.15, 0.6), PSD_SD_P: (0.15, 0.6)}
PSD_POINTS = int(os.environ.get("ORACLE_FIT_PSD_POINTS", "6"))

#: particle radial mesh (checklist 4: the default 20 uniform points are not converged for
#: the LiCoO2 surface gradients). env ORACLE_FIT_R_POINTS (points per particle, default:
#: PyBaMM's) and ORACLE_FIT_R_MESH ("uniform" or "surface": exponential, clustered at r = R)
R_POINTS = os.environ.get("ORACLE_FIT_R_POINTS")
R_MESH = os.environ.get("ORACLE_FIT_R_MESH", "uniform")
R_MESH_STRETCH = float(os.environ.get("ORACLE_FIT_R_STRETCH", "1.15"))
#: electrode/separator (x) mesh, as a factor on PyBaMM's points: the fit runs on
#: ORACLE_FIT_X_FACTOR (default 1); the final evaluation, the validity gate and the
#: oracle use ORACLE_FIT_VALIDATE_X_FACTOR (default 2). The x mesh moves the fitted
#: voltage by <= 0.3 mV (verify/mesh_domains.py) but the default one does not resolve
#: the salt flux at the negative|separator interface during fast charge (jump 0.10 of
#: the current at x1, 0.033 at x2; diag/validity_probe.py, 2026-10-18).
X_FACTOR = int(os.environ.get("ORACLE_FIT_X_FACTOR", "1"))
VALIDATE_X_FACTOR = int(os.environ.get("ORACLE_FIT_VALIDATE_X_FACTOR", "2"))


def radial_mesh(points=None, kind: str = "uniform", stretch: float = 1.15):
    """(var_pts, submesh_types) overrides for both particles' radial mesh."""
    var_pts = {"r_n": int(points), "r_p": int(points)} if points else {}
    if kind == "uniform":
        return var_pts, {}
    if kind != "surface":
        raise ValueError(f"radial mesh kind {kind!r}: use 'uniform' or 'surface'")
    import pybamm
    sub = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params={"side": "right", "stretch": stretch})
    return var_pts, {"negative particle": sub, "positive particle": sub}


def radial_mesh_setting(x_factor: int | None = None) -> dict | None:
    """The mesh set by ORACLE_FIT_R_* (and the x factor) as an oracle setting
    (``radial_mesh`` of PyBaMMOracle): {"points", "kind", "stretch", "x_factor"}; None
    for PyBaMM's default mesh. ``x_factor`` defaults to the validation mesh's."""
    xf = VALIDATE_X_FACTOR if x_factor is None else int(x_factor)
    if not R_POINTS and R_MESH == "uniform" and xf == 1:
        return None
    return {"points": int(R_POINTS) if R_POINTS else None, "kind": R_MESH, "stretch": R_MESH_STRETCH,
            "x_factor": xf}


def x_var_pts(model, factor: int) -> dict:
    """x points of ``model``'s default mesh times ``factor``."""
    d = model.default_var_pts
    return {k: int(d[k]) * int(factor) for k in ("x_n", "x_s", "x_p")} if factor != 1 else {}
#: Cycler voltage limits of the jones2022 protocol, from the raw GCPL files (PJ248,
#: PJ250; diag/cutoff_behaviour.py) and Jones et al. 2022 Methods: formation charge CC
#: to 4.2 V then CV (40 -> ~4 mA); every cycling charge stage CC with a hard stop at
#: 4.3 V (current to 0 within 1 s, no CV); every discharge a hard stop at 3.0 V. The
#: measured current is the real cell's response to these limits: a model cell that
#: reaches a limit earlier must stop (or hold, in CV) where the real cycler would have,
#: not be forced on (pilot 12: 2.76 V at the end of the first formation discharge,
#: measured 3.00 V).
#: Hard stops are EXACT: the cut-off is an input parameter CUT_INPUT (0 conducting, 1
#: stopped) and the solve runs in segments (ChainModel._solve_segments). A terminating
#: event stops the integration where the voltage crosses its limit in the current's
#: direction; the solve continues from that state with the current cut; a second event
#: releases the cut when the measured step ends (the real step sequence goes on). A
#: smooth latch state (until 2026-10-16) engaged gradually, so a model cell near its
#: limit hovered in a partial-cut, CV-like state; with a saturating LiCoO2 surface that
#: DAE was so stiff that solves crawled for hours (diag/crawl_bisect.py).
CYCLER_V_MIN = 3.0
CYCLER_V_MAX_CYCLING = 4.3
CYCLER_V_MAX_FORMATION = 4.2
VOLTAGE_LIMIT_WIDTH = 0.005          # V, smooth width of the formation CV hold
LATCH_I_REF = 1e-3                   # A: below this the cycler current counts as rest
V_HI_PARAM = "Cycler upper voltage limit [V]"
CV_PARAM = "Cycler charge CV flag"
CUT_INPUT = "Cycler cut-off"
LATCH_VAR = "Cycler cut-off state"
#: event names: the "[experiment]" tag lets PyBaMM's solver.step continue past them
#: (BaseSolver.step returns the old solution unchanged after any other event)
EVENT_LIMIT = "Cycler limit reached [experiment]"
EVENT_RELEASE = "Cycler limit released [experiment]"
MAX_SEGMENTS = 5000
#: scalar minimum of the electrolyte concentration (added to the chain model)
CE_MIN_VAR = "Minimum electrolyte concentration [mol.m-3]"
#: what a fit candidate stores at every output time (memory: the full state history of
#: one PJ248 candidate is ~1.7 GB on the converged mesh and was held several times over,
#: ~9 GB per candidate; diag/memory_profile.py, 2026-10-18). Full states are captured
#: only at the EIS linearisation times (segment ends).
OUTPUT_VARS = ("Voltage [V]", CE_MIN_VAR, "Volume-averaged cell temperature [K]",
               "Minimum negative particle surface stoichiometry", "Maximum negative particle surface stoichiometry",
               "Minimum positive particle surface stoichiometry", "Maximum positive particle surface stoichiometry")
#: smallest particle radius kept in the distribution, relative to the mean
PSD_R_MIN = 0.2
#: which electrodes get a size distribution: True / "both", or "positive" (LiCoO2
#: only; the PJ248 probes show the LiCoO2 spread is the one that changes the
#: end-of-discharge, and a single graphite radius halves the cost)
PSD_ELECTRODES = ("both", "positive", "negative")

#: Envelope probe (2026-10-19): the fit's electrolyte floor protects only the measured
#: history, but the oracle is asked for the whole jones2022 protocol space. With
#: ORACLE_FIT_ENVELOPE_MA="<charge mA>,<discharge mA>" every candidate's history is
#: followed by the most demanding protocol in that space -- CC charge to the 4.3 V cycler
#: stop, 20-min rest, CC discharge to the 3.0 V stop, 20-min rest -- and a candidate whose
#: electrolyte reaches the floor (or whose solve stops) in the probe is rejected. The
#: chemistry2-25C envelope is ~159 mA in both directions (~3.8C; 480 cycling files of 8
#: cells, diag/protocol_envelope.py).
ENVELOPE_MA = os.environ.get("ORACLE_FIT_ENVELOPE_MA")
PROBE_NAME = "envelope probe"
PROBE_STEP_S, PROBE_REST_S, PROBE_GAP_S, PROBE_SAMPLE_S = 3600.0, 1200.0, 600.0, 5.0
#: two-stage probe charge (ORACLE_FIT_ENVELOPE_MA="c1;c2;discharge", 2026-10-24): each stage
#: runs up to this long or to the 4.3 V cycler stop -- the oracle's demanding cycle
#: (physicality.check_dfn_cycle: 3.8C then 2C for <= 0.25 h each, 3.8C discharge), which the
#: single-stage probe did not cover (the second stage pushes in more lithium at high rate)
PROBE_STAGE_S = 900.0


def envelope_setting() -> tuple[float, ...] | None:
    """Envelope probe set by ORACLE_FIT_ENVELOPE_MA: "charge;discharge" -> (charge mA,
    discharge mA); "c1;c2;discharge" (two-stage charge) -> (c1 mA, discharge mA, c2 mA), the
    argument order of :func:`with_envelope_probe`."""
    if not ENVELOPE_MA:
        return None
    vals = [float(x) for x in ENVELOPE_MA.replace(";", ",").split(",")]
    if len(vals) == 2:
        return vals[0], vals[1]
    if len(vals) == 3:
        return vals[0], vals[2], vals[1]
    raise ValueError(f"ORACLE_FIT_ENVELOPE_MA needs 2 or 3 values, got {ENVELOPE_MA!r}")


def with_envelope_probe(data: dict, charge_ma: float, discharge_ma: float, charge2_ma: float | None = None) -> dict:
    """``data`` with the envelope probe appended as a last (synthetic, unmeasured) group.
    ``charge2_ma``: a second charge stage (two-stage protocol, PROBE_STAGE_S each)."""
    if any(g.get("probe") for g in data["groups"]):
        return data
    d = dict(data)
    groups = [dict(g) for g in data["groups"]]
    r = EDGE_RAMP_S
    i_c, i_d = -float(charge_ma) / 1000.0, float(discharge_ma) / 1000.0     # PyBaMM sign: discharge > 0
    if charge2_ma is None:
        t1 = PROBE_STEP_S
        t2 = t1 + PROBE_REST_S
        t3 = t2 + PROBE_STEP_S
        t4 = t3 + PROBE_REST_S
        tk = np.array([0.0, r, t1, t1 + r, t2, t2 + r, t3, t3 + r, t4])
        ik = np.array([0.0, i_c, i_c, 0.0, 0.0, i_d, i_d, 0.0, 0.0])
    else:
        i_c2 = -float(charge2_ma) / 1000.0
        s1 = PROBE_STAGE_S
        s2 = s1 + PROBE_STAGE_S
        t2 = s2 + PROBE_REST_S
        t3 = t2 + PROBE_STEP_S
        t4 = t3 + PROBE_REST_S
        tk = np.array([0.0, r, s1, s1 + r, s2, s2 + r, t2, t2 + r, t3, t3 + r, t4])
        ik = np.array([0.0, i_c, i_c, i_c2, i_c2, 0.0, 0.0, i_d, i_d, 0.0, 0.0])
    groups.append({"name": PROBE_NAME, "duration_s": t4 + r, "gap_before": PROBE_GAP_S, "formation": False,
                   "probe": True, "charge_ma": float(charge_ma), "discharge_ma": float(discharge_ma),
                   "charge2_ma": None if charge2_ma is None else float(charge2_ma)})
    d["groups"] = groups
    d["current"] = list(data["current"]) + [{"group": len(groups) - 1, "t": tk, "I": ik}]
    return d


def psd_mode(size_distribution) -> str | None:
    """Normalise a size_distribution flag (bool or PSD_ELECTRODES entry)."""
    if size_distribution in (None, False, "", "0", "none"):
        return None
    if size_distribution in (True, "1"):
        return "both"
    if size_distribution not in PSD_ELECTRODES:
        raise ValueError(f"size_distribution must be a bool or one of {PSD_ELECTRODES}, got {size_distribution!r}")
    return size_distribution


def chain_options(sei: str = "constant", hysteresis: bool = False, size_distribution: bool | str = False) -> dict:
    """Stage-A model options with the given SEI submodel (negative electrode only),
    optional graphite OCP hysteresis and particle-size distributions."""
    if sei not in SEI_MODES:
        raise ValueError(f"sei must be one of {SEI_MODES}, got {sei!r}")
    opts = {**CHAIN_OPTIONS, "SEI": sei if sei == "constant" else (sei, "none")}
    if hysteresis:
        opts["open-circuit potential"] = ("one-state hysteresis", "single")
    mode = psd_mode(size_distribution)
    if mode == "both":
        opts["particle size"] = "distribution"
    elif mode:
        opts["particle size"] = ("distribution", "single") if mode == "negative" else ("single", "distribution")
    return opts


def apply_voltage_limits(model, width: float = VOLTAGE_LIMIT_WIDTH):
    """Replace the EIS model's current equation I = I_app(t) with the jones2022 cycler
    limits (see CYCLER_V_MIN): I = I_app (1 - cut) [x a CV factor during the formation
    charge], with cut = CUT_INPUT switched by the two terminating events EVENT_LIMIT
    (3.0 V on discharge, V_HI_PARAM(t) on charge outside CV) and EVENT_RELEASE (the
    measured step ends). PyBaMM sign: I > 0 discharge. At rest cut = 0, so the
    impedance linearisation is unchanged (evaluate it with CUT_INPUT = 0)."""
    import pybamm

    w = float(width)
    sig = lambda x: 0.5 * (1 + pybamm.tanh(x))                      # noqa: E731
    i_var = model.variables["Current variable [A]"]
    v = model.variables["Voltage variable [V]"]
    i_app = pybamm.FunctionParameter("Current function [A]", {"Time [s]": pybamm.t})
    v_hi = pybamm.FunctionParameter(V_HI_PARAM, {"Time [s]": pybamm.t})
    cv = pybamm.FunctionParameter(CV_PARAM, {"Time [s]": pybamm.t})
    cut = pybamm.InputParameter(CUT_INPUT)
    dis = pybamm.tanh(pybamm.maximum(i_app, 0) / LATCH_I_REF)
    chg = pybamm.tanh(pybamm.maximum(-i_app, 0) / LATCH_I_REF)
    rest = 1 - pybamm.tanh((i_app / LATCH_I_REF) ** 2)
    s_cv = sig(((v_hi + 2 * w) - v) / w)                            # 0.98 at the limit
    i_ctrl = (1 - cut) * (pybamm.maximum(i_app, 0) + pybamm.minimum(i_app, 0) * (cv * s_cv + 1 - cv))
    model.algebraic[i_var] = model.variables["Current [A]"] - i_ctrl
    # > 0 while conducting inside the limits; crosses 0 where V passes the limit in the
    # current's direction (charge in CV: no hard stop, the CV factor holds the limit)
    model.events.append(pybamm.Event(EVENT_LIMIT, cut + (1 - cut) * (
        rest + dis * (v - CYCLER_V_MIN) + chg * ((1 - cv) * (v_hi - v) + cv))))
    # > 0 while stopped inside a measured step; crosses 0 when the step ends (rest)
    model.events.append(pybamm.Event(EVENT_RELEASE, (1 - cut) + cut * (0.5 - rest)))
    model.variables[LATCH_VAR] = cut * pybamm.Scalar(1)
    return model


def limit_schedule(groups, off: np.ndarray, t_end: float):
    """(V_HI_PARAM, CV_PARAM) as functions of time on the chain clock: formation groups
    4.2 V with CV, the others 4.3 V hard stop; a fitted gap keeps the previous group's."""
    import pybamm

    tk, vk, ck = [], [], []
    starts = list(off) + [max(float(t_end), float(off[-1])) + 1.0]
    for g, lo, hi in zip(groups, starts[:-1], starts[1:]):
        form = bool(g.get("formation"))
        v_hi, c = (CYCLER_V_MAX_FORMATION, 1.0) if form else (CYCLER_V_MAX_CYCLING, 0.0)
        for t in (lo, max(hi - 1e-3, lo + 1e-6)):
            if tk and t <= tk[-1]:
                continue
            tk.append(t); vk.append(v_hi); ck.append(c)
    tk, vk, ck = np.asarray(tk), np.asarray(vk), np.asarray(ck)
    return (pybamm.Interpolant(tk, vk, pybamm.t, interpolator="linear"),
            pybamm.Interpolant(tk, ck, pybamm.t, interpolator="linear"))


def hysteresis_values(pv, delta: float, decay: float = HYST_DECAY) -> dict:
    """PyBaMM one-state hysteresis parameters for the negative electrode: branches
    U(sto) -/+ delta around the set's graphite OCP."""
    u = pv["Negative electrode OCP [V]"]
    d = float(delta)
    return {"Negative electrode lithiation OCP [V]": lambda sto: u(sto) - d,
            "Negative electrode delithiation OCP [V]": lambda sto: u(sto) + d,
            "Negative particle lithiation hysteresis decay rate": float(decay),
            "Negative particle delithiation hysteresis decay rate": float(decay),
            "Negative particle hysteresis switching factor": 1.0,
            "Initial hysteresis state in negative electrode": 0.0}


def size_distribution_values(pv, sd_n: float, sd_p: float):
    """``pv`` with lognormal area-weighted particle-size distributions around its
    radii (relative widths sd_n, sd_p)."""
    import pybamm

    # PyBaMM's default lower cut, max(0, 1 - 5 sd) x R, reaches R = 0 for sd >= 0.2:
    # zero-radius particles have no capacity, empty at once at the end of a discharge and
    # stop the solver. Real powders have no such particles: cut the lognormal at
    # max(PSD_R_MIN, 1 - 3 sd) and 1 + 3 sd (relative to the mean radius).
    sd_n, sd_p = float(sd_n), float(sd_p)
    return pybamm.get_size_distribution_parameters(
        pv, R_n_av=float(pv["Negative particle radius [m]"]), R_p_av=float(pv["Positive particle radius [m]"]),
        sd_n=sd_n, sd_p=sd_p,
        R_min_n=max(PSD_R_MIN, 1.0 - 3.0 * sd_n), R_max_n=1.0 + 3.0 * sd_n,
        R_min_p=max(PSD_R_MIN, 1.0 - 3.0 * sd_p), R_max_p=1.0 + 3.0 * sd_p)


THICKNESS_SCALE = "Electrode thickness scale"
LI_FRACTION = "Cyclable lithium / positive electrode capacity"
H_TOTAL = "Total heat transfer coefficient [W.m-2.K-1]"
THICKNESSES = ("Negative electrode thickness [m]", "Positive electrode thickness [m]")
POROSITY_N = "Negative electrode porosity"
Q_P_RATIO = "Positive theoretical capacity / measured capacity"
TORTUOSITY_N = "Negative electrode tortuosity"
MACMULLIN_S = "Separator MacMullin number"
BRUGG_N = "Negative electrode Bruggeman coefficient (electrolyte)"
BRUGG_S = "Separator Bruggeman coefficient (electrolyte)"

#: Fit bounds from peer-reviewed values for LiCoO2/graphite cells (literature check
#: 2026-10-09; see _results/oracle_recal/report). Multipliers are relative to the
#: LiR2032 (Ai2020-template) values, absolute values in brackets.
LITERATURE_BOUNDS = {
    # j0 at c_e 1 M, sto 0.5: graphite 0.34-7.9 A/m2 (Chen 2020; Mohtat 2020; Ramadass
    # 2004; Ecker 2015; Marquis 2019), template 0.44 A/m2
    scale_parameter_name("Negative electrode exchange-current density [A.m-2]"): (0.75, 18.0),
    # LiCoO2 0.49-1.84 A/m2 (Marquis 2019; Ramadass 2004), template 0.76 A/m2
    scale_parameter_name("Positive electrode exchange-current density [A.m-2]"): (0.64, 2.4),
    # graphite D_s 1e-15 - 8e-13 m2/s (GITT; Ecker 2015), template 3.9e-14
    scale_parameter_name("Negative particle diffusivity [m2.s-1]"): (0.025, 20.0),
    # LiCoO2 D_s 1e-16 - 1e-13 m2/s (thin film .. DFN sets), template 5.39e-15
    scale_parameter_name("Positive particle diffusivity [m2.s-1]"): (0.02, 20.0),
    # 1 M LiPF6 carbonates 0.8-1.2 S/m (Valoen & Reimers 2005; Nyman 2008); V-R 1.15 S/m at 23 C
    scale_parameter_name("Electrolyte conductivity [S.m-1]"): (0.70, 1.05),
    # D_e 1.5-4e-10 m2/s at 1 M (Valoen & Reimers 2005; Nyman 2008); V-R 3.0e-10 at 23 C
    scale_parameter_name("Electrolyte diffusivity [m2.s-1]"): (0.5, 1.3),
    # double layer 0.05-0.6 F/m2 (models 0.1-0.2 by convention; weakly identifiable)
    scale_parameter_name("Negative electrode double-layer capacity [F.m-2]"): (0.25, 3.0),
    scale_parameter_name("Positive electrode double-layer capacity [F.m-2]"): (0.25, 3.0),
    CONTACT_R: (0.02, 0.3),                # measured high-frequency Re Z 0.28 Ohm; datasheets <= 0.75
    THICKNESS_SCALE: (0.6, 1.3),           # 40-100 um per electrode (Chen 2020; Ecker 2015; Ai 2020)
    EPS_N: (0.55, 0.74),                   # commercial graphite active fraction; porosity follows
    TORTUOSITY_N: (3.0, 10.0),             # graphite tau 3-10 (Landesfeind 2016; Ebner 2014)
    MACMULLIN_S: (2.5, 5.0),               # separators N_M ~2.8-4.5 (Landesfeind 2016; Thorat 2009)
    Q_P_RATIO: (1.6, 2.4),                 # LiCoO2 cycles ~42-62% of theoretical capacity
    LI_FRACTION: (0.85, 1.0),              # LiCoO2 made lithiated; 5-15% lost at formation
    H_TOTAL: (5.0, 25.0),                  # natural convection + holder conduction
    # particle radius: graphite 2.5-13.7 um (Mohtat 2020; Chen 2020; Ecker 2015), LiCoO2/
    # layered oxides 2-6.5 um (Chen 2020; Ecker 2015); template 5 / 3 um
    "Negative particle radius [m]": (2.5e-6, 13.7e-6),
    "Positive particle radius [m]": (2.0e-6, 6.5e-6),
}
RADII = ("Negative particle radius [m]", "Positive particle radius [m]")
#: lithiation dependence of the LiCoO2 diffusivity (parameter_sets.LICO2_DS_KNEE);
#: not in LITERATURE_BOUNDS by default: add via ``bounds`` (LICO2_DS_BOUNDS)
from battery_oracle.parameter_sets import (GRAPHITE_DS_DROP, GRAPHITE_DS_KNEE,  # noqa: E402
                                           J0_SHAPE_N, J0_SHAPE_P, LICO2_DS_DROP, LICO2_DS_KNEE)
LICO2_DS_SHAPE = (LICO2_DS_KNEE, LICO2_DS_DROP)
GRAPHITE_DS_SHAPE = (GRAPHITE_DS_KNEE, GRAPHITE_DS_DROP)
J0_SHAPE = (J0_SHAPE_N, J0_SHAPE_P)
#: knee 0.75-0.95; drop 0-3 decades (chemical D; the tracer D* of Hasegawa/Kuwata
#: 2021 falls ~4 decades over 0.4-1.0, the thermodynamic factor offsets part of it)
LICO2_DS_BOUNDS = {LICO2_DS_KNEE: (0.75, 0.95), LICO2_DS_DROP: (0.0, 3.0)}
#: graphite (parameter_sets.GRAPHITE_DS_KNEE): knee 0.5-0.9 (stage 2 -> 1 starts at
#: x ~0.5), drop 0-2 decades (GITT/PITT, Levi & Aurbach 1997; Persson et al. 2010)
GRAPHITE_DS_BOUNDS = {GRAPHITE_DS_KNEE: (0.5, 0.9), GRAPHITE_DS_DROP: (0.0, 2.0)}
#: exchange-current stoichiometry exponents (parameter_sets.J0_SHAPE_N/P): 0-2.5; at
#: theta = 0.95 (LiCoO2 near full) gamma = 2.5 lowers j0 by ~60x below the standard form,
#: within the 1-2 decades of SOC-dependent j0 reported for Li_xCoO2 (Hess et al. 2015)
J0_SHAPE_BOUNDS = {J0_SHAPE_N: (0.0, 2.5), J0_SHAPE_P: (0.0, 2.5)}
NP_RATIO = "Negative / positive theoretical capacity"
#: practical N:P = reversible graphite capacity / cell capacity, 1.0-1.4 (commercial
#: 1.05-1.2; same window as physicality's check); ~93% of graphite's theoretical
#: capacity is reversible (372 -> ~345 mAh/g)
NP_PRACTICAL_BOUNDS = (1.0, 1.4)
GRAPHITE_USABLE = 0.93
NEG_THICKNESS_RATIO = "Negative / positive electrode thickness"
#: commercial L_n/L_p ~1.0-1.4 (Chen 2020 1.13, Ecker 2015 1.37, Ai 2020 1.125); the
#: lower end is widened to 0.7 because the OCV-fixed N:P with LiCoO2's high volumetric
#: capacity can need a negative thinner than the positive (flagged by physicality)
NEG_THICKNESS_RATIO_BOUNDS = (0.7, 1.4)
POROSITY_P = "Positive electrode porosity"
EPS_P = "Positive electrode active material volume fraction"
#: Fitted electrode porosities with a closed solid volume fraction (user specification,
#: 2026-10-22): active fraction = SOLID_FRACTION - porosity in each electrode (the rest is
#: binder / conductive additive), porosities 0.25-0.35. With the OCV balance's N:P fixed, the
#: negative thickness then follows from N:P (it is no longer fitted: NEG_THICKNESS_RATIO is
#: dropped), and the electrode area from the positive capacity at the candidate's active
#: fraction. Enabled by ORACLE_FIT_POROSITY=1.
SOLID_FRACTION = {POROSITY_N: 0.90, POROSITY_P: 0.85}
POROSITY_BOUNDS = {POROSITY_N: (0.25, 0.35), POROSITY_P: (0.25, 0.35)}
#: The solid volume fractions themselves as fitted parameters (ORACLE_FIT_SOLID_FRACTION=1, with
#: ORACLE_FIT_POROSITY=1; user decision 2026-10-23): binder + conductive additive between the
#: user's 15 % (0.85) and the Ai2020 template's 6 % (0.94), each electrode independently. A
#: candidate's value replaces SOLID_FRACTION[...] for that electrode.
SOLID_N = "Negative electrode solid volume fraction"
SOLID_P = "Positive electrode solid volume fraction"
SOLID_OF = {POROSITY_N: SOLID_N, POROSITY_P: SOLID_P}
SOLID_BOUNDS = {SOLID_N: (0.85, 0.94), SOLID_P: (0.85, 0.94)}


def negative_porosity(pv_template, eps_n: float) -> float:
    """Porosity that keeps the negative electrode's volume fractions summing to 1 when
    its active fraction changes: the template's binder/additive fraction is kept
    (Ai2020: 1 - 0.61 - 0.33 = 0.06), the pore space takes up the rest."""
    filler = max(0.0, 1.0 - float(pv_template[EPS_N]) - float(pv_template[POROSITY_N]))
    return 1.0 - float(eps_n) - filler
F_AH = 96485.33212 / 3600.0
#: Stoichiometry beyond an OCP's data range that fails a candidate (extrapolated OCP).
OCP_GUARD_TOL = 0.02
#: Ramp [s] that replaces each current step edge (the logged current is the mean
#: over the sample interval; the step happens just after the previous sample).
EDGE_RAMP_S = 0.05


def _f(x) -> float:
    """Python float from a scalar or one-element array (PyBaMM evaluate / PyBOP inputs)."""
    return float(np.asarray(x, dtype=float).ravel()[0])


def gap_name(group: str) -> str:
    return f"Rest before {group} [s]"


# ---------------------------------------------------------------------------
# Electrode balance
# ---------------------------------------------------------------------------

def electrode_capacities_ah(pv) -> tuple[float, float]:
    """Theoretical capacities (stoichiometry 0 -> 1) of the negative and positive electrodes."""
    area = (float(pv["Electrode height [m]"]) * float(pv["Electrode width [m]"])
            * float(pv[N_PARALLEL]))
    out = []
    for d, dom in (("Negative", "negative"), ("Positive", "positive")):
        out.append(float(pv[f"{d} electrode active material volume fraction"])
                   * float(pv[f"{d} electrode thickness [m]"]) * area
                   * float(pv[f"Maximum concentration in {dom} electrode [mol.m-3]"]) * F_AH)
    return out[0], out[1]


def lithium_fraction(pv) -> float:
    """Cyclable lithium / positive capacity implied by the set's initial concentrations."""
    q_n, q_p = electrode_capacities_ah(pv)
    x = float(pv["Initial concentration in negative electrode [mol.m-3]"]) / float(
        pv["Maximum concentration in negative electrode [mol.m-3]"])
    y = float(pv["Initial concentration in positive electrode [mol.m-3]"]) / float(
        pv["Maximum concentration in positive electrode [mol.m-3]"])
    return (x * q_n + y * q_p) / q_p


def _ocp_evaluators(pv, temperature_k: float):
    """``U_n(x)``, ``U_p(y)`` [V] exactly as the model evaluates them (incl. the
    entropic correction at ``temperature_k``)."""
    import pybamm

    param = pybamm.LithiumIonParameters()
    sto = pybamm.InputParameter("sto")
    T = pybamm.Scalar(temperature_k)
    un = pv.process_symbol(param.n.prim.U(sto, T))
    up = pv.process_symbol(param.p.prim.U(sto, T))
    return (lambda x: _f(un.evaluate(inputs={"sto": x})),
            lambda y: _f(up.evaluate(inputs={"sto": y})))


def initial_stoichiometries(pv, voltage: float, li_fraction: float,
                            temperature_k: float) -> tuple[float, float]:
    """Stoichiometries ``(x, y)`` of a relaxed cell at open-circuit ``voltage`` holding
    ``li_fraction`` x (positive capacity) of cyclable lithium:
    ``x Q_n + y Q_p = li_fraction Q_p`` and ``U_p(y) - U_n(x) = voltage``."""
    from scipy.optimize import brentq

    q_n, q_p = electrode_capacities_ah(pv)
    q_li = li_fraction * q_p
    un, up = _ocp_evaluators(pv, temperature_k)
    eps = 1e-6
    x_lo = max(eps, (q_li - (1 - eps) * q_p) / q_n)
    x_hi = min(1 - eps, (q_li - eps * q_p) / q_n)
    if x_hi <= x_lo:
        raise ValueError(f"no electrode balance holds {li_fraction:.3f} x Q_p of lithium "
                         f"(Q_n {q_n * 1e3:.1f} mAh, Q_p {q_p * 1e3:.1f} mAh)")

    def g(x):
        return up((q_li - x * q_n) / q_p) - un(x) - voltage

    g_lo, g_hi = g(x_lo), g(x_hi)
    if g_lo * g_hi > 0:
        raise ValueError(f"open-circuit voltage {voltage:.3f} V unreachable with "
                         f"{li_fraction:.3f} x Q_p of lithium (range {g_lo + voltage:.3f}-"
                         f"{g_hi + voltage:.3f} V)")
    x = brentq(g, x_lo, x_hi, xtol=1e-10)
    return float(x), float((q_li - x * q_n) / q_p)


# ---------------------------------------------------------------------------
# Electrode balance from the measured rest voltages (closed-form OCV)
# ---------------------------------------------------------------------------

def history_rest_points(data: dict, min_rest_s: float = 600.0) -> np.ndarray:
    """(Q, V): cumulative discharge charge [Ah] and voltage at the end of every rest
    >= ``min_rest_s`` in the cell's history, plus the as-received first point (gaps
    between groups carry no current, so their length does not matter here)."""
    off, t = [], 0.0
    for g in data["groups"]:
        off.append(t)
        t += float(g["duration_s"])
    tk = np.concatenate([np.asarray(c["t"], float) + off[c["group"]] for c in data["current"]])
    ik = np.concatenate([np.asarray(c["I"], float) for c in data["current"]])
    o = np.argsort(tk, kind="stable")
    tk, ik = tk[o], ik[o]
    q = np.concatenate([[0.0], np.cumsum(0.5 * (ik[1:] + ik[:-1]) * np.diff(tk))]) / 3600.0
    pts = [(0.0, float(data["initial_voltage"]))]
    for r in data["records"]:
        tr = np.asarray(r["t"], float) + off[r["group"]]
        v = np.asarray(r["V"], float)
        rest = np.abs(np.interp(tr, tk, ik)) < 1e-5
        k = 0
        while k < tr.size:
            if rest[k]:
                j = k
                while j + 1 < tr.size and rest[j + 1]:
                    j += 1
                if tr[j] - tr[k] >= min_rest_s:
                    pts.append((float(np.interp(tr[j], tk, q)), float(v[j])))
                k = j + 1
            else:
                k += 1
    return np.asarray(pts)


#: dQ/dV feature of the measured formation cycle used by ``ocv_balance(features=True)``:
#: tolerances (the 0.95C pseudo-OCV is biased ~+/-20 mV by asymmetric polarisation and
#: hysteresis; the state of the feature is more robust) and the rest-point scale.
FEATURE_SIGMA_V, FEATURE_SIGMA_Q, REST_SIGMA_V = 0.020, 1.5e-3, 0.005
FEATURE_SMOOTH_AH, FEATURE_GRID_AH = 1.0e-3, 0.25e-3


def _smooth(y, n):
    if n < 3:
        return y
    return np.convolve(np.pad(y, n // 2, mode="edge"), np.ones(n) / n, mode="valid")[: y.size]


def curve_feature(qg, v, trim_ah: float = 2e-3):
    """(q*, V*) of the main dQ/dV peak (= interior minimum of |dV/dQ|) of a voltage
    curve on the uniform charge grid ``qg`` [Ah], after the same smoothing as the
    measured feature."""
    n = int(round(FEATURE_SMOOTH_AH / (qg[1] - qg[0])))
    vs = _smooth(np.asarray(v, float), n)
    d = np.abs(_smooth(np.gradient(vs, qg), n))
    m = (qg > qg.min() + trim_ah) & (qg < qg.max() - trim_ah)
    k = int(np.flatnonzero(m)[np.argmin(d[m])])
    return float(qg[k]), float(vs[k])


def measured_ocv_feature(data: dict, pair_index: int = 2, min_step_s: float = 600.0) -> dict | None:
    """Main dQ/dV feature of the measured pseudo-OCV (mean of a formation charge and
    the discharge after it at equal state; polarisation cancels to first order at
    equal |I|), in the cumulative-discharge coordinate of ``history_rest_points``."""
    tk = np.concatenate([np.asarray(c["t"], float) for c in data["current"] if c["group"] == 0])
    ik = np.concatenate([np.asarray(c["I"], float) for c in data["current"] if c["group"] == 0])
    o = np.argsort(tk, kind="stable")
    tk, ik = tk[o], ik[o]
    qk = np.r_[0.0, np.cumsum(0.5 * (ik[1:] + ik[:-1]) * np.diff(tk))] / 3600.0
    rec = next((r for r in data["records"] if r["group"] == 0 and r["kind"] == "formation"), None)
    if rec is None:
        return None
    t, v = np.asarray(rec["t"], float), np.asarray(rec["V"], float)
    i = np.interp(t, tk, ik)
    steps, k = [], 0
    while k < t.size:                                       # constant-sign current runs
        if abs(i[k]) < 1e-5:
            k += 1
            continue
        j = k
        while j + 1 < t.size and np.sign(i[j + 1]) == np.sign(i[k]) and abs(i[j + 1]) > 1e-5:
            j += 1
        if t[j] - t[k] >= min_step_s:
            steps.append((float(np.sign(i[k])), k, j))
        k = j + 1
    chg = [s_ for s_ in steps if s_[0] < 0]
    if not chg:
        return None
    c = chg[min(pair_index, len(chg) - 1)]
    d = next((s_ for s_ in steps if s_[0] > 0 and s_[1] > c[2]), None)
    if d is None:
        return None
    qc, vc = np.interp(t[c[1]:c[2] + 1], tk, qk), v[c[1]:c[2] + 1]
    qd, vd = np.interp(t[d[1]:d[2] + 1], tk, qk), v[d[1]:d[2] + 1]
    lo, hi = max(qc.min(), qd.min()), min(qc.max(), qd.max())
    qg = np.arange(lo, hi, FEATURE_GRID_AH)
    oc, od = np.argsort(qc), np.argsort(qd)
    vpo = 0.5 * (np.interp(qg, qc[oc], vc[oc]) + np.interp(qg, qd[od], vd[od]))
    q_star, v_star = curve_feature(qg, vpo)
    return {"q_star_Ah": q_star, "V_star": v_star, "q_grid": [float(lo), float(hi)],
            "half_gap_mV": float(500 * np.median(np.interp(qg, qc[oc], vc[oc]) - np.interp(qg, qd[od], vd[od])))}


def ocv_balance(data: dict, pv, temperature_k: float = 298.15, n_starts: int = 24, seed: int = 0,
                constrained: bool = True, features: bool = False) -> dict:
    """Electrode balance (x0, y0, Q_n, Q_p) that maps the measured rest voltages onto
    the set's OCV: OCV(Q) = U_p(y0 + Q/Q_p) - U_n(x0 - Q/Q_n). Least squares, multi-
    start; stoichiometries kept in [0, 1] and inside the OCP data ranges.

    ``constrained`` (needs ``data["cell_capacity_mah"]``): the balance is kept inside
    the literature ranges -- Q_p / Q_meas in LITERATURE_BOUNDS[Q_P_RATIO], practical
    N:P (GRAPHITE_USABLE x Q_n / Q_meas) in NP_PRACTICAL_BOUNDS, cyclable lithium in
    LITERATURE_BOUNDS[LI_FRACTION]. Unconstrained, the rest voltages alone can pick a
    graphite smaller than the cell's own capacity (PJ248 with Ai2020: N:P 0.38), which
    fills on the first formation charge. The unconstrained fit is reported beside it
    (``unconstrained``) so the cost of the literature limits is visible.

    ``features``: also match the measured main dQ/dV feature of a formation cycle
    (``measured_ocv_feature``: its state q* and voltage V*), so the stoichiometry
    window is set by where the graphite staging meets the LiCoO2 plateau, not only by
    end-of-rest voltages. Weighted as half of all rest points together, with
    tolerances FEATURE_SIGMA_Q / FEATURE_SIGMA_V."""
    q_meas = data.get("cell_capacity_mah")
    feat = measured_ocv_feature(data) if features else None
    if features and feat is None:
        log.warning("ocv_balance: no formation charge/discharge pair -> features ignored")
    if constrained and q_meas:
        out = _ocv_balance_fit(data, pv, temperature_k, n_starts, seed, float(q_meas) / 1000.0, feature=feat)
        # the constrained optimum is a feasible free point: start the free fit there too,
        # so the free RMSE can never exceed the constrained one (a multistart miss)
        start = [out["x0"], out["y0"], out["Q_n_Ah"], out["Q_p_Ah"]]
        out["unconstrained"] = _ocv_balance_fit(data, pv, temperature_k, n_starts, seed, None,
                                                extra_starts=[start], feature=feat)
        return out
    return _ocv_balance_fit(data, pv, temperature_k, n_starts, seed, None, feature=feat)


def _ocv_balance_fit(data: dict, pv, temperature_k: float, n_starts: int, seed: int,
                     q_meas_ah: float | None, extra_starts: list | None = None,
                     feature: dict | None = None) -> dict:
    from scipy.optimize import least_squares

    pts = history_rest_points(data)
    q, v = pts[:, 0], pts[:, 1]
    un, up = _ocp_evaluators(pv, temperature_k)
    sto = np.linspace(1e-4, 1 - 1e-4, 2001)
    un_t, up_t = np.array([un(x) for x in sto]), np.array([up(x) for x in sto])
    (xl, xh), (yl, yh) = ocp_data_range(pv, "negative"), ocp_data_range(pv, "positive")

    def model(p):
        x0, y0, qn, qp = p
        x, y = x0 - q / qn, y0 + q / qp
        return np.interp(y, sto, up_t) - np.interp(x, sto, un_t), x, y

    if q_meas_ah:
        (rl, rh), (nl, nh) = LITERATURE_BOUNDS[Q_P_RATIO], NP_PRACTICAL_BOUNDS
        lo = [0.0, 0.0, nl * q_meas_ah / GRAPHITE_USABLE, rl * q_meas_ah]
        hi = [1.0, 1.0, nh * q_meas_ah / GRAPHITE_USABLE, rh * q_meas_ah]
        li_lo, li_hi = LITERATURE_BOUNDS[LI_FRACTION]
    else:
        lo, hi, li_lo, li_hi = [0, 0, 0.005, 0.005], [1, 1, 1.0, 1.0], None, None

    def res(p):
        f, x, y = model(p)
        pen = 10.0 * (np.clip(xl - x, 0, None) + np.clip(x - xh, 0, None)
                      + np.clip(yl - y, 0, None) + np.clip(y - yh, 0, None))
        if li_lo is not None:
            li = (p[0] * p[2] + p[1] * p[3]) / p[3]
            pen = np.r_[pen, 10.0 * (max(li_lo - li, 0.0) + max(li - li_hi, 0.0))]
        if feature is not None:
            pen = np.r_[pen, feature_residuals(p)]
        return np.r_[f - v, pen]

    if feature is not None:
        qg_f = np.arange(*feature["q_grid"], FEATURE_GRID_AH)
        w_f = np.sqrt(q.size / 2.0)

        def feature_model(p):
            x0, y0, qn, qp = p
            vm = np.interp(y0 + qg_f / qp, sto, up_t) - np.interp(x0 - qg_f / qn, sto, un_t)
            return curve_feature(qg_f, vm)

        def feature_residuals(p):
            qs, vs = feature_model(p)
            return w_f * np.array([(vs - feature["V_star"]) * REST_SIGMA_V / FEATURE_SIGMA_V,
                                   (qs - feature["q_star_Ah"]) * REST_SIGMA_V / FEATURE_SIGMA_Q])

    rng = np.random.default_rng(seed)
    sols, starts = [], []
    for _ in range(n_starts):
        p0 = [rng.uniform(0.2, 0.95), rng.uniform(0.45, 0.9), rng.uniform(lo[2], hi[2]) if q_meas_ah else
              rng.uniform(0.04, 0.15), rng.uniform(lo[3], hi[3]) if q_meas_ah else rng.uniform(0.05, 0.15)]
        starts.append(p0)
    for p0 in starts + list(extra_starts or []):
        p0 = np.clip(np.asarray(p0, float), np.asarray(lo, float) + 1e-9, np.asarray(hi, float) - 1e-9)
        r = least_squares(res, p0, bounds=(lo, hi))
        f, _, _ = model(r.x)
        sols.append((float(1000 * np.sqrt(np.mean((f - v) ** 2))), r.x, float(r.cost)))
    # with a feature term the objective is not the rest RMSE alone: rank by total cost
    sols.sort(key=lambda t: t[2] if feature is not None else t[0])
    rmse, xb, cost_b = sols[0]
    x0, y0, qn, qp = (float(v_) for v_ in xb)
    f, x, y = model(xb)
    # identifiability: every multistart solution within 1 mV of the best (and, with a
    # feature term, within 10% of its cost)
    near = [p for e, p, c in sols if e <= rmse + 1.0 and (feature is None or c <= 1.1 * cost_b + 1e-12)]
    np_near = [p[2] / p[3] for p in near]
    qp_near = [p[3] for p in near]
    return {"x0": x0, "y0": y0, "Q_n_Ah": qn, "Q_p_Ah": qp, "np_ratio": qn / qp,
            "li_fraction": (x0 * qn + y0 * qp) / qp, "rmse_mV": rmse,
            "x_range": [float(x.min()), float(x.max())], "y_range": [float(y.min()), float(y.max())],
            "n_points": int(q.size), "n_near_optimal": len(near),
            "np_ratio_near_range": [float(min(np_near)), float(max(np_near))],
            "Q_p_near_range_Ah": [float(min(qp_near)), float(max(qp_near))],
            "constrained": bool(q_meas_ah),
            **({"feature_measured": feature,
                "feature_model": dict(zip(("q_star_Ah", "V_star"), feature_model(xb)))} if feature is not None else {}),
            **({"Q_p_ratio": qp / q_meas_ah, "np_practical": GRAPHITE_USABLE * qn / q_meas_ah,
                "at_bound": [n for n, val, a, b in (("Q_n", qn, lo[2], hi[2]), ("Q_p", qp, lo[3], hi[3]))
                             if min(val - a, b - val) <= 1e-3 * (b - a)]} if q_meas_ah else {})}


# ---------------------------------------------------------------------------
# Candidate parameter values
# ---------------------------------------------------------------------------

def geometry_overrides(pv_template, inputs: dict, measured_capacity_ah: float | None) -> dict:
    """Absolute geometry/microstructure values of a candidate.

    * thickness: both electrodes x THICKNESS_SCALE (bounded to 40-100 um);
    * negative active fraction; the porosity follows (volume fractions sum to 1);
    * negative Bruggeman exponent from the tortuosity at that porosity,
      tau = eps^(1-b)  ->  b = 1 - ln(tau) / ln(eps)  (so transport stays inside the
      measured graphite tortuosity whatever the porosity); separator Bruggeman from
      the MacMullin number N_M = tau / eps = eps^-b  ->  b = -ln(N_M) / ln(eps);
    * electrode area from the positive theoretical capacity (Q_P_RATIO x measured
      capacity), at the template positive loading and the candidate thickness.
    """
    ov: dict = {}
    s = _f(inputs.get(THICKNESS_SCALE, 1.0))
    if THICKNESS_SCALE in inputs:
        ov.update({k: float(pv_template[k]) * s for k in THICKNESSES})
    # fitted porosities, closed solid fraction (see SOLID_FRACTION)
    for por, eps in ((POROSITY_N, EPS_N), (POROSITY_P, EPS_P)):
        if por in inputs:
            ov[por] = _f(inputs[por])
            solid = _f(inputs[SOLID_OF[por]]) if SOLID_OF[por] in inputs else SOLID_FRACTION[por]
            ov[eps] = solid - ov[por]
    eps_p = ov.get(EPS_P, float(pv_template[EPS_P]))
    c_p = float(pv_template["Maximum concentration in positive electrode [mol.m-3]"])
    c_n = float(pv_template["Maximum concentration in negative electrode [mol.m-3]"])
    if NEG_THICKNESS_RATIO in inputs:          # L_n = ratio x L_p
        L_p = ov.get("Positive electrode thickness [m]", float(pv_template["Positive electrode thickness [m]"]))
        ov["Negative electrode thickness [m]"] = _f(inputs[NEG_THICKNESS_RATIO]) * L_p
    if NP_RATIO in inputs and POROSITY_N in inputs:   # porosity fixes eps_n: N:P fixes L_n
        L_p = ov.get("Positive electrode thickness [m]", float(pv_template["Positive electrode thickness [m]"]))
        ov["Negative electrode thickness [m]"] = _f(inputs[NP_RATIO]) * eps_p * L_p * c_p / (ov[EPS_N] * c_n)
    elif NP_RATIO in inputs:                   # active fraction from the fixed N:P (OCV balance)
        L_n = ov.get("Negative electrode thickness [m]", float(pv_template["Negative electrode thickness [m]"]))
        L_p = ov.get("Positive electrode thickness [m]", float(pv_template["Positive electrode thickness [m]"]))
        ov[EPS_N] = _f(inputs[NP_RATIO]) * eps_p * L_p * c_p / (L_n * c_n)
        ov[POROSITY_N] = negative_porosity(pv_template, ov[EPS_N])
    elif EPS_N in inputs:
        ov[EPS_N] = _f(inputs[EPS_N])
        ov[POROSITY_N] = negative_porosity(pv_template, ov[EPS_N])
    if TORTUOSITY_N in inputs:
        eps = ov.get(POROSITY_N, float(pv_template[POROSITY_N]))
        ov[BRUGG_N] = 1.0 - np.log(_f(inputs[TORTUOSITY_N])) / np.log(eps)
    if MACMULLIN_S in inputs:
        eps_s = float(pv_template["Separator porosity"])
        ov[BRUGG_S] = -np.log(_f(inputs[MACMULLIN_S])) / np.log(eps_s)
    if Q_P_RATIO in inputs and measured_capacity_ah:
        L_p = ov.get("Positive electrode thickness [m]", float(pv_template["Positive electrode thickness [m]"]))
        per_area = eps_p * L_p * c_p * F_AH
        area = _f(inputs[Q_P_RATIO]) * float(measured_capacity_ah) / per_area
        side = float(np.sqrt(area))
        ov.update({"Electrode height [m]": side, "Electrode width [m]": side, N_PARALLEL: 1.0})
    return ov


def candidate_values(pv_base, inputs: dict, initial_voltage: float,
                     measured_capacity_ah: float | None = None):
    """Plain-number ParameterValues of one candidate (no input parameters: geometry,
    thickness and initial concentrations all change per candidate)."""
    pv = pv_base.copy()
    scales = {n: _f(inputs[scale_parameter_name(n)]) for n in CHAIN_SCALES
              if scale_parameter_name(n) in inputs}
    apply_parameter_scales(pv, scales)
    direct = {k: _f(inputs[k]) for k in (CONTACT_R, N_PARALLEL, H_TOTAL, *RADII, *LICO2_DS_SHAPE, *GRAPHITE_DS_SHAPE, *J0_SHAPE)
              if k in inputs}
    pv.update(direct, check_already_exists=False)
    pv.update(geometry_overrides(pv_base, inputs, measured_capacity_ah), check_already_exists=False)
    if HYST_DELTA in inputs:
        pv.update(hysteresis_values(pv, _f(inputs[HYST_DELTA])), check_already_exists=False)
    if PSD_SD_N in inputs or PSD_SD_P in inputs:
        pv = size_distribution_values(pv, _f(inputs.get(PSD_SD_N, 0.3)), _f(inputs.get(PSD_SD_P, 0.3)))
    f_li = _f(inputs.get(LI_FRACTION, lithium_fraction(pv)))
    T0 = float(pv["Initial temperature [K]"])
    x0, y0 = initial_stoichiometries(pv, initial_voltage, f_li, T0)
    pv.update({
        "Initial concentration in negative electrode [mol.m-3]":
            x0 * float(pv["Maximum concentration in negative electrode [mol.m-3]"]),
        "Initial concentration in positive electrode [mol.m-3]":
            y0 * float(pv["Maximum concentration in positive electrode [mol.m-3]"]),
    })
    return pv


def chain_for_fit(data: dict, pv, fit: dict) -> "ChainModel":
    """The ChainModel a saved fit report was made with (ambient, SEI model and EC
    diffusivity factor, fixed inputs), for diagnostics and re-evaluation."""
    pv_base = base_values(pv, fit.get("ambient_k"))
    sei = fit.get("sei", "constant")
    factor = float(fit.get("ec_diffusivity_factor", 1.0))
    if sei != "constant" and factor != 1.0:
        pv_base[EC_DIFFUSIVITY] = float(pv_base[EC_DIFFUSIVITY]) * factor
    return ChainModel(data, pv_base, fixed=fit.get("fixed_inputs") or {}, sei=sei,
                      hysteresis=bool(fit.get("hysteresis")), size_distribution=fit.get("size_distribution") or False,
                      voltage_limits=bool(fit.get("voltage_limits")))


def base_values(pv, ambient_k: float | None = None):
    """The fit's base set: wide voltage window (the simulation follows the measured
    current; cut-off events must not stop it) and the measured ambient temperature."""
    pv = widened(pv)
    if ambient_k is not None:
        pv.update({"Ambient temperature [K]": float(ambient_k), "Initial temperature [K]": float(ambient_k)})
    return pv


# ---------------------------------------------------------------------------
# One integration through the whole history
# ---------------------------------------------------------------------------

class ChainModel:
    """Simulates one cell's continuous history for a candidate; caches results."""

    def __init__(self, data: dict, pv_base, *, cache_size: int = 256, rtol: float = 1e-6,
                 atol: float = 1e-6, fixed: dict | None = None, sei: str = "constant",
                 hysteresis: bool = False, size_distribution: bool | str = False,
                 voltage_limits: bool = False):
        env = envelope_setting()
        self.data = with_envelope_probe(data, *env) if env else data
        self.voltage_limits = bool(voltage_limits)
        self.options = chain_options(sei, hysteresis, size_distribution)
        self.var_pts = {"R_n": PSD_POINTS, "R_p": PSD_POINTS} if psd_mode(size_distribution) else {}
        r_pts, self.submesh_types = radial_mesh(R_POINTS, R_MESH, R_MESH_STRETCH)
        self.var_pts = {**self.var_pts, **r_pts} or None
        self.x_factor = X_FACTOR
        self.pv_base = pv_base
        self.fixed = dict(fixed or {})          # inputs held fixed (e.g. the OCV balance)
        self.rtol, self.atol = rtol, atol
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._model = None

    # -- time line -------------------------------------------------------------
    def offsets(self, inputs: dict) -> np.ndarray:
        """Start of each group on the global clock [s]."""
        off, t = [], 0.0
        for g in self.data["groups"]:
            gap = g.get("gap_before") or 0.0
            if gap == "fit":
                gap = _f(inputs[gap_name(g["name"])])
            t += float(gap)
            off.append(t)
            t += float(g["duration_s"])
        return np.asarray(off)

    def current_knots(self, off: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ts, Is = [], []
        for c in self.data["current"]:
            ts.append(np.asarray(c["t"], float) + off[c["group"]])
            Is.append(np.asarray(c["I"], float))
        t, i = np.concatenate(ts), np.concatenate(Is)
        # files that share a boundary (one ends where the next starts) overlap by
        # their zero-current end ramps: keep a strictly increasing knot sequence
        keep = np.zeros(t.size, dtype=bool)
        last = -np.inf
        for k, tk in enumerate(t):
            if tk > last + 1e-9:
                keep[k] = True
                last = tk
        return t[keep], i[keep]

    # -- model -----------------------------------------------------------------
    def model(self):
        if self._model is None:
            import pybamm

            from battery_oracle.eis_nonstationary import eis_model
            self._model = eis_model(pybamm.lithium_ion.DFN, self.options)
            if self.voltage_limits:
                apply_voltage_limits(self._model)
            # electrolyte floor as a solver EVENT: a candidate whose electrolyte runs out
            # stops at the floor instead of crawling on with ever smaller steps (a joint fit
            # spent > 100 min per candidate there); the truncated history is penalised below
            self._model.variables[CE_MIN_VAR] = pybamm.min(
                self._model.variables["Electrolyte concentration [mol.m-3]"])
            floor = float(os.environ.get("ORACLE_FIT_CE_FLOOR", CE_FLOOR_DEFAULT))
            if floor > 0:
                c_e = self._model.variables["Electrolyte concentration [mol.m-3]"]
                c_0 = pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]")
                self._model.events.append(
                    pybamm.Event("Electrolyte below floor", pybamm.min(c_e) - floor * c_0))
        return self._model

    @staticmethod
    def _solve_segments(sim, t_eval: np.ndarray, capture: np.ndarray | None = None):
        """Integrate with exact cycler hard stops (see CYCLER_V_MIN): segments between
        EVENT_LIMIT / EVENT_RELEASE, each continued from the last state with CUT_INPUT
        switched, on one built model. Segments also end at every ``capture`` time,
        where the full state is recorded (the EIS linearisation points when the
        solver stores only output variables). Returns (solution, number of stops,
        {capture time: state vector})."""
        import pybamm

        sim.build()
        solver, model = sim.solver, sim.built_model
        t_end = float(t_eval[-1])
        capture = np.unique(np.asarray(capture if capture is not None else [], float))
        capture = capture[(capture > 0.0) & (capture < t_end)]
        states: dict[float, np.ndarray] = {}
        sol, cut, n_cuts = None, 0.0, 0
        # every capture time ends a segment, and so does every cycler stop / release:
        # the cap must grow with them (a fixed 5000 silently truncated long histories at
        # ~14 cycle groups, 2026-10-19)
        max_segments = MAX_SEGMENTS + 2 * capture.size
        for _ in range(max_segments):
            t0 = 0.0 if sol is None else float(sol.t[-1])
            nxt = capture[capture > t0 + 1e-9]
            target = float(nxt[0]) if nxt.size else t_end
            dt = target - t0
            if t_end - t0 <= 1e-6:
                break
            rel = np.unique(np.r_[0.0, t_eval[(t_eval > t0 + 1e-9) & (t_eval <= target + 1e-9)] - t0, dt])
            try:
                sol = solver.step(sol, model, dt, t_eval=rel, inputs={CUT_INPUT: cut})
            except pybamm.SolverError as exc:
                # a new step that starts already beyond its limit (e.g. the second charge
                # stage after a stop at 4.3 V): the cycler stops it at once
                if "non-positive at initial conditions" not in str(exc):
                    raise
                cut = 1.0 - cut
                n_cuts += int(cut == 1.0)
                sol = solver.step(sol, model, dt, t_eval=rel, inputs={CUT_INPUT: cut})
            term = str(sol.termination)
            if EVENT_LIMIT in term:
                cut, n_cuts = 1.0, n_cuts + 1
            elif EVENT_RELEASE in term:
                cut = 0.0
            elif "final time" in term and float(sol.t[-1]) < t_end - 1e-6:
                states[float(sol.t[-1])] = np.asarray(sol.last_state.all_ys[0][:, 0], float).copy()
            else:
                break                               # final time, floor, or another event
        else:
            raise RuntimeError(f"segment limit ({max_segments}) reached at t = {float(sol.t[-1]):.0f} of "
                               f"{t_end:.0f} s: history truncated")
        return sol, n_cuts, states

    def solve(self, inputs: dict) -> dict:
        inputs = {**self.fixed, **inputs}
        key = tuple(sorted((k, round(_f(v), 12)) for k, v in inputs.items()))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        try:
            res = self._solve(inputs)
        except Exception as exc:                # noqa: BLE001 - becomes a failed candidate
            import traceback
            log.debug("chain solve failed: %s", traceback.format_exc())
            res = {"error": f"{type(exc).__name__}: {str(exc)[:300]}",
                   "where": traceback.format_exc().strip().splitlines()[-3:]}
        self._cache[key] = res
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return res

    def _solve(self, inputs: dict) -> dict:
        import pybamm

        from battery_oracle.eis_nonstationary import linearised_impedance
        from battery_oracle.patches import single_thread_solver

        t0 = time.time()
        cap = self.data.get("cell_capacity_mah")
        pv = candidate_values(self.pv_base, inputs, float(self.data["initial_voltage"]),
                              cap / 1000.0 if cap else None)
        off = self.offsets(inputs)
        tk, ik = self.current_knots(off)
        rec_t = [np.asarray(r["t"], float) + off[r["group"]] for r in self.data["records"]]
        sp_lin, sp_lo, sp_hi = [], [], []
        for s in self.data["spectra"]:
            lo = np.asarray(s["t_start"], float) + off[s["group"]]
            hi = np.asarray(s["t_end"], float) + off[s["group"]]
            sp_lo.append(lo)
            sp_hi.append(hi)
            sp_lin.append(0.5 * (lo + hi))
        t_end = max([tk[-1]] + [t[-1] for t in rec_t] + [h[-1] for h in sp_hi])
        if t_end > tk[-1]:          # e.g. a final EIS sweep after the last GCPL file: rest
            tk, ik = np.r_[tk, t_end + 1.0], np.r_[ik, 0.0]
        pv["Current function [A]"] = pybamm.Interpolant(tk, ik, pybamm.t, interpolator="linear")
        if self.voltage_limits:
            pv[V_HI_PARAM], pv[CV_PARAM] = limit_schedule(self.data["groups"], off, t_end)
        probe_g = [k for k, g in enumerate(self.data["groups"]) if g.get("probe")]
        t_probe = float(off[probe_g[0]]) if probe_g else None
        probe_t = [np.arange(t_probe, t_end, PROBE_SAMPLE_S)] if probe_g else []
        t_eval = np.unique(np.concatenate([[0.0, t_end]] + rec_t + sp_lin + sp_lo + sp_hi + probe_t))

        x_pts = x_var_pts(self.model(), getattr(self, "x_factor", 1))
        var_pts = ({**self.model().default_var_pts, **(self.var_pts or {}), **x_pts}
                   if (self.var_pts or x_pts) else None)
        submesh = {**self.model().default_submesh_types, **self.submesh_types} if self.submesh_types else None
        # a fit candidate stores only OUTPUT_VARS and the states at the EIS linearisation
        # times; a kept solution (diagnostics, validity checks) stores everything
        lean = self.voltage_limits and not getattr(self, "keep_solution", False)
        capture = np.concatenate(sp_lin) if (lean and sp_lin) else None
        sim = pybamm.Simulation(self.model(), parameter_values=pv, var_pts=var_pts, submesh_types=submesh,
                                solver=single_thread_solver(self.rtol, self.atol,
                                                            output_variables=list(OUTPUT_VARS) if lean else None))
        states: dict = {}
        if self.voltage_limits:
            sol, n_cuts, states = self._solve_segments(sim, t_eval, capture)
        else:
            sol, n_cuts = sim.solve(t_eval=t_eval), 0
        ts = np.asarray(sol.t)
        v_all = np.asarray(sol["Voltage [V]"].entries)
        t_last = float(ts[-1])

        diag = {"termination": str(sol.termination), "t_simulated_s": t_last, "t_total_s": float(t_end),
                "cycler_stops": int(n_cuts)}
        # Electrolyte exhausted (c_e <= 0 anywhere) or non-finite voltage: the cell could
        # not carry the measured current from here on -- the real one did -- so the
        # history ends here for scoring; the graded penalty scores what is left.
        try:
            ce_min_t = np.asarray(sol[CE_MIN_VAR].entries, float).ravel()
        except Exception:                       # noqa: BLE001
            ce_min_t = np.ones_like(ts)
        # optional physical floor (review R2): below CE_FLOOR x c0 the candidate is
        # treated as depleted, so the fit cannot buy rate limitation with an
        # electrolyte state no real cell has (pilot 3: 0.0025 c0)
        c0 = float(pv["Initial concentration in electrolyte [mol.m-3]"])
        floor = float(os.environ.get("ORACLE_FIT_CE_FLOOR", CE_FLOOR_DEFAULT)) * c0
        bad = ~np.isfinite(v_all) | (ce_min_t <= floor)
        if bad.any():
            k = int(np.argmax(bad))
            diag["stopped"] = ("electrolyte below floor" if ce_min_t[k] <= floor else "non-finite voltage")
            diag["t_stopped_s"] = float(ts[k])
            ts, v_all = ts[:max(k, 1)], v_all[:max(k, 1)]
            t_last = float(ts[-1])
            diag["t_simulated_s"] = t_last
        if probe_g:
            # envelope probe: a candidate that reaches it must come through it with the
            # electrolyte above the floor (a failure before it keeps the graded penalty)
            ts_all = np.asarray(sol.t, float)
            in_p = ts_all >= t_probe
            sim_end = float(ts_all[-1])
            g_p = self.data["groups"][probe_g[0]]
            pr = {"charge_ma": g_p["charge_ma"], "discharge_ma": g_p["discharge_ma"], "start_s": t_probe,
                  "reached": bool(sim_end >= t_probe), "completed": bool(sim_end >= t_end - 1.0),
                  "c_e_min_rel": float(np.min(ce_min_t[in_p[:ce_min_t.size]]) / c0) if in_p.any() else None}
            diag["envelope_probe"] = pr
            stop_t = diag.get("t_stopped_s")
            if pr["reached"] and (not pr["completed"] or (stop_t is not None and stop_t >= t_probe)):
                raise RuntimeError(
                    f"envelope probe ({pr['charge_ma']:.0f} mA charge / {pr['discharge_ma']:.0f} mA discharge) "
                    f"failed: electrolyte min {pr['c_e_min_rel']} c0, simulated {sim_end - t_probe:.0f} of "
                    f"{t_end - t_probe:.0f} s ({diag.get('stopped') or diag['termination']})")
        try:
            diag["T_max_K"] = float(np.max(sol["Volume-averaged cell temperature [K]"].entries))
            diag["c_e_min_rel"] = float(np.min(ce_min_t) / c0)
        except Exception:                       # noqa: BLE001 - diagnostics only
            pass
        # OCP guard: surface stoichiometry vs the OCP data range.
        ocp = {}
        for dom in ("negative", "positive"):
            lo_r, hi_r = ocp_data_range(pv, dom)
            smin = np.asarray(sol[f"Minimum {dom} particle surface stoichiometry"].entries)
            smax = np.asarray(sol[f"Maximum {dom} particle surface stoichiometry"].entries)
            out = (smin < lo_r) | (smax > hi_r)
            excursion = float(max(lo_r - smin.min(), smax.max() - hi_r, 0.0))
            ocp[dom] = {"data_range": [lo_r, hi_r], "min": float(smin.min()), "max": float(smax.max()),
                        "fraction_out": float(out.mean()), "excursion": excursion}
            if excursion > OCP_GUARD_TOL:
                raise RuntimeError(f"{dom} OCP extrapolated by {excursion:.3f} in stoichiometry "
                                   f"(data {lo_r:.3f}-{hi_r:.3f}, simulated {smin.min():.3f}-{smax.max():.3f})")
        diag["ocp"] = ocp

        records = []
        for t in rec_t:
            n = int(np.searchsorted(t, t_last + 1e-6, side="right"))
            records.append(np.interp(t[:n], ts, v_all))

        spectra = []
        model = self.model()
        v_scale = getattr(model.variables["Voltage [V]"], "scale", 1)
        i_scale = getattr(model.variables["Current [A]"], "scale", 1)
        z_scale = _f(pv.evaluate(v_scale / i_scale))
        amp = float(self.data.get("amplitude_a", 5e-3))
        if states:
            cap_t = np.array(sorted(states))
            state_at = lambda tt: np.column_stack([states[float(cap_t[np.argmin(np.abs(cap_t - x))])]  # noqa: E731
                                                   for x in tt])
        else:
            y = np.asarray(sol.y)
            state_at = lambda tt: y[:, np.clip(np.searchsorted(ts, tt), 0, ts.size - 1)]  # noqa: E731
        for s, t_lin, lo, hi in zip(self.data["spectra"], sp_lin, sp_lo, sp_hi):
            if hi[-1] > t_last + 1e-6:
                spectra.append(None)
                continue
            if states:
                t_at = np.asarray(t_lin, float)
                ys = state_at(t_at)
            else:
                idx = np.clip(np.searchsorted(ts, t_lin), 0, ts.size - 1)
                t_at, ys = ts[idx], state_at(t_lin)
            z = linearised_impedance(sim, t_at, ys, s["f"], z_scale,
                                     inputs={CUT_INPUT: 0.0} if self.voltage_limits else None)
            dvdt = (np.interp(hi, ts, v_all) - np.interp(lo, ts, v_all)) / np.maximum(hi - lo, 1e-9)
            z = z - 2.0 * dvdt / (2 * np.pi * np.asarray(s["f"], float) * amp)
            spectra.append(z)
        diag["solve_s"] = round(time.time() - t0, 2)
        out = {"records": records, "spectra": spectra, "diag": diag, "offsets": off.tolist()}
        if getattr(self, "keep_solution", False):      # diagnostics: internal states
            out.update(solution=sol, parameter_values=pv)
        return out


def chain_view_class():
    """``ChainView`` (defined lazily: PyBOP is imported only when fitting)."""
    import pybop
    from pybop.simulators.failed_solution import FailedSolution
    from pybop.simulators.solution import Solution

    class ChainView(pybop.BaseSimulator):
        """PyBOP simulator over one record (``"voltage"``) or spectrum (``"eis"``) of a
        :class:`ChainModel`."""

        def __init__(self, chain: ChainModel, parameters: dict, kind: str, index: int):
            super().__init__(dict(parameters))
            self.chain, self.kind, self.index = chain, kind, index
            self.variable = "Impedance" if kind == "eis" else "Voltage [V]"

        def set_output_variables(self, target):
            return None

        def solve_batch(self, inputs, calculate_sensitivities: bool = False):
            out = []
            for x in inputs:
                res = self.chain.solve(x)
                data = None
                if "error" not in res:
                    data = (res["spectra"] if self.kind == "eis" else res["records"])[self.index]
                if "error" in res:
                    out.append(FailedSolution([self.variable], list(x.keys())))
                    continue
                if data is None or len(data) == 0:
                    # not reached (the history stopped earlier): a finite penalty, so the
                    # optimiser can still rank how far each candidate got
                    if self.kind == "eis":
                        data = np.zeros(len(self.chain.data["spectra"][self.index]["f"]), dtype=complex)
                    else:
                        data = np.zeros(1)        # graded_failure pads the rest (+0.5 V per point)
                sol = Solution()
                sol.set_solution_variable(self.variable, data=np.asarray(data))
                out.append(sol)
            return out

    return ChainView


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def _centred_linear(bounds, value, min_half: float = 1e-3):
    lo, hi = bounds
    v = float(np.clip(value, lo, hi))
    d = max(min(v - lo, hi - v), min_half)
    return (v - d, v + d)


def fit_parameters_chain(pv, data: dict, warm: dict | None = None, *, bounds: dict | None = None,
                         gap_bounds=(60.0, 30 * 86400.0), free: list[str] | None = None) -> dict:
    """PyBOP parameters of the chain fit. Priors are uniform (log-uniform for the
    multiplicative and positive-only quantities) over the LITERATURE_BOUNDS ranges,
    so every fitted value stays inside published values; PyBOP's population
    optimisers start at the prior centre. Only parameters with no literature prior
    (the unrecorded rests) are centred on their warm values."""
    import pybop

    warm = dict(warm or {})
    bnd = {**LITERATURE_BOUNDS, **(bounds or {})}
    linear = {EPS_N, Q_P_RATIO, LI_FRACTION, POROSITY_N, POROSITY_P, SOLID_N, SOLID_P, *LICO2_DS_SHAPE, *GRAPHITE_DS_SHAPE, *J0_SHAPE, HYST_DELTA,
              PSD_SD_N, PSD_SD_P}
    fit = {}
    for name, (lo, hi) in bnd.items():
        if name in linear:
            fit[name] = pybop.Parameter(bounds=(float(lo), float(hi)), initial_value=0.5 * (lo + hi))
        else:
            fit[name] = pybop.Parameter(distribution=pybop.LogUniform(float(lo), float(hi)),
                                        initial_value=float(np.sqrt(lo * hi)),
                                        transformation=pybop.LogTransformation())
    for g in data["groups"]:
        if g.get("gap_before") == "fit":
            v0 = warm.get(gap_name(g["name"]), 3600.0)
            fit[gap_name(g["name"])] = pybop.Parameter(
                distribution=pybop.LogUniform(*_centred(gap_bounds, v0)), initial_value=float(v0),
                transformation=pybop.LogTransformation())
    if free is not None:
        missing = [k for k in free if k not in fit]
        if missing:
            raise KeyError(f"unknown free parameters {missing}; choose from {sorted(fit)}")
        fit = {k: v for k, v in fit.items() if k in free}
    return fit


def build_chain_problems(data: dict, pv_base, fit: dict, chain: ChainModel | None = None):
    """[(kind, label, problem, weight, target)] -- one per record and spectrum."""
    import pybop

    chain = chain or ChainModel(data, pv_base)
    View = chain_view_class()
    out = []
    for k, r in enumerate(data["records"]):
        ds = pybop.Dataset({"Time [s]": np.asarray(r["t"], float), "Current [A]": np.zeros(len(r["t"])),
                            "Voltage [V]": np.asarray(r["V"], float)})
        prob = pybop.Problem(View(chain, fit, "voltage", k), pybop.RootMeanSquaredError(ds))
        out.append((r["kind"], r["label"], prob, 1.0 / SIGMA_V, r))
    for k, s in enumerate(data["spectra"]):
        f = np.asarray(s["f"], float)
        ds = pybop.Dataset({"Frequency [Hz]": f, "Current [A]": np.zeros_like(f),
                            "Impedance": np.asarray(s["Z"], complex)}, domain="Frequency [Hz]")
        prob = pybop.Problem(View(chain, fit, "eis", k), pybop.RootMeanSquaredError(ds, target="Impedance"))
        sigma = SIGMA_Z_REL * float(np.median(np.abs(s["Z"])))
        out.append(("eis", s["label"], prob, 1.0 / sigma, s))
    return out, chain


def fit_cell_chain(cell_id: str, data: dict, pv, *, warm: dict | None = None,
                   max_iterations: int = 150, max_unchanged: int = 30, seed: int | None = 0,
                   free: list[str] | None = None, ambient_k: float | None = None,
                   bounds: dict | None = None, balance: str = "free", sei: str = "constant",
                   ec_diffusivity_factor: float = 1.0, hysteresis: bool = False,
                   size_distribution: bool | str = False, voltage_limits: bool = False) -> dict[str, Any]:
    """Chain fit of one cell (module docstring); returns a JSON-able report.
    ``bounds`` overrides entries of LITERATURE_BOUNDS (e.g. a hypothesis test).
    ``balance="ocv"``: two-step stage A -- the electrode balance (positive capacity,
    N:P, cyclable lithium) is fixed by the closed-form OCV fit to the measured rest
    voltages (``ocv_balance``); the DFN fit then adjusts kinetics, transport and
    architecture only, with the negative/positive thickness ratio replacing the
    negative active fraction (which follows from the fixed N:P).
    ``sei``: SEI submodel (SEI_MODES); for growth, ``ec_diffusivity_factor``
    multiplies the template EC diffusivity as the oracle does (0.25 x sei_rate_scale)."""
    import os

    import pybop

    from battery_oracle import patches

    patches.apply()
    pv_base = base_values(pv, ambient_k)
    if sei != "constant" and ec_diffusivity_factor != 1.0:
        pv_base[EC_DIFFUSIVITY] = float(pv_base[EC_DIFFUSIVITY]) * float(ec_diffusivity_factor)
    fixed, balance_report = {}, None
    if balance in ("ocv", "ocv_features"):
        balance_report = ocv_balance(data, pv_base, features=balance == "ocv_features")
        q_meas = float(data["cell_capacity_mah"]) / 1000.0
        fixed = {Q_P_RATIO: balance_report["Q_p_Ah"] / q_meas, NP_RATIO: balance_report["np_ratio"],
                 LI_FRACTION: balance_report["li_fraction"]}
        log.info("%s: OCV balance %s -> fixed %s", cell_id, balance_report, fixed)
        bounds = dict(bounds or {})
        if os.environ.get("ORACLE_FIT_POROSITY") == "1":
            bounds.update(POROSITY_BOUNDS)          # the negative thickness then follows from N:P
            if os.environ.get("ORACLE_FIT_SOLID_FRACTION") == "1":
                bounds.update(SOLID_BOUNDS)
        else:
            bounds[NEG_THICKNESS_RATIO] = NEG_THICKNESS_RATIO_BOUNDS
    extra = {}
    if hysteresis:
        extra[HYST_DELTA] = HYST_BOUNDS
    mode = psd_mode(size_distribution)
    if mode:
        extra.update({k: v for k, v in PSD_BOUNDS.items()
                      if mode == "both" or (k == PSD_SD_P) == (mode == "positive")})
    bounds = {**(bounds or {}), **extra}
    # warm start (ORACLE_FIT_WARM_JSON: a report or checkpoint with best_inputs): CMA-ES
    # starts at those values with a step ORACLE_FIT_WARM_SIGMA x the prior's, and every
    # parameter not freed (ORACLE_FIT_FREE, ";"-separated names) is held at its warm
    # value -- not at the template's, which a plain ``free`` list would give
    warm_inputs = None
    if os.environ.get("ORACLE_FIT_WARM_JSON"):
        import json
        warm_inputs = {k: float(v) for k, v in
                       json.loads(Path(os.environ["ORACLE_FIT_WARM_JSON"]).read_text())["best_inputs"].items()}
        warm = {**(warm or {}), **warm_inputs}
        if os.environ.get("ORACLE_FIT_FREE"):
            free = [k.strip() for k in os.environ["ORACLE_FIT_FREE"].split(";") if k.strip()]
    fit = fit_parameters_chain(pv_base, data, warm, free=free, bounds=bounds)
    for k in fixed:
        fit.pop(k, None)
    if fixed:
        fit.pop(EPS_N, None)
    if warm_inputs is not None:
        held = {k: v for k, v in warm_inputs.items() if k not in fit and k not in fixed and k != EPS_N}
        fixed = {**fixed, **held}
        for k, prm in fit.items():
            if k in warm_inputs:
                lo, hi = prm.bounds
                span = hi - lo
                prm.update_initial_value(float(np.clip(warm_inputs[k], lo + 1e-6 * span, hi - 1e-6 * span)))
        log.info("%s: warm start from %s; free %s; held %s", cell_id, os.environ["ORACLE_FIT_WARM_JSON"],
                 sorted(fit), sorted(held))
    chain = ChainModel(data, pv_base, fixed=fixed, sei=sei, hysteresis=hysteresis,
                       size_distribution=size_distribution, voltage_limits=voltage_limits)
    problems, chain = build_chain_problems(data, pv_base, fit, chain=chain)
    meta = pybop.MetaProblem(*[p[2] for p in problems], weights=[p[3] for p in problems])
    meta.set_workers(int(os.environ.get("ORACLE_FIT_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", 1))))
    if seed is not None:
        np.random.seed(seed)
    options = pybop.PintsOptions(max_iterations=max_iterations, max_unchanged_iterations=max_unchanged,
                                 verbose=True)
    t0 = time.time()
    resume = os.environ.get("ORACLE_FIT_FROM_CHECKPOINT") == "1"
    if resume:
        # the fit was stopped (e.g. a job time limit): report the best-so-far candidate
        # of its checkpoint (patches._checkpoint_best) instead of optimising again
        import json
        from types import SimpleNamespace
        ck = json.loads(Path(os.environ["ORACLE_FIT_CHECKPOINT"]).read_text())
        result = SimpleNamespace(best_inputs={k: v for k, v in ck["best_inputs"].items() if k in fit},
                                 best_cost=ck["best_cost"], n_iterations=None)
        log.info("%s: report from checkpoint %s (cost %s, written %s)", cell_id,
                 os.environ["ORACLE_FIT_CHECKPOINT"], ck["best_cost"], ck.get("written"))
    elif os.environ.get("ORACLE_FIT_POLISH_ONLY") == "1":
        # validate / apply the local polish alone, from the warm-start point (no CMA-ES)
        from types import SimpleNamespace
        if warm_inputs is None:
            raise ValueError("ORACLE_FIT_POLISH_ONLY needs ORACLE_FIT_WARM_JSON (the start point)")
        result = SimpleNamespace(best_inputs={k: float(warm_inputs[k]) for k in fit}, best_cost=np.nan,
                                 n_iterations=0)
    else:
        optim = pybop.CMAES(meta, options=options)
        if warm_inputs is not None:
            # PyBOP starts population optimisers at the prior mean: restart at the warm values
            f_sigma = float(os.environ.get("ORACLE_FIT_WARM_SIGMA", "0.3"))
            optim._optimiser = optim._pints_optimiser(
                x0=optim.problem.parameters.get_initial_values(transformed=True),
                sigma0=np.asarray(optim._std0, float) * f_sigma, boundaries=optim._boundaries)
        result = optim.run()
    polish_report = None
    n_polish = int(os.environ.get("ORACLE_FIT_POLISH", "0"))
    if n_polish > 0 or os.environ.get("ORACLE_FIT_POLISH_ONLY") == "1":
        # local polish from the CMA-ES best (battery_oracle.polish): replaces it only if better
        from battery_oracle.polish import polish as _polish
        names = list(fit)
        start = {k: float(np.asarray(result.best_inputs[k]).ravel()[0]) for k in names}
        pol = _polish(lambda cands: np.ravel(np.asarray(meta.evaluate_batch(cands).values, float)), names,
                      [tuple(map(float, fit[k].bounds)) for k in names], start, max_iter=max(n_polish, 1),
                      h=float(os.environ.get("ORACLE_FIT_POLISH_STEP", "5e-3")))
        cma_cost = float(np.asarray(result.best_cost).ravel()[0])
        polish_report = {k: pol[k] for k in ("start_cost", "best_cost", "iterations", "stop_reason", "history",
                                             "log_scaled")}
        polish_report["cmaes_cost"] = cma_cost
        log.info("%s: polish %s -> %s in %d iterations (%s)", cell_id, pol["start_cost"], pol["best_cost"],
                 pol["iterations"], pol["stop_reason"])
        if not np.isfinite(cma_cost) or pol["best_cost"] < cma_cost:
            from types import SimpleNamespace
            result = SimpleNamespace(best_inputs=pol["best_inputs"], best_cost=pol["best_cost"],
                                     n_iterations=getattr(result, "n_iterations", None))
    runtime = time.time() - t0
    best = {k: float(np.asarray(v).ravel()[0]) for k, v in result.best_inputs.items()}
    log.info("%s: chain fit cost %s in %.0f s: %s", cell_id, result.best_cost, runtime, best)
    fitted = dict(best)
    best = {**fixed, **best}                   # the full candidate (fixed balance included)
    return {"cell_id": cell_id, "method": "chain", "best_inputs": best, "fitted_inputs": fitted,
            "fixed_inputs": fixed, "ocv_balance": balance_report, "balance_mode": balance,
            "sei": sei, "ec_diffusivity_factor": float(ec_diffusivity_factor),
            "hysteresis": bool(hysteresis), "size_distribution": psd_mode(size_distribution) or False,
            "radial_mesh": radial_mesh_setting(),
            "envelope_probe_ma": envelope_setting(),
            "voltage_limits": ({"v_min": CYCLER_V_MIN, "v_max_cycling": CYCLER_V_MAX_CYCLING,
                                "v_max_formation_cv": CYCLER_V_MAX_FORMATION} if voltage_limits else None),
            "seed": seed,
            "warm_start": ({"from": os.environ["ORACLE_FIT_WARM_JSON"], "free": sorted(fit),
                            "sigma_factor": float(os.environ.get("ORACLE_FIT_WARM_SIGMA", "0.3"))}
                           if warm_inputs is not None else None),
            "cost": float(np.asarray(result.best_cost).ravel()[0]),
            "polish": polish_report,
            "param_bounds": {k: list(map(float, p.bounds)) for k, p in fit.items()},
            "log_params": [k for k, p in fit.items() if isinstance(p.transformation, pybop.LogTransformation)],
            "runtime_s": round(runtime, 1), "worker_pool": dict(patches.POOL_STATS),
            "n_iterations": (int(result.n_iterations) if getattr(result, "n_iterations", None) is not None else None),
            "from_checkpoint": bool(resume),
            **evaluate_chain(data, pv_base, fitted, chain=chain)}


def _joint_data(data: dict, cell_id: str) -> dict:
    """``data`` with its group names prefixed by the cell id, so every cell's unrecorded
    rests are separate fit parameters in a joint fit (gap_name follows the group name)."""
    groups = [dict(g, name=f"{cell_id}:{g['name']}") for g in data["groups"]]
    return {**data, "groups": groups}


def _unprefixed(inputs: dict, cell_id: str) -> dict:
    """A joint fit's inputs for one cell, with that cell's rest names as in its own data."""
    tag = f"{cell_id}:"
    out = {}
    for k, v in inputs.items():
        if k.startswith("Rest before "):
            if tag not in k:
                continue
            k = k.replace(tag, "")
        out[k] = v
    return out


def fit_cells_chain_joint(datas: dict, pv, *, warm: dict | None = None,
                          max_iterations: int = 150, max_unchanged: int = 30, seed: int | None = 0,
                          ambient_k: float | None = None, bounds: dict | None = None,
                          balance: str = "ocv_features", sei: str = "constant",
                          ec_diffusivity_factor: float = 1.0, hysteresis: bool = False,
                          size_distribution: bool | str = False,
                          voltage_limits: bool = False) -> dict[str, dict]:
    """Joint chain fit of several cells: ONE set of kinetic, transport and geometric
    parameters shared by all cells, scored on every cell's records and spectra.

    Per cell: the electrode balance (step one, ``ocv_balance``: capacities differ
    between cells) and the unrecorded rests. A single-cell fit can tune the shared
    physics to one cell's history; the joint fit cannot. Returns {cell_id: report},
    each report shaped like ``fit_cell_chain``'s (``best_inputs`` with the cell's own
    rest names), so ``aggregate_chain`` works unchanged. Warm start and checkpoint
    environment variables as in ``fit_cell_chain``; ORACLE_FIT_DRY_RUN=1 builds the
    problems and returns their sizes without optimising."""
    import json

    import pybop

    from battery_oracle import patches

    patches.apply()
    if balance not in ("ocv", "ocv_features"):
        raise ValueError("a joint fit needs a per-cell OCV balance (balance='ocv' or 'ocv_features')")
    pv_base = base_values(pv, ambient_k)
    if sei != "constant" and ec_diffusivity_factor != 1.0:
        pv_base[EC_DIFFUSIVITY] = float(pv_base[EC_DIFFUSIVITY]) * float(ec_diffusivity_factor)
    cells = list(datas)
    jdata = {c: _joint_data(d, c) for c, d in datas.items()}
    fixed, balances = {}, {}
    for c in cells:
        rep = ocv_balance(jdata[c], pv_base, features=balance == "ocv_features")
        q_meas = float(jdata[c]["cell_capacity_mah"]) / 1000.0
        fixed[c] = {Q_P_RATIO: rep["Q_p_Ah"] / q_meas, NP_RATIO: rep["np_ratio"], LI_FRACTION: rep["li_fraction"]}
        balances[c] = rep
        log.info("joint fit %s: OCV balance -> fixed %s", c, fixed[c])
    bounds = dict(bounds or {})
    if os.environ.get("ORACLE_FIT_POROSITY") == "1":
        bounds.update(POROSITY_BOUNDS)
        if os.environ.get("ORACLE_FIT_SOLID_FRACTION") == "1":
            bounds.update(SOLID_BOUNDS)
    else:
        bounds[NEG_THICKNESS_RATIO] = NEG_THICKNESS_RATIO_BOUNDS
    if hysteresis:
        bounds[HYST_DELTA] = HYST_BOUNDS
    mode = psd_mode(size_distribution)
    if mode:
        bounds.update({k: v for k, v in PSD_BOUNDS.items()
                       if mode == "both" or (k == PSD_SD_P) == (mode == "positive")})
    warm = dict(warm or {})
    warm_inputs = None
    if os.environ.get("ORACLE_FIT_WARM_JSON"):
        warm_inputs = {k: float(v) for k, v in
                       json.loads(Path(os.environ["ORACLE_FIT_WARM_JSON"]).read_text())["best_inputs"].items()}
        warm.update(warm_inputs)
    fit = {}
    for c in cells:                          # shared names coincide; rests are per cell
        fit.update(fit_parameters_chain(pv_base, jdata[c], warm, bounds=bounds))
    for k in (Q_P_RATIO, NP_RATIO, LI_FRACTION, EPS_N):
        fit.pop(k, None)
    if warm_inputs is not None:
        for k, prm in fit.items():
            if k in warm_inputs:
                lo, hi = prm.bounds
                span = hi - lo
                prm.update_initial_value(float(np.clip(warm_inputs[k], lo + 1e-6 * span, hi - 1e-6 * span)))
    chains, problems = {}, []
    for c in cells:
        own = {k: v for k, v in fit.items() if not k.startswith("Rest before ") or f"{c}:" in k}
        chains[c] = ChainModel(jdata[c], pv_base, fixed=fixed[c], sei=sei, hysteresis=hysteresis,
                               size_distribution=size_distribution, voltage_limits=voltage_limits)
        probs, _ = build_chain_problems(jdata[c], pv_base, own, chain=chains[c])
        problems += probs
    shared = sorted(k for k in fit if not k.startswith("Rest before "))
    log.info("joint fit %s: %d shared parameters, %d rests, %d problems", cells, len(shared),
             len(fit) - len(shared), len(problems))
    if os.environ.get("ORACLE_FIT_DRY_RUN") == "1":
        return {"dry_run": {"cells": cells, "shared": shared, "rests": sorted(set(fit) - set(shared)),
                            "n_problems": len(problems), "fixed": fixed}}
    meta = pybop.MetaProblem(*[p[2] for p in problems], weights=[p[3] for p in problems])
    meta.set_workers(int(os.environ.get("ORACLE_FIT_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", 1))))
    if seed is not None:
        np.random.seed(seed)
    options = pybop.PintsOptions(max_iterations=max_iterations, max_unchanged_iterations=max_unchanged,
                                 verbose=True)
    t0 = time.time()
    resume = os.environ.get("ORACLE_FIT_FROM_CHECKPOINT") == "1"
    if resume:
        from types import SimpleNamespace
        ck = json.loads(Path(os.environ["ORACLE_FIT_CHECKPOINT"]).read_text())
        result = SimpleNamespace(best_inputs={k: v for k, v in ck["best_inputs"].items() if k in fit},
                                 best_cost=ck["best_cost"], n_iterations=None)
    else:
        optim = pybop.CMAES(meta, options=options)
        if warm_inputs is not None:
            f_sigma = float(os.environ.get("ORACLE_FIT_WARM_SIGMA", "0.3"))
            optim._optimiser = optim._pints_optimiser(
                x0=optim.problem.parameters.get_initial_values(transformed=True),
                sigma0=np.asarray(optim._std0, float) * f_sigma, boundaries=optim._boundaries)
        result = optim.run()
    runtime = time.time() - t0
    best = {k: float(np.asarray(v).ravel()[0]) for k, v in result.best_inputs.items()}
    cost = float(np.asarray(result.best_cost).ravel()[0])
    log.info("joint fit %s: cost %s in %.0f s: %s", cells, cost, runtime, best)
    reports = {}
    for c in cells:
        own = {k: v for k, v in best.items() if not k.startswith("Rest before ") or f"{c}:" in k}
        ev = evaluate_chain(jdata[c], pv_base, own, chain=chains[c])
        mine = _unprefixed(best, c)
        reports[c] = {
            "cell_id": c, "method": "chain-joint", "best_inputs": {**fixed[c], **mine}, "fitted_inputs": mine,
            "fixed_inputs": fixed[c], "ocv_balance": balances[c], "balance_mode": balance,
            "sei": sei, "ec_diffusivity_factor": float(ec_diffusivity_factor),
            "hysteresis": bool(hysteresis), "size_distribution": psd_mode(size_distribution) or False,
            "radial_mesh": radial_mesh_setting(),
            "envelope_probe_ma": envelope_setting(),
            "voltage_limits": ({"v_min": CYCLER_V_MIN, "v_max_cycling": CYCLER_V_MAX_CYCLING,
                                "v_max_formation_cv": CYCLER_V_MAX_FORMATION} if voltage_limits else None),
            "seed": seed,
            "warm_start": ({"from": os.environ["ORACLE_FIT_WARM_JSON"],
                            "sigma_factor": float(os.environ.get("ORACLE_FIT_WARM_SIGMA", "0.3"))}
                           if warm_inputs is not None else None),
            "joint": {"cells": cells, "cost": cost, "shared": shared},
            "cost": cost,
            "param_bounds": {k: list(map(float, p.bounds)) for k, p in fit.items()
                             if not k.startswith("Rest before ") or f"{c}:" in k},
            "log_params": [k for k, p in fit.items() if isinstance(p.transformation, pybop.LogTransformation)],
            "runtime_s": round(runtime, 1), "worker_pool": dict(patches.POOL_STATS),
            "n_iterations": (int(result.n_iterations) if getattr(result, "n_iterations", None) is not None else None),
            "from_checkpoint": bool(resume),
            **ev}
        # report the cell's own rest names (as in its unprefixed data)
        reports[c]["param_bounds"] = _unprefixed(reports[c]["param_bounds"], c)
        reports[c]["log_params"] = [k.replace(f"{c}:", "") for k in reports[c]["log_params"]
                                    if not k.startswith("Rest before ") or f"{c}:" in k]
    return reports


def _mesh_agreement(fit_res: dict, val_res: dict, x_fit: int, cell: str) -> dict:
    """Validity record: the fit mesh's voltages vs the validation mesh's (the fitted
    parameters must not depend on resolving the interfaces better)."""
    name = f"validity[{cell}] fit mesh (x{x_fit}) agrees with validation mesh (x{VALIDATE_X_FACTOR})"
    if "error" in fit_res:
        return {"name": name, "status": "fail", "value": None, "limit": [1.0, 3.0], "note": fit_res["error"]}
    dv = [float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in
          zip(fit_res["records"], val_res["records"]) if len(a) and len(a) == len(b)]
    # stop times may shift a sample: compare away from the largest single-point jumps
    worst = float(np.percentile(np.concatenate([np.abs(np.asarray(a) - np.asarray(b)) for a, b in
                  zip(fit_res["records"], val_res["records"]) if len(a) and len(a) == len(b)]), 99)) if dv else np.nan
    v = 1e3 * worst
    # a mesh-dependent fit is a resolution matter, not unphysical: warn above the 1 mV noise
    status = "skip" if not np.isfinite(v) else ("pass" if v < 1.0 else "warn")
    return {"name": name, "status": status, "value": v, "limit": 1.0,
            "note": f"p99 |dV| mV over all records (max {1e3 * max(dv):.1f} mV, at cycler stops)" if dv else ""}


def evaluate_chain(data: dict, pv_base, inputs: dict, chain: ChainModel | None = None) -> dict:
    """Measured vs simulated curves/spectra + diagnostics at ``inputs`` (plot/physicality),
    including the physical-validity gate (:mod:`battery_oracle.validity`, under
    ``diagnostics["validity"]``): the full internal solution on the deployed mesh
    (x VALIDATE_X_FACTOR, the oracle's) and on that mesh refined x2, judged by exact
    laws, convergence of the interface continuity, and whether the deployed mesh
    resolves the outputs to the measurement noise. Curves are the deployed mesh's."""
    from battery_oracle import validity

    chain = chain or ChainModel(data, pv_base)
    cell = str(data.get("cell_id", ""))
    fit_res = chain.solve(inputs)               # on the fit mesh (usually cached)
    keep, xf = getattr(chain, "keep_solution", False), getattr(chain, "x_factor", 1)

    def full(x_factor):
        """Full-storage solve past the candidate cache on x ``x_factor``; returns the
        result with the solution replaced by its validity measures."""
        chain.keep_solution, chain.x_factor = True, x_factor
        try:
            r = chain._solve({**chain.fixed, **inputs})
        except Exception as exc:                # noqa: BLE001
            r = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        finally:
            chain.keep_solution, chain.x_factor = keep, xf
        if "error" not in r:
            try:
                r["measures"] = validity.measure(r.pop("solution"), r.pop("parameter_values"))
            except Exception as exc:            # noqa: BLE001
                r["measures_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return r

    res = full(VALIDATE_X_FACTOR)
    if "error" in res:
        return {"records": [{"kind": r["kind"], "label": r["label"], "error": res["error"]}
                            for r in data["records"]], "diagnostics": {"error": res["error"]}}
    fine = full(2 * VALIDATE_X_FACTOR)
    names = (f"x{VALIDATE_X_FACTOR}", f"x{2 * VALIDATE_X_FACTOR}")
    if "error" in fine or "measures" not in res or "measures" not in fine:
        why = fine.get("error") or res.get("measures_error") or fine.get("measures_error")
        res["diag"]["validity"] = [{"name": f"validity[{cell}] two-mesh gate evaluated", "status": "fail",
                                    "value": None, "limit": None, "note": f"refined mesh: {why}"}]
    else:
        res["diag"]["validity"] = (validity.judge(res["measures"], fine["measures"], factor=2, rtol=chain.rtol,
                                                  cell=cell, mesh=names)
                                   + validity.relevance(res, fine, cell=cell, mesh=names)
                                   + [_mesh_agreement(fit_res, res, xf, cell)])
        res["diag"]["validity_measures"] = {names[0]: res["measures"], names[1]: fine["measures"]}
    recs = []
    for r, v in zip(data["records"], res["records"]):
        vm = np.asarray(r["V"], float)
        n = len(v)
        step = max(1, len(vm) // 600)
        recs.append({"kind": r["kind"], "label": r["label"], "weight": 1.0 / SIGMA_V,
                     "t": np.asarray(r["t"])[::step].tolist(), "v": vm[::step].tolist(),
                     "vfit": np.pad(np.asarray(v, float), (0, len(vm) - n),
                                    constant_values=np.nan)[::step].tolist(),
                     "n_simulated": int(n), "n_points": int(len(vm)),
                     "rmse_v": float(np.sqrt(np.mean((np.asarray(v) - vm[:n]) ** 2))) if n else float("nan")})
    for s, z in zip(data["spectra"], res["spectra"]):
        zt = np.asarray(s["Z"], complex)
        e = {"kind": "eis", "label": s["label"], "f": np.asarray(s["f"]).tolist(),
             "z_real": zt.real.tolist(), "z_imag": zt.imag.tolist()}
        if z is None:
            e["error"] = "spectrum after the simulation stopped"
        else:
            e.update(zfit_real=z.real.tolist(), zfit_imag=z.imag.tolist(),
                     rmse_ohm=float(np.sqrt(np.mean(np.abs(z - zt) ** 2))),
                     rel_rmse=float(np.sqrt(np.mean(np.abs(z - zt) ** 2)) / np.median(np.abs(zt))))
        recs.append(e)
    return {"records": recs, "diagnostics": res["diag"], "group_offsets_s": res["offsets"]}


# ---------------------------------------------------------------------------
# From a fit to the oracle's bol section
# ---------------------------------------------------------------------------

def ocv_limits_in_data(pv, li_fraction: float, temperature_k: float, *, margin: float = 0.01,
                       nominal: tuple[float, float] = (3.0, 4.2)) -> dict:
    """0% / 100% SOC open-circuit voltages that keep PyBaMM's electrode-SOH window
    inside both OCPs' data, for a cell holding ``li_fraction`` x Q_p of lithium.

    The jones2022 protocol window (3.0 V discharge cut-off, 4.2 V formation CV) is
    used where the fitted balance reaches it inside the data; otherwise the limit is
    the OCV where an electrode reaches its data edge (Ai2020 LiCoO2 data stop at
    y = 0.4, below which the cubic extrapolation is unphysical). Returns the limits
    and whether each was clamped.
    """
    q_n, q_p = electrode_capacities_ah(pv)
    q_li = li_fraction * q_p
    un, up = _ocp_evaluators(pv, temperature_k)
    (x_lo, x_hi), (y_lo, y_hi) = ocp_data_range(pv, "negative"), ocp_data_range(pv, "positive")
    x_lo, x_hi, y_lo, y_hi = x_lo + margin, x_hi - margin, y_lo + margin, y_hi - margin

    def state(y_target, x_clip):
        x = (q_li - y_target * q_p) / q_n
        if x_clip(x) != x:
            x = x_clip(x)
            y_target = (q_li - x * q_n) / q_p
        return x, y_target

    xt, yt = state(y_lo, lambda x: min(x, x_hi))           # top of charge: positive emptiest
    xb, yb = state(y_hi, lambda x: max(x, x_lo))           # bottom: positive fullest
    top, bot = up(yt) - un(xt), up(yb) - un(xb)
    ocv100, ocv0 = min(nominal[1], top), max(nominal[0], bot)
    return {"ocv0": float(ocv0), "ocv100": float(ocv100), "ocv0_clamped": bool(bot > nominal[0]),
            "ocv100_clamped": bool(top < nominal[1]), "ocv_top_in_data": float(top),
            "ocv_bottom_in_data": float(bot)}


def overrides_from_chain(pv, best: dict, measured_capacity_mah: float | None = None
                         ) -> tuple[dict, dict, float, dict, dict]:
    """(parameter_overrides, parameter_scales, contact_resistance, thermal kwargs, esoh info).

    Geometry becomes one square layer of the fitted area; thickness, N:P, and the
    cyclable lithium (as initial concentrations at 100% SOC, so PyBaMM's electrode-
    SOH solve recovers the fitted lithium) are absolute overrides. The 0% / 100%
    OCVs stay inside the OCP data (``ocv_limits_in_data``). The nominal capacity is
    the MEASURED cell capacity (the oracle maps real mA through it, so the replayed
    currents are the real ones); the capacity between the OCV limits is reported.
    """
    import pybamm

    ov = geometry_overrides(pv, best, measured_capacity_mah / 1000.0 if measured_capacity_mah else None)
    ov.update({k: float(best[k]) for k in (*RADII, *LICO2_DS_SHAPE, *GRAPHITE_DS_SHAPE, *J0_SHAPE) if k in best})
    if "Electrode height [m]" not in ov:        # legacy fits: area as a parallel-layer count
        area = float(pv["Electrode height [m]"]) * float(pv["Electrode width [m]"]) * float(best.get(N_PARALLEL, 1.0))
        side = float(np.sqrt(area))
        ov.update({"Electrode height [m]": side, "Electrode width [m]": side, N_PARALLEL: 1.0})
    pvf = apply_parameter_overrides(pv.copy(), ov)
    f_li = float(best.get(LI_FRACTION, lithium_fraction(pvf)))
    T = float(pvf["Initial temperature [K]"])
    lim = ocv_limits_in_data(pvf, f_li, T)
    ov[OCV0], ov[OCV100] = lim["ocv0"], lim["ocv100"]
    x100, y100 = initial_stoichiometries(pvf, ov[OCV100], f_li, T)
    ov["Initial concentration in negative electrode [mol.m-3]"] = x100 * float(
        pvf["Maximum concentration in negative electrode [mol.m-3]"])
    ov["Initial concentration in positive electrode [mol.m-3]"] = y100 * float(
        pvf["Maximum concentration in positive electrode [mol.m-3]"])
    pvf = apply_parameter_overrides(pvf, ov)
    esoh = dict(lim)
    try:
        x0, x1, y1, y0 = (float(v) for v in pybamm.lithium_ion.get_min_max_stoichiometries(pvf))
        q_n, _ = electrode_capacities_ah(pvf)
        esoh.update(x_0=x0, x_100=x1, y_100=y1, y_0=y0, capacity_ah=float((x1 - x0) * q_n))
    except Exception as exc:                    # noqa: BLE001
        esoh["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        log.warning("electrode SOH on the fitted cell failed (%s)", exc)
    if measured_capacity_mah:
        ov["Nominal cell capacity [A.h]"] = float(measured_capacity_mah) / 1000.0
    elif "capacity_ah" in esoh:
        ov["Nominal cell capacity [A.h]"] = esoh["capacity_ah"]
    scales = {n: float(best[scale_parameter_name(n)]) for n in CHAIN_SCALES if scale_parameter_name(n) in best}
    thermal = {"thermal": "lumped", "h_total_W_per_m2K": float(best.get(H_TOTAL, pv[H_TOTAL]))}
    return ov, scales, float(best.get(CONTACT_R, 0.0)), thermal, esoh


def aggregate_chain(cell_reports: list[dict], pv) -> dict:
    """Median over cells (geometric for log-scale parameters) -> bol section. Rests
    between sequences are measurement history, not cell properties: reported only."""
    ok = [c for c in cell_reports if c.get("best_inputs")]
    failed = [{"cell_id": c["cell_id"], "reason": c.get("error", "no fit")} for c in cell_reports
              if not c.get("best_inputs")]
    if not ok:
        raise ValueError(f"no cell has a chain fit: {failed}")
    keys = [k for k in ok[0]["best_inputs"] if not k.startswith("Rest before")]
    logs = set(ok[0].get("log_params", []))
    med = {}
    for k in keys:
        x = np.asarray([c["best_inputs"][k] for c in ok if k in c["best_inputs"]], float)
        med[k] = float(np.exp(np.median(np.log(x)))) if k in logs else float(np.median(x))
    caps = [c["cell_capacity_mah"] for c in ok if c.get("cell_capacity_mah")]
    ov, scales, cr, thermal, esoh = overrides_from_chain(pv, med, float(np.median(caps)) if caps else None)
    dist = {}
    for k in ok[0]["best_inputs"]:
        x = np.asarray([c["best_inputs"][k] for c in ok if k in c["best_inputs"]], float)
        dist[k] = {"values": x.tolist(), "median": float(np.median(x)),
                   "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}
    # submodels of the fitted model the oracle must use too (same parameter functions)
    model_kw = {}
    if HYST_DELTA in med:
        model_kw["hysteresis_half_gap_V"] = med[HYST_DELTA]
    mode = psd_mode(ok[0].get("size_distribution"))
    if mode:
        model_kw["size_distribution"] = {"mode": mode, "sd_n": med.get(PSD_SD_N), "sd_p": med.get(PSD_SD_P)}
    # the fit's particle mesh (a report written before reports recorded it: this run's setting)
    mesh = ok[0]["radial_mesh"] if "radial_mesh" in ok[0] else radial_mesh_setting()
    if mesh:
        model_kw["radial_mesh"] = mesh
    return {"parameter_overrides": ov, "parameter_scales": scales, "contact_resistance_ohm": cr,
            **thermal, **model_kw,
            "fit": {"method": "chain DFN fit per cell (one continuous history: formation, cycling, "
                              "rests, non-stationary EIS; lumped thermal); oracle values = cross-cell median",
                    "cells": [c["cell_id"] for c in ok], "failed_cells": failed, "distributions": dist,
                    "param_bounds": ok[0]["param_bounds"], "log_params": sorted(logs),
                    "esoh": esoh, "measured_capacity_mah": float(np.median(caps)) if caps else None,
                    "per_cell_cost": {c["cell_id"]: c["cost"] for c in ok},
                    "diagnostics": {c["cell_id"]: c.get("diagnostics") for c in ok}}}
