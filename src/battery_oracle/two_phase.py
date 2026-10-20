"""Phase-coexistence (two-phase) LiCoO2 positive electrode for the DFN, via PyBaMM's MSMR.

Why. Whole-trajectory residuals of the jones2022 fits show a missing discharge loss that
is zero at a current step, builds under load over ~10-100 s, is superlinear in the rate,
and is gone within ~10 s of rest without a trace in the rest OCV. A leading hypothesis is
Li_yCoO2 phase coexistence: the H1/H2 (O3(I)/O3(II)) miscibility gap at y ~ 0.75-0.93
(Reimers & Dahn 1992, J. Electrochem. Soc. 139, 2091) and the vacancy-poor phase as
y -> 1. In a two-phase particle the rest voltage is pinned at the plateau (it "recovers"
at once), while a growing lithium-rich shell adds a transport loss under load -- the
shrinking-core picture of LiCoO2 discharge (Zhang & White 2007, J. Electrochem. Soc.
154, A587, doi 10.1149/1.2728733: a spherical-particle moving-boundary model of
diffusion-controlled phase transitions in LiCoO2, embedded in a porous electrode).

Upstream first. PyBaMM's MSMR framework (Verbrugge et al. 2017, J. Electrochem. Soc.
164, E3243, doi 10.1149/2.0341708jes; Baker & Verbrugge 2018, J. Electrochem. Soc. 165,
A3952, doi 10.1149/2.0771816jes), selectable per electrode in PyBaMM 26.7, writes the
occupancy as a sum of logistic reactions x(U) = sum_j X_j / (1 + exp(f (U - U0_j) / w_j))
and solves the particle in the chemical-potential variable U (mu = -F U) with the Onsager
flux N = c_max x (1 - x) (F/RT) D grad U. That is a thermodynamically consistent
diffusion law; its chemical diffusivity is D Theta(x), Theta = (F/RT) x (1 - x) |dU/dx|.

* With reactions fitted freely to the measured (smoothly sloping) Ai2020 OCP, MSMR is a
  single-phase solid solution -- the same law as the "thermo-D" variant of
  :mod:`battery_oracle.parameter_sets` (``lico2_ds="thermo"``).
* A reaction of width w -> 0 makes dU/dx -> 0 over its occupancy X: a flat plateau of
  width X, inside which Theta -> 0 and a small change of U moves a large amount of
  lithium. Solving for U (not c) through such a plateau is the enthalpy (chemical-
  potential) formulation of a two-phase Stefan problem (Crank 1984, Free and Moving
  Boundary Problems, Oxford; Voller & Cross 1981, Int. J. Heat Mass Transfer 24, 545):
  the plateau region carries no gradient of U, a steep composition front between the
  two end-member phases forms and moves with the Stefan speed, and lithium, flux and U
  are continuous everywhere -- no jump condition has to be imposed, it emerges in the
  limit w -> 0. With w > 0 it is the diffuse-interface regularisation of that limit; it
  cannot represent a non-monotone (spinodal) free energy, nucleation barriers or
  coherency strain.

So the two-phase particle here is MSMR with a narrow "plateau" reaction of occupancy
X = y_beta - y_alpha (the miscibility gap) and the other reactions refitted so that the
total stays close to the literature OCP outside the gap. Nothing is fitted to cell data:
:func:`fit_msmr_reactions` only re-expresses the literature Ai2020 LiCoO2 OCP (the one
the chemistry already uses) in MSMR form.

Interface for the fit chain (``bol_chain``): :func:`msmr_options` adds the per-electrode
MSMR options to ``chain_options(...)``; :func:`msmr_parameter_values` converts a built
parameter set (``parameter_sets.resolve_parameter_values(name)``) into the MSMR positive
electrode. Validity: :func:`add_lithium_flux_counter` + :func:`particle_lithium_balance`
(particle inventory vs integrated interfacial flux), :func:`stoichiometry_range`, and the
existing two-mesh gate :func:`battery_oracle.validity.measure` / ``judge``.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np

F = 96485.33212
R_GAS = 8.314462618
T_REF_K = 298.15

#: H1/H2 two-phase region of Li_yCoO2 at room temperature (Reimers & Dahn 1992,
#: J. Electrochem. Soc. 139, 2091; the same bounds -- metallic Li_0.75CoO2 to insulating
#: Li_0.93CoO2 -- are quoted in later work on the insulator-metal transition).
GAP_REIMERS_DAHN = (0.75, 0.93)
#: Width of the plateau reaction in the "two-phase limit" demonstration: the 10-90 %
#: occupancy span of a logistic is 2 ln 9 w RT/F = 113 w mV, so w = 0.05 is a plateau flat
#: to ~6 mV (a slow-rate LiCoO2 H1/H2 plateau is flat to a few mV). Not fitted.
PLATEAU_W_DEFAULT = 0.05
#: MSMR charge-transfer coefficient (symmetric, as the template's Butler-Volmer)
ALPHA_DEFAULT = 0.5

POS = "Positive electrode"
FLUX_COUNTER = "Integrated positive electrode lithium uptake [mol]"


@dataclass(frozen=True)
class Reaction:
    """One MSMR host-site reaction: standard potential U0 [V], occupancy X, ideality w."""
    U0: float
    X: float
    w: float
    alpha: float = ALPHA_DEFAULT


# ---------------------------------------------------------------------------
# MSMR thermodynamics (numpy)
# ---------------------------------------------------------------------------

def _f(T: float) -> float:
    return F / (R_GAS * T)


def msmr_x(U, reactions, T: float = T_REF_K):
    """Occupancy x(U) = sum_j X_j / (1 + exp(f (U - U0_j) / w_j)) (Baker 2018 eq. 5-6)."""
    U = np.asarray(U, float)
    x = np.zeros_like(U)
    for r in reactions:
        z = np.clip(_f(T) * (U - r.U0) / r.w, -700.0, 700.0)
        x = x + r.X / (1.0 + np.exp(z))
    return x


def msmr_dxdU(U, reactions, T: float = T_REF_K):
    """dx/dU [1/V] (< 0)."""
    U = np.asarray(U, float)
    d = np.zeros_like(U)
    for r in reactions:
        z = np.clip(_f(T) * (U - r.U0) / r.w, -700.0, 700.0)
        e = np.exp(-np.abs(z))                    # symmetric, overflow-free form of e/(1+e)^2
        d = d - (_f(T) / r.w) * r.X * e / (1.0 + e) ** 2
    return d


def msmr_ocp(y, reactions, T: float = T_REF_K, lo: float = 0.0, hi: float = 6.0):
    """U(y): inverse of :func:`msmr_x` by bisection (x is strictly decreasing in U)."""
    y = np.atleast_1d(np.asarray(y, float))
    a = np.full_like(y, lo)
    b = np.full_like(y, hi)
    for _ in range(64):                            # (hi - lo) / 2**64: far below round-off
        m = 0.5 * (a + b)
        above = msmr_x(m, reactions, T) > y        # still too lithiated -> U must rise
        a = np.where(above, m, a)
        b = np.where(above, b, m)
    return 0.5 * (a + b)


def thermodynamic_factor(y, reactions, T: float = T_REF_K):
    """Theta(y) = (F/RT) y (1 - y) |dU/dy|: MSMR's chemical diffusivity is D Theta."""
    y = np.asarray(y, float)
    U = msmr_ocp(y, reactions, T)
    return _f(T) * y * (1.0 - y) / np.abs(msmr_dxdU(U, reactions, T))


# ---------------------------------------------------------------------------
# Re-expressing a literature OCP in MSMR form
# ---------------------------------------------------------------------------

def lico2_ocp_data():
    """(y, U) of the Ai2020 LiCoO2 OCP over its real data range (no continuation)."""
    from battery_oracle.parameter_sets import LICO2_MONOTONE_NAME, OCP_DATA_RANGES, _lico2_monotone_table

    x, u = _lico2_monotone_table()
    lo, hi = OCP_DATA_RANGES[LICO2_MONOTONE_NAME]
    o = np.argsort(x)
    x, u = np.asarray(x, float)[o], np.asarray(u, float)[o]
    keep = (x >= lo - 1e-12) & (x <= hi + 1e-12)
    return x[keep], u[keep]


def _unpack(p, n_free, plateau, total):
    """Free vector -> reactions. X of the free reactions by softmax onto ``total``."""
    U0 = p[:n_free]
    w = np.exp(p[n_free:2 * n_free])
    s = p[2 * n_free:3 * n_free]
    s = np.exp(s - s.max())
    X = total * s / s.sum()
    out = [Reaction(float(a), float(b), float(c)) for a, b, c in zip(U0, X, w)]
    if plateau is not None:
        out.append(Reaction(float(p[3 * n_free]), float(plateau[1] - plateau[0]), float(plateau[2])))
    return out


def fit_msmr_reactions(y, U, n: int = 5, *, plateau: tuple[float, float, float] | None = None,
                       T: float = T_REF_K, n_starts: int = 6, seed: int = 0,
                       w_bounds: tuple[float, float] = (0.05, 8.0)) -> tuple[list[Reaction], dict]:
    """Represent an OCP table (y, U) by MSMR reactions with sum X_j = 1 (x -> 1 as U -> -inf).

    ``plateau=(y_alpha, y_beta, w)``: one reaction is fixed to occupancy y_beta - y_alpha and
    width w (its U0 free) -- the two-phase plateau; the other n - 1 are free. Least squares
    on the occupancy x(U_i) - y_i plus the voltage residual of the inverse, the latter
    weighted to keep the steep ends. Returns (reactions sorted by U0, fit report)."""
    from scipy.optimize import least_squares

    y = np.asarray(y, float)
    U = np.asarray(U, float)
    n_free = n - (1 if plateau is not None else 0)
    total = 1.0 - (plateau[1] - plateau[0] if plateau is not None else 0.0)
    rng = np.random.default_rng(seed)
    lw, hw = np.log(w_bounds[0]), np.log(w_bounds[1])
    lo = np.r_[np.full(n_free, U.min() - 0.3), np.full(n_free, lw), np.full(n_free, -8.0)]
    hi = np.r_[np.full(n_free, U.max() + 0.3), np.full(n_free, hw), np.full(n_free, 8.0)]
    if plateau is not None:
        sel = (y >= plateau[0]) & (y <= plateau[1])
        u_mid = float(np.median(U[sel])) if sel.any() else float(np.median(U))
        lo = np.r_[lo, U.min()]
        hi = np.r_[hi, U.max()]
    u_scale = max(float(np.ptp(U)), 1e-3)

    def res(p):
        rx = _unpack(p, n_free, plateau, total)
        r1 = msmr_x(U, rx, T) - y
        r2 = (msmr_ocp(y, rx, T) - U) / u_scale
        return np.r_[r1, 0.3 * r2]

    best = None
    for k in range(n_starts):
        U0 = np.sort(rng.uniform(U.min(), U.max(), n_free))[::-1]
        p0 = np.r_[U0, rng.uniform(np.log(0.3), np.log(2.0), n_free), rng.normal(0, 0.3, n_free)]
        if plateau is not None:
            p0 = np.r_[p0, u_mid]
        p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
        try:
            r = least_squares(res, p0, bounds=(lo, hi), x_scale="jac", max_nfev=4000)
        except Exception:                                  # noqa: BLE001
            continue
        if best is None or r.cost < best.cost:
            best = r
    rx = sorted(_unpack(best.x, n_free, plateau, total), key=lambda r: -r.U0)
    dU = msmr_ocp(y, rx, T) - U
    rep = {"rmse_U_V": float(np.sqrt(np.mean(dU ** 2))), "max_abs_U_V": float(np.max(np.abs(dU))),
           "rmse_x": float(np.sqrt(np.mean((msmr_x(U, rx, T) - y) ** 2))),
           "sum_X": float(sum(r.X for r in rx)), "n": n, "plateau": plateau,
           "reactions": [asdict(r) for r in rx]}
    return rx, rep


# ---------------------------------------------------------------------------
# PyBaMM configuration
# ---------------------------------------------------------------------------

def msmr_options(base: dict | None, n_reactions: int) -> dict:
    """``base`` model options with the positive electrode switched to MSMR (OCP, particle,
    intercalation kinetics); the negative electrode keeps its own choices. PyBaMM 26.7
    accepts mixed full cells when all three options name MSMR for that electrode."""
    opts = dict(base or {})

    def pos(key, neg_default):
        v = opts.get(key, neg_default)
        neg = v[0] if isinstance(v, (tuple, list)) else v
        return (neg, "MSMR")

    opts["open-circuit potential"] = pos("open-circuit potential", "single")
    opts["particle"] = pos("particle", "Fickian diffusion")
    opts["intercalation kinetics"] = pos("intercalation kinetics", "symmetric Butler-Volmer")
    opts["number of MSMR reactions"] = ("none", str(int(n_reactions)))
    return opts


def _eval(pv, expr) -> float:
    return float(np.asarray(pv.process_symbol(expr).evaluate()).ravel()[0])


def template_j0(pv, y: float, c_e: float | None = None, T: float = T_REF_K) -> float:
    """The parameter set's positive exchange-current density [A/m2] at stoichiometry y."""
    import pybamm

    c_max = float(pv["Maximum concentration in positive electrode [mol.m-3]"])
    c_e = float(pv["Initial concentration in electrolyte [mol.m-3]"]) if c_e is None else c_e
    fn = pv["Positive electrode exchange-current density [A.m-2]"]
    return _eval(pv, fn(pybamm.Scalar(c_e), pybamm.Scalar(y * c_max), pybamm.Scalar(c_max), pybamm.Scalar(T)))


def msmr_j0_total(U, reactions, j0_ref: float, T: float = T_REF_K, ce_ratio: float = 1.0):
    """Sum over reactions of PyBaMM's MSMR exchange-current density (lithium_ion_parameters
    j0_j: j0_ref x_j^w_j exp(f (1 - a_j) (U - U0_j)) (c_e/c_e0)^(1 - a_j))."""
    U = np.asarray(U, float)
    tot = np.zeros_like(U)
    for r in reactions:
        xj = msmr_x(U, [r], T)
        tot = tot + j0_ref * xj ** r.w * np.exp(_f(T) * (1 - r.alpha) * (U - r.U0)) * ce_ratio ** (1 - r.alpha)
    return tot


def msmr_parameter_values(pv, reactions, *, y_init: float | None = None, j0_match_y: float = 0.5):
    """A copy of ``pv`` with an MSMR positive electrode built from ``reactions``.

    * reactions -> "Positive electrode host site ... (j)" parameters;
    * one reference exchange-current density for all reactions, chosen so the summed MSMR
      j0 equals the set's own j0 at stoichiometry ``j0_match_y`` and the initial
      electrolyte concentration (a unit mapping, not a fit);
    * "Positive particle diffusivity" is kept: in MSMR it is the D of the Onsager flux
      (chemical diffusivity D Theta), the same number as the set's;
    * initial state: ``y_init`` (default: the set's initial positive stoichiometry) ->
      "Initial voltage in positive electrode [V]" = U_msmr(y_init). The negative electrode
      keeps its initial concentration unless the caller shifts it (see
      :func:`shift_initial_state`)."""
    import pybamm

    pv = pv.copy()
    c_max = float(pv["Maximum concentration in positive electrode [mol.m-3]"])
    if y_init is None:
        y_init = float(pv["Initial concentration in positive electrode [mol.m-3]"]) / c_max
    U_ref = float(msmr_ocp(j0_match_y, reactions)[0])
    j0_ref = template_j0(pv, j0_match_y) / float(np.atleast_1d(msmr_j0_total(U_ref, reactions, 1.0))[0])
    upd = {"Initial voltage in positive electrode [V]": float(msmr_ocp(y_init, reactions)[0]),
           "Number of reactions in positive electrode": len(reactions)}
    for j, r in enumerate(reactions):
        upd[f"{POS} host site standard potential ({j}) [V]"] = r.U0
        upd[f"{POS} host site occupancy fraction ({j})"] = r.X
        upd[f"{POS} host site ideality factor ({j})"] = r.w
        upd[f"{POS} host site charge transfer coefficient ({j})"] = r.alpha
        upd[f"{POS} host site reference exchange-current density ({j}) [A.m-2]"] = j0_ref
    pv.update(upd, check_already_exists=False)
    # MSMR derives the initial concentration from the initial voltage; keep the old entry
    # consistent for code that reads it (e.g. capacity bookkeeping)
    pv.update({"Initial concentration in positive electrode [mol.m-3]":
               float(np.atleast_1d(msmr_x(upd["Initial voltage in positive electrode [V]"], reactions))[0]) * c_max})
    _ = pybamm  # imported for parity with other builders (process_symbol above)
    return pv


def electrode_capacities_mol(pv) -> tuple[float, float]:
    """Lithium sites [mol] of the negative and positive electrodes (eps_s L c_max A)."""
    area = (float(pv["Electrode height [m]"]) * float(pv["Electrode width [m]"])
            * float(pv["Number of electrodes connected in parallel to make a cell"]))
    out = []
    for d, D in (("negative", "Negative"), ("positive", "Positive")):
        eps = pv[f"{D} electrode active material volume fraction"]
        eps = float(eps) if not callable(eps) else None
        if eps is None:
            raise ValueError("x-dependent active material fraction not supported here")
        out.append(eps * float(pv[f"{D} electrode thickness [m]"])
                   * float(pv[f"Maximum concentration in {d} electrode [mol.m-3]"]) * area)
    return out[0], out[1]


def shift_initial_state(pv, y_new: float) -> tuple[float, float]:
    """(x_neg, y_pos) uniform initial stoichiometries that move ``pv``'s state to positive
    stoichiometry ``y_new`` at constant cyclable lithium (Coulomb counting)."""
    n_n, n_p = electrode_capacities_mol(pv)
    x0 = float(pv["Initial concentration in negative electrode [mol.m-3]"]) / float(
        pv["Maximum concentration in negative electrode [mol.m-3]"])
    y0 = float(pv["Initial concentration in positive electrode [mol.m-3]"]) / float(
        pv["Maximum concentration in positive electrode [mol.m-3]"])
    return x0 - (y_new - y0) * n_p / n_n, y_new


def set_uniform_state(pv, x_neg: float, y_pos: float, reactions=None):
    """In place: uniform initial stoichiometries; MSMR positive via its initial voltage."""
    pv.update({"Initial concentration in negative electrode [mol.m-3]":
               x_neg * float(pv["Maximum concentration in negative electrode [mol.m-3]"]),
               "Initial concentration in positive electrode [mol.m-3]":
               y_pos * float(pv["Maximum concentration in positive electrode [mol.m-3]"])})
    if reactions is not None:
        pv.update({"Initial voltage in positive electrode [V]": float(msmr_ocp(y_pos, reactions)[0])})
    return pv


# ---------------------------------------------------------------------------
# Validity helpers
# ---------------------------------------------------------------------------

def add_msmr_thermal_correction(model, n_reactions: int, domain: str = "positive"):
    """Adds the missing temperature term to PyBaMM's MSMR particle equation (in place).

    PyBaMM (26.7.1.0 and 26.10.1.0; develop as of 2026-10-20) solves the MSMR particle for
    U with dU/dt = -div N / (c_max dx/dU), i.e. c_max dx/dt = -div N only while T is
    constant: x = x(U, T), so with a time-varying temperature lithium appears/disappears at
    rate c_max (dx/dT)|_U dT/dt (~1 % of the delivered lithium for a 1.8 K rise in a 120 s
    3C pulse; diagnosed by scripts/oracle_recal/verify/two_phase_msmr_thermal.py). The
    consistent equation is

        dU/dt = [-div N / c_max - (dx/dT)|_U dT/dt] / (dx/dU).

    Implemented for the lumped thermal model (one cell temperature, dT/dt = its rhs) and
    temperature-independent host-site parameters (X_j, U0_j, w_j constants; otherwise their
    T-derivatives would be needed too). No-op for an isothermal model. Report upstream."""
    import pybamm

    D = domain.capitalize()
    tvar = model.variables.get("Volume-averaged cell temperature [K]")
    if tvar is None or tvar not in model.rhs:
        return model                                   # isothermal: nothing to correct
    if model.options["thermal"] != "lumped":
        raise NotImplementedError("MSMR thermal correction implemented for 'thermal': 'lumped' only")
    U = model.variables[f"{D} particle potential [V]"]
    if U in model.rhs:                                 # x-resolved particles (DFN)
        T_e = model.variables[f"{D} electrode temperature [K]"]
        dxdU = model.variables[f"{D} particle differential stoichiometry [V-1]"]
        x_resolved = True
    else:                                              # x-averaged particles (SPM, SPMe)
        U = model.variables[f"X-averaged {domain} particle potential [V]"]
        if U not in model.rhs:
            raise ValueError(f"no MSMR {domain} particle potential state in this model")
        T_e = model.variables[f"X-averaged {domain} electrode temperature [K]"]
        dxdU = model.variables[f"X-averaged {domain} particle differential stoichiometry [V-1]"]
        x_resolved = False
    T = pybamm.PrimaryBroadcast(T_e, f"{domain} particle")
    f = pybamm.constants.F / (pybamm.constants.R * T)
    dxdT = 0
    for j in range(int(n_reactions)):
        inputs = {"Temperature [K]": T}
        U0 = pybamm.FunctionParameter(f"{D} electrode host site standard potential ({j}) [V]", inputs)
        X = pybamm.FunctionParameter(f"{D} electrode host site occupancy fraction ({j})", inputs)
        w = pybamm.FunctionParameter(f"{D} electrode host site ideality factor ({j})", inputs)
        z = f * (U - U0) / w
        # x_j = X / (1 + e^z), dz/dT|_U = -z / T  ->  dx_j/dT = X e^z / (1 + e^z)^2 * z / T
        s = pybamm.sigmoid(0, z, 0.5)                  # (1 + tanh(z/2))/2 = e^z / (1 + e^z), overflow-safe
        dxdT = dxdT + X * s * (1 - s) * z / T
    dTdt = model.rhs[tvar]
    if x_resolved:
        dTdt = pybamm.PrimaryBroadcast(dTdt, f"{domain} electrode")
    dTdt_b = pybamm.PrimaryBroadcast(dTdt, f"{domain} particle")
    model.rhs[U] = model.rhs[U] - dxdT * dTdt_b / dxdU
    model.variables[f"{D} particle MSMR thermal correction [V.s-1]"] = -dxdT * dTdt_b / dxdU
    return model


def add_lithium_flux_counter(model):
    """Adds the state FLUX_COUNTER(t) = -(A/F) int_0^t int_{x_p} a j_main dx dt' -- the lithium
    the main positive reaction has put into the particles -- integrated by the solver itself,
    so particle inventory change vs integrated flux is compared to solver tolerance (not to
    the quadrature error of saved samples). Call before discretisation."""
    import pybamm

    aj = model.variables["Positive electrode volumetric interfacial current density [A.m-3]"]
    x = pybamm.SpatialVariable("x_p", domain=["positive electrode"], auxiliary_domains={
        "secondary": "current collector"}, coord_sys="cartesian")
    area = pybamm.Parameter("Electrode height [m]") * pybamm.Parameter("Electrode width [m]") * pybamm.Parameter(
        "Number of electrodes connected in parallel to make a cell")
    q = pybamm.Variable(FLUX_COUNTER, domain="current collector")
    model.rhs[q] = -area / pybamm.constants.F * pybamm.Integral(aj, x)
    model.initial_conditions[q] = pybamm.Scalar(0.0)
    model.variables[FLUX_COUNTER] = q
    return model


#: PyBaMM < 26.10 reports a wrong MSMR particle inventory: ``_BaseAverage._try_separable``
#: rewrote avg(a / b) as avg(a) / avg(b) whenever EITHER operand was constant, so the
#: r-average of x(U) = sum_j X_j / (1 + exp(...)) became sum_j X_j / avg(1 + exp(...)).
#: Only derived outputs are affected ("R-averaged ... stoichiometry", "Total lithium in
#: positive electrode [mol]", ...); the MSMR dynamics conserve lithium (diagnosed
#: 2026-10-20, scripts/oracle_recal/verify/two_phase_msmr_conservation*.py: 5-18 % of the
#: delivered lithium "missing" on 26.7.1.0, independent of mesh and tolerance; 1e-8 on
#: 26.10.1.0). Fixed upstream in PyBaMM 26.10.0.0 (pybamm-team/PyBaMM#5813, fixes #5810:
#: "split a Division only when its denominator is constant"). :func:`particle_inventory`
#: recomputes the inventory from the solved field on the mesh, independent of the version.
MSMR_AVERAGE_FIX_PYBAMM = "26.10.0.0"


def _fv_volume_weights(mesh, domain: str) -> np.ndarray:
    """Normalised finite-volume weights of a particle mesh (spherical shells)."""
    e = np.asarray(mesh[domain].edges, float)
    w = e[1:] ** 3 - e[:-1] ** 3
    return w / w.sum()


def particle_inventory(sol, mesh, pv, reactions=None) -> np.ndarray:
    """Lithium [mol] in the positive particles at each saved time, summed on the mesh
    (spherical finite volumes in r, the electrode cells in x): MSMR from x(U) of the
    solved potential field (``reactions``), Fickian from the concentration field. The
    exact discrete inventory the finite-volume scheme conserves; version-independent
    (see MSMR_AVERAGE_FIX_PYBAMM)."""
    w_r = _fv_volume_weights(mesh, "positive particle")
    dx = np.asarray(mesh["positive electrode"].d_edges, float)
    c_max = float(pv["Maximum concentration in positive electrode [mol.m-3]"])
    if reactions is not None:
        U = np.asarray(sol["Positive particle potential [V]"].entries, float)
        T = np.asarray(sol["Positive electrode temperature [K]"].entries, float)   # (x, t)
        sto = msmr_x(U, reactions, np.broadcast_to(T[None, :, :], U.shape))
    else:
        sto = np.asarray(sol["Positive particle concentration [mol.m-3]"].entries, float) / c_max
    eps = float(pv["Positive electrode active material volume fraction"])
    area = (float(pv["Electrode height [m]"]) * float(pv["Electrode width [m]"])
            * float(pv["Number of electrodes connected in parallel to make a cell"]))
    y_rav = np.einsum("r,rxt->xt", w_r, sto)
    return eps * c_max * area * np.einsum("x,xt->t", dx, y_rav)


def particle_lithium_balance(sol, *, judged: np.ndarray | None = None, inventory: np.ndarray | None = None) -> dict:
    """Positive particle inventory change vs the integrated main-reaction flux.

    ``inventory``: the particle lithium [mol] at each time (default: PyBaMM's "Total lithium in
    positive electrode [mol]"; pass :func:`particle_inventory` for MSMR on PyBaMM < 26.10).
    ``residual_rel`` = max |dn_particles - integrated flux| / max(|integrated flux|, n_p * 1e-6)."""
    n = (np.asarray(inventory, float).ravel() if inventory is not None else
         np.asarray(sol["Total lithium in positive electrode [mol]"].entries, float).ravel())
    q = np.asarray(sol[FLUX_COUNTER].entries, float).ravel()
    dn = n - n[0]
    r = np.abs(dn - q)
    if judged is not None:
        r = r[judged]
    scale = max(float(np.max(np.abs(q))), 1e-6 * abs(float(n[0])))
    return {"residual_rel": float(np.max(r) / scale), "delivered_mol": float(q[-1]), "inventory_change_mol": float(dn[-1])}


def stoichiometry_range(sol) -> dict:
    """Min / max of the positive and negative particle stoichiometries over the history."""
    out = {}
    for d, D in (("negative", "Negative"), ("positive", "Positive")):
        s = np.asarray(sol[f"{D} particle stoichiometry"].entries, float)
        out[d] = (float(s.min()), float(s.max()))
    return out


def surface_flux_consistency(sol) -> float:
    """max over time and x of |N_s(r = R) - j/F| / max|j/F|: the particle's surface flux
    (the outermost flux edge) equals the interfacial reaction rate (no flux discontinuity
    between the particle and its reaction)."""
    N = np.asarray(sol["Positive particle flux [mol.m-2.s-1]"].entries, float)
    j = np.asarray(sol["Positive electrode interfacial current density [A.m-2]"].entries, float)
    # N has shape (r edges, x, t) or (r edges * x, t)
    if N.ndim == 2:
        nx = j.shape[0]
        N = N.reshape(-1, nx, N.shape[-1], order="F") if N.shape[0] % nx == 0 else N[:, None, :]
    surf = N[-1]
    jf = j / F
    scale = max(float(np.max(np.abs(jf))), 1e-30)
    return float(np.max(np.abs(surf - jf)) / scale)


def equilibrium_ocv(sol, pv, reactions=None, T: float = T_REF_K, positive_inventory=None) -> np.ndarray:
    """Cell OCV at each saved time from the bulk (volume-averaged) stoichiometries --
    what the cell would relax to: U_p(y_bar) - U_n(x_bar). ``positive_inventory`` [mol]
    overrides PyBaMM's positive total (see MSMR_AVERAGE_FIX_PYBAMM)."""
    import pybamm

    n_n, n_p = electrode_capacities_mol(pv)
    x_bar = np.asarray(sol["Total lithium in negative electrode [mol]"].entries, float).ravel() / n_n
    y_bar = (np.asarray(positive_inventory, float).ravel() if positive_inventory is not None else
             np.asarray(sol["Total lithium in positive electrode [mol]"].entries, float).ravel()) / n_p

    def ocp(name, s):
        fn = pv[name]
        if not callable(fn):
            return np.full(s.shape, float(fn))
        expr = pv.process_symbol(fn(pybamm.Vector(np.asarray(s, float))))
        return np.asarray(expr.evaluate(), float).ravel()

    U_n = ocp("Negative electrode OCP [V]", x_bar)
    U_p = msmr_ocp(y_bar, reactions, T) if reactions is not None else ocp("Positive electrode OCP [V]", y_bar)
    return U_p - U_n


__all__ = [
    "GAP_REIMERS_DAHN", "PLATEAU_W_DEFAULT", "Reaction", "msmr_x", "msmr_dxdU", "msmr_ocp",
    "thermodynamic_factor", "lico2_ocp_data", "fit_msmr_reactions", "msmr_options",
    "msmr_parameter_values", "msmr_j0_total", "template_j0", "electrode_capacities_mol",
    "shift_initial_state", "set_uniform_state", "add_lithium_flux_counter",
    "particle_lithium_balance", "stoichiometry_range", "surface_flux_consistency", "equilibrium_ocv",
]
