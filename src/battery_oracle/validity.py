"""Physical validity of a solved DFN history (the acceptance gate of a fit).

A fit need not reproduce the cell exactly, but its internal solution must obey the
continuum equations it discretises (user rule, 2026-10-18). From one PyBaMM solution
this measures, at every saved time:

* flux continuity across both electrode|separator interfaces: the electrolyte current
  density i_e, the electrolyte molar flux N_e, the concentration c_e and the potential
  phi_e, each extrapolated linearly to the interface from the two nearest samples on
  either side (a jump means the interface is not resolved or the coefficients there
  are inconsistent);
* charge balance: i_e + i_s = i_app in both electrodes, i_e = i_app in the separator, and no electrolyte current at
  the current collectors (i_e(0) = i_e(L) = 0);
* mesh oscillation: alternating-sign second differences of c_e and phi_e inside a
  domain (an odd-even mode of the finite volumes, not physics);
* lithium conservation: particles + electrolyte + lithium consumed by side reactions
  and lost active material stays at its initial value.

Continuity is not judged within ``STEP_WINDOW_S`` after a step in the applied current
(a cycler stop or a new step): there the double layer redistributes charge through a
boundary layer at the separator that starts with zero thickness (transmission-line
time constant ~ L^2 a C_dl / kappa_eff, ~0.01-1 s here), so no mesh resolves it at the
step itself (diag/validity_probe.py, 2026-10-18: a 0.69 i_1C "jump" right after a stop,
unchanged by refining x). In this model (differential surface form) each electrode's
i_e is written with its own surface potential difference and i_e = i_app at the
separator, so no coefficient is averaged across an interface for i_e.

Fluxes are scaled by the larger of the applied and the 1C current density.

Judgement (:func:`judge`, 2026-10-19). Fixed tolerances on the jumps would be arbitrary:
in finite volumes the discrete interface flux is single-valued by construction, and the
one-sided reconstruction measures how well the field is RESOLVED there. So the gate
compares two meshes (the deployed one and one refined by ``factor``) in three tiers:

1. exact laws -- limits from the solver tolerance, no tuning: charge balance
   <= 10 rtol; lithium drift <= 1 % of the lithium consumed by side reactions (the
   stage-B signal it would corrupt), or <= 10 rtol of the total when there is none.
   Above the limit a residual still passes if it is a tolerance error: it shrinks
   >= 10x when the solver tolerances are tightened 100x (an algebraic surface form
   solves the current split only to the potential tolerance);
2. continuity -- converged if each jump shrinks at least first-order (by ``factor``)
   under refinement, or is already at round-off (<= 10 rtol); a jump that does not
   shrink is a real inconsistency and fails the fit;
3. relevance -- the deployed mesh must resolve what the oracle reports: refining it
   changes the voltage by less than the instrument noise (1 mV, p99 over the records),
   the impedance by less than the EIS noise (1 %), the number of cycler stops by 0.
   A failure here means the oracle must run the refined mesh, not that the fit is
   unphysical (reported as such).

Mesh oscillation (odd-even modes) has no tolerance-free test here: it is reported as
a warning when a sign-alternating run of >= 4 does not shrink under refinement.
:func:`check_solution` (one mesh, the earlier fixed limits) is kept for diagnostics.
"""
from __future__ import annotations

import numpy as np

F = 96485.33212
#: seconds after a step in the applied current excluded from the continuity checks
STEP_WINDOW_S = 2.0

#: noise levels the relevance tier is judged against (jones2022 instrument / EIS)
VOLTAGE_NOISE_V = 1e-3
EIS_NOISE_REL = 0.01

#: legacy single-mesh limits (check_solution only; provisional, not used by judge)
LIMITS = {
    "i_e jump": (0.02, 0.05),
    "N_e jump": (0.02, 0.05),
    "c_e jump": (0.01, 0.03),
    "phi_e jump [mV]": (2.0, 5.0),
    "charge balance": (1e-3, 1e-2),
    "collector current": (0.02, 0.05),
    "oscillation run": (3, 5),
    "lithium drift": (1e-4, 1e-3),
}


def _check(name, value, key, note=""):
    lo, hi = LIMITS[key]
    if value is None or not np.isfinite(value):
        return {"name": name, "status": "skip", "value": None, "limit": [lo, hi], "note": note or "not evaluable"}
    status = "pass" if value < lo else ("warn" if value < hi else "fail")
    return {"name": name, "status": status, "value": float(value), "limit": [lo, hi], "note": note}


def _entries(sol, name):
    return np.asarray(sol[name].entries, dtype=float)


def _layout(sol, pv):
    """Node and edge positions of the electrolyte mesh [m] and the two interfaces."""
    n = [_entries(sol, f"x_{d} [m]").shape[0] for d in ("n", "s", "p")]
    L = [float(pv[f"{d} thickness [m]"]) for d in ("Negative electrode", "Separator", "Positive electrode")]
    starts = np.cumsum([0.0] + L[:-1])
    edges = np.unique(np.concatenate([s + np.linspace(0.0, l, k + 1) for s, l, k in zip(starts, L, n)]))
    nodes = 0.5 * (edges[:-1] + edges[1:])
    return {"n": n, "nodes": nodes, "edges": edges, "interfaces": (starts[1], starts[2]), "L": sum(L)}


def _positions(arr, lay):
    if arr.ndim != 2:
        return None
    if arr.shape[0] == lay["nodes"].size:
        return lay["nodes"]
    if arr.shape[0] == lay["edges"].size:
        return lay["edges"]
    return None


def _one_sided(arr, pos, xi):
    """Linear extrapolation to ``xi`` from the two nearest samples on each side (a
    sample exactly at ``xi`` -- a shared edge -- is used by neither side)."""
    tol = 1e-12 * max(abs(xi), 1e-9) + 1e-15
    left = np.flatnonzero(pos < xi - tol)[-2:]
    right = np.flatnonzero(pos > xi + tol)[:2]
    if left.size < 2 or right.size < 2:
        return None, None

    def ext(idx):
        x0, x1 = pos[idx]
        w = (xi - x0) / (x1 - x0)
        return arr[idx[0]] + w * (arr[idx[1]] - arr[idx[0]])
    return ext(left), ext(right)


def _boundary(arr, pos, xb):
    """Linear extrapolation of a field to a boundary ``xb`` from its two nearest samples."""
    idx = np.argsort(np.abs(pos - xb))[:2]
    idx = np.sort(idx)
    if np.isclose(pos[idx[0]], xb) or np.isclose(pos[idx[1]], xb):
        return arr[idx[0]] if np.isclose(pos[idx[0]], xb) else arr[idx[1]]
    x0, x1 = pos[idx]
    w = (xb - x0) / (x1 - x0)
    return arr[idx[0]] + w * (arr[idx[1]] - arr[idx[0]])


def _to_nodes(arr, pos, lay):
    if pos is lay["nodes"]:
        return arr
    return 0.5 * (arr[:-1] + arr[1:])


def _max_alternating_run(arr, tol):
    """Longest run of consecutive second differences that alternate in sign (each
    larger than ``tol``) along axis 0, maximised over time."""
    if arr.shape[0] < 4:
        return 0
    d2 = np.diff(arr, 2, axis=0)
    s = np.sign(d2) * (np.abs(d2) > tol)
    alt = (s[1:] * s[:-1]) < 0                          # sign flip between neighbours
    best = 0
    run = np.zeros(arr.shape[1], dtype=int)
    for k in range(alt.shape[0]):
        run = np.where(alt[k], run + 1, 0)
        best = max(best, int(run.max()))
    return best + 1 if best else 0


def check_solution(sol, pv, *, cell: str = "") -> list[dict]:
    """Validity checks of one solved history; ``pv`` the parameter values it was
    solved with."""
    tag = f"validity[{cell}] " if cell else "validity "
    out: list[dict] = []
    try:
        lay = _layout(sol, pv)
    except Exception as exc:                            # noqa: BLE001
        return [{"name": tag + "electrolyte mesh layout", "status": "fail", "value": None,
                 "limit": None, "note": f"{type(exc).__name__}: {str(exc)[:160]}"}]
    area = (float(pv["Electrode height [m]"]) * float(pv["Electrode width [m]"])
            * float(pv["Number of electrodes connected in parallel to make a cell"]))
    i_app = _entries(sol, "Current [A]").ravel() / area
    i_1c = float(pv["Nominal cell capacity [A.h]"]) / area
    scale_i = np.maximum(np.abs(i_app), i_1c)
    c0 = float(pv["Initial concentration in electrolyte [mol.m-3]"])
    t = np.asarray(sol.t, float)
    steps = t[1:][np.abs(np.diff(i_app)) > 0.01 * i_1c]
    judged = np.ones(t.size, bool)
    for ts_ in steps:
        judged &= ~((t >= ts_ - 1e-9) & (t <= ts_ + STEP_WINDOW_S))

    fields = {}
    for key, name in (("i_e", "Electrolyte current density [A.m-2]"),
                      ("N_e", "Electrolyte flux [mol.m-2.s-1]"),
                      ("c_e", "Electrolyte concentration [mol.m-3]"),
                      ("phi_e", "Electrolyte potential [V]")):
        try:
            arr = _entries(sol, name)
            pos = _positions(arr, lay)
            fields[key] = (arr, pos) if pos is not None else None
        except Exception:                               # noqa: BLE001
            fields[key] = None

    # 1. continuity across both interfaces
    for key, label, scale, unit in (("i_e", "i_e jump", scale_i, ""),
                                    ("N_e", "N_e jump", scale_i / F, ""),
                                    ("c_e", "c_e jump", np.full_like(scale_i, c0), ""),
                                    ("phi_e", "phi_e jump [mV]", np.full_like(scale_i, 1e-3), " mV")):
        f = fields.get(key)
        worst, where = None, ""
        if f is not None:
            arr, pos = f
            for xi, side in zip(lay["interfaces"], ("negative|separator", "separator|positive")):
                a, b = _one_sided(arr, pos, xi)
                if a is None:
                    continue
                rel = np.where(judged, np.abs(a - b) / scale, 0.0)
                k = int(np.argmax(rel))
                if worst is None or rel[k] > worst:
                    worst, where = float(rel[k]), f"{side} at t = {float(sol.t[k]):.0f} s"
        where += f"; {int((~judged).sum())} samples within {STEP_WINDOW_S:g} s after {steps.size} current steps not judged"
        out.append(_check(tag + f"{key} continuous across interfaces", worst, label,
                          f"largest jump{unit} ({'/ max(|i_app|, i_1C)' if key in ('i_e', 'N_e') else ''}"
                          f"{'/ c0' if key == 'c_e' else ''}) {where}"))

    # 2. charge balance and collector boundary conditions
    f = fields.get("i_e")
    if f is not None:
        arr, pos = f
        ie_nodes = _to_nodes(arr, pos, lay)
        nn, ns, _ = lay["n"]
        worst = 0.0
        try:
            for dom, sl in (("Negative", slice(0, nn)), ("Positive", slice(nn + ns, None))):
                i_s = _entries(sol, f"{dom} electrode current density [A.m-2]")
                if i_s.shape[0] == ie_nodes[sl].shape[0] + 1:     # stored on the domain's edges
                    i_s = 0.5 * (i_s[:-1] + i_s[1:])
                res = np.abs(ie_nodes[sl] + i_s - i_app[None, :]) / scale_i[None, :]
                worst = max(worst, float(res.max()))
            res = np.abs(ie_nodes[nn:nn + ns] - i_app[None, :]) / scale_i[None, :]   # separator: i_e = i_app
            worst = max(worst, float(res.max()))
            out.append(_check(tag + "charge balance i_e + i_s = i_app", worst, "charge balance"))
        except Exception as exc:                        # noqa: BLE001
            out.append(_check(tag + "charge balance i_e + i_s = i_app", None, "charge balance",
                              f"{type(exc).__name__}: {str(exc)[:120]}"))
        # integral charge balance: all reactions (main + SEI + plating) in each electrode
        # carry the applied current (global charge conservation)
        try:
            edges = lay["edges"]
            worst_int = 0.0
            for dom, sl, sign in (("negative", slice(0, nn), 1.0), ("positive", slice(nn + ns, None), -1.0)):
                aj = _entries(sol, f"Sum of {dom} electrode volumetric interfacial current densities [A.m-3]")
                total = (aj * np.diff(edges)[sl][:, None]).sum(axis=0)
                # judged outside the post-step window: with the differential surface form the
                # double layer carries part of i_app transiently (not in the summed reactions)
                worst_int = max(worst_int, float(np.max(np.where(judged, np.abs(total - sign * i_app) / scale_i, 0.0))))
            out.append(_check(tag + "integral charge balance (all reactions carry i_app)", worst_int, "charge balance",
                              "max |sum over the electrode of a j (main + side reactions) -/+ i_app| / max(|i_app|, i_1C)"))
        except Exception as exc:                        # noqa: BLE001
            out.append(_check(tag + "integral charge balance (all reactions carry i_app)", None, "charge balance",
                              f"{type(exc).__name__}: {str(exc)[:120]}"))
        # (judged outside the post-step window, like the interface continuity)
        coll = max(float(np.max(np.where(judged, np.abs(_boundary(arr, pos, 0.0)) / scale_i, 0.0))),
                   float(np.max(np.where(judged, np.abs(_boundary(arr, pos, lay["L"])) / scale_i, 0.0))))
        out.append(_check(tag + "no electrolyte current at the collectors", coll, "collector current"))
    else:
        out.append(_check(tag + "charge balance i_e + i_s = i_app", None, "charge balance"))

    # 3. mesh oscillation inside each domain
    run = 0
    for key in ("c_e", "phi_e"):
        f = fields.get(key)
        if f is None:
            continue
        arr, pos = f
        if pos is not lay["nodes"]:
            continue
        nn, ns, _ = lay["n"]
        span = float(np.ptp(arr)) or 1.0
        for sl in (slice(0, nn), slice(nn, nn + ns), slice(nn + ns, None)):
            run = max(run, _max_alternating_run(arr[sl], 1e-6 * span))
    out.append(_check(tag + "no mesh oscillation (alternating 2nd differences)", run, "oscillation run",
                      "longest run of sign-alternating second differences of c_e, phi_e"))

    # 4. lithium conservation
    try:
        names = list(sol.all_models[0].variables.keys())
        held = _entries(sol, "Total lithium in particles [mol]") + _entries(sol, "Total lithium in electrolyte [mol]")
        lost = _entries(sol, "Total lithium lost to side reactions [mol]")
        for n in names:
            if n.startswith("Loss of lithium due to loss of active material") and n.endswith("[mol]"):
                lost = lost + _entries(sol, n)
        total = held + lost
        drift = float(np.max(np.abs(total - total[0])) / abs(total[0]))
        out.append(_check(tag + "lithium conserved", drift, "lithium drift",
                          "particles + electrolyte + side reactions + lost active material"))
    except Exception as exc:                            # noqa: BLE001
        out.append(_check(tag + "lithium conserved", None, "lithium drift", str(exc)[:120]))
    return out


# ---------------------------------------------------------------------------
# Two-mesh judgement
# ---------------------------------------------------------------------------

JUMPS = ("i_e", "N_e", "c_e", "phi_e", "collector")


def measure(sol, pv) -> dict:
    """Raw validity measures of one solution (no pass/fail): the jumps, charge balance,
    collector current, oscillation run and lithium balance (see :func:`check_solution`)."""
    m = {}
    for c in check_solution(sol, pv):
        n = c["name"].split("] ", 1)[-1] if c["name"].startswith("validity[") else c["name"]
        n = n.replace("validity ", "", 1) if n.startswith("validity ") else n
        field = n.split(" ", 1)[0]               # exact first word: i_e / N_e / c_e / phi_e ...
        key = (field if (field in ("i_e", "N_e", "c_e", "phi_e") and "continuous across interfaces" in n) else
               "charge_integral" if "integral charge balance" in n else
               "charge" if "charge balance" in n else
               "collector" if "collectors" in n else
               "oscillation" if "oscillation" in n else
               "lithium" if "lithium conserved" in n else None)
        if key:
            m[key] = c["value"]
            m[key + "_note"] = c["note"]
    try:
        tot = (_entries(sol, "Total lithium in particles [mol]") + _entries(sol, "Total lithium in electrolyte [mol]"))
        side = _entries(sol, "Total lithium lost to side reactions [mol]")
        m["side_loss_rel"] = float((side[-1] - side[0]) / (tot[0] + side[0]))
    except Exception:                                   # noqa: BLE001
        m["side_loss_rel"] = None
    return m


def _rec(name, status, value, limit, note):
    return {"name": name, "status": status, "value": value, "limit": limit, "note": note}


def _exact_law(tag, name, value, limit, tight, what):
    """An exact law: pass at or below ``limit``; above it, pass only if the residual is
    a solver-tolerance error, i.e. it shrinks >= 10x when the tolerances are tightened
    100x (``tight``: the residual of that solve); fail if it does not (a real violation)."""
    if value is None:
        return _rec(tag + name, "skip", None, limit, "not evaluable")
    if value <= limit:
        return _rec(tag + name, "pass", value, limit, what)
    if tight is None:
        return _rec(tag + name, "fail", value, limit, what + "; above the limit and no tighter-tolerance solve to "
                    "show it is a tolerance error")
    ok = tight <= value / 10.0
    return _rec(tag + name, "pass" if ok else "fail", value, limit,
                what + f"; tolerances x100 tighter: {value:.2e} -> {tight:.2e} "
                + ("(a solver-tolerance error)" if ok else "(does not shrink: a real violation)"))


def judge(coarse: dict, fine: dict, *, factor: int = 2, rtol: float = 1e-6, cell: str = "",
          mesh: tuple[str, str] = ("deployed", "refined"), tight: dict | None = None) -> list[dict]:
    """Validity records from the measures of the deployed (``coarse``) and refined
    (``fine``) solutions (see the module doc for the three tiers); ``tight``: the
    measures of the deployed mesh solved with 100x tighter tolerances, which decides
    whether an exact-law residual above its limit is a tolerance error."""
    tag = f"validity[{cell}] " if cell else "validity "
    floor = 10.0 * rtol
    tight = tight or {}
    out = []
    # 1. exact laws (both meshes)
    vals = [v for v in (coarse.get("charge"), fine.get("charge")) if v is not None]
    out.append(_exact_law(tag, "charge balance (exact law)", max(vals) if vals else None, floor, tight.get("charge"),
                          "max |i_e + i_s - i_app| / max(|i_app|, i_1C), both meshes; limit 10 rtol"))
    vals = [v for v in (coarse.get("charge_integral"), fine.get("charge_integral")) if v is not None]
    out.append(_exact_law(tag, "integral charge balance (exact law)", max(vals) if vals else None, floor,
                          tight.get("charge_integral"),
                          "max |sum of a j over each electrode (all reactions) -/+ i_app| / max(|i_app|, i_1C), "
                          "both meshes; limit 10 rtol"))
    vals = [v for v in (coarse.get("lithium"), fine.get("lithium")) if v is not None]
    side = coarse.get("side_loss_rel") or 0.0
    li_lim = max(0.01 * abs(side), floor)
    out.append(_exact_law(tag, "lithium conserved (exact law)", max(vals) if vals else None, li_lim, tight.get("lithium"),
                          f"max drift / total, both meshes; limit 1 % of the lithium consumed by side reactions "
                          f"({abs(side):.2e} of the total) or 10 rtol"))
    # 2. continuity: converges under refinement
    for key in JUMPS:
        a, b = coarse.get(key), fine.get(key)
        if a is None or b is None:
            out.append(_rec(tag + f"{key} continuity converges", "skip", None, None, "not evaluable"))
            continue
        if b <= floor:                                  # relative (phi_e: mV), round-off level
            st, note = "pass", "at round-off on the refined mesh"
        else:
            ratio = a / b if b > 0 else np.inf
            st = "pass" if ratio >= factor else "fail"
            note = (f"{mesh[0]} {a:.3g} -> {mesh[1]} {b:.3g}: shrinks x{ratio:.2f} for a x{factor} refinement"
                    + ("" if st == "pass" else " -- does not converge: a real inconsistency, not resolution"))
        out.append(_rec(tag + f"{key} continuity converges", st, b, f"ratio >= {factor} or <= {floor:g}", note))
    a, b = coarse.get("oscillation"), fine.get("oscillation")
    if a is not None and b is not None:
        st = "warn" if (b >= 4 and b >= a) else "pass"
        out.append(_rec(tag + "no persistent mesh oscillation", st, b, "run < 4 or shrinking",
                        f"sign-alternating run {mesh[0]} {a:g} -> {mesh[1]} {b:g} (heuristic, warning only)"))
    return out


def relevance(coarse_res: dict, fine_res: dict, *, cell: str = "", mesh: tuple[str, str] = ("deployed", "refined")) -> list[dict]:
    """Tier 3: does refining the deployed mesh change the reported outputs by less than
    the measurement noise? ``*_res`` are ChainModel.solve results."""
    tag = f"validity[{cell}] " if cell else "validity "
    out = []
    dv = [np.abs(np.asarray(a) - np.asarray(b)) for a, b in zip(coarse_res["records"], fine_res["records"])
          if len(a) and len(a) == len(b)]
    if dv:
        p99 = float(np.percentile(np.concatenate(dv), 99))
        out.append(_rec(tag + f"{mesh[0]} mesh resolves the voltage", "pass" if p99 < VOLTAGE_NOISE_V else "fail",
                        1e3 * p99, 1e3 * VOLTAGE_NOISE_V, f"p99 |dV| mV {mesh[0]} vs {mesh[1]} (max "
                        f"{1e3 * float(max(x.max() for x in dv)):.1f} mV, at cycler stops); limit the instrument noise"))
    dz = [float(np.max(np.abs(a - b) / np.abs(b))) for a, b in zip(coarse_res["spectra"], fine_res["spectra"])
          if a is not None and b is not None]
    if dz:
        out.append(_rec(tag + f"{mesh[0]} mesh resolves the impedance", "pass" if max(dz) < EIS_NOISE_REL else "fail",
                        max(dz), EIS_NOISE_REL, "max |dZ| / |Z|; limit the EIS noise"))
    sa, sb = coarse_res["diag"].get("cycler_stops"), fine_res["diag"].get("cycler_stops")
    out.append(_rec(tag + f"{mesh[0]} mesh gives the same cycler stops", "pass" if sa == sb else "fail",
                    [sa, sb], "equal", ""))
    for r in out:
        if r["status"] == "fail":
            r["note"] += " -- the oracle must run the refined mesh (resolution, not physics)"
    return out
