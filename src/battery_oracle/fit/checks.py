"""Stage-A physicality of a chain fit (:mod:`battery_oracle.fit.chain`): is the fitted
cell physically valid?

Validity, not accuracy: these checks judge the internal solution and the fitted cell,
never how closely it reproduces the measurement (the caller reports accuracy on its
own; a fit need not be exact, but it must be physical). Every check is a record of
:mod:`battery_oracle.physicality`.

Per cell (from its report's ``diagnostics``): the internal-solution checks of
:mod:`battery_oracle.validity`, whole history simulated, OCP data-range excursions,
temperature rise, electrolyte depletion. Population (the aggregate report): electrode
area and active fraction, volume fractions, thickness, lithium inventory, heat
transfer, tortuosities, utilisation and N:P, kinetics/transport windows, the
electrode-SOH solve and one demanding DFN cycle on the deployed oracle
(:func:`battery_oracle.physicality.check_dfn_cycle`).
"""
from __future__ import annotations

from battery_oracle.fit.chain import (EPS_N, GRAPHITE_USABLE, H_TOTAL, LI_FRACTION, NP_PRACTICAL_BOUNDS,
                                      THICKNESS_SCALE, electrode_capacities_ah)
from battery_oracle.params import apply_parameter_overrides, apply_parameter_scales
from battery_oracle.physicality import (CE_FAIL_REL, CE_WARN_REL, _check, _range_check, check_dfn_cycle, check_esoh, check_fitted_values,
                                        electrolyte_status, fitted_oracle_kwargs)


def cell_checks(cell_report: dict, *, ambient_k: float | None = None) -> list[dict]:
    """Physical validity of one cell's fitted history."""
    cid = cell_report["cell_id"]
    d = cell_report.get("diagnostics") or {}
    if "error" in d:
        return [_check(f"fit[{cid}] history simulated", "fail", None, None, d["error"])]
    checks = list(d.get("validity") or [])
    if not checks:
        checks.append(_check(f"validity[{cid}] internal-solution checks evaluated", "fail", None, None,
                             "no validity checks in the cell report (re-evaluate with chain.evaluate_chain)"))
    done = d.get("t_simulated_s", 0.0) / max(d.get("t_total_s", 1.0), 1.0)
    checks.append(_check(f"fit[{cid}] whole history simulated", "pass" if done > 0.999 else "fail",
                         round(done, 4), 1.0, d.get("termination", "")))
    for dom, o in (d.get("ocp") or {}).items():
        st = "pass" if o["fraction_out"] == 0 else ("warn" if o["excursion"] <= 0.02 else "fail")
        checks.append(_check(f"fit[{cid}] {dom} surface stoichiometry inside OCP data", st,
                             round(o["fraction_out"], 4), 0.0,
                             f"simulated {o['min']:.3f}-{o['max']:.3f}, data {o['data_range'][0]:.3f}-"
                             f"{o['data_range'][1]:.3f} (fraction of time outside)"))
    if "T_max_K" in d and ambient_k is not None:
        rise = d["T_max_K"] - ambient_k
        checks.append(_check(f"fit[{cid}] cell temperature rise", "pass" if rise < 10 else
                             ("warn" if rise < 20 else "fail"), round(rise, 2), [0, 10],
                             "lumped thermal; coin cell at <=4C should stay within ~10 K"))
    if "c_e_min_rel" in d:
        ce = d["c_e_min_rel"]
        checks.append(_check(f"fit[{cid}] electrolyte not depleted", electrolyte_status(ce), round(ce, 4),
                             [CE_FAIL_REL, CE_WARN_REL], "minimum electrolyte concentration / initial over the history"))
    return checks


def population_checks(fit_report: dict, pv_template, oracle_kwargs: dict, *,
                      ambient_k: float | None = None) -> list[dict]:
    """Physical validity of the aggregate fitted cell, including one demanding DFN cycle on
    the oracle built from it (``oracle_kwargs`` plus the fitted values)."""
    checks: list[dict] = []
    ov = fit_report["parameter_overrides"]
    t_area = float(pv_template["Electrode height [m]"]) * float(pv_template["Electrode width [m]"])
    checks.append(_range_check("geometry: electrode area / template",
                               ov["Electrode height [m]"] * ov["Electrode width [m]"] / t_area,
                               0.5, 2.0, "template area scaled from Ai2020 area-per-capacity", fail_factor=2.0))
    checks.append(_range_check("geometry: negative active fraction / template", ov[EPS_N] / float(pv_template[EPS_N]),
                               0.7, 1.3, "N:P knob; commercial anodes 0.5-0.7", fail_factor=1.5))
    for dom in ("Negative", "Positive"):
        act = float(ov.get(f"{dom} electrode active material volume fraction",
                           pv_template[f"{dom} electrode active material volume fraction"]))
        por = float(ov.get(f"{dom} electrode porosity", pv_template[f"{dom} electrode porosity"]))
        checks.append(_check(f"geometry: {dom.lower()} volume fractions (active + pores) <= 1",
                             "pass" if act + por <= 1.0 + 1e-9 and por >= 0.15 else "fail",
                             round(act + por, 4), 1.0, f"active {act:.3f}, porosity {por:.3f}"))
    med = {k: v["median"] for k, v in fit_report["fit"]["distributions"].items()}
    if THICKNESS_SCALE in med:
        checks.append(_range_check("geometry: electrode thickness / template", med[THICKNESS_SCALE], 0.52, 1.31,
                                   "40-100 um per electrode (Chen 2020; Ecker 2015); Ai2020 68-77 um",
                                   fail_factor=1.2))
    if LI_FRACTION in med:
        checks.append(_range_check("cyclable lithium / positive capacity", med[LI_FRACTION], 0.85, 1.0,
                                   "LiCoO2 is made fully lithiated (1.0); 5-15% lost at formation",
                                   fail_factor=1.1))
    if H_TOTAL in med:
        checks.append(_range_check("heat-transfer coefficient [W/(m2 K)]", med[H_TOTAL], 5.0, 25.0,
                                   "natural convection 5-25 W/(m2 K), plus conduction into the holder",
                                   fail_factor=2.0))
    esoh = fit_report["fit"].get("esoh") or {}
    meas = fit_report["fit"].get("measured_capacity_mah")
    if "capacity_ah" in esoh and meas:
        checks.append(_range_check("capacity between OCV limits / measured 1C capacity",
                                   1000.0 * esoh["capacity_ah"] / meas, 1.0, 1.3,
                                   "open-circuit capacity must hold the measured ~1C capacity, "
                                   "with a modest rate loss", fail_factor=1.3))
    elif "error" in esoh:
        checks.append(_check("electrode SOH on the fitted cell", "fail", None, None, esoh["error"]))
    for key, nom in (("ocv100", 4.2), ("ocv0", 3.0)):
        if key in esoh:
            clamped = esoh.get(f"{key}_clamped")
            checks.append(_check(f"SOC window {key} inside OCP data", "warn" if clamped else "pass",
                                 round(esoh[key], 4), nom,
                                 "clamped to the electrode OCP data edge" if clamped else "protocol value"))
    pv_fitted = apply_parameter_overrides(pv_template.copy(), ov)
    apply_parameter_scales(pv_fitted, fit_report["parameter_scales"])
    pv_fitted.update({"Contact resistance [Ohm]": fit_report["contact_resistance_ohm"]}, check_already_exists=False)
    # bound hits are the caller's (accuracy / identifiability), not validity
    shim = {"fit": {"eis": {}}, "parameter_scales": fit_report["parameter_scales"],
            "contact_resistance_ohm": fit_report["contact_resistance_ohm"]}
    checks += microstructure_checks(pv_fitted, meas)
    checks += check_fitted_values(pv_fitted, shim) + check_esoh(pv_fitted)
    checks += check_dfn_cycle(fitted_oracle_kwargs(fit_report, oracle_kwargs, ambient_k))
    return checks


def microstructure_checks(pv, measured_capacity_mah: float | None) -> list[dict]:
    """Tortuosities, positive utilisation and N:P of the fitted cell vs measured
    ranges (graphite tau 3-10: Landesfeind et al. 2016, Ebner et al. 2014; separator
    MacMullin 2.5-5: Landesfeind 2016, Thorat et al. 2009; LiCoO2 cycles ~40-65% of
    its theoretical capacity; practical N:P, reversible negative / measured)."""
    out = []
    for dom, lo, hi, ref in (("Negative electrode", 3.0, 10.0, "graphite tau 3-10"),
                             ("Positive electrode", 1.5, 6.0, "LiCoO2 electrodes tau ~2-5"),
                             ("Separator", None, None, "")):
        eps = float(pv[f"{dom} porosity"])
        b = float(pv[f"{dom} Bruggeman coefficient (electrolyte)"])
        if dom == "Separator":
            out.append(_range_check("separator MacMullin number", eps ** (-b), 2.5, 5.0,
                                    "Celgard-type separators ~2.8-4.5", fail_factor=1.5))
        else:
            out.append(_range_check(f"{dom.lower()} tortuosity", eps ** (1.0 - b), lo, hi, ref, fail_factor=1.5))
    try:
        q_n, q_p = electrode_capacities_ah(pv)
        if measured_capacity_mah:
            q = measured_capacity_mah / 1000.0
            out.append(_range_check("positive utilisation (measured / theoretical)", q / q_p, 0.40, 0.65,
                                    "LiCoO2 cycles ~0.45-1.0 of its lithium", fail_factor=1.3))
            out.append(_range_check(f"practical N:P ({GRAPHITE_USABLE} Q_n / measured)", GRAPHITE_USABLE * q_n / q,
                                    *NP_PRACTICAL_BOUNDS, "commercial 1.05-1.2", fail_factor=1.3))
    except Exception as exc:                    # noqa: BLE001
        out.append(_check("electrode capacities", "skip", None, None, str(exc)[:120]))
    return out
