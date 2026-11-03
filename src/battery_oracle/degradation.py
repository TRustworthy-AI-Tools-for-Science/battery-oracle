"""PyBaMM degradation options and parameter edits of each degradation preset."""
from __future__ import annotations

import copy

import pybamm as _pb


# ---------------------------------------------------------------------------
# Degradation configuration helper
# ---------------------------------------------------------------------------

# Numeric per-preset degradation constants, mirroring
# config_oracle_defaults.yml's degradation.preset_constants section (which is
# the actual, YAML-overridable source — see experiment.oracle_kwargs_from_config).
# This module-level dict is only the fallback used when _build_degradation_config
# is called directly with preset_constants=None (e.g. from a bare PyBaMMOracle()).
_DEFAULT_PRESET_CONSTANTS: dict[str, dict] = {
    "nominal": {},
    "accelerated": {
        "plating_kinetic_rate_constant_m_s": 1e-8,
        "dead_lithium_decay_constant_s": 4e-6,
        "initial_plated_lithium_concentration_mol_m3": 0.0,
    },
    "severe": {
        "plating_kinetic_rate_constant_m_s": 1e-7,
        "dead_lithium_decay_constant_s": 1e-4,
        "initial_plated_lithium_concentration_mol_m3": 0.0,
    },
}


def _build_degradation_config(
    preset: str,
    pv,
    kinetics_scale: float = 1.0,
    sei_rate_scale: float = 1.0,
    dead_li_decay_scale: float = 1.0,
    plating_rate_scale: float = 1.0,
    ec_diffusivity_base_factor: float = 0.25,
    preset_constants: dict | None = None,
) -> tuple[dict, object]:
    """Return (model_options, pv_modified) for the requested degradation preset.

    All presets apply a ×0.25 correction to the Chen2020 EC diffusivity, calibrated
    to match commercial CR2032 cells with FEC/VC-containing electrolytes that form a
    more passivating SEI than the LP30 baseline (Reniers et al., 2019).  Preset
    lifetimes differ only through which physical mechanisms are active.

    ``kinetics_scale`` (default 1.0, no change) multiplies both electrodes'
    exchange-current density. Chen2020 is parameterised from a high-power
    automotive NMC/graphite pouch cell; jones2022's small research coin cells
    have measurably slower charge-transfer kinetics (real R3/R1 EIS ratios are
    roughly an order of magnitude larger than the oracle's at kinetics_scale=1.0
    — see jones_oracle_study.py comparisons). Pass a value < 1 to slow the
    kinetics and grow the relative charge-transfer arc.

    ``sei_rate_scale`` (default 1.0, no change) multiplies the EC diffusivity
    correction below — additional factor on top of the existing ×0.25, so the
    final factor is ``0.25 * sei_rate_scale``. Has sharply diminishing effect on
    R1 growth once D_EC is low enough that growth becomes reaction-limited
    (confirmed empirically on PJ121: 0.5/0.25/0.1 only reduced 15-cycle R1
    growth 133%/111%/82.5%, far from linear, against a real target of ~10%).
    Scaling "SEI kinetic rate constant [m.s-1]" (the actual reaction-kinetics
    parameter for "ec reaction limited") was tried as a second lever and
    reverted — it moved R1 growth in the *wrong* direction (236% at 0.1x), so
    R1/SEI growth-rate calibration is not yet fully solved by this knob alone.
    Added because lowering
    ``kinetics_scale`` has a side effect on R1/SEI growth rate that's independent
    of its intended effect on the EIS arc shape: lower exchange-current density
    raises the electrode overpotential needed for the same applied current, and
    the SEI growth law is exponential in that overpotential (η_SEI; see
    ORACLE.md §2.1), so kinetics_scale alone *also* accelerates SEI growth as an
    unintended side effect. sei_rate_scale lets R1 growth rate be tuned back down
    independently, without re-perturbing the arc-shape calibration kinetics_scale
    was actually tuned for.

    ``"nominal"``
        SEI (``"ec reaction limited"``) + particle cracking, using Ai2020's
        unmodified critical stress (60 MPa).  PyBaMM's LAM rate is a smooth power
        law ``beta_LAM * (sigma_h/sigma_critical)^m_LAM`` with no threshold — there
        is no C-rate "onset"; sensitivity comes entirely from how sigma_hyd scales
        with C-rate, which Ai2020's stress model already provides.  No plating,
        no SEI-on-cracks.
        Target EOL: ~80–150 cycles at 1C (commercial CR2032-calibrated).

    ``"accelerated"``  *(default)*
        SEI + cracking (Ai2020 default critical stress) + partially-reversible
        lithium plating.  No "SEI on cracks": the three-way crack→SEI-on-cracks→
        plating coupling causes IDA solver stiffness (h → 0); cracking + plating
        together already provide C-rate sensitivity.
        Plating kinetic rate 1e-8 m/s, dead-Li decay 4e-6 s⁻¹.
        Target EOL: ~40–70 cycles at 1C.

    ``"severe"``
        Same model options as accelerated; faster plating (1e-7 m/s, decay 1e-4 s⁻¹).
        Target EOL: ~20–50 cycles at 1C.

    An earlier version of this function reduced critical stress to 15 MPa for all
    presets, reasoning that this would create a C-rate "activation threshold" for
    cracking.  That reasoning was wrong: the LAM rate has no threshold/switch at
    sigma_critical (it is a smooth power law), so the C-rate sensitivity RATIO
    (sigma_1C/sigma_critical)^2 / (sigma_0.3C/sigma_critical)^2 =
    (sigma_1C/sigma_0.3C)^2 is invariant to sigma_critical — it cancels out.
    Lowering sigma_critical 4x (60->15 MPa) bought zero C-rate differentiation
    while amplifying the Jacobian sensitivity d(rate)/d(sigma) ~ 1/sigma_critical^2
    by 16x, which is what was causing repeated IDA_ERR_FAIL convergence failures
    in the IDA solver.  Reverted to Ai2020 defaults for all presets.

    Physical basis: Reniers et al. (2019) for SEI; Ai et al. (2020) for crack/LAM
    parameters and the underlying sigma_hyd(C-rate) stress model; OKane et al.
    (2022) for plating kinetics.

    ``ec_diffusivity_base_factor`` (default 0.25) is the universal EC
    diffusivity correction factor described below; ``preset_constants``
    (default ``None`` -> :data:`_DEFAULT_PRESET_CONSTANTS`) supplies the
    numeric per-preset plating/dead-Li constants used in the accelerated/severe
    branches. Both mirror config_oracle_defaults.yml's ``degradation`` section
    (``ec_diffusivity_base_factor`` and ``preset_constants`` respectively) —
    ``PyBaMMOracle`` passes its own (YAML-overridable) values here.
    """

    if preset not in ("nominal", "accelerated", "severe"):
        raise ValueError(
            f"Unknown degradation_preset {preset!r}; "
            "choose 'nominal', 'accelerated', or 'severe'."
        )

    pv = copy.deepcopy(pv)
    pc = preset_constants if preset_constants is not None else _DEFAULT_PRESET_CONSTANTS[preset]

    # Universal EC diffusivity correction: commercial-grade electrolyte (FEC/VC additives)
    # forms a denser, more passivating SEI, reducing effective EC transport through the
    # film by ~4× relative to Chen2020's LP30 baseline (Reniers et al., 2019;
    # Pinson & Bazant, 2013).  Applied identically to all presets because electrolyte
    # quality is a cell property, not a degradation severity setting.
    pv["EC diffusivity [m2.s-1]"] = (
        float(pv["EC diffusivity [m2.s-1]"]) * ec_diffusivity_base_factor * sei_rate_scale
    )

    # sei_rate_scale scales EC diffusivity only. Scaling "SEI kinetic rate constant
    # [m.s-1]" too (the parameter "ec reaction limited" reads for its kinetics) is not
    # monotonic in growth rate: 0.1x made SEI growth faster.

    # Charge-transfer kinetics correction (see kinetics_scale docstring above).
    # Wraps Chen2020's concentration-dependent exchange-current-density functions
    # rather than overwriting with a scalar, since they're functions of
    # (c_e, c_s_surf, c_s_max, T), not plain floats.
    if kinetics_scale != 1.0:
        for key in (
            "Negative electrode exchange-current density [A.m-2]",
            "Positive electrode exchange-current density [A.m-2]",
        ):
            pv[key] = (lambda fn, s: lambda *a, **kw: fn(*a, **kw) * s)(pv[key], kinetics_scale)

    if preset == "nominal":
        # PyBaMM 26.6.2.0 regression: particle mechanics ('swelling and cracking')
        # fails at t=0 with IDAGetDky: IDA_BAD_K when the Chen2020 LGM50 graphite
        # OCP ('graphite_LGM50_ocp_Chen2020') is used. The cracking stress submodel
        # internally evaluates dOCP/dc at initial conditions; the LGM50 function
        # returns inconsistent derivatives at the initial stoichiometry that make
        # the IDA DAE initialization fail. Ai2020's OCP works; Chen2020's does not.
        # Replacing only 'Negative electrode OCP [V]' with Chen2020's value causes
        # the failure; all other Ai2020 params are fine.
        # Workaround: drop 'particle mechanics' from all presets; use plating
        # kinetics for C-rate sensitivity in accelerated/severe instead.
        opts = {
            "SEI": "ec reaction limited",
        }
        _override: dict = {}

    elif preset == "accelerated":
        # SEI + plating.  Particle cracking dropped (see nominal comment above for
        # the PyBaMM 26.6.2.0 regression). Partially-reversible lithium plating
        # provides strong C-rate sensitivity: plating rate scales exponentially
        # with the local overpotential, which rises steeply at high C-rates.
        opts = {
            "SEI": "ec reaction limited",
            "lithium plating": "partially reversible",
        }
        _override = {
            "Lithium plating kinetic rate constant [m.s-1]": (
                pc.get("plating_kinetic_rate_constant_m_s", 1e-8) * plating_rate_scale
            ),
            "Dead lithium decay constant [s-1]": (
                pc.get("dead_lithium_decay_constant_s", 4e-6) * dead_li_decay_scale
            ),
            # Zero out OKane2022's formation-cycle plated Li: the oracle starts with
            # a fresh cell.  Non-zero initial concentration strips during early cycling
            # at low C-rates and drives "Loss of capacity to ... plating" negative.
            "Initial plated lithium concentration [mol.m-3]": (
                pc.get("initial_plated_lithium_concentration_mol_m3", 0.0)
            ),
        }

    else:  # severe
        opts = {
            "SEI": "ec reaction limited",
            "lithium plating": "partially reversible",
            # Particle cracking also dropped for severe — same PyBaMM 26.6.2.0 OCP
            # regression applies here (see nominal preset comment above).
        }
        _override = {
            "Lithium plating kinetic rate constant [m.s-1]": (
                pc.get("plating_kinetic_rate_constant_m_s", 1e-7) * plating_rate_scale
            ),
            "Dead lithium decay constant [s-1]": (
                pc.get("dead_lithium_decay_constant_s", 1e-4) * dead_li_decay_scale
            ),
            "Initial plated lithium concentration [mol.m-3]": (
                pc.get("initial_plated_lithium_concentration_mol_m3", 0.0)
            ),
        }

    # Borrow plating parameters that Chen2020 lacks from OKane2022.
    # Only copy keys that are genuinely absent so caller-supplied values win.
    _PLATING_BORROW_KEYS = [
        "Exchange-current density for plating [A.m-2]",
        "Exchange-current density for stripping [A.m-2]",
        "Initial plated lithium concentration [mol.m-3]",
        "Typical plated lithium concentration [mol.m-3]",
        "Lithium plating transfer coefficient",
        "Lithium plating kinetic rate constant [m.s-1]",
        "Dead lithium decay constant [s-1]",
        "Dead lithium decay rate [s-1]",
        "Lithium metal partial molar volume [m3.mol-1]",     # plated / dead Li thickness
    ]
    pv_ref = _pb.ParameterValues("OKane2022")
    for k in _PLATING_BORROW_KEYS:
        if k not in pv:
            pv[k] = pv_ref[k]

    for key, val in _override.items():
        pv[key] = val

    return opts, pv
