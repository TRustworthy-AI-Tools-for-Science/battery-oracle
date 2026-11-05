"""PyBaMM degradation options and parameter edits of each degradation preset.

The values and their sources come from the battery-forecast Stage B review (B5, 2026-11-05):

* **SEI**, every preset: ``"ec reaction limited"``, the mixed reaction/EC-diffusion law of
  Yang et al. 2017 (J. Power Sources 360:28, eqs 10-11), on the parameter set's own SEI
  values (for LiR2032 see :func:`battery_oracle.parameter_sets.lir2032`).
  ``sei_rate_scale`` multiplies the EC diffusivity, the law's transport constant.
* **Lithium plating**, ``"accelerated"``: ``"partially reversible"`` (O'Kane et al. 2022,
  Phys. Chem. Chem. Phys. 24:7909). The defaults are O'Kane 2022's (SI Table S5):
  plating kinetic rate constant 1e-9 m/s and dead-lithium decay constant 1e-6 s-1.
  ``plating_rate_scale`` and ``dead_li_decay_scale`` multiply them.
* **Particle mechanics**: swelling, cracking, stress-driven loss of active material and
  SEI on cracks are off in every preset. On PyBaMM 26.10 with the LiR2032 set they all
  solve (B5d; the Chen2020 start-up failure recorded on PyBaMM 26.6.2 was not re-run), but:
  - stress-driven LAM at the template's values is negligible (positive LAM 2e-7 % in
    three cycles; the negative proportional term is 0) and makes each solve about 5x
    slower;
  - SEI on cracks starts from bare crack surfaces (5e-13 m) and grows 1.25 mAh in three
    cycles, about 11x the measured total loss of a jones2022 cell; no source gives the
    SEI thickness on the cracks of a formed cell;
  - the initial crack length differs 1000x between the O'Kane 2022 SI (2e-5 m as
    printed) and PyBaMM (2e-8 m).
"""
from __future__ import annotations

import copy

import pybamm as _pb

from battery_oracle.params import _scaled

#: Degradation presets: ``"nominal"`` is SEI only, ``"accelerated"`` adds lithium plating.
PRESETS = ("nominal", "accelerated")

# Numeric per-preset degradation constants, mirroring config_oracle_defaults.yml's
# degradation.preset_constants section (the YAML-overridable source -- see
# experiment.oracle_kwargs_from_config). This dict is the fallback when
# _build_degradation_config is called with preset_constants=None (a bare PyBaMMOracle()).
_DEFAULT_PRESET_CONSTANTS: dict[str, dict] = {
    "nominal": {},
    "accelerated": {
        "plating_kinetic_rate_constant_m_s": 1e-9,       # O'Kane 2022 SI Table S5
        "dead_lithium_decay_constant_s": 1e-6,           # O'Kane 2022 SI Table S5
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
    preset_constants: dict | None = None,
) -> tuple[dict, object]:
    """Return (model_options, pv_modified) for the requested degradation preset.

    ``"nominal"``
        SEI (``"ec reaction limited"``) only.

    ``"accelerated"``  *(default)*
        SEI + partially-reversible lithium plating. Plating is exponential in the local
        overpotential, so it is the mechanism that makes the oracle C-rate sensitive.

    ``sei_rate_scale`` multiplies ``EC diffusivity [m2.s-1]``; ``plating_rate_scale`` and
    ``dead_li_decay_scale`` multiply the plating rate constant and the dead-lithium decay
    constant of the preset. ``kinetics_scale`` multiplies both electrodes'
    exchange-current density; it also changes SEI growth, which is exponential in the
    electrode overpotential. ``preset_constants`` (default ``None`` ->
    :data:`_DEFAULT_PRESET_CONSTANTS`) supplies the numeric plating constants; it mirrors
    config_oracle_defaults.yml's ``degradation.preset_constants``, and ``PyBaMMOracle``
    passes its own (YAML-overridable) values here. Sources: module docstring.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown degradation_preset {preset!r}; choose one of {PRESETS}.")

    pv = copy.deepcopy(pv)
    pc = preset_constants if preset_constants is not None else _DEFAULT_PRESET_CONSTANTS[preset]

    pv["EC diffusivity [m2.s-1]"] = float(pv["EC diffusivity [m2.s-1]"]) * sei_rate_scale
    # sei_rate_scale scales EC diffusivity only. Scaling "SEI kinetic rate constant
    # [m.s-1]" too (the parameter "ec reaction limited" reads for its kinetics) is not
    # monotonic in growth rate: 0.1x made SEI growth faster.

    # Wraps the concentration-dependent exchange-current-density functions rather than
    # overwriting them with a scalar: they are functions of (c_e, c_s_surf, c_s_max, T).
    if kinetics_scale != 1.0:
        for key in (
            "Negative electrode exchange-current density [A.m-2]",
            "Positive electrode exchange-current density [A.m-2]",
        ):
            pv[key] = _scaled(pv[key], kinetics_scale)

    if preset == "nominal":
        return {"SEI": "ec reaction limited"}, pv

    opts = {
        "SEI": "ec reaction limited",
        "lithium plating": "partially reversible",
    }
    # Plating parameters the set lacks come from OKane2022; keys already present win.
    for k in (
        "Exchange-current density for plating [A.m-2]",
        "Exchange-current density for stripping [A.m-2]",
        "Initial plated lithium concentration [mol.m-3]",
        "Typical plated lithium concentration [mol.m-3]",
        "Lithium plating transfer coefficient",
        "Lithium plating kinetic rate constant [m.s-1]",
        "Dead lithium decay constant [s-1]",
        "Dead lithium decay rate [s-1]",
        "Lithium metal partial molar volume [m3.mol-1]",     # plated / dead Li thickness
    ):
        if k not in pv:
            pv[k] = _pb.ParameterValues("OKane2022")[k]
    default = _DEFAULT_PRESET_CONSTANTS["accelerated"]
    pv.update({
        "Lithium plating kinetic rate constant [m.s-1]":
            pc.get("plating_kinetic_rate_constant_m_s", default["plating_kinetic_rate_constant_m_s"])
            * plating_rate_scale,
        "Dead lithium decay constant [s-1]":
            pc.get("dead_lithium_decay_constant_s", default["dead_lithium_decay_constant_s"])
            * dead_li_decay_scale,
        # A fresh cell: OKane2022's formation-cycle plated lithium would strip during early
        # low-rate cycling and drive "Loss of capacity to ... plating" negative.
        "Initial plated lithium concentration [mol.m-3]":
            pc.get("initial_plated_lithium_concentration_mol_m3", 0.0),
    })
    return opts, pv
