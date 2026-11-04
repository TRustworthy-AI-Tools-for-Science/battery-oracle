"""Battery oracle: PyBaMM DFN simulations + EIS + ECM fitting.

PyBaMMOracle            — stateful battery oracle (DFN → EIS → ECM); see ``run_cycle``
oracle_parameter_values — the parameter values an oracle simulates

Its parts: settings ``oracle_config``, failures ``failures``, output layout and
``CycleResult`` ``state``, degradation presets ``degradation``, ECM fits ``ecm``,
protocol vectors ``protocol``.
"""
from __future__ import annotations

import logging
import threading
import types
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pybamm as _pb
from scipy.signal import find_peaks as _find_peaks

from battery_oracle.circuit import (
    ACTION_FEATURE_NAMES as _ACTION_NAMES,
)
from battery_oracle.circuit import (
    ACTION_FEATURE_NAMES_TEMPERATURE as _ACTION_NAMES_TEMPERATURE,
)
from battery_oracle.circuit import (
    DEFAULT_CIRCUIT as _PROJECT_CIRCUIT,
)
from battery_oracle.circuit import (
    param_labels_from_circuit as param_labels_from_circuit,
)
from battery_oracle._eis.kk import linkk_rmse as _linkk_rmse
from battery_oracle._eis.noise import (
    add_flicker_noise as _add_flicker_noise,
)

# DRT (get_drt_impedance) needs hybrid-drt (the optional [drt] extra); it is
# imported lazily inside build_record's try/except so DRT peaks are simply
# omitted when the extra is absent.
from battery_oracle._eis.noise import (
    add_white_noise as _add_white_noise,
)
from battery_oracle.degradation import _build_degradation_config
from battery_oracle.ecm import _AUTOEIS_AVAILABLE, autoeis_ecm, randles_stub_ecm
from battery_oracle.eis import (
    drift_leakage,
    eis_model,
    equilibrium_impedance,
    impedance_scale,
    linearised_impedance,
    sweep_windows,
)
from battery_oracle.failures import FailureKind, OracleFailure, _SilentTruncation
from battery_oracle.oracle_config import OracleConfig
from battery_oracle.parameter_sets import check_chemistry, resolve_parameter_values
from battery_oracle.params import apply_parameter_overrides, apply_parameter_scales
from battery_oracle.protocol import PROTOCOL_FIELD_NAMES, Protocol
from battery_oracle.state import CycleResult, state_vector_schema

log = logging.getLogger(__name__)

# The cycling and EIS models are both PyBaMM's DFN, built with the degradation option
# dict from ``_build_degradation_config`` (and, for EIS, ``surface form: differential``).
# A model that does not solve fails loudly (OracleFailure); it is never swapped for another.
_MODEL_CLS = _pb.lithium_ion.DFN

# Ideal gas constant (J/mol/K), for the Arrhenius EIS temperature correction (#11).
_R_GAS = 8.314462618

# Chemistry name / alias -> parameter-set name: PyBaMM's sets, the custom sets of
# battery_oracle.parameter_sets (each its own name), and friendly aliases. Degradation
# presets and protocol bounds are chemistry-specific: a non-Chen2020 chemistry needs its
# own oracle config (voltage window, capacity, preset); LFP (Prada2013) has a ~2.0-3.6 V
# window, incompatible with the default 4.3 V / 3.0 V bounds.


# ---------------------------------------------------------------------------
# PyBaMMOracle
# ---------------------------------------------------------------------------

def _final_value(sol, name: str, *, absent: float) -> float:
    """Last value of solution variable ``name``; ``absent`` when the solved model has no
    such variable (its mechanism is not modelled). Any other failure propagates."""
    if name not in sol.all_models[-1].variables:
        return absent
    return float(sol[name].entries[-1])


class PyBaMMOracle:
    """Battery oracle using PyBaMM DFN simulations + EIS + ECM fitting.

    The oracle is **stateful**: each call runs ``n_cycles`` charge/discharge
    cycles starting from the cell state left by the previous call, so
    degradation accumulates across active learning iterations. The
    constructor already builds a fresh cell, so a new ``PyBaMMOracle(...)``
    is ready to call immediately — :meth:`reset` is only needed to *reuse*
    this instance for a new run starting from a fresh cell.

    If the PyBaMM simulation fails (even after an emergency retry with the emergency
    solver on the same model), :class:`OracleFailure` is raised -- the model is never
    swapped for a reduced one.  The caller should
    treat this as battery end-of-life and end the active learning loop; the
    oracle state is **not** reset automatically.

    For each queried protocol the oracle:

    1. Builds a PyBaMM ``Experiment`` from the 6-D protocol vector
       ``[C_rate_1_mA, C_rate_2_mA, dur_1_h, dur_2_h, D_rate_mA, dur_d_h]``.
    2. Runs ``n_cycles`` DFN charge/discharge cycles, continuing from the
       cell state accumulated so far (``starting_solution``).
    3. Reads post-charge and post-discharge SOC from the cycling solution.
    4. Simulates EIS at each SOC using a separate model of the same class with
       ``surface form = differential``, then applies post-hoc ohmic corrections
       for SEI, crack-SEI, dead-Li, and LAM degradation.
    5. Calls ``ecm_model_fn(frequencies, Z_real, Z_imag)`` independently on the
       charge and discharge spectra, taking ``n_params`` parameters (one half)
       from each, then appends the per-parameter AutoEIS posterior std for each
       state. The returned state vector is therefore
       ``[means_charge | means_discharge | std_charge | std_discharge]`` with
       length ``4 * n_params`` (``n_params`` derived from the ECM circuit, e.g.
       7 for the default ``R1-[R2,P3]-[R4,P5]`` → length 28). Slice it by name
       via :attr:`state_vector_schema` rather than by index; std slots are NaN
       when AutoEIS is unavailable (Randles stub).

    Parameters
    ----------
    config : OracleConfig, optional
        Every setting below except ``ecm_model_fn`` / ``parameter_values``
        (:class:`battery_oracle.oracle_config.OracleConfig`, which holds the one
        default of each). Keyword ``**settings`` (field names) override it; an
        unknown name raises ``TypeError``. The oracle keeps it as ``self.config``.
    ecm_model_fn : callable, optional
        Maps ``(frequencies, Z_real, Z_imag)`` to a full circuit-derived vector
        (charge half ++ discharge half); the oracle takes ``n_params`` per half.
        Defaults to :func:`autoeis_ecm`; pass :func:`randles_stub_ecm` for a
        fast no-AutoEIS fallback.
    n_cycles : int
        Number of charge/discharge cycles to simulate per oracle call.
    frequencies : np.ndarray, optional
        EIS frequencies in Hz (default: 60 log-spaced from 0.01 to 10 000 Hz).
    parameter_values : pybamm.ParameterValues, optional
        PyBaMM parameter set (default: Chen2020).
    chemistry : str
        Parameter-set name from ``parameter_sets.SUPPORTED_PARAMETER_SETS``:
        ``"Chen2020"`` (LG M50, NMC811/graphite, default) or a LiR2032 coin-cell
        set. Ignored if an explicit ``parameter_values`` is passed. A non-default
        chemistry needs its own calibration (degradation preset, voltage window) —
        see :meth:`from_config` with a matching ``config_oracle_*.yml``.
    degradation_preset : str
        Degradation physics preset — ``"nominal"``, ``"accelerated"`` (default),
        or ``"severe"``.  See :func:`_build_degradation_config`. The
        particle-cracking/LAM submodels are disabled in every
        preset (see the comment in :func:`_build_degradation_config` — an
        upstream PyBaMM regression workaround, not a design choice).
    thermal : str
        Thermal submodel — ``"isothermal"`` (default, cell held at
        ``temperature_K``) or ``"lumped"`` (Timms 2021 lumped thermal ODE;
        couples cell temperature to SEI/plating/electrolyte-conductivity
        rates and surfaces ``T_cell_K`` in the state vector). See
        ``T_ambient_K``/``h_total_W_per_m2K``.
    eol_capacity_fraction : float
        SOH threshold below which :class:`OracleFailure` is raised (default 0.80).
    capacity_check : bool
        If ``True``, append a C/20 reference discharge after each cycle to
        measure usable capacity directly.  Slower but more accurate.
    temperature_K : float
        Ambient temperature in Kelvin (default 298.15 K = 25 °C) when
        ``thermal="isothermal"``.  Sets the PyBaMM ``"Ambient temperature [K]"``
        parameter.
    use_temperature_protocol : bool
        If ``True`` (requires ``thermal="lumped"``), accept a 7-D protocol
        vector whose slot 6 overrides the ambient temperature for that cycle
        (see :class:`battery_oracle.protocol.Protocol`'s ``T_ambient_K`` field).
        Default ``False`` (6-D protocols only).
    eis_noise_level : float
        Relative noise amplitude added to synthesised EIS spectra (default
        0.02). See ``eis_noise_model`` for the noise mixture.
    seed : int, optional
        Seeds the synthetic EIS noise generator (``_eis/noise.py``'s white/
        flicker terms). When the default ``ecm_model_fn``
        (:func:`autoeis_ecm`) is used, each fit also draws a fresh int seed
        from this same RNG stream and forwards it to AutoEIS's
        ``perform_bayesian_inference(..., seed=...)``, so a fixed ``seed``
        reproduces both the EIS noise *and* the AutoEIS MCMC fit. The PyBaMM
        solver itself has its own, separate (unseeded) internal stochasticity.
        Default ``None`` (unseeded, ``np.random.default_rng()``).

    Notes
    -----
    Every setting not documented above (solver tolerances, protocol
    current/voltage/duration bounds, the first-call ``initial_soc``, SOC/LAM
    clip bounds, the combined-noise split, linKK validation params, the
    degradation preset's numeric physics constants, and the ECM CPE fit
    seeds/AutoEIS sampler settings) is a lower-level constant documented in
    ``config_oracle_defaults.yml``, grouped as:

    * ``dfn_solver_*`` / ``emergency_solver_*`` — solver tolerances
      (YAML section ``solver``).
    * ``c_min_mA`` / ``c_max_mA`` / ``dur_min_s`` / ``dur_max_s`` / ``v_charge_max`` /
      ``v_discharge_min`` / ... — protocol sanitisation bounds (YAML ``protocol_bounds``).
    * ``eis_*`` / ``noise_combined_*`` / ``linkk_*`` — EIS synthesis + validation
      (YAML ``eis``).
    * ``ecm_*`` / ``cpe_*`` / ``autoeis_*`` — ECM fit seeds and AutoEIS sampler
      settings (YAML ``ecm``).
    * ``*_scale`` (``kinetics_scale``, ``sei_rate_scale``, ``dead_li_decay_scale``,
      ``plating_rate_scale``, ...) — degradation calibration scales (YAML
      ``degradation`` / ``protocol_scaling``).

    * The YAML location of each is ``oracle_config.YAML_KEYS``. Rather than
      passing these individually, prefer :meth:`from_config` (or
      ``experiment.build_oracle_from_config``) to build a fully-configured oracle
      from that YAML (packaged or a user-supplied ``config_oracle_*.yml``).
    """

    def __init__(
        self,
        config: OracleConfig | None = None,
        *,
        ecm_model_fn=None,
        parameter_values=None,
        **settings,
    ) -> None:
        # every setting and its default: battery_oracle.oracle_config.OracleConfig
        cfg = (config or OracleConfig()).replace(**settings)
        self.config = cfg
        self._pb = _pb
        self._init_interface(cfg, ecm_model_fn)
        self._init_settings(cfg)
        self._init_parameter_values(cfg, parameter_values)

        # EIS model is built fresh each __call__ so degraded ParameterValues can
        # be injected for LAM before each simulation.

        # Harmonic averaging of coefficients at the electrode|separator interfaces
        # (battery_oracle.patches). Non-strict: on an untested PyBaMM it warns and the
        # stock (arithmetic) averaging is used.
        from battery_oracle import patches as _patches
        _patches.apply(["harmonic_flux_coefficients"], strict=False)

        self._build_native_state()

        self._lock = threading.Lock()
        self._eis_lin_ok: bool | None = None       # state layout check (first spectrum)
        self._clear_cell_state()
        self._init_film_resistances()

    def _init_interface(self, cfg: OracleConfig, ecm_model_fn) -> None:
        """ECM circuit and fitter, action names and the returned-state layout."""
        # ECM circuit fitted to every spectrum + the protocol/action feature
        # names. Default to the package canonical circuit (battery_oracle.circuit
        # DEFAULT_CIRCUIT); a study pipeline passes its own so featurized records
        # line up. ecm_param_names are derived from the circuit (AutoEIS labels).
        self._circuit = cfg.circuit or _PROJECT_CIRCUIT
        self._ecm_param_names = param_labels_from_circuit(self._circuit)
        # 6-D actions by default; 7-D (adds T_ambient_K) only when the opt-in
        # temperature protocol is enabled. An explicit action_names= wins.
        self._use_temperature_protocol = bool(cfg.use_temperature_protocol)
        if cfg.action_names:
            self._action_names = list(cfg.action_names)
        elif self._use_temperature_protocol:
            self._action_names = list(_ACTION_NAMES_TEMPERATURE)
        else:
            self._action_names = list(_ACTION_NAMES)
        # Returned-state layout, derived from the circuit + active feature flags
        # (never a hardcoded length). Rebuilt via _refresh_state_schema() whenever
        # a length-affecting flag (thermal, in Phase 2) changes.
        self._refresh_state_schema()
        if ecm_model_fn is not None:
            self.ecm_model_fn = ecm_model_fn
        elif _AUTOEIS_AVAILABLE:
            self.ecm_model_fn = autoeis_ecm
        else:
            # Bare install (no [autoeis] extra): fall back to the analytic
            # Randles stub rather than raising on the first cycle. Posterior-std
            # slots in the state vector are NaN under the stub.
            log.warning(
                "autoeis is not installed; ECM fits use the analytic Randles stub "
                "(posterior-std state slots will be NaN). Install "
                "battery-oracle[autoeis] for Bayesian ECM fitting."
            )
            self.ecm_model_fn = randles_stub_ecm

    def _init_settings(self, cfg: OracleConfig) -> None:
        """Scalar settings: protocol, noise, solver, sanitisation and ECM-fit options."""
        # OCV rest duration (s) on each side of a cycle. Default 1200 s (20 min)
        # per Jones, Stimming & Lee 2022 (Nat. Commun. 13:4806): each cycle rests
        # 20 min at OCV in both the discharged and charged states before EIS. (The
        # CR2032 config's 10 s charged-state value is potentiostat switching
        # overhead, not a protocol rest.) See _protocol_to_experiment.
        self._rest_s = float(cfg.rest_s)
        self.n_cycles = cfg.n_cycles
        # Real-cell-specific protocol scaling: jones2022 cells vary in actual
        # measured capacity (e.g. PJ121 ~42 mAh, not the ~200 mAh assumed by the
        # old hardcoded _CAP_SCALE), so this is exposed per-instance rather than
        # being a fixed class constant. Same-C-rate scaling onto the simulated
        # cell's nominal capacity (5 Ah for Chen2020, ~42 mAh for LiR2032): set
        # once the parameter values are resolved below; see _protocol_to_experiment.
        self._real_cell_capacity_mah = float(cfg.real_cell_capacity_mah)

        # Depth-of-discharge / charge-induced-stress LAM term: an explicit,
        # additive capacity-loss pathway that makes degradation respond to DoD
        # and the (charge-rate x DoD) coupling, the lever the dynamic-cycling
        # literature identifies as dominant for NMC/graphite cells. DISABLED BY
        # DEFAULT (dod_lam_scale=0.0).
        #
        # Motivation. Li et al. 2024 (Cell Reports Physical Science 5:101891,
        # 225 NMC/Gr cells) and Smith et al. find that cycling capacity fade is
        # driven by DoD and a charge-induced stress Stress = (C_chg*DoD)^0.5
        # (a diffusion-induced-stress proxy), accumulating as sqrt of Ah
        # throughput -- NOT by C-rate alone, which is null on its own. A
        # cell-level regression on the real jones2022 variable-discharge cells
        # (AutoEIS R_ct = R3+R5 as the protocol-independent health signal;
        # discharge capacity itself is unusable because it equals the commanded
        # DoD) reproduces the CONTRAST -- DoD/stress predict R_ct growth where
        # C-rate is null -- but only modestly (rho~0.4 among the cells with a
        # clean degradation signal) because every jones2022 variable-discharge
        # cell is shallow (4-17% DoD): there is no deep-cycling arm to size the
        # magnitude from.
        #
        # Mechanism (Smith-style sqrt-throughput law, integrated incrementally):
        #   stress_i      = sqrt( (C_rate_1/Q) * (DoD_ah_i/Q) )   [dimensionless]
        #   T            += DoD_ah_i                              [cumulative Ah]
        #   dLoss_i       = dod_lam_scale * stress_i * DoD_ah_i / (2*sqrt(T))
        # which integrates to ~ dod_lam_scale * <stress> * sqrt(T): irreversible
        # active-material loss that grows with depth and throughput and is
        # additive to delta_lli/LAM. The cumulative loss fraction is fed into the
        # EIS (it reduces the negative-electrode active-material volume fraction,
        # growing the R3+R5 charge-transfer arc) so this degradation appears in
        # the observable impedance/state vector the surrogate sees -- the way
        # real jones2022 cells age via R_ct growth -- not only in the hidden SOH.
        # This stands in for the physical particle-cracking/LAM submodel that the
        # PyBaMM 26.6.2.0 OCP regression forces off (see _build_degradation_config).
        # dod_lam_scale must be calibrated
        # to a literature magnitude or a deep-DoD dataset BEFORE enabling --
        # jones2022's shallow range cannot fix it. 0.0 = contributes nothing.
        self._dod_lam_scale = float(cfg.dod_lam_scale)
        self.frequencies = (
            np.asarray(cfg.frequencies, float) if cfg.frequencies is not None else np.logspace(-2, 4, 60)
        )
        self.eol_capacity_fraction = cfg.eol_capacity_fraction
        self.capacity_check = cfg.capacity_check
        self._eis_noise_level = float(cfg.eis_noise_level)
        self._eis_noise_model = cfg.eis_noise_model
        if cfg.eis_mode not in ("equilibrium", "nonstationary"):
            raise ValueError(f"Unknown eis_mode {cfg.eis_mode!r}; choose 'equilibrium' or 'nonstationary'.")
        self._eis_mode = str(cfg.eis_mode)
        self._eis_amplitude_mA = float(cfg.eis_amplitude_mA)
        self._init_eis_sweep(cfg)
        # Seedable randomness (#A4): covers only the synthetic EIS noise terms
        # (_eis/noise.py) and the AutoEIS sampler seeds; the PyBaMM solver is
        # unaffected. The generator itself is (re)created from the seed in
        # _clear_cell_state, so reset()->run reproduces a fresh instance.
        self._seed = cfg.seed
        # Solver tolerances (see _build_native_state).
        self._emergency_solver_rtol = float(cfg.emergency_solver_rtol)
        self._emergency_solver_atol = float(cfg.emergency_solver_atol)
        self._dfn_solver_rtol = float(cfg.dfn_solver_rtol)
        self._dfn_solver_atol = float(cfg.dfn_solver_atol)
        self._dfn_solver_dt_max_s = float(cfg.dfn_solver_dt_max_s)
        self._dfn_max_crate = float(cfg.dfn_max_crate)

        # Protocol sanitisation bounds (see _sanitise_current/_sanitise_duration/
        # _protocol_to_experiment/_cap_check_steps). Instance attributes (not
        # class constants) so a YAML-driven build can override them per oracle.
        self._C_MIN_mA = float(cfg.c_min_mA)
        self._C_MAX_mA = float(cfg.c_max_mA)
        self._C2_MIN_mA = float(cfg.c2_min_mA)
        self._C2_MAX_mA = float(cfg.c2_max_mA)
        self._DUR_MIN_s = float(cfg.dur_min_s)
        self._DUR_MAX_s = float(cfg.dur_max_s)
        self._V_CHARGE_MAX = float(cfg.v_charge_max)
        self._V_DISCHARGE_MIN = float(cfg.v_discharge_min)
        self._CHARGE_STAGE_MAX_s = float(cfg.charge_stage_max_s)

        # First-call SOC + internal clip bounds (see __call__).
        self._initial_soc = float(cfg.initial_soc)
        self._soc_clip_min = float(cfg.soc_clip_min)
        self._soc_clip_max = float(cfg.soc_clip_max)
        self._lam_ceiling = float(cfg.lam_ceiling)

        # Combined-noise split + linKK validation params (see __call__/_eis_and_correct).
        self._noise_combined_flicker_frac = float(cfg.noise_combined_flicker_frac)
        self._noise_combined_white_frac = float(cfg.noise_combined_white_frac)
        self._linkk_c = float(cfg.linkk_c)
        self._linkk_max_M = int(cfg.linkk_max_M)

        # ECM CPE fit seeds + AutoEIS sampler settings (see _fit_half).
        self._cpe_w_seed = cfg.cpe_w_seed
        self._cpe_n_seed = cfg.cpe_n_seed
        self._cpe_w_default = float(cfg.cpe_w_default)
        self._cpe_n_default = float(cfg.cpe_n_default)
        self._ecm_rescale_target_r0 = (
            float(cfg.ecm_rescale_target_r0) if cfg.ecm_rescale_target_r0 is not None else None
        )
        self._autoeis_num_warmup = int(cfg.autoeis_num_warmup)
        self._autoeis_num_samples = int(cfg.autoeis_num_samples)

    def _init_eis_sweep(self, cfg: OracleConfig) -> None:
        """Sweep timing of the frequency grid (non-stationary EIS windows)."""
        # Offsets from the sweep start, in self.frequencies order.
        self._eis_win_lo, self._eis_win_hi = sweep_windows(
            self.frequencies, cfg.eis_sweep_periods, cfg.eis_sweep_overhead_s)
        self._eis_sweep_s = float(self._eis_win_hi.max()) + 1.0

    def _init_parameter_values(self, cfg: OracleConfig, parameter_values) -> None:
        """Parameter values and model options: chemistry, preset, BOL fit, contact
        resistance, thermal submodel and mesh."""
        # Chemistry (#14): validate early (before any heavy PyBaMM work). An
        # explicit parameter_values wins; otherwise resolve the chemistry name.
        self._chemistry = str(cfg.chemistry)
        check_chemistry(self._chemistry)
        self._pv = parameter_values or resolve_parameter_values(self._chemistry)
        self._temperature_K = float(cfg.temperature_K)
        if self._temperature_K != 298.15:
            self._pv["Ambient temperature [K]"] = self._temperature_K
        # Build degradation model options and (optionally) override parameters.
        # "ec reaction limited" is always the SEI base; accelerated/severe add
        # lithium plating and particle cracking on top of it.
        self._deg_opts, self._pv = _build_degradation_config(
            cfg.degradation_preset, self._pv,
            kinetics_scale=cfg.kinetics_scale, sei_rate_scale=cfg.sei_rate_scale,
            dead_li_decay_scale=cfg.dead_li_decay_scale,
            plating_rate_scale=cfg.plating_rate_scale,
            ec_diffusivity_base_factor=cfg.ec_diffusivity_base_factor,
            preset_constants=cfg.preset_constants,
        )

        # Beginning-of-life fit (see battery_oracle.params): applied AFTER the
        # preset so fitted multipliers compose with kinetics_scale rather than
        # being overwritten by it.
        self._parameter_scales = dict(cfg.parameter_scales or {})
        self._parameter_overrides = dict(cfg.parameter_overrides or {})
        apply_parameter_scales(self._pv, self._parameter_scales)
        apply_parameter_overrides(self._pv, self._parameter_overrides)
        # Real current [mA] x _cap_scale = simulated current [mA] at the same
        # C-rate. After the BOL overrides, which may set the nominal capacity.
        self._nominal_mAh = 1000.0 * float(self._pv["Nominal cell capacity [A.h]"])
        self._cap_scale = self._nominal_mAh / self._real_cell_capacity_mah
        self._c20_A = self._nominal_mAh / 1000.0 / 20.0
        self._contact_resistance_ohm = (
            float(cfg.contact_resistance_ohm) if cfg.contact_resistance_ohm is not None else None
        )
        if self._contact_resistance_ohm is not None:
            self._pv.update({"Contact resistance [Ohm]": self._contact_resistance_ohm},
                            check_already_exists=False)
            self._deg_opts = {**self._deg_opts, "contact resistance": "true"}
        if cfg.discharge_mode not in ("cutoff", "protocol"):
            raise ValueError(
                f"Unknown discharge_mode {cfg.discharge_mode!r}; choose 'cutoff' or 'protocol'."
            )
        self._discharge_mode = cfg.discharge_mode

        # Thermal submodel. Lumped (Timms 2021) adds one ODE coupling cell
        # temperature to the electrochemistry; merge it into the degradation option
        # dict so _build_native_state's model picks it up, and write the ambient /
        # heat-transfer parameters. Isothermal keeps the existing behaviour.
        self._thermal = str(cfg.thermal)
        if self._thermal not in ("isothermal", "lumped"):
            raise ValueError(
                f"Unknown thermal {cfg.thermal!r}; choose 'isothermal' or 'lumped'."
            )
        self._T_ambient_K = float(cfg.T_ambient_K)
        self._h_total = float(cfg.h_total_W_per_m2K)
        if self._use_temperature_protocol and self._thermal != "lumped":
            raise ValueError(
                "use_temperature_protocol=True requires thermal='lumped' "
                "(the per-cycle ambient slot only affects the lumped thermal ODE)."
            )
        if self._thermal == "lumped":
            self._pv["Ambient temperature [K]"] = self._T_ambient_K
            self._pv["Initial temperature [K]"] = self._T_ambient_K
            self._pv["Total heat transfer coefficient [W.m-2.K-1]"] = self._h_total
            self._deg_opts = {**self._deg_opts, "thermal": "lumped"}
            # A set with a whole-cell heat capacity (LiR2032: the coin can, see
            # parameter_sets) heats the real cell, not the electrode stack alone.
            if "Cell heat capacity [J.K-1.m-3]" in self._pv.keys():
                self._deg_opts["use lumped thermal capacity"] = "true"
        self._var_pts = None
        self._radial_mesh = dict(cfg.radial_mesh) if cfg.radial_mesh else None
        self._submesh = None
        if self._radial_mesh:
            from battery_oracle.model_options import radial_mesh as _radial_mesh
            r_pts, self._submesh = _radial_mesh(self._radial_mesh.get("points"),
                                                self._radial_mesh.get("kind", "uniform"),
                                                float(self._radial_mesh.get("stretch") or 1.15))
            self._var_pts = {**(self._var_pts or {}), **r_pts}
        self._x_factor = int((self._radial_mesh or {}).get("x_factor") or 1)
        # Arrhenius EIS temperature dependence (#11).
        self._E_a = float(cfg.E_a_J_per_mol)
        self._E_a_el = float(cfg.E_a_electrolyte_J_per_mol)
        self._T_ref_K = 298.15
        # Thermal changes the returned-state layout (adds the T_cell_K slot), so
        # rebuild the schema now that self._thermal is known.
        self._refresh_state_schema()

    def _clear_cell_state(self) -> None:
        """Fresh-cell state: accumulated degradation, history, cached status, and the
        EIS noise generator re-seeded from ``seed`` (shared by __init__ and reset)."""
        self._rng = np.random.default_rng(self._seed)
        self._prev_solution = None
        self._last_cycle_pv = None
        # Per-call history for diagnostic plots and EOL tracking.
        self._history: list[dict] = []
        self._last_Z: np.ndarray | None = None
        self._initial_capacity_ah: float | None = None
        self._initial_lli_ah: float | None = None  # formation-cycle LLI baseline
        self._cumulative_dod_lam_frac = 0.0   # dimensionless cumulative loss fraction
        self._cumulative_dod_lam_ah = 0.0     # = frac * nominal_cap (eol_loss budget)
        self._cumulative_dod_throughput_ah = 0.0
        # Per-call fidelity status (reset each run_cycle; downgraded by the fallback).
        self._last_failure_kind: FailureKind | None = None
        self._last_fidelity: str = "full"
        # Most recent CycleResult (None until the first successful run_cycle()).
        self._last_result: CycleResult | None = None
        # Clamp *signatures* (field + reason, not exact values) already warned
        # about at WARNING level, so a policy that repeatedly sends the same
        # out-of-range protocol doesn't spam every cycle in an AL loop — see
        # _warn_clamps / _protocol_to_experiment.
        self._warned_clamp_signatures: set[str] = set()

    def _init_film_resistances(self) -> None:
        """SEI / dead-lithium thickness -> ohmic resistance factors [Ohm/m], and the
        nominal ε_s for the LAM injection."""
        # Pre-compute SEI-thickness → ohmic resistance factor (Ohm / m).
        # Specific surface area of spherical particles: a_s = 3 * eps_s / r_p
        # [Doyle, Fuller & Newman, J. Electrochem. Soc. 140(6):1526, 1993, §2.1]
        # SEI film resistance: R_SEI = delta / (sigma_SEI * a_s * L * A_cc)
        # where r_f = delta / sigma_SEI is resistance per unit interfacial area (Ohm.m2)
        # [Christensen & Newman, J. Electrochem. Soc. 151(11):A1977, 2004, eq. (5)]
        #
        # PyBaMM's sets carry "SEI resistivity [Ohm.m]" (Chen2020: 200 kΩ·m); a set
        # without it is an error, not a disabled EIS correction.
        sigma_sei = 1.0 / float(self._pv["SEI resistivity [Ohm.m]"])
        eps_s = float(self._pv["Negative electrode active material volume fraction"])
        r_p   = float(self._pv["Negative particle radius [m]"])
        L_neg = float(self._pv["Negative electrode thickness [m]"])
        W     = float(self._pv["Electrode width [m]"])
        H     = float(self._pv["Electrode height [m]"])
        a_s   = 3.0 * eps_s / r_p          # m⁻¹  [Doyle, Fuller & Newman 1993]
        V_el  = L_neg * W * H               # m³

        self._eps_s_nominal = eps_s         # saved for LAM a-priori injection
        self._sei_to_R      = 1.0 / (sigma_sei * a_s * V_el)
        self._dead_li_to_R  = 5e-2 / (a_s * V_el)  # ρ_dead≈0.05 Ω·m [OKane2022]

    @classmethod
    def from_config(cls, source: "str | Path | dict | None" = None) -> "PyBaMMOracle":
        """Build a calibrated oracle from an oracle-config YAML (recommended entry point).

        The friendliest way to get a working, calibrated ``PyBaMMOracle``
        without hand-picking kwargs: delegates to
        :func:`battery_oracle.experiment.build_oracle_from_oracle_config`,
        which resolves ``source`` and maps every ``config_oracle_*.yml``
        field to the matching constructor kwarg (falling back to this
        class's own defaults for anything the YAML omits).

        Parameters
        ----------
        source : str, Path, dict, or None, optional
            * ``None`` (default) — the packaged ``config_oracle_defaults.yml``.
            * A path to a custom ``config_oracle_*.yml`` (e.g. a
              ``tune.write_oracle_config`` output).
            * An already-parsed config ``dict``.

        Returns
        -------
        PyBaMMOracle

        Examples
        --------
        >>> oracle = PyBaMMOracle.from_config()          # packaged defaults
        >>> oracle = PyBaMMOracle.from_config("my_cell.yml")  # custom YAML
        """
        from battery_oracle.experiment import build_oracle_from_oracle_config

        # build_oracle_from_oracle_config's `source: str | Path | dict` falls
        # through to load_oracle_config(source) for anything that isn't a dict
        # -- and load_oracle_config(None) loads the
        # packaged config_oracle_defaults.yml -- so None is passed through as-is.
        return build_oracle_from_oracle_config(source)

    def _model_var_pts(self, model):
        """Mesh for a PyBaMM model of this oracle: the defaults, with the coarse
        size-distribution mesh of the chain fit when one is set (None otherwise)."""
        from battery_oracle.model_options import x_var_pts

        x_pts = x_var_pts(model, self._x_factor)
        if not self._var_pts and not x_pts:
            return None
        return {**model.default_var_pts, **(self._var_pts or {}), **x_pts}

    def _model_submesh(self, model):
        """Submesh types for a PyBaMM model of this oracle: the chain fit's radial mesh
        when one is set (None otherwise: PyBaMM's defaults)."""
        if not self._submesh:
            return None
        return {**model.default_submesh_types, **self._submesh}

    def _refresh_state_schema(self) -> None:
        """Recompute the returned-state layout from the circuit + active flags.

        Called from ``__init__`` and whenever a length-affecting flag changes.
        ``has_t_cell`` is on only when lumped thermal is active (Phase 2); read
        defensively so this is safe to call before ``_thermal`` is assigned.
        """
        self.state_vector_schema = state_vector_schema(
            len(self._ecm_param_names),
            has_std=True,
            has_t_cell=(getattr(self, "_thermal", "isothermal") == "lumped"),
        )

    @property
    def state_vector_len(self) -> int:
        """Length of the vector returned by ``__call__`` — derived, never a literal."""
        return max(hi for _, hi in self.state_vector_schema.values())

    @property
    def history(self) -> list[dict]:
        """Append-only per-cycle diagnostic log.

        One dict per successful :meth:`run_cycle` call (in call order),
        carrying every field ``CycleResult`` surfaces plus lower-level
        diagnostics (raw EIS arrays, AutoEIS posterior samples, cumulative
        degradation bookkeeping) not promoted to named ``CycleResult``
        fields. Cleared by :meth:`reset`. Prefer :attr:`last_result` /
        :meth:`run_cycle`'s return value for the common case — this is for
        diagnostics across the whole run (battery-forecast exports and plots it).
        """
        return self._history

    @property
    def last_result(self) -> "CycleResult | None":
        """The :class:`CycleResult` from the most recent successful ``run_cycle``
        call (or ``__call__``, which delegates to it); ``None`` before the
        first call or immediately after :meth:`reset`."""
        return self._last_result

    @property
    def circuit(self) -> str:
        """ECM circuit the state vector's ECM blocks are fitted against."""
        return self._circuit

    @property
    def action_names(self) -> list[str]:
        """Names of the protocol (action) slots, in order."""
        return list(self._action_names)

    @property
    def last_solution(self):
        """PyBaMM solution of the most recent cycle (``None`` before the first)."""
        return self._prev_solution

    @property
    def last_parameter_values(self):
        """PyBaMM parameter values the most recent cycle was solved with (``None`` before
        the first): what :func:`battery_oracle.validity.check_solution` needs alongside
        :attr:`last_solution`."""
        return self._last_cycle_pv

    @property
    def degradation_options(self) -> dict:
        """PyBaMM model options of the cycling model (SEI, plating, thermal, ...)."""
        return dict(self._deg_opts)

    @property
    def dod_lam_scale(self) -> float:
        """Scale of the oracle's depth-of-discharge LAM term (0 = off)."""
        return self._dod_lam_scale

    def _build_native_state(self) -> None:
        """(Re)create the cycling model and the primary/emergency IDAKLUSolvers.

        Every ``__call__`` builds a fresh ``pb.Simulation`` and rediscretizes it
        against a new ``experiment``, but historically bound this to the *same*
        ``self._cycling_model`` and solver objects for the whole oracle lifetime
        (which spans every seed and every policy in one process — hundreds of
        solves).  PyBaMM's discretization step is known to attach processed/cached
        symbols onto the model object across repeated builds, and the IDAKLU C++
        extension's internal workspace does not appear to fully release the
        previous model's compiled CasADi function handles when rebound.  Both
        accumulate over enough solves and eventually corrupt memory, segfaulting
        inside ``idaklu_solver.py:_integrate`` with no Python-catchable exception.
        Rebuilding the model + solvers periodically (call this from ``reset()``,
        i.e. once per policy) keeps native object lifetimes bounded to one
        policy's iterations instead of the whole multi-seed experiment.
        """
        self._cycling_model = _MODEL_CLS(options=self._deg_opts)
        # DFN is a stiff coupled DAE: tight IDAKLU tolerances, and a low dt_max on the
        # Casadi emergency solver (dt_max is a no-op on IDAKLU itself).
        _rtol, _atol = self._dfn_solver_rtol, self._dfn_solver_atol
        self._solver = self._pb.IDAKLUSolver(rtol=_rtol, atol=_atol)

        # Emergency solver: deliberately a *different solver family*, not just
        # looser IDAKLU tolerances.  The CC->CV switch (charge step -> "Hold at
        # 4.1 V until C/20") reproducibly trips IDA_ERR_FAIL in IDAKLUSolver
        # regardless of rtol/atol (verified at rtol 1e-3, 1e-2, and 1e-6) — the
        # algebraic constraint at that control-mode boundary is the problem,
        # not solver precision.  CasadiSolver(mode="safe") integrates through
        # it reliably, so it is always the emergency solver here. For DFN this is
        # the #4 fallback tier-2 (loose Casadi with the low DFN dt_max).
        self._solver_emerg = self._pb.CasadiSolver(
            mode="safe",
            dt_max=self._dfn_solver_dt_max_s,
            rtol=self._emergency_solver_rtol,
            atol=self._emergency_solver_atol,
        )

    def reset(self) -> None:
        """Reset accumulated cell state so the next call starts with a fresh cell.

        The constructor already builds a fresh cell (``_build_native_state``
        runs at the end of ``__init__``), so ``reset()`` is **not** required
        after construction — a brand-new ``PyBaMMOracle(...)`` is ready to
        call immediately. It exists to *reuse* an existing instance for a new
        run (e.g. a new active-learning policy/seed) without paying the cost
        of rebuilding the PyBaMM model/solvers from scratch: it clears
        ``history``, accumulated degradation state, and cached fidelity/
        fallback status, then rebuilds the native model. It also re-seeds the
        EIS noise generator from the original ``seed`` kwarg, so a
        ``reset()``-then-run sequence reproduces the same noise realisation
        as a fresh instance.
        """
        self._clear_cell_state()
        self._build_native_state()

    # Protocol bounds -- _C_MIN_mA/_C_MAX_mA/_C2_MIN_mA/_C2_MAX_mA/_DUR_MIN_s/
    # _DUR_MAX_s/_V_CHARGE_MAX/_V_DISCHARGE_MIN/_CHARGE_STAGE_MAX_s -- are INSTANCE
    # attributes set in __init__ (from the c_min_mA/... kwargs, YAML-overridable
    # via config_oracle_defaults.yml's protocol_bounds section). Upper duration:
    # 8 h is generous for any step that also has a voltage-cutoff termination
    # condition. The charge step 2 (taper stage) shares the C1 ceiling. Voltage/duration
    # cutoffs from Jones, Stimming & Lee 2022 (Nat. Commun. 13:4806), Methods
    # "Battery cycling": two-stage CC charge (<=15 min/stage) stopping at
    # 4.3 V (no CV hold), single-stage CC discharge until 3.0 V.
    def _sanitise_current(self, val_mA: float, default_mA: float,
                          lo: float, hi: float, *,
                          label: str | None = None,
                          clamps: list | None = None) -> float:
        """Return a finite, physically bounded current in Amperes.

        When ``label``/``clamps`` are supplied, any value actually changed by
        NaN->default substitution or the ``[lo, hi]`` clip is recorded onto
        ``clamps`` as ``(signature, message)`` for :meth:`_warn_clamps`
        (loud sanitisation, see the class docstring).
        """
        used_default = not np.isfinite(val_mA)
        v = float(default_mA if used_default else val_mA)
        out = float(np.clip(v, lo, hi))
        if clamps is not None and label is not None:
            if used_default:
                clamps.append((f"{label}:nan_default",
                                f"{label} NaN→{default_mA:.0f} mA (default)"))
            elif out != v:
                clamps.append((f"{label}:bounds",
                                f"{label} {v:.0f}→{out:.0f} mA (bounds [{lo:.0f}, {hi:.0f}])"))
        return out / 1000.0

    def _current_ceiling_mA(self, base_hi: float) -> float:
        """Model-aware upper current bound (in the 5 Ah PyBaMM frame, mA).

        Tightened to ``_dfn_max_crate`` — computed from the cell's own C/20 current, so
        no mA value is hardcoded (``nominal_A = 20 * self._c20_A``).
        """
        dfn_hi_mA = self._dfn_max_crate * (20.0 * self._c20_A) * 1000.0
        return min(base_hi, dfn_hi_mA)

    def _sanitise_duration(self, val_h: float, default_h: float, *,
                           label: str | None = None,
                           clamps: list | None = None) -> float:
        """Return a finite, physically bounded duration in seconds.

        See :meth:`_sanitise_current` for the ``label``/``clamps`` contract.
        """
        used_default = not np.isfinite(val_h)
        v = float(default_h if used_default else val_h)
        v_s = v * 3600.0
        out = float(np.clip(v_s, self._DUR_MIN_s, self._DUR_MAX_s))
        if clamps is not None and label is not None:
            if used_default:
                clamps.append((f"{label}:nan_default",
                                f"{label} NaN→{default_h:.3f} h (default)"))
            elif out != v_s:
                clamps.append((f"{label}:bounds",
                                f"{label} {v_s:.0f}→{out:.0f} s (bounds "
                                f"[{self._DUR_MIN_s:.0f}, {self._DUR_MAX_s:.0f}])"))
        return out

    def _warn_clamps(self, clamps: list[tuple[str, str]]) -> None:
        """Emit ONE ``log.warning`` per oracle call summarising protocol
        sanitisation that actually changed a value (bounds clips, NaN->default
        substitution, the charge-stage caps, the discharge floor).

        Deduplicated by clamp *signature* (field + reason, not exact values)
        via ``self._warned_clamp_signatures``: the first time a given
        signature is seen it is logged at WARNING; every later call whose
        clamps are all already-seen signatures logs the same summary at DEBUG
        instead, so a policy that repeatedly sends the same out-of-range
        protocol doesn't spam every cycle of an active-learning loop.
        """
        if not clamps:
            return
        signatures = [sig for sig, _ in clamps]
        is_new = [sig not in self._warned_clamp_signatures for sig in signatures]
        self._warned_clamp_signatures.update(signatures)
        summary = "; ".join(msg for _, msg in clamps)
        if any(is_new):
            log.warning("[PyBaMMOracle] protocol sanitised: %s", summary)
        else:
            log.debug("[PyBaMMOracle] protocol sanitised (repeat): %s", summary)

    def _cap_check_steps(self) -> tuple[str, ...]:
        """Four-step C/20 capacity check appended after regular cycling.

        Fully charges the cell at C/20 (CV hold to C/20 taper), rests, then
        discharges at C/20 to the lower voltage cutoff.  The final discharge
        capacity is the usable capacity at that point in the cell's life.
        The 25-hour (90 000 s) time limit is a safe upper bound at C/20.
        """
        c = self._c20_A
        vc, vd = self._V_CHARGE_MAX, self._V_DISCHARGE_MIN
        return (
            f"Charge at {c:.4f} A for 90000 seconds or until {vc} V",
            "Hold at {} V until C/50".format(vc),
            "Rest for 300 seconds",
            f"Discharge at {c:.4f} A for 90000 seconds or until {vd} V",
        )

    def _protocol_to_experiment(self, protocol: np.ndarray):
        """Build the PyBaMM ``Experiment`` for one call, sanitising as needed.

        Returns
        -------
        experiment : pybamm.Experiment
        protocol_applied : Protocol
            The sanitised protocol actually simulated (post scaling/clamping),
            in the real-cell mA/h frame — see :class:`CycleResult.protocol_applied`.

        Raises
        ------
        ValueError
            If ``protocol`` is not length 6, or length 7 without
            ``use_temperature_protocol=True`` — previously this silently
            sliced ``protocol[:6]``.
        """
        pb = self._pb
        n = len(protocol)
        if not (n == 6 or (n == 7 and self._use_temperature_protocol)):
            raise ValueError(
                f"protocol must have length {len(PROTOCOL_FIELD_NAMES)} (the core "
                f"protocol), or {len(PROTOCOL_FIELD_NAMES) + 1} when this oracle was "
                f"constructed with use_temperature_protocol=True (currently "
                f"{self._use_temperature_protocol}); got length {n}."
            )
        # Slots 0-5 are the electrochemical protocol; an optional slot 6 (#10) is
        # the per-cycle ambient temperature, consumed in __call__, not here.
        C1_mA, C2_mA, dur1_h, dur2_h, D_mA, dur_d_h = protocol[:6]
        s = self._cap_scale
        c_hi  = self._current_ceiling_mA(self._C_MAX_mA)
        c2_hi = self._current_ceiling_mA(self._C2_MAX_mA)

        clamps: list[tuple[str, str]] = []
        # NaN fallbacks: 0.1C / 0.05C of the simulated cell (500 / 250 mA on 5 Ah).
        C1    = self._sanitise_current(C1_mA * s, 0.1 * self._nominal_mAh, self._C_MIN_mA,  c_hi,
                                        label="C1", clamps=clamps)
        C2    = self._sanitise_current(C2_mA * s, 0.05 * self._nominal_mAh, self._C2_MIN_mA, c2_hi,
                                        label="C2", clamps=clamps)
        D     = self._sanitise_current(D_mA  * s, 0.1 * self._nominal_mAh, self._C_MIN_mA,  c_hi,
                                        label="D", clamps=clamps)
        # Discharge is governed by the 3.0 V cutoff (Jones 2022): use a generous
        # timeout (>= time to reach 3.0 V even at ~1C) rather than the real
        # duration, so the cell actually fully discharges. No 2 h floor. The
        # 3600 s (1 h) floor below is INTENTIONAL, not a bug: it exists so the
        # discharge step always has time to reach v_discharge_min rather than
        # being cut off early by a too-short requested duration.
        dur_d_pre = self._sanitise_duration(dur_d_h, 1.0, label="dur_d", clamps=clamps)
        # discharge_mode="protocol" keeps the requested duration, for protocols
        # whose discharges are time-limited (jones2022's are not: they run to
        # 3.0 V; their records only *look* time-limited because ~10 s logging
        # misses the last part of the voltage drop).
        dur_d = dur_d_pre if self._discharge_mode == "protocol" else max(dur_d_pre, 3_600.0)
        if dur_d != dur_d_pre:
            clamps.append(("dur_d:discharge_floor",
                            f"dur_d {dur_d_pre:.0f}→{dur_d:.0f} s "
                            "(1 h discharge floor, intentional)"))
        # Two-stage CC charge: each stage capped at 15 min (paper limit).
        dur1_pre = self._sanitise_duration(dur1_h, 0.25, label="dur1", clamps=clamps)
        dur1 = min(dur1_pre, self._CHARGE_STAGE_MAX_s)
        if dur1 != dur1_pre:
            clamps.append(("dur1:stage_cap",
                            f"dur1 {dur1_pre:.0f}→{dur1:.0f} s (charge-stage cap)"))
        dur2_pre = self._sanitise_duration(dur2_h, 0.25, label="dur2", clamps=clamps)
        dur2 = min(dur2_pre, self._CHARGE_STAGE_MAX_s)
        if dur2 != dur2_pre:
            clamps.append(("dur2:stage_cap",
                            f"dur2 {dur2_pre:.0f}→{dur2:.0f} s (charge-stage cap)"))

        self._warn_clamps(clamps)

        protocol_applied = Protocol(
            charge_current_1_mA=C1 * 1000.0,
            charge_current_2_mA=C2 * 1000.0,
            charge_duration_1_h=dur1 / 3600.0,
            charge_duration_2_h=dur2 / 3600.0,
            discharge_current_mA=D * 1000.0,
            discharge_duration_h=dur_d / 3600.0,
            T_ambient_K=(float(protocol[6]) if n == 7 else None),
        )

        vc, vd, rest = self._V_CHARGE_MAX, self._V_DISCHARGE_MIN, self._rest_s
        # Non-stationary EIS: the sweep is a zero-current period after the 20 min
        # rest, so the rest step runs on through it (the cycle keeps the real timing).
        rest_step = (f"Rest for {rest + self._eis_sweep_s:.0f} seconds"
                     if self._eis_mode == "nonstationary" else f"Rest for {rest:.0f} seconds")
        # Faithful jones2022 cycle (Jones, Stimming & Lee 2022, Nat. Commun.
        # 13:4806, Methods). Discharge-first order (cyclically equivalent to the
        # paper's discharged-start; preserves the initial_soc=0.8 first-call
        # logic). EIS is read at the relaxed post-rest states (steps[1], steps[4]).
        steps = (
            f"Discharge at {D:.4f} A for {dur_d:.0f} seconds or until {vd} V",   # [0]
            rest_step,                                                            # [1] discharged-state OCV rest -> EIS@discharged
            f"Charge at {C1:.4f} A for {dur1:.0f} seconds or until {vc} V",       # [2] CC stage 1
            f"Charge at {C2:.4f} A for {dur2:.0f} seconds or until {vc} V",       # [3] CC stage 2
            rest_step,                                                            # [4] charged-state OCV rest -> EIS@charged
        )
        # Each oracle call is ONE cycle from PyBaMM's perspective.
        # Wrapping all steps in a tuple tells PyBaMM to treat them as a single
        # cycle; a flat list would make each step a separate "cycle" and cause
        # LLI summary variables to accumulate once per step, inflating totals.
        cycles = [steps] * self.n_cycles
        if self.capacity_check:
            cycles = [steps + self._cap_check_steps()] * self.n_cycles
        # Per-call trace of the sanitized step strings (not just the raw protocol):
        # if the native solver crashes — leaving no Python traceback through this
        # function — this shows exactly what currents/durations were fed in just
        # before the crash. Emitted at DEBUG so it stays silent by default: a
        # per-call WARNING floods notebooks and any library consumer (and, under
        # process-parallel tuning, every worker). Opt back in for crash debugging
        # with logging.getLogger("battery_oracle.oracle").setLevel(logging.DEBUG).
        log.debug("[PyBaMMOracle] call %d: raw_protocol=%s steps=%s",
                  len(self._history), np.array2string(np.asarray(protocol), precision=4),
                  steps)
        return pb.Experiment(cycles), protocol_applied

    def _add_eis_noise(self, Z: np.ndarray) -> np.ndarray:
        """Measurement noise on a spectrum (config eis.noise_*)."""
        if self._eis_noise_level > 0.0 and self._eis_noise_model != "none":
            nl = self._eis_noise_level
            if self._eis_noise_model == "white":
                Z = _add_white_noise(Z, nl, rng=self._rng)
            elif self._eis_noise_model == "flicker":
                Z = _add_flicker_noise(self.frequencies, Z, nl, rng=self._rng)
            else:  # "combined": split budget across flicker and white terms
                Z = _add_flicker_noise(
                    self.frequencies, Z,
                    noise_level=nl * self._noise_combined_flicker_frac,
                    rng=self._rng,
                )
                Z = _add_white_noise(
                    Z, noise_level=nl * self._noise_combined_white_frac, rng=self._rng
                )
        return Z

    def _nonstationary_spectrum(self, step, eis_pv) -> np.ndarray:
        """The EIS sweep after a 20 min rest, measured as performed.

        ``step`` is the cycling solution's rest step. A double-layer DFN with the
        oracle's own options and the EIS voltage/current variables (PyBOP's EIS
        model) starts from the rest's first state -- current already zero; mapped by
        variable name in physical units (``set_initial_conditions_from``), so PyBaMM's
        state references are handled -- and is integrated at zero current through
        the rest and the sweep. (Starting from the end of the step before, still
        under current at a voltage cut-off, has no consistent zero-current initial
        state; the double layer relaxes in milliseconds, so nothing is lost.) Frequency ``k`` is linearised at
        the simulated state at the centre of its window (``self._eis_win_lo/hi``
        after the rest; :func:`battery_oracle.eis.linearised_impedance`)
        and the lock-in leakage of the simulated voltage drift is added. Same physics
        as stage A (:mod:`battery_oracle.fit.chain`). The cycling model itself keeps
        its algebraic surface form: the double layer only matters at EIS frequencies,
        and in the experiment's instantaneous current steps it makes the DAE fail.
        """
        import pybamm
        pv = eis_pv.copy()
        pv["Ambient temperature [K]"] = self._pv["Ambient temperature [K]"]
        pv["Current function [A]"] = 0.0
        lo, hi = self._rest_s + self._eis_win_lo, self._rest_s + self._eis_win_hi
        lin = 0.5 * (lo + hi)
        t_end = float(hi.max()) + 1.0

        def _measure(start_state, t0: float):
            """Double-layer EIS model from ``start_state`` at rest time ``t0``, to t_end."""
            opts = {**self._deg_opts, "surface form": "differential"}
            model = eis_model(_MODEL_CLS, opts, initial_state=start_state)
            # dense output in the first second after t0: the stiffest part of the rest
            rel = np.unique(np.concatenate([[0.0, t_end - t0], np.logspace(-4, 0, 9),
                                            np.linspace(0.0, max(self._rest_s - t0, 1.0), 61),
                                            lo - t0, hi - t0, lin - t0]))
            rel = rel[rel >= 0.0]
            last_exc = None
            # A failure raises with its reason (caught and recorded by the caller).
            # Tier 2 relaxes the algebraic error test (PyBOP's own IDAKLU setting).
            for options in ({"num_threads": 1}, {"num_threads": 1, "suppress_algebraic_error": True}):
                try:
                    solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6, options=options)
                    sim = pybamm.Simulation(model, parameter_values=pv, solver=solver,
                                            var_pts=self._model_var_pts(model), submesh_types=self._model_submesh(model))
                    return model, sim, sim.solve(t_eval=rel)
                except pybamm.SolverError as exc:
                    last_exc = exc
            raise last_exc

        try:
            model, sim, sol = _measure(step.first_state, 0.0)
            t_shift = 0.0
        except pybamm.SolverError:
            # The first instants after the current stops (double layer and
            # electrolyte relaxing) can defeat the double-layer DAE. Run them with
            # the cycling model itself (no double layer; it integrated this rest
            # already), then measure from there: the double layer relaxes in ms
            # and the first EIS window starts after the 20 min rest, so nothing
            # measured changes.
            t_shift = 30.0
            m0 = _MODEL_CLS(options=self._deg_opts)
            m0.set_initial_conditions_from(step.first_state, inplace=True)
            pre = pybamm.Simulation(m0, parameter_values=pv, var_pts=self._model_var_pts(m0), submesh_types=self._model_submesh(m0),
                                    solver=pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6,
                                                               options={"num_threads": 1}))
            pre_sol = pre.solve(t_eval=np.linspace(0.0, t_shift, 31))
            model, sim, sol = _measure(pre_sol, t_shift)
        ts = np.asarray(sol.t, float) + t_shift
        if ts[-1] < t_end - 1e-6:
            raise RuntimeError(f"rest + sweep simulation stopped at {ts[-1]:.0f} s of {t_end:.0f} s "
                               f"({sol.termination})")
        v = np.asarray(sol["Voltage [V]"].entries, float)
        idx = np.clip(np.searchsorted(ts, lin), 0, ts.size - 1)
        self._eis_lin_ok = bool(np.allclose(ts[idx], lin, atol=1e-6))
        z = linearised_impedance(sim, ts[idx], np.asarray(sol.y)[:, idx], self.frequencies,
                                 impedance_scale(model, pv))
        amp_a = self._eis_amplitude_mA / 1000.0 * self._cap_scale
        leak = drift_leakage(ts, v, lo, hi, self.frequencies, amp_a)
        self._last_eis_leakage = leak            # real [Ohm]; recorded in the history row
        return z + leak

    # Kinds of the five protocol steps built by _protocol_to_experiment.
    _STEP_KINDS: tuple[str, ...] = ("dis", "rest", "chg", "chg", "rest")

    def _step_stats(self, last_cycle) -> list[dict]:
        """Per-step summary of the protocol steps of the last simulated cycle.

        One dict per protocol step (capacity-check steps excluded): ``kind``,
        ``current_A`` (mean |I|), ``crate`` (vs. the parameter set's nominal
        capacity), ``duration_s``, ``throughput_ah``, ``frac`` (throughput /
        nominal capacity), ``v_end`` and ``hit_cutoff``. A step PyBaMM skipped
        because its cutoff was already exceeded at t=0 (an empty step) reports
        zero duration/throughput -- the same near-empty step the real cells show
        when they hit 4.3 V / 3.0 V within the first sample.
        """
        nominal_ah = 20.0 * self._c20_A
        out: list[dict] = []
        steps = last_cycle.steps
        for i, kind in enumerate(self._STEP_KINDS):
            row = {"kind": kind, "current_A": 0.0, "crate": 0.0, "duration_s": 0.0,
                   "throughput_ah": 0.0, "frac": 0.0, "v_end": float("nan"),
                   "hit_cutoff": False}
            step = steps[i]
            if isinstance(step, _pb.EmptySolution):     # skipped: cutoff met at t=0
                out.append(row)
                continue
            t = np.asarray(step["Time [s]"].entries, dtype=float)
            I = np.asarray(step["Current [A]"].entries, dtype=float)
            V = np.asarray(step["Voltage [V]"].entries, dtype=float)
            if t.size:
                q = float(abs(np.trapezoid(I, t)) / 3600.0) if t.size > 1 else 0.0
                i_mean = float(np.mean(np.abs(I)))
                row.update(current_A=i_mean, crate=i_mean / nominal_ah,
                           duration_s=float(t[-1] - t[0]), throughput_ah=q,
                           frac=q / nominal_ah, v_end=float(V[-1]))
                if kind == "chg":
                    row["hit_cutoff"] = bool(V[-1] >= self._V_CHARGE_MAX - 5e-3)
                elif kind == "dis":
                    row["hit_cutoff"] = bool(V[-1] <= self._V_DISCHARGE_MIN + 5e-3)
            out.append(row)
        return out

    def _is_truncated(self, sol, experiment) -> bool:
        """True if PyBaMM silently dropped steps from the cycle(s) this call added.

        The Experiment runner catches internal solver errors (e.g. IDA_ERR_FAIL at
        the CC->CV switch) on its own callback path and returns whatever was
        integrated up to the failure WITHOUT raising, so a truncated cycle looks
        identical to success unless the step count is checked explicitly.
        """
        if not sol.cycles:
            return True
        n = self.n_cycles
        expected = experiment.cycle_lengths[-1]
        added = sol.cycles[-n:] if n <= len(sol.cycles) else sol.cycles
        return any(len(c.steps) < expected for c in added)

    def _solve_with_fallbacks(self, experiment, protocol, solve_kw):
        """Solve ``experiment``, returning ``(solution, fidelity, failure_kind)``.

        Primary solver, then the emergency (looser) solver on the same model and cell
        state. The model is never swapped: when both solvers fail, ``OracleFailure``
        is raised.

        ``FailureKind.SOLVER_DEGRADED`` is *recorded* (returned) when the emergency
        solver was needed (same model, looser tolerances). Must be called holding
        ``self._lock``.
        """
        pb = self._pb
        proto = np.asarray(protocol).copy()

        def _solve(model, solver):
            sim = pb.Simulation(model, experiment=experiment, var_pts=self._model_var_pts(model), submesh_types=self._model_submesh(model),
                                parameter_values=self._pv, solver=solver)
            sol = sim.solve(starting_solution=self._prev_solution, **solve_kw)
            # The values this solve was built with (incl. initial_soc's initial
            # concentrations): PyBaMM stores potentials relative to references
            # set by them, so the non-stationary EIS model must use the same.
            self._last_cycle_pv = sim.parameter_values
            if self._is_truncated(sol, experiment):
                raise _SilentTruncation()
            return sol

        # Same model throughout: primary solver, then the emergency (looser) solver on the
        # SAME model and cell state. No model swap: if both fail, OracleFailure is raised.
        active_model = self._cycling_model
        primary_solver, emerg_solver, base_fid = self._solver, self._solver_emerg, "full"

        try:
            sol = _solve(active_model, primary_solver)
            return sol, base_fid, None
        except Exception as exc1:
            log.warning("[PyBaMMOracle] primary solver failed (%s: %s); retrying "
                        "with emergency solver (same cell state)",
                        type(exc1).__name__, exc1)
        try:
            sol = _solve(active_model, emerg_solver)
        except _SilentTruncation as exc2:
            raise OracleFailure(
                "Emergency solver also silently truncated the cycle",
                protocol=proto, failure_kind=FailureKind.SOLVER_TRUNCATION,
            ) from exc2
        except Exception as exc2:
            kind = (FailureKind.VOLTAGE_INFEASIBLE if "voltage" in str(exc2).lower()
                    else FailureKind.SOLVER_FAILURE)
            raise OracleFailure(
                f"Battery simulation failed after two attempts: {exc2}",
                protocol=proto, failure_kind=kind,
            ) from exc2
        # Emergency (looser Casadi) succeeded — record the degrade, keep base fidelity.
        return sol, base_fid, FailureKind.SOLVER_DEGRADED

    def __call__(self, protocol: "Protocol | np.ndarray | Sequence[float]") -> np.ndarray:
        """Run one cycle and return the raw state vector.

        This is the hard compatibility contract consumed by the downstream
        active-learning/DMDc loop: array (or :class:`Protocol`) in, array out
        — unchanged behaviour, equivalent to ``self.run_cycle(protocol).state``.
        Prefer :meth:`run_cycle` for named access to SOH, SOC, EIS spectra, and
        fitted ECM parameters instead of reaching into :attr:`history`.
        """
        return self.run_cycle(protocol).state

    def run_cycle(self, protocol: "Protocol | np.ndarray | Sequence[float]") -> "CycleResult":
        """Run one charge/discharge cycle and return the full :class:`CycleResult`.

        Parameters
        ----------
        protocol : Protocol, np.ndarray, or sequence of float
            The charge/discharge protocol for this cycle. A :class:`Protocol`
            instance is converted via :meth:`Protocol.to_array`; a bare array
            or sequence is used as-is (length 6, or 7 with ``T_ambient_K`` in
            slot 6 when this oracle was built with
            ``use_temperature_protocol=True``).

        Returns
        -------
        CycleResult

        Raises
        ------
        ValueError
            If the protocol vector has the wrong length.
        OracleFailure
            If the PyBaMM simulation fails outright, or the cell reaches
            end-of-life (SOH below ``eol_capacity_fraction``). The oracle
            state is **not** reset — see the class docstring.
        """
        protocol = (
            protocol.to_array() if isinstance(protocol, Protocol)
            else np.asarray(protocol, dtype=float)
        )
        # Per-call fidelity/degradation status, recorded into this call's history
        # row. Reset to full fidelity every call; _solve_with_fallbacks records
        # SOLVER_DEGRADED when it needed the emergency solver (same model).
        self._last_failure_kind = None
        self._last_fidelity = "full"
        experiment, protocol_applied = self._protocol_to_experiment(protocol)
        # On the first call (no prior solution), start at 80 % SOC so the initial
        # OCV (~3.9–4.0 V) sits safely inside the experiment voltage window.
        first_call = self._prev_solution is None
        solve_kw: dict = {"initial_soc": self._initial_soc} if first_call else {}
        protocol_applied = self._apply_ambient(protocol, protocol_applied, first_call)

        # Lock protects _prev_solution so sequential state accumulation is safe
        # even if callers inadvertently share this oracle across threads.
        with self._lock:
            sol, self._last_fidelity, self._last_failure_kind = \
                self._solve_with_fallbacks(experiment, protocol, solve_kw)
            self._prev_solution = sol

        # With tuple-based experiment, steps live in sol.cycles[-1].steps[i].
        # Faithful 5-step cycle (see _protocol_to_experiment):
        #   [0] discharge  [1] discharged-state rest  [2] charge CC1
        #   [3] charge CC2  [4] charged-state rest
        # EIS is read at the relaxed post-rest states: step 4 (charged), step 1
        # (discharged), matching the paper's "rest 20 min then EIS". Step indices are
        # stable whether or not capacity_check is enabled (its steps are appended).
        last_cycle = sol.cycles[-1] if sol.cycles else None
        soc_charge, soc_discharge = self._rest_socs(sol, last_cycle)
        T_cell_chg, T_cell_dis = self._rest_temperatures(sol, last_cycle)
        T_cell_K = T_cell_chg

        # DoD damage is accumulated BEFORE the EIS linearisation so it reduces ε_s
        # there and shows up in the impedance (see _accumulate_dod_lam).
        step_stats = self._step_stats(last_cycle)
        self._accumulate_dod_lam(step_stats)
        eis_pv, lam_frac = self._eis_parameter_values(sol)
        sei_thick, dead_li_thick, delta_R_ohmic = self._ohmic_films(sol)

        self._last_eis_mode_used = "equilibrium"
        steps = last_cycle.steps if last_cycle is not None else None
        # non-stationary EIS starts from the first state of each rest (current already
        # zero): step 4 for the charged spectrum, step 1 for the discharged one
        self._last_eis_leakage = None
        Z_charge = self._cycle_spectrum(soc_charge, T_cell_chg, steps[4] if steps else None,
                                        eis_pv, sei_thick, delta_R_ohmic)
        leak_charge, self._last_eis_leakage = self._last_eis_leakage, None
        Z_discharge = self._cycle_spectrum(soc_discharge, T_cell_dis, steps[1] if steps else None,
                                           eis_pv, sei_thick, delta_R_ohmic)
        leak_discharge = self._last_eis_leakage

        cap_ah = self._capacity_ah(sol, last_cycle)
        if self._initial_capacity_ah is None:
            self._initial_capacity_ah = cap_ah
        soh, budget = self._state_of_health(sol, lam_frac, protocol)

        self._history.append({
            "call_idx":           len(self._history),
            "chemistry":          self._chemistry,
            "protocol":           np.asarray(protocol).copy(),
            # Non-fatal per-step status; SOLVER_DEGRADED (Phase 1) is recorded
            # here, never raised. Successful full-fidelity steps carry None/"full".
            "failure_kind":       self._last_failure_kind,
            "fidelity":           self._last_fidelity,
            "eis_mode_used":      self._last_eis_mode_used,
            # lock-in drift leakage included in Z (non-stationary EIS only) [Ohm]
            "eis_leakage_charge": None if leak_charge is None else np.asarray(leak_charge).copy(),
            "eis_leakage_discharge": None if leak_discharge is None else np.asarray(leak_discharge).copy(),
            "sei_thickness_nm":   sei_thick * 1e9 if self._sei_to_R > 0.0 else float("nan"),
            "dead_li_nm":         dead_li_thick * 1e9,
            "lam_pct":            lam_frac * 100.0,
            "lam_cap_loss_ah":    budget["lam_cap_loss_ah"],
            "end_soh":            soh,
            "T_cell_K":           T_cell_K,
            "capacity_ah":              cap_ah,
            "cumulative_sei_loss_ah":   budget["cumulative_sei_loss_ah"],
            "cumulative_plating_ah":    budget["cumulative_plating_ah"],
            "cumulative_lli_total_ah":  budget["cumulative_lli_total_ah"],
            "delta_lli_ah":             budget["delta_lli_ah"],
            "cumulative_dod_lam_ah":    self._cumulative_dod_lam_ah,
            # Per-step throughput / duration / end voltage (see _step_stats) --
            # what the tune engine's step-shape term compares to real GCPL steps.
            "step_stats":         step_stats,
            # EIS spectra at this call, jones2022 raw-data convention
            # (freq/Hz, Re(Z)/Ohm, -Im(Z)/Ohm) so a Nyquist plot is just
            # ax.plot(Z_charge_real, Z_charge_neg_imag).
            "Z_charge_real":        Z_charge.real.copy(),
            "Z_charge_neg_imag":    -Z_charge.imag.copy(),
            "Z_discharge_real":     Z_discharge.real.copy(),
            "Z_discharge_neg_imag": -Z_discharge.imag.copy(),
        })
        self._last_Z = Z_charge.copy()

        # linKK and DRT use the post-charge spectrum (primary measurement)
        linkk_rmse = _linkk_rmse(self.frequencies, Z_charge, c=self._linkk_c, max_M=self._linkk_max_M)
        drt_peaks = self._drt_peaks(Z_charge)

        # ECM fit: charge and discharge spectra fitted independently
        diag_chg: dict = {}
        diag_dis: dict = {}
        params_charge = self._fit_ecm_half(Z_charge, diag_chg)
        params_discharge = self._fit_ecm_half(Z_discharge, diag_dis)
        std_charge = self._ecm_std_half(diag_chg)
        std_discharge = self._ecm_std_half(diag_dis)
        # Layout matches self.state_vector_schema: means_charge | means_discharge |
        # std_charge | std_discharge [| T_cell_K when lumped thermal]. Every block
        # is registered in the schema; the length is derived, never hardcoded.
        blocks = [params_charge, params_discharge, std_charge, std_discharge]
        if "T_cell_K" in self.state_vector_schema:
            blocks.append(np.array([T_cell_K], dtype=float))
        state = np.concatenate(blocks)

        self._history[-1].update({
            "linkk_rmse":          linkk_rmse,
            "max_cv":              diag_chg.get("max_cv",    float("nan")),
            "converged":           diag_chg.get("converged", True),
            "ecm_params":          diag_chg.get("ecm_params", None),
            "drt_peaks":           drt_peaks,
            "ecm_params_charge":   params_charge.copy(),
            "ecm_params_discharge": params_discharge.copy(),
            "ecm_std_charge":      std_charge.copy(),
            "ecm_std_discharge":   std_discharge.copy(),
            # Raw AutoEIS MCMC posterior samples behind ecm_params_{charge,discharge}
            # above (dict: AutoEIS variable name -> (num_samples,) array), so callers
            # comparing oracle vs. real posterior distributions (e.g.
            # jones_oracle_study.py) don't need to refit — None if AutoEIS fell back
            # to the Randles stub for that state.
            "ecm_samples_charge":      diag_chg.get("_raw_samples"),
            "ecm_samples_discharge":   diag_dis.get("_raw_samples"),
            "ecm_variables_charge":    diag_chg.get("_raw_variables", []),
            "ecm_variables_discharge": diag_dis.get("_raw_variables", []),
        })

        result = CycleResult(
            state=state,
            schema=self.state_vector_schema,
            soh=soh,
            soc_charge=soc_charge,
            soc_discharge=soc_discharge,
            capacity_ah=cap_ah,
            fidelity=self._last_fidelity,
            failure_kind=self._last_failure_kind,
            protocol=np.asarray(protocol).copy(),
            protocol_applied=protocol_applied,
            eis_charge=(self.frequencies, Z_charge.copy()),
            eis_discharge=(self.frequencies, Z_discharge.copy()),
            ecm_charge=dict(zip(self._ecm_param_names, params_charge.tolist())),
            ecm_discharge=dict(zip(self._ecm_param_names, params_discharge.tolist())),
            ecm_std_charge=dict(zip(self._ecm_param_names, std_charge.tolist())),
            ecm_std_discharge=dict(zip(self._ecm_param_names, std_discharge.tolist())),
            raw=types.MappingProxyType(self._history[-1]),
        )
        self._last_result = result
        return result

    # ── run_cycle steps ──────────────────────────────────────────────────────

    def _apply_ambient(self, protocol: np.ndarray, protocol_applied: Protocol,
                       first_call: bool) -> Protocol:
        """Per-cycle ambient temperature (#10): a 7-D protocol's slot 6 overrides the
        lumped-thermal ambient for this call.

        The fresh Simulation built in _solve_with_fallbacks re-processes self._pv, so
        mutating the ambient here takes effect. Bounded to a physical window; T_cell is
        then read from the solution (self-heated above ambient).
        """
        if not (self._use_temperature_protocol and len(protocol) >= 7):
            return protocol_applied
        self._T_ambient_K = float(np.clip(protocol[6], 253.15, 333.15))
        self._pv["Ambient temperature [K]"] = self._T_ambient_K
        if first_call:
            self._pv["Initial temperature [K]"] = self._T_ambient_K
        # Keep protocol_applied's T_ambient_K in sync with the clamp above
        # (frozen dataclass, so replace rather than mutate).
        return replace(protocol_applied, T_ambient_K=self._T_ambient_K)

    def _rest_socs(self, sol, last_cycle) -> tuple[float, float]:
        """SOC at the end of the charged (step 4) and discharged (step 1) rests,
        from the negative-electrode stoichiometry relative to its initial value."""
        x100 = (
            float(self._pv["Initial concentration in negative electrode [mol.m-3]"])
            / float(self._pv["Maximum concentration in negative electrode [mol.m-3]"])
        )
        x_neg_chg = float(last_cycle.steps[4]["Average negative particle stoichiometry"].entries[-1])
        x_neg_dis = float(last_cycle.steps[1]["Average negative particle stoichiometry"].entries[-1])
        return (float(np.clip(x_neg_chg / x100, self._soc_clip_min, self._soc_clip_max)),
                float(np.clip(x_neg_dis / x100, self._soc_clip_min, self._soc_clip_max)))

    def _rest_temperatures(self, sol, last_cycle) -> tuple[float, float]:
        """Volume-averaged cell temperature (#9) at the charged and discharged rests.

        Isothermal runs return the setpoint; lumped thermal reads the self-heated
        temperature from the solution. These feed the Arrhenius EIS correction (#11).
        """
        if self._thermal != "lumped":
            return self._temperature_K, self._temperature_K
        T_VAR = "Volume-averaged cell temperature [K]"
        return (float(last_cycle.steps[4][T_VAR].entries[-1]),
                float(last_cycle.steps[1][T_VAR].entries[-1]))

    def _accumulate_dod_lam(self, step_stats: list[dict]) -> None:
        """DoD / charge-induced-stress LAM increment (see __init__), from the simulated
        cycle: the depth of discharge is the discharge step's throughput / nominal
        capacity, the stress uses the first charge stage's C-rate (a step stopped early
        at a voltage limit counts only what it passed).

        Called before the EIS linearisation, so depth-of-discharge damage reduces ε_s
        there and therefore shows up in the impedance as a grown charge-transfer arc
        (R3+R5), the way real jones2022 cells age, rather than only in the hidden SOH.
        Accumulated as a dimensionless cell fraction; mapped onto the nominal capacity
        for the eol_loss budget in _state_of_health.
        """
        if self._dod_lam_scale <= 0.0:
            return
        dis = next(r for r in step_stats if r["kind"] == "dis")
        chg = next(r for r in step_stats if r["kind"] == "chg")
        dod_frac = max(0.0, float(dis["frac"]))
        stress = float(np.sqrt(max(0.0, float(chg["crate"])) * dod_frac))
        self._cumulative_dod_throughput_ah += dod_frac
        self._cumulative_dod_lam_frac += (
            self._dod_lam_scale * stress * dod_frac
            / (2.0 * np.sqrt(max(self._cumulative_dod_throughput_ah, 1e-9)))
        )

    def _eis_parameter_values(self, sol) -> tuple:
        """Parameter values for this cycle's EIS, and the cracking LAM fraction.

        LAM a priori: ε_s is reduced so the EIS linearisation sees the degraded
        electrode microstructure [Doyle et al., 1993].
        """
        eis_pv = self._pv.copy()
        # For lumped thermal (#11), synthesise the EIS baseline at T_ref so the
        # analytic Arrhenius scaling (T_ref -> T_cell in _cycle_spectrum) is the
        # SOLE temperature effect — otherwise PyBaMM's own T-dependent parameters
        # (evaluated at the pv ambient = T_ambient_K) would double-count it.
        if self._thermal == "lumped":
            eis_pv["Ambient temperature [K]"] = self._T_ref_K
        lam_pct = _final_value(sol, "Loss of active material in negative electrode [%]", absent=0.0)
        lam_frac = float(np.clip(lam_pct / 100.0, 0.0, self._lam_ceiling))
        # Total active-material loss seen by the spectrum = physical
        # cracking (lam_frac, ~0 while particle mechanics is disabled)
        # plus the DoD/charge-induced-stress LAM. lam_frac alone still
        # feeds lam_cap_loss; the DoD part is added to eol_loss
        # separately, so combining them here does not double-count.
        eff_lam = float(np.clip(lam_frac + self._cumulative_dod_lam_frac, 0.0, self._lam_ceiling))
        if eff_lam > 1e-4:
            eis_pv["Negative electrode active material volume fraction"] = (
                self._eps_s_nominal * (1.0 - eff_lam)
            )
        return eis_pv, lam_frac

    def _ohmic_films(self, sol) -> tuple[float, float, float]:
        """SEI and dead-lithium thickness [m] and the ohmic resistance they add [Ohm].

        The SEI film is purely ohmic [Christensen & Newman, J. Electrochem. Soc.
        151:A1977, 2004].
        """
        # absent without the mechanism (no plating -> no dead lithium)
        sei_thick = _final_value(sol, "X-averaged negative SEI thickness [m]", absent=0.0)
        dead_li_thick = _final_value(sol, "X-averaged negative dead lithium thickness [m]", absent=0.0)
        delta_R_ohmic = sei_thick * self._sei_to_R
        delta_R_ohmic += dead_li_thick * self._dead_li_to_R
        return sei_thick, dead_li_thick, delta_R_ohmic

    def _cycle_spectrum(self, soc: float, T_cell: float, step, eis_pv,
                        sei_thick: float, delta_R_ohmic: float) -> np.ndarray:
        """One noisy EIS spectrum: non-stationary from ``step`` when configured (and
        the solve was full fidelity), otherwise the equilibrium spectrum at ``soc``
        with the Arrhenius and ohmic-film corrections."""
        native = False
        if self._eis_mode == "nonstationary" and step is not None and self._last_fidelity == "full":
            try:
                Z = self._nonstationary_spectrum(step, eis_pv)
                native = True
                self._last_eis_mode_used = "nonstationary"
            except Exception as exc:              # noqa: BLE001 - recorded, stock path below
                log.warning("[PyBaMMOracle] non-stationary EIS failed (%s: %s); "
                            "equilibrium EIS for this spectrum", type(exc).__name__, str(exc)[:200])
                self._last_eis_mode_used = "equilibrium (non-stationary failed)"
        if native:
            # The model itself carries temperature, the SEI film and contact
            # resistance and the relaxation drift: only effects it does not
            # model (crack SEI, dead lithium) and the measurement noise follow.
            d_r = delta_R_ohmic - sei_thick * self._sei_to_R
            if d_r > 0.0:
                Z = (Z.real + d_r) + 1j * Z.imag
            return self._add_eis_noise(Z)
        # the cycling model's contact resistance and mesh; isothermal and without SEI
        # (temperature and the films enter analytically below)
        opts = {"surface form": "differential"}
        if self._deg_opts.get("contact resistance") == "true":
            opts["contact resistance"] = "true"
        model = _MODEL_CLS(options=opts)
        Z = equilibrium_impedance(model, eis_pv, self.frequencies, soc, var_pts=self._model_var_pts(model),
                                  submesh_types=self._model_submesh(model))
        # Arrhenius temperature dependence (#11): the EIS model is solved
        # isothermally, so temperature enters analytically. Scale the
        # charge-transfer arc by exp(E_a/R (1/T - 1/T_ref)) and the
        # electrolyte/ohmic HF intercept by the electrolyte E_a. Applied to the
        # native spectrum BEFORE the ohmic film add and noise, so both the ECM
        # means and the posterior std are consistently corrected (the AutoEIS
        # target-R0 rescale is scale-equivariant and does not cancel it). R0 =
        # HF real intercept (highest frequency), the file-wide convention.
        if self._thermal == "lumped" and abs(T_cell - self._T_ref_K) > 1e-9:
            inv = 1.0 / T_cell - 1.0 / self._T_ref_K
            arr_ct = float(np.exp(self._E_a / _R_GAS * inv))
            arr_el = float(np.exp(self._E_a_el / _R_GAS * inv))
            R0 = float(Z.real[int(np.argmax(self.frequencies))])
            Z = (R0 * arr_el) + arr_ct * (Z - R0)
        if delta_R_ohmic > 0.0:
            Z = (Z.real + delta_R_ohmic) + 1j * Z.imag
        return self._add_eis_noise(Z)

    def _capacity_ah(self, sol, last_cycle) -> float:
        """Discharge capacity of this cycle [A.h].

        With capacity_check, read from the dedicated C/20 discharge step (last step),
        which is sensitive to LLI and protocol. Without it, the voltage-limited
        discharge capacity of the regular cycling step (less sensitive, ~20× faster).
        """
        if self.capacity_check:
            step = last_cycle.steps[-1]
            if isinstance(step, _pb.EmptySolution):     # skipped: already at the cutoff
                return 0.0
            return float(step["Discharge capacity [A.h]"].entries[-1])
        return float(np.max(sol["Discharge capacity [A.h]"].entries))

    def _state_of_health(self, sol, lam_frac: float, protocol: np.ndarray) -> tuple[float, dict]:
        """SOH from the irreversible-loss budget, and the budget's terms [A.h].

        Raises OracleFailure(END_OF_LIFE) below eol_capacity_fraction.
        """
        # EOL criterion: total irreversible lithium inventory loss from all active
        # mechanisms.  Each variable is a cumulative integrator that carries
        # forward via _prev_solution, so it reflects total loss since battery birth.
        # SEI on cracks and dead-Li plating are protocol-sensitive (higher C-rate
        # → more cracking → faster crack-SEI; higher C-rate → more plating → more
        # dead Li), so including them makes EOL respond to the chosen protocol.
        def try_get(key: str) -> float:     # 0 for a mechanism this model does not include
            return _final_value(sol, key, absent=0.0)

        # "Loss of capacity to ... plating [A.h]" integrates the NET plating current
        # (plating minus stripping).  In the "partially reversible" model this can be
        # negative when stripping dominates (e.g. low C-rate cycling strips reversible
        # plated Li faster than new Li plates).  Clamp to ≥ 0: stripping returns Li to
        # the active pool — it is not a capacity gain.  The true irreversible loss from
        # plating is the dead-Li component, which trends positive over time.
        cumulative_sei_loss = (
            try_get("Loss of capacity to negative SEI [A.h]")
            + try_get("Loss of capacity to negative SEI on cracks [A.h]")
            + max(0.0, try_get("Loss of capacity to negative lithium plating [A.h]"))
        )
        if cumulative_sei_loss == 0.0:
            cumulative_sei_loss = float("nan")

        # Use the parameter-set nominal capacity (5 Ah for Chen2020) rather than the
        # first-call measured capacity.  The first-call discharge may be truncated if
        # the initial SOC causes early voltage-event termination, making cap_ah ~1 Ah
        # and inflating all subsequent SOH values by ~4×.
        nominal_cap = float(self._pv["Nominal cell capacity [A.h]"])

        # Zero-reference the LLI at call 1 to exclude formation-cycle loss.  PyBAMM
        # may accumulate non-zero LLI from the initial simulation even before any
        # cycling damage; subtracting this baseline keeps SOH = 1.0 for a fresh cell.
        if self._initial_lli_ah is None:
            self._initial_lli_ah = (
                cumulative_sei_loss if np.isfinite(cumulative_sei_loss) else 0.0
            )

        delta_lli = (
            max(cumulative_sei_loss - self._initial_lli_ah, 0.0)
            if np.isfinite(cumulative_sei_loss) else 0.0
        )

        # LAM (particle cracking) reduces the electrode's active surface area and
        # capacity independently of LLI.  Add the estimated capacity loss due to LAM
        # so that cracking-dominated degradation also drives EOL.
        # Conservative upper bound: full lam_frac × Q_nominal (graphite-limited cell).
        lam_cap_loss = lam_frac * nominal_cap

        # DoD / charge-induced-stress LAM term: the per-cycle increment and its
        # cumulative loss FRACTION are computed earlier (_accumulate_dod_lam, before
        # the EIS step, so it also reduces ε_s and grows R3+R5). Here we just map that
        # dimensionless fraction onto the oracle's nominal_cap for the eol_loss budget,
        # keeping it scale-consistent with delta_lli. Smith-style sqrt(throughput) law;
        # the running total only grows -> irreversible active-material loss.
        self._cumulative_dod_lam_ah = self._cumulative_dod_lam_frac * nominal_cap

        eol_loss = delta_lli + lam_cap_loss + self._cumulative_dod_lam_ah
        soh = 1.0 - eol_loss / nominal_cap

        if soh < self.eol_capacity_fraction:
            raise OracleFailure(
                f"Battery reached end-of-life: SOH {soh:.1%} < "
                f"{self.eol_capacity_fraction:.0%} threshold "
                f"(LLI {delta_lli:.3f} Ah + LAM {lam_cap_loss:.3f} Ah "
                f"+ DoD-LAM {self._cumulative_dod_lam_ah:.3f} Ah "
                f"= {eol_loss:.3f} Ah = {eol_loss / nominal_cap:.1%} of nominal)",
                protocol=np.asarray(protocol).copy(),
                failure_kind=FailureKind.END_OF_LIFE,
            )
        return soh, {
            "lam_cap_loss_ah":         lam_cap_loss,
            "cumulative_sei_loss_ah":  try_get("Loss of capacity to negative SEI [A.h]"),
            "cumulative_plating_ah":   try_get("Loss of capacity to negative lithium plating [A.h]"),
            "cumulative_lli_total_ah": cumulative_sei_loss,
            "delta_lli_ah":            delta_lli,
        }

    def _drt_peaks(self, Z: np.ndarray) -> np.ndarray:
        """Relaxation times [s] of the DRT peaks of ``Z`` (empty if the DRT fails)."""
        try:
            from battery_oracle._eis.drt import get_drt_impedance
            drt_out, _, _, _ = get_drt_impedance(
                (self.frequencies,), (Z,), (None,), trim=False
            )
            tau = drt_out[0][:, 0]
            gamma = drt_out[0][:, 1]
            idx, _ = _find_peaks(gamma)
            if len(idx):
                return tau[idx]
        except Exception as exc:                  # noqa: BLE001 - a diagnostic, not the state
            log.warning("[PyBaMMOracle] DRT failed (%s: %s); no DRT peaks this cycle",
                        type(exc).__name__, str(exc)[:200])
        return np.array([], dtype=float)

    def _fit_ecm_half(self, Z: np.ndarray, diag: dict) -> np.ndarray:
        """ECM parameter means for one spectrum.

        ecm_model_fn(frequencies, Z_real, Z_imag) returns a full circuit-derived vector
        (charge half ++ discharge half); the first n_params (one half) are kept. n_params
        is the per-half width, derived from the circuit, never hardcoded.
        """
        if self.ecm_model_fn is autoeis_ecm:
            # Draw a fresh int seed from the oracle's noise RNG for AutoEIS's
            # sampler (autoeis.perform_bayesian_inference's own `seed` kwarg),
            # so both the EIS noise and the MCMC fit are reproducible from the
            # same PyBaMMOracle(seed=...) under a single deterministic stream.
            autoeis_seed = int(self._rng.integers(0, 2**31 - 1))
            full, raw_samples, raw_variables = autoeis_ecm(
                self.frequencies, Z.real, Z.imag,
                circuit=self._circuit, _diag=diag, return_samples=True,
                cpe_w_seed=self._cpe_w_seed, cpe_n_seed=self._cpe_n_seed,
                cpe_w_default=self._cpe_w_default, cpe_n_default=self._cpe_n_default,
                rescale_target_r0=self._ecm_rescale_target_r0,
                num_warmup=self._autoeis_num_warmup, num_samples=self._autoeis_num_samples,
                seed=autoeis_seed,
            )
            diag["_raw_samples"] = raw_samples
            diag["_raw_variables"] = raw_variables
        elif self.ecm_model_fn is randles_stub_ecm:
            full = randles_stub_ecm(
                self.frequencies, Z.real, Z.imag,
                circuit=self._circuit, cpe_n_seed=self._cpe_n_seed,
                cpe_n_default=self._cpe_n_default,
            )
        else:
            full = self.ecm_model_fn(self.frequencies, Z.real, Z.imag)
        return full[:len(self._ecm_param_names)]

    def _ecm_std_half(self, diag: dict) -> np.ndarray:
        """Per-parameter AutoEIS posterior std, aligned to the mean vector.

        The mean half is built by iterating ``_raw_variables`` (see ``autoeis_ecm``:
        ``half = [median(samples[k]) for k in _variables]``), so the std MUST iterate
        the same order to line up with the means. Returns all-NaN when AutoEIS fell
        back to the Randles stub (no samples), which a downstream DMDc/audit consumer
        drops or imputes.
        """
        n_params = len(self._ecm_param_names)
        samples = diag.get("_raw_samples")
        variables = diag.get("_raw_variables") or []
        out = np.full(n_params, np.nan)
        if not samples:
            return out
        vals = [float(np.std(np.asarray(samples[k]))) for k in variables[:n_params]]
        out[:len(vals)] = vals
        return out


def oracle_parameter_values(**oracle_kwargs):
    """The parameter values a :class:`PyBaMMOracle` built with ``oracle_kwargs`` simulates:
    the degradation preset's edits and ``kinetics_scale`` included, with the oracle's own
    voltage limits as the cut-offs (so fitted values are relative to exactly that)."""
    oracle = PyBaMMOracle(ecm_model_fn=randles_stub_ecm, **oracle_kwargs)
    pv = oracle._pv.copy()
    pv.update({"Upper voltage cut-off [V]": oracle._V_CHARGE_MAX,
               "Lower voltage cut-off [V]": oracle._V_DISCHARGE_MIN})
    return pv
