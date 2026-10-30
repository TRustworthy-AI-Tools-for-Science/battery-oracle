"""Settings of a stage-A chain fit (:mod:`battery_oracle.bol_chain`), built once by the
caller and recorded in the fit report, so a report can be re-evaluated with the
settings it was made with rather than whatever the environment holds later."""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field

#: default electrolyte floor (fraction of c0) for the depletion stop
CE_FLOOR_DEFAULT = 0.1


@dataclass(frozen=True)
class FitConfig:
    # particle radial mesh (the default 20 uniform points are not converged for the
    # LiCoO2 surface gradients): points per particle (None: PyBaMM's), "uniform" or
    # "surface" (exponential, clustered at r = R) and its stretch
    r_points: int | None = None
    r_mesh: str = "uniform"
    r_stretch: float = 1.15
    # electrode/separator (x) mesh as a factor on PyBaMM's points: the fit runs on
    # x_factor; the final evaluation, the validity gate and the oracle on
    # validate_x_factor. The x mesh moves the fitted voltage by <= 0.3 mV but the default
    # one does not resolve the salt flux at the negative|separator interface during fast
    # charge (jump 0.10 of the current at x1, 0.033 at x2).
    x_factor: int = 1
    validate_x_factor: int = 2
    # envelope probe after each candidate's history, in the argument order of
    # bol_chain.with_envelope_probe: (charge mA, discharge mA) or
    # (c1 mA, discharge mA, c2 mA) for a two-stage charge; None: no probe
    envelope_ma: tuple[float, ...] | None = None
    # electrolyte floor (fraction of c0): below it a candidate is treated as depleted
    ce_floor: float = CE_FLOOR_DEFAULT
    # optimiser run control (not part of the fitted model)
    workers: int = 1
    candidate_timeout_s: float = 600.0
    checkpoint: str | None = None
    from_checkpoint: bool = False
    # warm start: a report or checkpoint with best_inputs; CMA-ES starts there with a step
    # warm_sigma x the prior's, and every parameter not in ``free`` is held at its warm value
    warm_json: str | None = None
    warm_sigma: float = 0.3
    free: tuple[str, ...] = field(default_factory=tuple)
    dry_run: bool = False

    def __post_init__(self):
        if self.r_mesh not in ("uniform", "surface"):
            raise ValueError(f"radial mesh kind {self.r_mesh!r}: use 'uniform' or 'surface'")
        if self.envelope_ma is not None and len(self.envelope_ma) not in (2, 3):
            raise ValueError(f"envelope_ma needs 2 or 3 values, got {self.envelope_ma!r}")

    def radial_mesh_setting(self, x_factor: int | None = None) -> dict | None:
        """The mesh as an oracle setting (``radial_mesh`` of PyBaMMOracle):
        {"points", "kind", "stretch", "x_factor"}; None for PyBaMM's default mesh.
        ``x_factor`` defaults to the validation mesh's."""
        xf = self.validate_x_factor if x_factor is None else int(x_factor)
        if not self.r_points and self.r_mesh == "uniform" and xf == 1:
            return None
        return {"points": int(self.r_points) if self.r_points else None, "kind": self.r_mesh,
                "stretch": self.r_stretch, "x_factor": xf}

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["envelope_ma"] = list(self.envelope_ma) if self.envelope_ma is not None else None
        d["free"] = list(self.free)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FitConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        if kw.get("envelope_ma") is not None:
            kw["envelope_ma"] = tuple(float(x) for x in kw["envelope_ma"])
        if "free" in kw:
            kw["free"] = tuple(kw["free"])
        return cls(**kw)

    @classmethod
    def from_report(cls, report: dict) -> "FitConfig":
        """The config a fit report was made with."""
        if not report.get("fit_config"):
            raise ValueError("fit report has no fit_config: regenerate it with this version")
        return cls.from_dict(report["fit_config"])

    @classmethod
    def from_env(cls, environ=None) -> "FitConfig":
        """The ORACLE_FIT_* environment variables of the battery-forecast launchers."""
        env = os.environ if environ is None else environ
        g = env.get
        env_ma = g("ORACLE_FIT_ENVELOPE_MA")
        envelope = None
        if env_ma:
            vals = [float(x) for x in env_ma.replace(";", ",").split(",")]
            if len(vals) not in (2, 3):
                raise ValueError(f"ORACLE_FIT_ENVELOPE_MA needs 2 or 3 values, got {env_ma!r}")
            # "charge;discharge" or "c1;c2;discharge" -> with_envelope_probe's order
            envelope = (vals[0], vals[1]) if len(vals) == 2 else (vals[0], vals[2], vals[1])
        return cls(
            r_points=int(g("ORACLE_FIT_R_POINTS")) if g("ORACLE_FIT_R_POINTS") else None,
            r_mesh=g("ORACLE_FIT_R_MESH", "uniform"),
            r_stretch=float(g("ORACLE_FIT_R_STRETCH", "1.15")),
            x_factor=int(g("ORACLE_FIT_X_FACTOR", "1")),
            validate_x_factor=int(g("ORACLE_FIT_VALIDATE_X_FACTOR", "2")),
            envelope_ma=envelope,
            ce_floor=float(g("ORACLE_FIT_CE_FLOOR", CE_FLOOR_DEFAULT)),
            workers=int(g("ORACLE_FIT_WORKERS", g("SLURM_CPUS_PER_TASK", "1"))),
            candidate_timeout_s=float(g("ORACLE_FIT_CANDIDATE_TIMEOUT", "600")),
            checkpoint=g("ORACLE_FIT_CHECKPOINT") or None,
            from_checkpoint=g("ORACLE_FIT_FROM_CHECKPOINT") == "1",
            warm_json=g("ORACLE_FIT_WARM_JSON") or None,
            warm_sigma=float(g("ORACLE_FIT_WARM_SIGMA", "0.3")),
            free=tuple(k.strip() for k in (g("ORACLE_FIT_FREE") or "").split(";") if k.strip()),
            dry_run=g("ORACLE_FIT_DRY_RUN") == "1",
        )
