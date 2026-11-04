"""battery_oracle.fit.config: a fit's settings survive the report round trip, and a report
without them is refused."""
import pytest

from battery_oracle.fit.config import FitConfig


def test_report_round_trip_preserves_every_setting():
    cfg = FitConfig(r_points=40, r_mesh="surface", r_stretch=2.5, x_factor=1, validate_x_factor=3,
                    envelope_ma=(10.0, 8.0, 5.0), ce_floor=0.2, workers=4, candidate_timeout_s=30.0,
                    checkpoint="ck.json", from_checkpoint=True, warm_json="w.json", warm_sigma=0.1,
                    free=("a", "b"), dry_run=True)
    assert FitConfig.from_report({"fit_config": cfg.to_dict()}) == cfg


def test_report_without_config_is_refused():
    with pytest.raises(ValueError):
        FitConfig.from_report({"radial_mesh": {"points": 40}})


def test_environment_maps_onto_the_config():
    env = {"ORACLE_FIT_R_POINTS": "40", "ORACLE_FIT_R_MESH": "surface", "ORACLE_FIT_X_FACTOR": "2",
           "ORACLE_FIT_CE_FLOOR": "0.05", "SLURM_CPUS_PER_TASK": "6", "ORACLE_FIT_FREE": "a; b;",
           "ORACLE_FIT_FROM_CHECKPOINT": "1"}
    cfg = FitConfig.from_env(env)
    assert (cfg.r_points, cfg.r_mesh, cfg.x_factor, cfg.ce_floor) == (40, "surface", 2, 0.05)
    assert cfg.workers == 6 and cfg.free == ("a", "b") and cfg.from_checkpoint
    assert FitConfig.from_env({}) == FitConfig()


def test_radial_mesh_setting_is_none_only_for_the_default_mesh():
    assert FitConfig(validate_x_factor=1).radial_mesh_setting() is None
    assert FitConfig().radial_mesh_setting() == {"points": None, "kind": "uniform", "stretch": 1.15, "x_factor": 2}
    assert FitConfig(r_points=40).radial_mesh_setting(x_factor=1)["points"] == 40


def test_unknown_mesh_kind_and_envelope_length_are_rejected():
    with pytest.raises(ValueError):
        FitConfig(r_mesh="bogus")
    with pytest.raises(ValueError):
        FitConfig(envelope_ma=(1.0,))
