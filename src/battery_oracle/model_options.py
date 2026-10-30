"""Mesh settings shared by the oracle and the stage-A chain fit, so that the oracle
replays protocols on the mesh the fit converged on without importing the fit code."""
from __future__ import annotations


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


def x_var_pts(model, factor: int) -> dict:
    """x points of ``model``'s default mesh times ``factor``."""
    d = model.default_var_pts
    return {k: int(d[k]) * int(factor) for k in ("x_n", "x_s", "x_p")} if factor != 1 else {}
