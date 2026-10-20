"""patches.nonuniform_edge_extrapolation: a Fickian particle with a concentration-dependent
D conserves lithium on a non-uniform (surface-clustered) mesh."""
import numpy as np
import pybamm
import pytest

from battery_oracle import patches

R, CMAX, D0, J = 2.6e-6, 5.0e4, 1e-14, 2e-4


def leak(mesh_kind):
    m = pybamm.BaseModel()
    c = pybamm.Variable("c", domain="particle")
    s = pybamm.maximum((c / CMAX - 0.95) / 0.05, 0.0)
    Dc = D0 * 10 ** (-3.0 * s ** 2)
    m.rhs = {c: -pybamm.div(-Dc * pybamm.grad(c))}
    m.boundary_conditions = {c: {"left": (pybamm.Scalar(0), "Neumann"), "right": (J / pybamm.surf(Dc), "Neumann")}}
    m.initial_conditions = {c: pybamm.Scalar(0.80 * CMAX)}
    r = pybamm.SpatialVariable("r", domain="particle", coord_sys="spherical polar")
    m.variables = {"total": pybamm.Integral(c, r)}
    m.events = [pybamm.Event("near full", 0.995 * CMAX - pybamm.surf(c))]
    sub = {"particle": pybamm.Uniform1DSubMesh} if mesh_kind == "uniform" else \
        {"particle": pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params={"side": "right", "stretch": 3.0})}
    mesh = pybamm.Mesh({"particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(R)}}}, sub, {r: 40})
    pybamm.Discretisation(mesh, {"particle": pybamm.FiniteVolume()}).process_model(m)
    sol = pybamm.IDAKLUSolver(rtol=1e-9, atol=1e-9).solve(m, np.linspace(0, 3000, 3001))
    tot, t = sol["total"].entries, sol.t
    delivered = 4 * np.pi * R ** 2 * J * (t[-1] - t[0])
    return (tot[-1] - tot[0] - delivered) / delivered


@pytest.fixture
def patched():
    patches.apply(["nonuniform_edge_extrapolation"])
    yield
    patches.revert(["nonuniform_edge_extrapolation"])


def _upstream_fixed():
    import inspect
    return "exterior_edge_rows" in inspect.getsource(pybamm.FiniteVolume.shift)


@pytest.mark.skipif(_upstream_fixed(), reason="PyBaMM >= 26.10 has the fix (PyBaMM#5755)")
def test_stock_pybamm_leaks_on_a_clustered_mesh():
    assert patches.applied().isdisjoint({"nonuniform_edge_extrapolation"})
    assert abs(leak("surface")) > 0.1          # the bug this patch fixes


def test_patch_conserves_lithium_on_a_clustered_mesh(patched):
    assert abs(leak("surface")) < 1e-10


def test_patch_leaves_a_uniform_mesh_unchanged(patched):
    assert abs(leak("uniform")) < 1e-10


def _shifted(mesh_kind, method, f):
    r = pybamm.SpatialVariable("r", domain="particle", coord_sys="spherical polar")
    sub = {"particle": pybamm.Uniform1DSubMesh} if mesh_kind == "uniform" else \
        {"particle": pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params={"side": "right", "stretch": 3.0})}
    mesh = pybamm.Mesh({"particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1.0)}}}, sub, {r: 12})
    fvm = pybamm.FiniteVolume()
    fvm.build(mesh)
    nodes, edges = mesh["particle"].nodes, mesh["particle"].edges
    v = pybamm.Vector(f(nodes), domain="particle")
    return np.ravel(fvm.node_to_edge(v, method=method).evaluate()), edges


@pytest.mark.parametrize("method", ["arithmetic", "harmonic"])
def test_uniform_mesh_matches_stock_pybamm(method):
    f = lambda x: 1.0 + 3.0 * x ** 2               # noqa: E731
    stock, _ = _shifted("uniform", method, f)
    patches.apply(["nonuniform_edge_extrapolation"])
    try:
        new, _ = _shifted("uniform", method, f)
    finally:
        patches.revert(["nonuniform_edge_extrapolation"])
    assert np.allclose(new, stock, rtol=1e-12, atol=1e-14)


def test_stretched_mesh_is_exact_for_a_linear_profile(patched):
    f = lambda x: 2.0 + 5.0 * x                    # noqa: E731
    vals, edges = _shifted("surface", "arithmetic", f)
    assert np.allclose(vals, f(edges), rtol=1e-12)
    vals_h, _ = _shifted("surface", "harmonic", f)
    assert np.allclose(vals_h[[0, -1]], f(edges)[[0, -1]], rtol=1e-12)   # exterior edges exact
