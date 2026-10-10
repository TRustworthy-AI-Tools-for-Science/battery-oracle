"""Interface averaging of flux coefficients (battery_oracle.patches harmonic_flux_coefficients)."""
import numpy as np
import pytest


def _discretised_divergences(patched: bool):
    import pybamm

    from battery_oracle import patches

    if patched:
        patches.apply(["harmonic_flux_coefficients"])
    else:
        patches.revert(["harmonic_flux_coefficients"])
    x_n, x_s = pybamm.standard_spatial_vars.x_n, pybamm.standard_spatial_vars.x_s
    geometry = {"negative electrode": {x_n: {"min": 0.0, "max": 1.0}},
                "separator": {x_s: {"min": 1.0, "max": 2.0}}}
    mesh = pybamm.Mesh(geometry, {"negative electrode": pybamm.Uniform1DSubMesh,
                                  "separator": pybamm.Uniform1DSubMesh}, {x_n: 6, x_s: 6})
    disc = pybamm.Discretisation(mesh, {"negative electrode": pybamm.FiniteVolume(),
                                        "separator": pybamm.FiniteVolume()})
    whole = ["negative electrode", "separator"]
    phi = pybamm.Variable("phi", domain=whole)
    psi = pybamm.Variable("psi", domain=whole)            # zero: grad(phi) - grad(psi) == grad(phi)
    disc.set_variable_slices([phi, psi])
    # boundary fluxes: the gradient then spans every edge, like the DFN's
    zero_flux = {"left": (pybamm.Scalar(0.0), "Neumann"), "right": (pybamm.Scalar(0.0), "Neumann")}
    disc.bcs = {phi: zero_flux, psi: zero_flux}
    kappa = pybamm.concatenation(pybamm.PrimaryBroadcast(1.0, "negative electrode"),
                                 pybamm.PrimaryBroadcast(9.0, "separator"))
    bare = pybamm.div(kappa * pybamm.grad(phi))
    combo = pybamm.div(kappa * (pybamm.grad(phi) - pybamm.grad(psi)))     # as the DFN electrolyte current
    y = np.r_[np.linspace(0.0, 1.0, 12) ** 2, np.zeros(12)]
    out = [np.asarray(disc.process_symbol(e).evaluate(y=y)).ravel() for e in (bare, combo)]
    patches.revert(["harmonic_flux_coefficients"])
    return out


def test_flux_combination_uses_harmonic_interface_mean():
    bare, combo = _discretised_divergences(patched=True)
    assert np.allclose(bare, combo, rtol=1e-12, atol=1e-12)


def test_stock_pybamm_averages_flux_combinations_arithmetically():
    """Documents the stock behaviour the patch corrects (fails if PyBaMM changes it)."""
    bare, combo = _discretised_divergences(patched=False)
    assert not np.allclose(bare, combo)
    assert np.allclose(bare[:5], combo[:5]) and np.allclose(bare[-5:], combo[-5:])   # interface cells only


def test_fit_checkpoint_keeps_the_best_candidate(tmp_path, monkeypatch):
    """parallel_batches writes the best-so-far candidate (ORACLE_FIT_CHECKPOINT), so a
    fit killed at the job time limit still leaves its parameters."""
    import json

    from battery_oracle import patches

    path = tmp_path / "cell.checkpoint.json"
    monkeypatch.setenv("ORACLE_FIT_CHECKPOINT", str(path))
    patches._BEST.clear()
    patches._BEST["cost"] = np.inf
    patches._checkpoint_best([{"a": 1.0}, {"a": np.array([2.0])}], np.array([3.0, 1.0]))
    first = json.loads(path.read_text())
    assert first["best_cost"] == 1.0 and first["best_inputs"] == {"a": 2.0}
    patches._checkpoint_best([{"a": 5.0}], np.array([4.0]))            # worse: unchanged
    assert json.loads(path.read_text())["best_inputs"] == {"a": 2.0}
    patches._checkpoint_best([{"a": 7.0}], np.array([np.inf]))         # failed: unchanged
    assert json.loads(path.read_text())["best_cost"] == 1.0
