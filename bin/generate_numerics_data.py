#!/usr/bin/env python3
"""Generate the data + figures behind the numerical-stability docs page.

For each instability the oracle works around, this reproduces the behaviour
**with the workaround disabled vs enabled** and writes a summary record plus a
figure into ``docs/_static/numerics/``. PNGs under ``docs/`` are git-whitelisted,
so the figures commit and ``docs/numerics.md`` embeds them.

Crash-prone demonstrations (particle-cracking init failure, the AutoEIS/JAX XLA
cache, mpire fork corruption) run in **isolated subprocesses** so a hard crash is
recorded as an exit code instead of aborting the whole run. Each worker writes its
results incrementally, so partial data survives a crash.

Usage
-----
    python bin/generate_numerics_data.py --all
    python bin/generate_numerics_data.py --only solver_family
    python bin/generate_numerics_data.py --all --skip-crashy
    # internal (subprocess) entry point:
    python bin/generate_numerics_data.py --worker jax_cache --worker-out /tmp/x.json

This is a manual dev tool: it is not imported by the package and is not run in CI.
It needs the core install; the ``jax_cache`` / ``mpire_fork`` demos additionally
need the ``[autoeis]`` extra (they are skipped with a note otherwise).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / "docs" / "_static" / "numerics"
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Slipstream palette for consistent figures (fall back to defaults if unavailable).
try:
    from battery_oracle._plotting import slipstream
    C_FAIL, C_OK = slipstream(0.9), slipstream(0.4)
except Exception:  # pragma: no cover
    C_FAIL, C_OK = "#F14124", "#4E67C8"

plt.rcParams.update({"font.family": "serif", "figure.dpi": 120,
                     "axes.grid": True, "grid.alpha": 0.3})


# ---------------------------------------------------------------------------
# In-process demos (safe — no crash risk)
# ---------------------------------------------------------------------------
def demo_solver_family(outdir: Path) -> dict:
    """CC->CV control-mode switch: IDAKLUSolver vs CasadiSolver(mode='safe').

    The algebraic constraint at the CC->CV boundary is where IDAKLU reproducibly
    trips IDA_ERR_FAIL; CasADi 'safe' integrates through it. We solve the same
    experiment with each and record which completed, how many steps, and the time.
    """
    import pybamm

    model = pybamm.lithium_ion.SPMe()
    experiment = pybamm.Experiment([
        "Discharge at C/2 until 3.0 V",
        "Charge at C/2 until 4.2 V",
        "Hold at 4.2 V until C/20",   # <- CC->CV switch + CV hold
    ])
    record = {"name": "solver_family", "solvers": {}}
    traces = {}
    for label, make in (
        ("IDAKLUSolver", lambda: pybamm.IDAKLUSolver(rtol=1e-3, atol=1e-6)),
        ("CasadiSolver(safe)", lambda: pybamm.CasadiSolver(mode="safe", dt_max=60.0,
                                                           rtol=1e-3, atol=1e-6)),
    ):
        entry = {"completed": False, "n_steps": None, "solve_s": None, "error": None}
        t0 = time.time()
        try:
            sim = pybamm.Simulation(model, experiment=experiment, solver=make())
            sol = sim.solve()
            entry["completed"] = True
            entry["n_steps"] = int(len(sol.cycles[-1].steps)) if sol.cycles else 0
            traces[label] = (np.asarray(sol["Time [s]"].entries) / 3600.0,
                             np.asarray(sol["Terminal voltage [V]"].entries))
        except Exception as exc:  # noqa: BLE001 - we want to record any failure
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["solve_s"] = round(time.time() - t0, 2)
        record["solvers"][label] = entry

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, (t, v) in traces.items():
        c = C_OK if "Casadi" in label else C_FAIL
        ax.plot(t, v, "-", color=c, lw=1.3, label=f"{label} ({record['solvers'][label]['solve_s']}s)")
    ax.set_xlabel("time [h]"); ax.set_ylabel("terminal voltage [V]")
    ax.set_title("CC→CV switch: solver family comparison"); ax.legend(fontsize=8)
    _savefig(fig, outdir / "solver_family.png")
    return record


def demo_silent_truncation(outdir: Path) -> dict:
    """A truncated solution reports fewer steps than requested.

    PyBaMM catches internal solver errors on its own callback path and returns
    whatever integrated, WITHOUT raising — so a truncated cycle looks 'successful'
    unless the completed step count is compared to the requested count (the oracle's
    _is_truncated guard). We report the requested vs completed step counts.
    """
    import pybamm

    model = pybamm.lithium_ion.SPMe()
    steps = ["Discharge at C/2 until 3.0 V", "Rest for 600 seconds",
             "Charge at C/2 until 4.2 V", "Rest for 600 seconds"]
    experiment = pybamm.Experiment([tuple(steps)])
    expected = len(steps)
    record = {"name": "silent_truncation", "expected_steps": expected,
              "completed_steps": None, "silently_truncated": None, "error": None}
    try:
        sim = pybamm.Simulation(model, experiment=experiment,
                                solver=pybamm.CasadiSolver(mode="safe", dt_max=60.0))
        sol = sim.solve(initial_soc=0.8)
        completed = int(len(sol.cycles[-1].steps)) if sol.cycles else 0
        record["completed_steps"] = completed
        record["silently_truncated"] = completed < expected
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"

    fig, ax = plt.subplots(figsize=(5, 3.6))
    vals = [record["expected_steps"], record["completed_steps"] or 0]
    ax.bar(["requested", "completed"], vals, color=[C_OK, C_FAIL if record["silently_truncated"] else C_OK])
    ax.set_ylabel("steps in final cycle")
    ax.set_title("Silent truncation: step-count guard")
    _savefig(fig, outdir / "silent_truncation.png")
    return record


# ---------------------------------------------------------------------------
# Subprocess (crash-prone) demos — dispatched via --worker
# ---------------------------------------------------------------------------
def worker_cracking(out: Path) -> None:
    """Particle cracking + Chen2020 OCP: reproduce the IDA init failure."""
    import pybamm
    rec = {"name": "cracking_regression", "with_cracking": {}, "without_cracking": {}}
    # without cracking — the oracle's actual config: completes
    try:
        m = pybamm.lithium_ion.SPMe(options={"SEI": "ec reaction limited"})
        sim = pybamm.Simulation(m, experiment=pybamm.Experiment(
            [("Discharge at C/2 until 3.0 V", "Charge at C/2 until 4.2 V")]),
            solver=pybamm.CasadiSolver(mode="safe", dt_max=60.0))
        sim.solve(initial_soc=0.8)
        rec["without_cracking"] = {"completed": True, "error": None}
    except Exception as exc:  # noqa: BLE001
        rec["without_cracking"] = {"completed": False, "error": f"{type(exc).__name__}: {exc}"}
    out.write_text(json.dumps(rec))  # checkpoint before the risky path

    # with cracking on Chen2020 — expected to fail at init (IDA_BAD_K) or complain
    try:
        m = pybamm.lithium_ion.SPMe(options={
            "SEI": "ec reaction limited",
            "particle mechanics": "swelling and cracking",
        })
        pv = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(m, parameter_values=pv, experiment=pybamm.Experiment(
            [("Discharge at C/2 until 3.0 V", "Charge at C/2 until 4.2 V")]),
            solver=pybamm.IDAKLUSolver(rtol=1e-3, atol=1e-6))
        sim.solve(initial_soc=0.8)
        rec["with_cracking"] = {"completed": True, "error": None}
    except Exception as exc:  # noqa: BLE001
        rec["with_cracking"] = {"completed": False, "error": f"{type(exc).__name__}: {exc}"}
    out.write_text(json.dumps(rec))


def worker_jax_cache(out: Path) -> None:
    """AutoEIS/JAX XLA cache: RSS growth over repeated inferences w/ vs w/o clear_caches."""
    import resource

    rec = {"name": "jax_cache", "available": False, "with_clear": [], "without_clear": [], "note": None}
    try:
        from battery_oracle.oracle import _AUTOEIS_AVAILABLE, _autoeis_ecm
    except Exception as exc:  # noqa: BLE001
        rec["note"] = f"import failed: {exc}"; out.write_text(json.dumps(rec)); return
    if not _AUTOEIS_AVAILABLE:
        rec["note"] = "autoeis not installed; run with `uv run --extra autoeis`."
        out.write_text(json.dumps(rec)); return
    rec["available"] = True

    import jax  # noqa: F401 - imported for clear_caches
    freq = np.logspace(-2, 4, 40)
    # a simple synthetic Randles-ish spectrum
    Z = 0.05 + 0.02 / (1 + 1j * freq / 10.0) + 0.015 / (1 + 1j * freq / 0.5)

    def rss_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # kB->MB on Linux

    n_iter = 25
    for clear in (False, True):
        key = "with_clear" if clear else "without_clear"
        for i in range(n_iter):
            try:
                _autoeis_ecm(freq, Z.real, Z.imag)
            except Exception as exc:  # noqa: BLE001
                rec[key].append({"iter": i, "rss_mb": rss_mb(), "error": f"{type(exc).__name__}"})
                out.write_text(json.dumps(rec)); break
            if clear:
                import jax as _jax
                _jax.clear_caches()
            rec[key].append({"iter": i, "rss_mb": round(rss_mb(), 1)})
            out.write_text(json.dumps(rec))  # checkpoint every iteration (survives a crash)
    out.write_text(json.dumps(rec))


def worker_mpire_fork(out: Path) -> None:
    """mpire fork vs spawn: best-effort record (fork corruption is nondeterministic)."""
    rec = {"name": "mpire_fork", "available": False, "note": None}
    try:
        import autoeis  # noqa: F401
        import mpire  # noqa: F401
        rec["available"] = True
        rec["note"] = (
            "battery_oracle patches mpire.WorkerPool to start_method='spawn' at import "
            "(oracle.py), so forked SUNDIALS/CasADi JIT state is never inherited. This "
            "demo is documentation-only: the fork-time corruption is nondeterministic and "
            "environment-dependent, so it is described rather than asserted."
        )
    except Exception as exc:  # noqa: BLE001
        rec["note"] = f"autoeis/mpire not installed ({exc}); demo skipped."
    out.write_text(json.dumps(rec))


WORKERS = {
    "cracking_regression": worker_cracking,
    "jax_cache": worker_jax_cache,
    "mpire_fork": worker_mpire_fork,
}


def run_subprocess_demo(name: str, outdir: Path, timeout: int = 900) -> dict:
    """Run a crash-prone demo in a subprocess; return its record + exit status."""
    tmp = outdir / f"_worker_{name}.json"
    tmp.write_text(json.dumps({"name": name, "note": "worker did not write"}))
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", name,
         "--worker-out", str(tmp)],
        capture_output=True, text=True, timeout=timeout,
    )
    try:
        rec = json.loads(tmp.read_text())
    except Exception:  # noqa: BLE001
        rec = {"name": name, "note": "no worker output"}
    rec["subprocess_returncode"] = proc.returncode
    rec["subprocess_crashed"] = proc.returncode < 0 or proc.returncode not in (0,)
    if proc.returncode != 0:
        rec["subprocess_stderr_tail"] = proc.stderr[-500:]
    tmp.unlink(missing_ok=True)

    # figure for the jax cache demo
    if name == "jax_cache" and (rec.get("with_clear") or rec.get("without_clear")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for key, c, lab in (("without_clear", C_FAIL, "without clear_caches"),
                            ("with_clear", C_OK, "with clear_caches")):
            series = [(d["iter"], d.get("rss_mb")) for d in rec.get(key, []) if d.get("rss_mb")]
            if series:
                xs, ys = zip(*series)
                ax.plot(xs, ys, "-o", color=c, ms=3, label=lab)
        ax.set_xlabel("AutoEIS inference #"); ax.set_ylabel("max RSS [MB]")
        ax.set_title("JAX/XLA cache growth"); ax.legend(fontsize=8)
        _savefig(fig, outdir / "jax_cache.png")
    return rec


# ---------------------------------------------------------------------------
def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


IN_PROCESS = {"solver_family": demo_solver_family, "silent_truncation": demo_silent_truncation}
CRASHY = ("cracking_regression", "jax_cache", "mpire_fork")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="run every demo (default if none selected)")
    ap.add_argument("--only", action="append", default=[],
                    help="run only the named demo(s); repeatable")
    ap.add_argument("--skip-crashy", action="store_true",
                    help="skip the subprocess (crash-prone) demos")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--worker", help=argparse.SUPPRESS)      # internal subprocess entry
    ap.add_argument("--worker-out", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    # Subprocess worker mode
    if args.worker:
        WORKERS[args.worker](Path(args.worker_out))
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    selected = args.only or list(IN_PROCESS) + list(CRASHY)
    if args.skip_crashy:
        selected = [s for s in selected if s not in CRASHY]

    results = {}
    for name in selected:
        print(f"[{name}]")
        if name in IN_PROCESS:
            results[name] = IN_PROCESS[name](outdir)
        elif name in CRASHY:
            results[name] = run_subprocess_demo(name, outdir)
        else:
            print(f"  unknown demo {name!r}; skipping"); continue

    data_path = outdir / "numerics_data.json"
    data_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {data_path.relative_to(REPO_ROOT)} ({len(results)} demo(s))")


if __name__ == "__main__":
    main()
