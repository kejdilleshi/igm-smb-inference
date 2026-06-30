#!/usr/bin/env python3
"""
Multi-objective Optuna sweep over the two *scalar* DA physics parameters,
scored jointly on the surface-velocity AND the thickness misfit.

This is the thickness-aware sibling of ``optuna_sweep_da_nsga.py``. The latter
is single-objective (velsurf only) and is used for the OGGM glaciers (rhone,
aletsch) whose DA cost has no thickness term. This script is for glaciers that
DO carry measured thickness observations (e.g. Glacier d'Argentière, with a
radar/GPR ``thkobs`` profile) and lets BOTH the velocity profile and the
thickness profile decide the optimal scalars.

What it does NOT do: it never lets DA optimize slidingco/arrhenius as 2D fields.
``control_list=[thk]`` is forced, so the two parameters stay fixed uniform
scalars at the values picked per-trial — only ice thickness is inverted. The
output is the pair of scalars to plug into the final combined run.

Search space (log-uniform):
    processes.iceflow.physics.init_slidingco : [0.05, 2.0]
        ↑ τ_ref (reference basal shear stress in MPa for u_ref=100 m/yr).
          HIGHER = stiffer bed = LESS sliding.
    processes.iceflow.physics.init_arrhenius : [30, 200]
        ↑ flow-law constant A (MPa^-3 yr^-1). IGM default 78 (~temperate).
          LOWER = colder/stiffer ice.

Two objectives (both minimized, last row of costs.dat):
    velsurf — surface-velocity misfit (vs uvelsurfobs/vvelsurfobs)
    thk     — thickness misfit        (vs thkobs / the GPR profile)

These are NOT comparable in absolute units, so we do NOT weight-sum them.
NSGA-II works on Pareto dominance (scale-invariant per objective). The single
recommended pick is the "balanced knee": min-max normalize each objective over
the completed trials, then take the trial closest to the ideal corner (0, 0).

The DA fitting weights / regularization come straight from the ``_da`` YAML
(``thkobs_std``, ``velsurfobs_std``, ``regularization.thk``) — i.e. the same
weighting the existing pipeline already uses — so the relative velsurf/thk
tradeoff matches that configuration. Override ``--thkobs-std`` / ``--reg-thk``
if you want a different balance.

Outputs
-------
    sweep_results_<experiment>_velthk/
        optuna_study.db
        trial_XXXX/                 # one per evaluated candidate
        sweep_summary.json          # pareto front + recommended pick
        plots/
            pareto_front.png
            param_landscape.png     # one panel per objective
            convergence.png

Usage
-----
    conda activate igm-pretrain
    python optuna_sweep_da_velthk.py                       # argentiere, 40 trials
    python optuna_sweep_da_velthk.py --n-trials 60
    python optuna_sweep_da_velthk.py --experiment params_oggm_argentiere_da
    python optuna_sweep_da_velthk.py --plots-only
"""

import argparse
import json
import os
import queue
import subprocess
import threading
import time

import numpy as np
import optuna
from optuna.samplers import NSGAIISampler

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Defaults (overridable via CLI) ──────────────────────────────────────────

DEFAULT_EXPERIMENT = "params_oggm_argentiere_velthk_da"
DEFAULT_N_TRIALS = 40
DEFAULT_SEED = 42
# NSGA-II population: small so ~3 generations fit in the default trial budget
# (2 continuous params → simple front; evolutionary pressure still helps).
DEFAULT_POP_SIZE = 12

# Last-row costs.dat columns used as the two objectives, in order.
OBJECTIVES = ["velsurf", "thk"]


def _experiment_paths(experiment: str):
    sweep_dir = os.path.join(PROJECT_DIR, f"sweep_results_{experiment}_velthk")
    return {
        "sweep_dir": sweep_dir,
        "db_path": os.path.join(sweep_dir, "optuna_study.db"),
        "plots_dir": os.path.join(sweep_dir, "plots"),
        "study_name": f"da_velthk_{experiment}",
    }


# ── Metric extraction ───────────────────────────────────────────────────────


def _read_da_costs(trial_dir: str):
    costs_path = os.path.join(trial_dir, "costs.dat")
    if not os.path.exists(costs_path):
        return {}, []

    with open(costs_path) as f:
        header = f.readline().split()

    data = np.loadtxt(costs_path, skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    histories = {name: data[:, i].tolist() for i, name in enumerate(header)}
    return histories, header


# ── Objective ───────────────────────────────────────────────────────────────


def objective(trial: optuna.Trial, gpu_id: int, sweep_dir: str, args):
    params = {
        "init_slidingco": trial.suggest_float("init_slidingco", 0.05, 2.0, log=True),
        "init_arrhenius": trial.suggest_float("init_arrhenius", 30.0, 200.0, log=True),
    }
    trial.set_user_attr("gpu_id", gpu_id)

    trial_dir = os.path.join(sweep_dir, f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    overrides = [
        f"+experiment={args.experiment}",
        # Force scalars: only thickness is a DA control. slidingco/arrhenius
        # stay fixed at the suggested values (the whole point of this sweep).
        "processes.data_assimilation.control_list=[thk]",
        f"processes.iceflow.physics.init_slidingco={params['init_slidingco']}",
        f"processes.iceflow.physics.init_arrhenius={params['init_arrhenius']}",
        f"hydra.run.dir={trial_dir}",
    ]
    if args.reg_thk is not None:
        overrides.append(
            f"processes.data_assimilation.regularization.thk={args.reg_thk}")
    if args.thkobs_std is not None:
        overrides.append(
            f"processes.data_assimilation.fitting.thkobs_std={args.thkobs_std}")
    if args.da_nbitmax is not None:
        overrides.append(
            f"processes.data_assimilation.optimization.nbitmax={args.da_nbitmax}")

    cmd = ["igm_run"] + overrides

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    print(
        f"\n[trial {trial.number} | gpu {gpu_id}] start  "
        f"init_slid={params['init_slidingco']:.3f}  "
        f"init_arrh={params['init_arrhenius']:.2f}",
        flush=True,
    )

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=7200,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print(f"[trial {trial.number} | gpu {gpu_id}] TIMED OUT", flush=True)
        _save_meta(trial_dir, params, "timeout", time.time() - t0)
        return float("inf"), float("inf")

    elapsed = time.time() - t0

    with open(os.path.join(trial_dir, "stdout.log"), "w") as f:
        f.write(result.stdout)
    with open(os.path.join(trial_dir, "stderr.log"), "w") as f:
        f.write(result.stderr)

    if result.returncode != 0:
        print(
            f"[trial {trial.number} | gpu {gpu_id}] FAILED "
            f"(exit {result.returncode})",
            flush=True,
        )
        tail = result.stderr[-500:] if result.stderr else result.stdout[-500:]
        print(f"    {tail}", flush=True)
        _save_meta(trial_dir, params, "failed", elapsed)
        return float("inf"), float("inf")

    histories, header = _read_da_costs(trial_dir)
    missing = [c for c in OBJECTIVES if c not in histories]
    if missing:
        print(
            f"[trial {trial.number} | gpu {gpu_id}] missing {missing} "
            f"in costs.dat (found columns: {header})",
            flush=True,
        )
        _save_meta(trial_dir, params, "missing_output", elapsed)
        return float("inf"), float("inf")

    final = {c: float(histories[c][-1]) for c in OBJECTIVES}

    for c, hist in histories.items():
        trial.set_user_attr(f"history_{c}", hist)
    for c, v in final.items():
        trial.set_user_attr(f"final_{c}", v)

    _save_meta(
        trial_dir,
        params,
        "completed",
        elapsed,
        final_velsurf=final["velsurf"],
        final_thk=final["thk"],
        n_iters=len(histories["velsurf"]),
    )

    print(
        f"[trial {trial.number} | gpu {gpu_id}] done  "
        f"velsurf={final['velsurf']:.4f}  thk={final['thk']:.4f}  "
        f"time={elapsed:.0f}s",
        flush=True,
    )

    return final["velsurf"], final["thk"]


def _save_meta(trial_dir, params, status, elapsed, **extra):
    meta = {**params, "status": status, "elapsed_s": round(elapsed, 1), **extra}
    with open(os.path.join(trial_dir, "trial_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ── Parallel runner ─────────────────────────────────────────────────────────


_study_lock = threading.Lock()


def _run_one_trial(study: optuna.Study, gpu_id: int, sweep_dir: str, args):
    with _study_lock:
        trial = study.ask()

    try:
        values = objective(trial, gpu_id=gpu_id, sweep_dir=sweep_dir, args=args)
        with _study_lock:
            study.tell(trial, values)
    except Exception as exc:
        with _study_lock:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        print(
            f"[trial {trial.number} | gpu {gpu_id}] EXCEPTION: {exc!r}",
            flush=True,
        )


def _worker_loop(study, gpu_id, work_q, sweep_dir, args):
    while True:
        try:
            work_q.get_nowait()
        except queue.Empty:
            return
        try:
            _run_one_trial(study, gpu_id, sweep_dir, args)
        finally:
            work_q.task_done()


def run_parallel(study, n_trials, gpus, workers_per_gpu, sweep_dir, args):
    work_q: queue.Queue = queue.Queue()
    for _ in range(n_trials):
        work_q.put(None)

    threads = []
    for gpu_id in gpus:
        for w in range(workers_per_gpu):
            t = threading.Thread(
                target=_worker_loop,
                args=(study, gpu_id, work_q, sweep_dir, args),
                name=f"gpu{gpu_id}-w{w}",
                daemon=True,
            )
            t.start()
            threads.append(t)

    for t in threads:
        t.join()


# ── Selection: balanced knee ─────────────────────────────────────────────────


def _completed(study):
    return [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and all(np.isfinite(v) for v in t.values)]


def balanced_knee(trials):
    """Min-max normalize both objectives and return the trial closest to the
    ideal corner (0, 0). Ties broken by lower velsurf.

    Degenerate Pareto points are excluded first: a solution where one objective
    is an extreme upper outlier (e.g. a near-no-sliding boundary pick with a
    huge velsurf but marginally lower thk) is non-dominated yet useless, and
    would otherwise blow up the min-max range and drag the knee toward it. We
    drop any trial above an objective's Tukey upper fence (Q3 + 1.5·IQR)."""
    if not trials:
        return None
    vel = np.array([t.values[0] for t in trials])
    thk = np.array([t.values[1] for t in trials])

    def _upper_fence(a):
        q25, q75 = np.percentile(a, [25, 75])
        return q75 + 1.5 * (q75 - q25)

    keep = (vel <= _upper_fence(vel)) & (thk <= _upper_fence(thk))
    if keep.sum() < 2:            # not enough to discriminate → keep all
        keep = np.ones_like(vel, dtype=bool)
    idx = np.where(keep)[0]
    v, t = vel[idx], thk[idx]

    def _norm(a):
        lo, hi = a.min(), a.max()
        return np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo)

    dist = np.hypot(_norm(v), _norm(t))
    order = np.lexsort((v, dist))  # primary: dist, secondary: vel
    return trials[idx[order[0]]]


# ── Plots ────────────────────────────────────────────────────────────────────


def make_plots(study, plots_dir, pick_number=None):
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)
    completed = _completed(study)
    if not completed:
        print("No completed trials — skipping plots.")
        return

    vel = np.array([t.values[0] for t in completed])
    thk = np.array([t.values[1] for t in completed])
    slid = np.array([t.params["init_slidingco"] for t in completed])
    arrh = np.array([t.params["init_arrhenius"] for t in completed])
    nums = np.array([t.number for t in completed])
    pareto_nums = {t.number for t in study.best_trials}
    on_front = np.array([n in pareto_nums for n in nums])
    is_pick = nums == pick_number if pick_number is not None else np.zeros_like(nums, bool)

    # 1. Pareto front: velsurf vs thk
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(vel[~on_front], thk[~on_front], s=45, color="lightgray",
               edgecolor="gray", linewidth=0.3, label="dominated")
    # connect the front
    fo = np.argsort(vel[on_front])
    ax.plot(vel[on_front][fo], thk[on_front][fo], "-o", color="steelblue",
            ms=7, lw=1.2, label="Pareto front")
    if is_pick.any():
        ax.scatter(vel[is_pick], thk[is_pick], s=260, marker="*",
                   facecolor="none", edgecolor="red", linewidth=1.8,
                   label=f"recommended (trial {pick_number})", zorder=5)
    ax.set_xlabel("final velsurf misfit")
    ax.set_ylabel("final thk misfit")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("DA velocity vs thickness misfit — scalar (slidingco, arrhenius)")
    fig.tight_layout()
    p = os.path.join(plots_dir, "pareto_front.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  saved {p}")

    # 2. Param landscape: slid × arrh, one panel per objective
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, vals, name in zip(axes, (vel, thk), ("velsurf", "thk")):
        sc = ax.scatter(slid, arrh, c=vals, cmap="viridis_r", s=70,
                        edgecolor="gray", linewidth=0.3)
        if is_pick.any():
            ax.scatter(slid[is_pick], arrh[is_pick], s=240, marker="*",
                       facecolor="none", edgecolor="red", linewidth=1.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("init_slidingco (MPa)")
        ax.set_ylabel("init_arrhenius (MPa⁻³ yr⁻¹)")
        plt.colorbar(sc, ax=ax).set_label(f"final {name} misfit")
        ax.grid(True, alpha=0.3)
        ax.set_title(name)
    fig.suptitle("Search-space landscape (★ = recommended)")
    fig.tight_layout()
    p = os.path.join(plots_dir, "param_landscape.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  saved {p}")

    # 3. Convergence histories (both objectives), pick highlighted
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, name in zip(axes, OBJECTIVES):
        for t in completed:
            is_p = t.number == pick_number
            hist = t.user_attrs.get(f"history_{name}", [])
            if hist:
                ax.plot(hist, color="red" if is_p else "gray",
                        lw=1.3 if is_p else 0.4, alpha=0.9 if is_p else 0.25)
        ax.set_xlabel("DA iteration"); ax.set_ylabel(f"{name} cost")
        ax.set_yscale("log"); ax.grid(True, alpha=0.3)
        ax.set_title(f"{name} convergence")
    axes[0].plot([], [], color="red", lw=1.3, label=f"recommended (trial {pick_number})")
    axes[0].plot([], [], color="gray", lw=0.4, label="others")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(plots_dir, "convergence.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  saved {p}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Multi-objective (velsurf + thk) NSGA-II DA sweep over the "
                    "two scalar physics params (init_slidingco, init_arrhenius)."
    )
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT,
                        help=f"Hydra DA experiment name (default: {DEFAULT_EXPERIMENT})")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE,
                        help=f"NSGA-II population size (default: {DEFAULT_POP_SIZE})")
    parser.add_argument("--gpus", type=str, default="0,1",
                        help="Comma-separated GPU ids (default: 0,1).")
    parser.add_argument("--workers-per-gpu", type=int, default=3,
                        help="Concurrent trials per GPU (default: 3).")
    parser.add_argument("--reg-thk", type=float, default=None,
                        help="Override regularization.thk (default: use YAML).")
    parser.add_argument("--thkobs-std", type=float, default=None,
                        help="Override fitting.thkobs_std (default: use YAML).")
    parser.add_argument("--da-nbitmax", type=int, default=None,
                        help="Override DA nbitmax (smoke tests).")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip running trials; just regenerate plots + pick.")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]

    paths = _experiment_paths(args.experiment)
    sweep_dir = paths["sweep_dir"]; db_path = paths["db_path"]
    plots_dir = paths["plots_dir"]; study_name = paths["study_name"]
    os.makedirs(sweep_dir, exist_ok=True)

    storage = f"sqlite:///{db_path}"
    sampler = NSGAIISampler(seed=args.seed, population_size=args.pop_size)

    study = optuna.create_study(
        study_name=study_name, sampler=sampler,
        directions=["minimize", "minimize"],
        storage=storage, load_if_exists=True,
    )

    n_done = len(_completed(study))
    total_workers = len(gpus) * args.workers_per_gpu

    print("DA sweep — NSGA-II (velsurf + thk)")
    print(f"  Experiment         : {args.experiment}")
    print(f"  Study name         : {study_name}")
    print(f"  Target n_trials    : {args.n_trials}")
    print(f"  Already completed  : {n_done}")
    print(f"  GPUs               : {gpus}")
    print(f"  Workers per GPU    : {args.workers_per_gpu}")
    print(f"  Concurrent trials  : {total_workers}")
    print(f"  reg.thk override   : {args.reg_thk}")
    print(f"  thkobs_std override: {args.thkobs_std}")
    print(f"  Results directory  : {sweep_dir}")
    print()

    n_remaining = max(args.n_trials - n_done, 0)
    if not args.plots_only and n_remaining > 0:
        run_parallel(study, n_trials=n_remaining, gpus=gpus,
                     workers_per_gpu=args.workers_per_gpu,
                     sweep_dir=sweep_dir, args=args)

    completed = _completed(study)
    if not completed:
        print("No completed trials.")
        return

    pick = balanced_knee(study.best_trials or completed)
    front = sorted(study.best_trials, key=lambda t: t.values[0])

    print(f"\n{'=' * 80}")
    print(f"  PARETO FRONT ({len(front)} non-dominated of {len(completed)} completed)")
    print(f"{'=' * 80}")
    print(f"  {'trial':>5}  {'init_slid':>9}  {'init_arrh':>9}  {'velsurf':>9}  {'thk':>9}")
    for t in front:
        mark = "*" if t.number == pick.number else " "
        print(f"  {t.number:>5}{mark} {t.params['init_slidingco']:>9.4f}  "
              f"{t.params['init_arrhenius']:>9.3f}  "
              f"{t.values[0]:>9.4f}  {t.values[1]:>9.4f}")
    print(f"\n  RECOMMENDED (balanced knee): trial {pick.number}")
    print(f"      init_slidingco = {pick.params['init_slidingco']:.4f}")
    print(f"      init_arrhenius = {pick.params['init_arrhenius']:.3f}")
    print(f"      velsurf = {pick.values[0]:.4f}   thk = {pick.values[1]:.4f}")

    rows = [{
        "trial": t.number,
        "init_slidingco": t.params["init_slidingco"],
        "init_arrhenius": t.params["init_arrhenius"],
        "velsurf": t.values[0],
        "thk": t.values[1],
        "on_pareto_front": t.number in {f.number for f in front},
        "is_recommended": t.number == pick.number,
    } for t in sorted(completed, key=lambda t: t.values[0])]

    with open(os.path.join(sweep_dir, "sweep_summary.json"), "w") as f:
        json.dump({
            "experiment": args.experiment,
            "sampler": "NSGAIISampler",
            "objectives": OBJECTIVES,
            "seed": args.seed,
            "completed_trials": len(completed),
            "pareto_front_size": len(front),
            "recommended": {
                "trial": pick.number,
                "init_slidingco": pick.params["init_slidingco"],
                "init_arrhenius": pick.params["init_arrhenius"],
                "velsurf": pick.values[0],
                "thk": pick.values[1],
            },
            "results": rows,
        }, f, indent=2)

    make_plots(study, plots_dir, pick_number=pick.number)
    print(f"\n  summary: {os.path.join(sweep_dir, 'sweep_summary.json')}")


if __name__ == "__main__":
    main()
