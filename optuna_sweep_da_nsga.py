#!/usr/bin/env python3
"""
Single-objective Optuna sweep over DA hyperparameters (TPE sampler).

Default target is Rhonegletscher (OGGM/COP-DEM pipeline,
``params_oggm_rhone_da.yaml``); pass ``--experiment <name>`` to sweep a
different DA config (e.g. ``params_oggm_aletsch_da``). The sweep dir and
Optuna study name are derived from the experiment name, so different targets
don't clobber each other's databases.

(Filename retained as ``optuna_sweep_da_nsga.py`` for backward compatibility
with the pipeline; the implementation is now single-objective TPE, not
NSGA-II.)

Search space (log-uniform):
    processes.iceflow.physics.init_slidingco : [0.05, 2.0]
        ↑ τ_ref (reference basal shear stress in MPa for u_ref=100 m/yr).
          HIGHER value = stiffer bed = LESS sliding.
    processes.iceflow.physics.init_arrhenius : [30, 200]
        ↑ flow-law constant A (MPa^-3 yr^-1). IGM default 78 (~temperate).
          LOWER = colder/stiffer ice.

Neither parameter is optimized by DA itself — both are fixed scalars at the
values picked here. ``regularization.thk`` is NOT swept either; it is chosen
separately via an L-curve in step 2.2 of the pipeline.

Single cost term (minimized, last row of costs.dat):
    velsurf — surface-velocity misfit (m yr⁻¹ scale, see velsurfobs_std).

Thickness observations are intentionally NOT used: with ``cost_list=[velsurf,
icemask]`` (no ``thk``) the DA inverts thickness purely from velocity and
outline matching, regularized by ``reg.thk``.

Outputs
-------
    sweep_results_<experiment>/
        optuna_study.db
        trial_XXXX/                 # one per evaluated candidate
        sweep_summary.json
        plots/
            velsurf_vs_params.png
            param_landscape.png
            convergence.png

Usage
-----
    conda activate igm-pretrain
    python optuna_sweep_da_nsga.py                                # Rhone default, 30 trials
    python optuna_sweep_da_nsga.py --n-trials 60
    python optuna_sweep_da_nsga.py --experiment params_oggm_aletsch_da
    python optuna_sweep_da_nsga.py --plots-only
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
from optuna.samplers import TPESampler

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Defaults (overridable via CLI) ──────────────────────────────────────────

DEFAULT_EXPERIMENT = "params_oggm_rhone_da"
DEFAULT_N_TRIALS = 30
DEFAULT_SEED = 42

# Search bounds for the two swept iceflow scalars (log-uniform).
# The Arrhenius upper end is 120: values above that are not physical for these
# Alpine glaciers. The earlier 200 let Aletsch settle at 182, which fits the
# velocities but is not a defensible ice rheology.
DEFAULT_SLIDINGCO_BOUNDS = (0.05, 2.0)
DEFAULT_ARRHENIUS_BOUNDS = (30.0, 120.0)


def _experiment_paths(experiment: str):
    sweep_dir = os.path.join(PROJECT_DIR, f"sweep_results_{experiment}")
    return {
        "sweep_dir": sweep_dir,
        "db_path": os.path.join(sweep_dir, "optuna_study.db"),
        "plots_dir": os.path.join(sweep_dir, "plots"),
        "study_name": f"da_tpe_{experiment}",
    }


# Last row of costs.dat columns we treat as objectives. With cost_list=[velsurf,
# icemask] (no thk) and control_list=[thk] the costs.dat header is typically:
#     velsurf  icemask  thk_regu  glen
# We minimise the final velsurf term; the others are kept as user_attrs.
DA_COST_COLUMNS = ["velsurf"]


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


def objective(trial: optuna.Trial, gpu_id: int, sweep_dir: str, experiment: str,
              bounds=None):
    slid_lo, slid_hi = (bounds or {}).get("slidingco", DEFAULT_SLIDINGCO_BOUNDS)
    arrh_lo, arrh_hi = (bounds or {}).get("arrhenius", DEFAULT_ARRHENIUS_BOUNDS)
    params = {
        "init_slidingco": trial.suggest_float("init_slidingco", slid_lo, slid_hi, log=True),
        "init_arrhenius": trial.suggest_float("init_arrhenius", arrh_lo, arrh_hi, log=True),
    }
    trial.set_user_attr("gpu_id", gpu_id)

    trial_dir = os.path.join(sweep_dir, f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    overrides = [
        f"+experiment={experiment}",
        # control_list=[thk] is already set in the DA config; keep this
        # override as a defensive belt-and-braces guard in case the YAML
        # is re-edited.
        "processes.data_assimilation.control_list=[thk]",
        f"processes.iceflow.physics.init_slidingco={params['init_slidingco']}",
        f"processes.iceflow.physics.init_arrhenius={params['init_arrhenius']}",
        f"hydra.run.dir={trial_dir}",
    ]

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
        return float("inf")

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
        return float("inf")

    histories, header = _read_da_costs(trial_dir)
    missing = [c for c in DA_COST_COLUMNS if c not in histories]
    if missing:
        print(
            f"[trial {trial.number} | gpu {gpu_id}] missing {missing} "
            f"in costs.dat (found columns: {header})",
            flush=True,
        )
        _save_meta(trial_dir, params, "missing_output", elapsed)
        return float("inf")

    final_velsurf = float(histories["velsurf"][-1])

    for c, hist in histories.items():
        trial.set_user_attr(f"history_{c}", hist)

    _save_meta(
        trial_dir,
        params,
        "completed",
        elapsed,
        final_velsurf=final_velsurf,
        n_iters=len(histories["velsurf"]),
    )

    print(
        f"[trial {trial.number} | gpu {gpu_id}] done  "
        f"velsurf={final_velsurf:.4f}  time={elapsed:.0f}s",
        flush=True,
    )

    return final_velsurf


def _save_meta(trial_dir, params, status, elapsed, **extra):
    meta = {**params, "status": status, "elapsed_s": round(elapsed, 1), **extra}
    with open(os.path.join(trial_dir, "trial_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ── Parallel runner ─────────────────────────────────────────────────────────


_study_lock = threading.Lock()


def _run_one_trial(study: optuna.Study, gpu_id: int, sweep_dir: str, experiment: str,
                   bounds=None):
    with _study_lock:
        trial = study.ask()

    try:
        value = objective(trial, gpu_id=gpu_id, sweep_dir=sweep_dir,
                          experiment=experiment, bounds=bounds)
        with _study_lock:
            study.tell(trial, value)
    except Exception as exc:
        with _study_lock:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        print(
            f"[trial {trial.number} | gpu {gpu_id}] EXCEPTION: {exc!r}",
            flush=True,
        )


def _worker_loop(study, gpu_id, work_q, sweep_dir, experiment, bounds=None):
    while True:
        try:
            work_q.get_nowait()
        except queue.Empty:
            return
        try:
            _run_one_trial(study, gpu_id, sweep_dir, experiment, bounds=bounds)
        finally:
            work_q.task_done()


def run_parallel(study, n_trials, gpus, workers_per_gpu, sweep_dir, experiment,
                 bounds=None):
    work_q: queue.Queue = queue.Queue()
    for _ in range(n_trials):
        work_q.put(None)

    threads = []
    for gpu_id in gpus:
        for w in range(workers_per_gpu):
            t = threading.Thread(
                target=_worker_loop,
                args=(study, gpu_id, work_q, sweep_dir, experiment, bounds),
                name=f"gpu{gpu_id}-w{w}",
                daemon=True,
            )
            t.start()
            threads.append(t)

    for t in threads:
        t.join()


# ── Plots ────────────────────────────────────────────────────────────────────


PARAM_NAMES = ["init_slidingco", "init_arrhenius"]
LOG_PARAMS = {"init_slidingco", "init_arrhenius"}


def make_plots(study: optuna.Study, plots_dir: str):
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials — skipping plots.")
        return

    best_number = study.best_trial.number
    rows = []
    for t in completed:
        rows.append({
            "trial": t.number,
            **{p: t.params[p] for p in PARAM_NAMES},
            "velsurf": t.value,
            "is_best": t.number == best_number,
        })
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0].keys()}
    best = arr["is_best"].astype(bool)

    # 1. Search-space scatter: init_slidingco × init_arrhenius, colored by velsurf
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(arr["init_slidingco"], arr["init_arrhenius"],
                    c=arr["velsurf"], cmap="viridis_r", s=60,
                    edgecolor="gray", linewidth=0.3)
    ax.scatter(arr["init_slidingco"][best], arr["init_arrhenius"][best],
               s=200, marker="*", edgecolor="red", linewidth=1.5,
               facecolor="none", label=f"best (trial {best_number})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("init_slidingco (MPa)")
    ax.set_ylabel("init_arrhenius (MPa⁻³ yr⁻¹)")
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label("final velsurf cost")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title("Search-space landscape")
    fig.tight_layout()
    path = os.path.join(plots_dir, "param_landscape.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  saved {path}")

    # 2. velsurf vs each param (1×2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, p in zip(axes, PARAM_NAMES):
        ax.scatter(arr[p], arr["velsurf"], s=30, alpha=0.5,
                   color="steelblue", edgecolor="none")
        ax.scatter(arr[p][best], arr["velsurf"][best], s=120, marker="*",
                   color="red", edgecolor="k", linewidth=0.5,
                   label=f"best (trial {best_number})")
        ax.set_xlabel(p)
        if p in LOG_PARAMS: ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("final velsurf cost")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle("Final velsurf cost vs each parameter")
    fig.tight_layout()
    path = os.path.join(plots_dir, "velsurf_vs_params.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  saved {path}")

    # 3. velsurf history per trial; best in red.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for t in completed:
        is_b = t.number == best_number
        col = "red" if is_b else "gray"
        alpha = 0.9 if is_b else 0.25
        lw = 1.2 if is_b else 0.4
        hist = t.user_attrs.get("history_velsurf", [])
        if hist:
            ax.plot(hist, color=col, lw=lw, alpha=alpha)
    ax.plot([], [], color="red", lw=1.2, label=f"best (trial {best_number})")
    ax.plot([], [], color="gray", lw=0.4, label="others")
    ax.set_xlabel("DA iteration"); ax.set_ylabel("velsurf cost")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("velsurf convergence per trial")
    fig.tight_layout()
    path = os.path.join(plots_dir, "convergence.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Single-objective TPE DA sweep over 2 hyperparameters "
                    "(init_slidingco, init_arrhenius) → 1 cost (velsurf)."
    )
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT,
                        help=f"Hydra experiment name (default: {DEFAULT_EXPERIMENT})")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gpus", type=str, default="0,1",
                        help="Comma-separated GPU ids (default: 0,1).")
    parser.add_argument("--workers-per-gpu", type=int, default=3,
                        help="Concurrent trials per GPU (default: 3).")
    parser.add_argument("--arrhenius-min", type=float,
                        default=DEFAULT_ARRHENIUS_BOUNDS[0])
    parser.add_argument("--arrhenius-max", type=float,
                        default=DEFAULT_ARRHENIUS_BOUNDS[1],
                        help=f"Upper bound on init_arrhenius "
                             f"(default: {DEFAULT_ARRHENIUS_BOUNDS[1]:g}). The "
                             f"study DB records the sampled distribution, so "
                             f"changing this needs a fresh sweep dir.")
    parser.add_argument("--slidingco-min", type=float,
                        default=DEFAULT_SLIDINGCO_BOUNDS[0])
    parser.add_argument("--slidingco-max", type=float,
                        default=DEFAULT_SLIDINGCO_BOUNDS[1])
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip running trials; just regenerate plots.")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    bounds = {"slidingco": (args.slidingco_min, args.slidingco_max),
              "arrhenius": (args.arrhenius_min, args.arrhenius_max)}

    paths = _experiment_paths(args.experiment)
    sweep_dir = paths["sweep_dir"]; db_path = paths["db_path"]
    plots_dir = paths["plots_dir"]; study_name = paths["study_name"]
    os.makedirs(sweep_dir, exist_ok=True)

    storage = f"sqlite:///{db_path}"
    sampler = TPESampler(seed=args.seed)

    study = optuna.create_study(
        study_name=study_name, sampler=sampler,
        direction="minimize", storage=storage, load_if_exists=True,
    )

    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    total_workers = len(gpus) * args.workers_per_gpu

    print("DA sweep — TPE (velsurf)")
    print(f"  Experiment         : {args.experiment}")
    print(f"  Study name         : {study_name}")
    print(f"  Target n_trials    : {args.n_trials}")
    print(f"  Already completed  : {n_done}")
    print(f"  GPUs               : {gpus}")
    print(f"  Workers per GPU    : {args.workers_per_gpu}")
    print(f"  init_slidingco     : [{args.slidingco_min:g}, {args.slidingco_max:g}]")
    print(f"  init_arrhenius     : [{args.arrhenius_min:g}, {args.arrhenius_max:g}]")
    print(f"  Concurrent trials  : {total_workers}")
    print(f"  Results directory  : {sweep_dir}")
    print()

    n_remaining = max(args.n_trials - n_done, 0)
    if not args.plots_only and n_remaining > 0:
        run_parallel(study, n_trials=n_remaining, gpus=gpus,
                     workers_per_gpu=args.workers_per_gpu,
                     sweep_dir=sweep_dir, experiment=args.experiment,
                     bounds=bounds)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best = study.best_trial
        rows = [{
            "trial": t.number,
            "init_slidingco": t.params["init_slidingco"],
            "init_arrhenius": t.params["init_arrhenius"],
            "velsurf": t.value,
            "is_best": t.number == best.number,
        } for t in completed]
        rows.sort(key=lambda r: r["velsurf"])

        print(f"\n{'=' * 80}")
        print(f"  BEST TRIAL (of {len(completed)})")
        print(f"{'=' * 80}")
        print(f"  trial {best.number}: velsurf={best.value:.4f}")
        print(f"      init_slidingco = {best.params['init_slidingco']:.4f}")
        print(f"      init_arrhenius = {best.params['init_arrhenius']:.3f}")
        print(f"\n  Top 5 trials:")
        print(f"  {'trial':>5}  {'init_slid':>9}  {'init_arrh':>9}  {'velsurf':>9}")
        for r in rows[:5]:
            mark = "*" if r["is_best"] else " "
            print(f"  {r['trial']:>5}{mark} {r['init_slidingco']:>9.4f}  "
                  f"{r['init_arrhenius']:>9.3f}  {r['velsurf']:>9.4f}")

        with open(os.path.join(sweep_dir, "sweep_summary.json"), "w") as f:
            json.dump({
                "experiment": args.experiment,
                "sampler": "TPESampler",
                "seed": args.seed,
                "best_trial": best.number,
                "best_params": dict(best.params),
                "best_velsurf": best.value,
                "completed_trials": len(completed),
                "results": rows,
            }, f, indent=2)

        make_plots(study, plots_dir)


if __name__ == "__main__":
    main()
