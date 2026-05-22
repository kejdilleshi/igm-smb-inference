#!/usr/bin/env python3
"""
Multi-objective Optuna NSGA-II sweep over DA hyperparameters. Default target
is Rhonegletscher (OGGM/COP-DEM pipeline, params_oggm_rhone_da.yaml); pass
`--experiment <name>` to sweep a different DA config (e.g. params_aletsch_da,
params_oggm_aletsch_da). The sweep dir and Optuna study name are derived
automatically from the experiment name, so different targets don't clobber
each other's databases.

Search space (continuous, log where appropriate):
    processes.data_assimilation.regularization.thk        : [1000, 4000] (log)
    processes.data_assimilation.fitting.thkobs_std        : [0.5, 10]    (log)
    processes.iceflow.physics.init_slidingco              : [0.05, 2.0]  (log)
        ↑ this is the τ_ref (reference basal shear stress in MPa for
          u_ref=100 m/yr). HIGHER value = stiffer bed = LESS sliding.
          Sliding is NOT optimized by DA here (control_list=[thk]); we
          sweep init_slidingco to find a good fixed scalar value.

Two objectives (both minimized, last row of costs.dat):
    1. velsurf    — surface-velocity misfit
    2. thk        — GlaThiDa thickness-observation misfit

thk_regu (the curvature regularization on thk) is intentionally *not* an
objective: it's an internal regularization term, not data fidelity, so
including it would just push the Pareto front toward over-smoothed
thickness fields that fit neither velsurf nor thk well.

NSGA-II is a genetic algorithm — it evolves a population of candidates by
selection / crossover / mutation, retaining a non-dominated Pareto front
across the two objectives. Output is the Pareto-optimal trials, not a
single best.

Outputs
-------
    sweep_results_<experiment>/
        optuna_study.db
        trial_XXXX/                 # one per evaluated candidate
        sweep_summary.json
        plots/
            pareto_velsurf_thk.png
            param_vs_costs.png
            convergence.png

Usage
-----
    conda activate igm-pretrain
    python optuna_sweep_da_nsga.py                                    # Rhone, default 30 trials
    python optuna_sweep_da_nsga.py --n-trials 60
    python optuna_sweep_da_nsga.py --experiment params_aletsch_da    # different target
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
from optuna.samplers import NSGAIISampler

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Defaults (overridable via CLI) ──────────────────────────────────────────

# Default to the OGGM/COP-DEM Rhone DA config — see experiment/params_oggm_rhone_da.yaml.
# Each experiment name gets its own sweep_results_<name>/ dir and Optuna
# study (keyed off the experiment), so flipping --experiment doesn't load a
# prior sweep's DB back into a mismatched search space.
DEFAULT_EXPERIMENT = "params_oggm_rhone_da"
DEFAULT_N_TRIALS = 30
DEFAULT_POPULATION = 10
DEFAULT_SEED = 42


def _experiment_paths(experiment: str):
    """Derive per-experiment output dirs from the Hydra experiment name."""
    sweep_dir = os.path.join(PROJECT_DIR, f"sweep_results_{experiment}")
    return {
        "sweep_dir": sweep_dir,
        "db_path": os.path.join(sweep_dir, "optuna_study.db"),
        "plots_dir": os.path.join(sweep_dir, "plots"),
        "study_name": f"da_nsga2_{experiment}",
    }

# Last row of costs.dat columns we treat as objectives. The IGM DA loop emits
# one column per active cost term; with cost_list=[velsurf, icemask, thk] and
# control_list=[thk] (slidingco NOT optimized) the header is:
#   velsurf  thk  thk_regu  glen
# We only treat the two data-fidelity terms as objectives — thk_regu is a
# regularization, not data, and is still computed and saved in costs.dat
# (via t.user_attrs["history_thk_regu"]) for diagnostics.
DA_COST_COLUMNS = ["velsurf", "thk"]


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


def objective(trial: optuna.Trial, gpu_id: int, sweep_dir: str, experiment: str):
    params = {
        "regularization_thk": trial.suggest_float("regularization_thk", 500.0, 4000.0, log=True),
        "thkobs_std": trial.suggest_float("thkobs_std", 0.5, 10.0, log=True),
        "init_slidingco": trial.suggest_float("init_slidingco", 0.05, 2.0, log=True),
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
        f"processes.data_assimilation.regularization.thk={params['regularization_thk']}",
        f"processes.data_assimilation.fitting.thkobs_std={params['thkobs_std']}",
        f"processes.iceflow.physics.init_slidingco={params['init_slidingco']}",
        f"hydra.run.dir={trial_dir}",
    ]

    cmd = ["igm_run"] + overrides

    # Pin this subprocess to a single GPU and let TF grow its memory allocation
    # on demand (default behaviour pre-grabs all VRAM, which prevents stacking
    # multiple trials per GPU).
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    print(
        f"\n[trial {trial.number} | gpu {gpu_id}] start  "
        f"reg_thk={params['regularization_thk']:.2e}  "
        f"thko_std={params['thkobs_std']:.3f}  "
        f"init_slid={params['init_slidingco']:.3f}",
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
    missing = [c for c in DA_COST_COLUMNS if c not in histories]
    if missing:
        print(
            f"[trial {trial.number} | gpu {gpu_id}] missing {missing} "
            f"in costs.dat (found columns: {header})",
            flush=True,
        )
        _save_meta(trial_dir, params, "missing_output", elapsed)
        return float("inf"), float("inf")

    finals = {c: float(histories[c][-1]) for c in DA_COST_COLUMNS}

    for c, hist in histories.items():
        trial.set_user_attr(f"history_{c}", hist)

    _save_meta(
        trial_dir,
        params,
        "completed",
        elapsed,
        final_velsurf=finals["velsurf"],
        final_thk=finals["thk"],
        n_iters=len(histories[DA_COST_COLUMNS[0]]),
    )

    print(
        f"[trial {trial.number} | gpu {gpu_id}] done  "
        f"velsurf={finals['velsurf']:.4f}  thk={finals['thk']:.4f}  "
        f"time={elapsed:.0f}s",
        flush=True,
    )

    return finals["velsurf"], finals["thk"]


def _save_meta(trial_dir, params, status, elapsed, **extra):
    meta = {**params, "status": status, "elapsed_s": round(elapsed, 1), **extra}
    with open(os.path.join(trial_dir, "trial_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ── Parallel runner ─────────────────────────────────────────────────────────


_study_lock = threading.Lock()  # serialise ask/tell into SQLite


def _run_one_trial(study: optuna.Study, gpu_id: int, sweep_dir: str, experiment: str):
    """Pull one trial, evaluate it on the given GPU, and report back."""
    with _study_lock:
        trial = study.ask()

    try:
        values = objective(trial, gpu_id=gpu_id, sweep_dir=sweep_dir, experiment=experiment)
        with _study_lock:
            study.tell(trial, values)
    except Exception as exc:  # objective itself shouldn't throw, but be safe
        with _study_lock:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        print(
            f"[trial {trial.number} | gpu {gpu_id}] EXCEPTION: {exc!r}",
            flush=True,
        )


def _worker_loop(study, gpu_id, work_q, sweep_dir, experiment):
    while True:
        try:
            work_q.get_nowait()
        except queue.Empty:
            return
        try:
            _run_one_trial(study, gpu_id, sweep_dir, experiment)
        finally:
            work_q.task_done()


def run_parallel(study: optuna.Study, n_trials: int, gpus, workers_per_gpu: int,
                 sweep_dir: str, experiment: str):
    """Run n_trials concurrently across (gpus × workers_per_gpu) threads.

    Each worker thread pins its igm_run subprocess to a single GPU via
    CUDA_VISIBLE_DEVICES, and lets TF grow its memory allocation on demand
    so multiple trials per GPU can coexist.
    """
    work_q: queue.Queue = queue.Queue()
    for _ in range(n_trials):
        work_q.put(None)

    threads = []
    for gpu_id in gpus:
        for w in range(workers_per_gpu):
            t = threading.Thread(
                target=_worker_loop,
                args=(study, gpu_id, work_q, sweep_dir, experiment),
                name=f"gpu{gpu_id}-w{w}",
                daemon=True,
            )
            t.start()
            threads.append(t)

    for t in threads:
        t.join()


# ── Plots ────────────────────────────────────────────────────────────────────


PARAM_NAMES = [
    "regularization_thk",
    "thkobs_std",
    "init_slidingco",
]
LOG_PARAMS = {
    "regularization_thk",
    "thkobs_std",
    "init_slidingco",
}


def make_plots(study: optuna.Study, plots_dir: str):
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)

    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        print("No completed trials — skipping plots.")
        return

    pareto_numbers = {t.number for t in study.best_trials}

    rows = []
    for t in completed:
        rows.append(
            {
                "trial": t.number,
                **{p: t.params[p] for p in PARAM_NAMES},
                "velsurf": t.values[0],
                "thk": t.values[1],
                "is_pareto": t.number in pareto_numbers,
            }
        )
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0].keys()}
    is_pareto = arr["is_pareto"].astype(bool)

    # 1. Pareto scatter velsurf vs thk, Pareto trials highlighted, color = thkobs_std
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(
        arr["velsurf"][~is_pareto],
        arr["thk"][~is_pareto],
        c=arr["thkobs_std"][~is_pareto],
        cmap="viridis",
        s=40,
        alpha=0.4,
        edgecolor="gray",
        linewidth=0.3,
        label="dominated",
    )
    sc = ax.scatter(
        arr["velsurf"][is_pareto],
        arr["thk"][is_pareto],
        c=arr["thkobs_std"][is_pareto],
        cmap="viridis",
        s=120,
        edgecolor="red",
        linewidth=1.2,
        marker="*",
        label="Pareto",
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("thkobs_std")
    ax.set_xlabel("Final velsurf cost")
    ax.set_ylabel("Final thk cost")
    ax.set_title("NSGA-II Pareto front: velsurf vs thk")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plots_dir, "pareto_velsurf_thk.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")

    # 2. param vs each cost, with Pareto highlighted
    costs_list = ["velsurf", "thk"]
    fig, axes = plt.subplots(
        len(PARAM_NAMES), len(costs_list), figsize=(9, 9), sharey="col"
    )
    for i, p in enumerate(PARAM_NAMES):
        for j, c in enumerate(costs_list):
            ax = axes[i, j]
            ax.scatter(
                arr[p][~is_pareto],
                arr[c][~is_pareto],
                s=20,
                alpha=0.4,
                color="gray",
                edgecolor="none",
            )
            ax.scatter(
                arr[p][is_pareto],
                arr[c][is_pareto],
                s=70,
                color="red",
                edgecolor="k",
                linewidth=0.5,
                marker="*",
                label="Pareto" if (i == 0 and j == 0) else None,
            )
            ax.set_xlabel(p)
            ax.set_ylabel(f"final {c}")
            ax.set_yscale("log")
            if p in LOG_PARAMS:
                ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Final DA cost terms vs each parameter (Pareto = red stars)")
    fig.tight_layout()
    path = os.path.join(plots_dir, "param_vs_costs.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")

    # 3. Convergence curves (one subplot per cost term, Pareto highlighted)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for t in completed:
        is_p = t.number in pareto_numbers
        col = "red" if is_p else "gray"
        alpha = 0.9 if is_p else 0.25
        lw = 1.0 if is_p else 0.4
        for ax, name in zip(axes, costs_list):
            hist = t.user_attrs.get(f"history_{name}", [])
            if hist:
                ax.plot(hist, color=col, lw=lw, alpha=alpha)
    for ax, name in zip(axes, costs_list):
        ax.set_title(f"{name} history")
        ax.set_xlabel("DA iteration")
        ax.set_ylabel(name)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].plot([], [], color="red", lw=1.0, label="Pareto")
    axes[0].plot([], [], color="gray", lw=0.4, label="dominated")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(plots_dir, "convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="NSGA-II DA sweep over 3 hyperparameters → 3 cost terms"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=DEFAULT_EXPERIMENT,
        help=(
            "Hydra experiment name to sweep over (without .yaml). "
            f"Default: {DEFAULT_EXPERIMENT}. The sweep dir and Optuna study "
            "name are derived from this — flipping --experiment opens a "
            "separate DB so search spaces don't get cross-loaded."
        ),
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Comma-separated GPU ids to use (default: 0,1).",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=3,
        help="Concurrent trials per GPU (default: 3).",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip running trials; just regenerate plots from the existing DB.",
    )
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]

    paths = _experiment_paths(args.experiment)
    sweep_dir = paths["sweep_dir"]
    db_path = paths["db_path"]
    plots_dir = paths["plots_dir"]
    study_name = paths["study_name"]

    os.makedirs(sweep_dir, exist_ok=True)

    storage = f"sqlite:///{db_path}"
    sampler = NSGAIISampler(population_size=args.population, seed=args.seed)

    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        directions=["minimize", "minimize"],
        storage=storage,
        load_if_exists=True,
    )

    n_done = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )

    total_workers = len(gpus) * args.workers_per_gpu

    print("DA sweep — NSGA-II (velsurf, thk)")
    print(f"  Experiment         : {args.experiment}")
    print(f"  Study name         : {study_name}")
    print(f"  Population size    : {args.population}")
    print(f"  Target n_trials    : {args.n_trials}")
    print(f"  Already completed  : {n_done}")
    print(f"  GPUs               : {gpus}")
    print(f"  Workers per GPU    : {args.workers_per_gpu}")
    print(f"  Concurrent trials  : {total_workers}")
    print(f"  Results directory  : {sweep_dir}")
    print()

    n_remaining = max(args.n_trials - n_done, 0)
    if not args.plots_only and n_remaining > 0:
        run_parallel(
            study,
            n_trials=n_remaining,
            gpus=gpus,
            workers_per_gpu=args.workers_per_gpu,
            sweep_dir=sweep_dir,
            experiment=args.experiment,
        )

    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if completed:
        pareto = study.best_trials
        pareto_numbers = {t.number for t in pareto}

        rows = [
            {
                "trial": t.number,
                **{p: t.params[p] for p in PARAM_NAMES},
                "velsurf": t.values[0],
                "thk": t.values[1],
                "is_pareto": t.number in pareto_numbers,
            }
            for t in completed
        ]
        rows.sort(key=lambda r: (r["velsurf"], r["thk"]))

        print(f"\n{'=' * 100}")
        print(f"  PARETO TRIALS ({len(pareto)} of {len(completed)})")
        print(f"{'=' * 100}")
        header = (
            f"  {'trial':>5}  {'reg_thk':>10}  {'thko_std':>9}  "
            f"{'init_slid':>9}  "
            f"{'velsurf':>9}  {'thk':>9}"
        )
        print(header)
        for r in rows:
            if r["is_pareto"]:
                print(
                    f"  {r['trial']:>5}  "
                    f"{r['regularization_thk']:>10.2e}  {r['thkobs_std']:>9.3f}  "
                    f"{r['init_slidingco']:>9.3f}  "
                    f"{r['velsurf']:>9.4f}  {r['thk']:>9.4f}"
                )

        with open(os.path.join(sweep_dir, "sweep_summary.json"), "w") as f:
            json.dump(
                {
                    "experiment": args.experiment,
                    "sampler": "NSGAIISampler",
                    "population": args.population,
                    "seed": args.seed,
                    "results": rows,
                    "pareto_trials": sorted(pareto_numbers),
                    "completed_trials": len(completed),
                },
                f,
                indent=2,
            )

        make_plots(study, plots_dir)


if __name__ == "__main__":
    main()
