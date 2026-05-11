#!/usr/bin/env python3
"""
Multi-objective Optuna NSGA-II sweep over DA hyperparameters (Argentière).

Search space (continuous, log where appropriate):
    processes.data_assimilation.regularization.thk        : [300, 1000]  (log)
    processes.data_assimilation.fitting.thkobs_std        : [0.5, 20]    (log)
    processes.data_assimilation.regularization.slidingco  : [1e7, 1e10]  (log)
    processes.data_assimilation.scaling.slidingco         : [1e-2, 1.0]  (log)

Three objectives (all minimized, last row of costs.dat):
    1. velsurf
    2. thk
    3. thk_regu

NSGA-II is a genetic algorithm — it evolves a population of candidates by
selection / crossover / mutation, retaining a non-dominated Pareto front
across the three objectives. Output is the Pareto-optimal trials, not a
single best.

Outputs
-------
    sweep_results_da_nsga/
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
    python optuna_sweep_da_nsga.py                  # 60 trials, 20-strong population
    python optuna_sweep_da_nsga.py --n-trials 100
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
SWEEP_DIR = os.path.join(PROJECT_DIR, "sweep_results_da_nsga")
DB_PATH = os.path.join(SWEEP_DIR, "optuna_study.db")
PLOTS_DIR = os.path.join(SWEEP_DIR, "plots")

# ── Defaults (overridable via CLI) ──────────────────────────────────────────

DEFAULT_N_TRIALS = 100
DEFAULT_POPULATION = 20
DEFAULT_SEED = 42

# Last row of costs.dat columns we treat as objectives. The IGM DA loop emits
# one column per active cost term; with cost_list=[velsurf, icemask, thk] and
# control_list=[thk, slidingco] the header is:
#   velsurf  thk  thk_regu  slid_regu  glen
DA_COST_COLUMNS = ["velsurf", "thk", "thk_regu"]


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


def objective(trial: optuna.Trial, gpu_id: int = 0):
    params = {
        "regularization_thk": trial.suggest_float("regularization_thk", 300.0, 1000.0, log=True),
        "thkobs_std": trial.suggest_float("thkobs_std", 0.5, 20.0, log=True),
        "regularization_slidingco": trial.suggest_float("regularization_slidingco", 1.0e7, 1.0e10, log=True),
        "scaling_slidingco": trial.suggest_float("scaling_slidingco", 1.0e-2, 1.0, log=True),
    }
    trial.set_user_attr("gpu_id", gpu_id)

    trial_dir = os.path.join(SWEEP_DIR, f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    overrides = [
        "+experiment=params_argentiere_da",
        f"processes.data_assimilation.regularization.thk={params['regularization_thk']}",
        f"processes.data_assimilation.regularization.slidingco={params['regularization_slidingco']}",
        f"processes.data_assimilation.scaling.slidingco={params['scaling_slidingco']}",
        f"processes.data_assimilation.fitting.thkobs_std={params['thkobs_std']}",
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
        f"reg_slid={params['regularization_slidingco']:.2e}  "
        f"sc_slid={params['scaling_slidingco']:.2e}",
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
        return float("inf"), float("inf"), float("inf")

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
        return float("inf"), float("inf"), float("inf")

    histories, header = _read_da_costs(trial_dir)
    missing = [c for c in DA_COST_COLUMNS if c not in histories]
    if missing:
        print(
            f"[trial {trial.number} | gpu {gpu_id}] missing {missing} "
            f"in costs.dat (found columns: {header})",
            flush=True,
        )
        _save_meta(trial_dir, params, "missing_output", elapsed)
        return float("inf"), float("inf"), float("inf")

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
        final_thk_regu=finals["thk_regu"],
        n_iters=len(histories[DA_COST_COLUMNS[0]]),
    )

    print(
        f"[trial {trial.number} | gpu {gpu_id}] done  "
        f"velsurf={finals['velsurf']:.4f}  thk={finals['thk']:.4f}  "
        f"thk_regu={finals['thk_regu']:.4f}  time={elapsed:.0f}s",
        flush=True,
    )

    return finals["velsurf"], finals["thk"], finals["thk_regu"]


def _save_meta(trial_dir, params, status, elapsed, **extra):
    meta = {**params, "status": status, "elapsed_s": round(elapsed, 1), **extra}
    with open(os.path.join(trial_dir, "trial_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ── Parallel runner ─────────────────────────────────────────────────────────


_study_lock = threading.Lock()  # serialise ask/tell into SQLite


def _run_one_trial(study: optuna.Study, gpu_id: int):
    """Pull one trial, evaluate it on the given GPU, and report back."""
    with _study_lock:
        trial = study.ask()

    try:
        values = objective(trial, gpu_id=gpu_id)
        with _study_lock:
            study.tell(trial, values)
    except Exception as exc:  # objective itself shouldn't throw, but be safe
        with _study_lock:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        print(
            f"[trial {trial.number} | gpu {gpu_id}] EXCEPTION: {exc!r}",
            flush=True,
        )


def _worker_loop(study: optuna.Study, gpu_id: int, work_q: queue.Queue):
    while True:
        try:
            work_q.get_nowait()
        except queue.Empty:
            return
        try:
            _run_one_trial(study, gpu_id)
        finally:
            work_q.task_done()


def run_parallel(study: optuna.Study, n_trials: int, gpus, workers_per_gpu: int):
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
                args=(study, gpu_id, work_q),
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
    "regularization_slidingco",
    "scaling_slidingco",
]
LOG_PARAMS = {
    "regularization_thk",
    "thkobs_std",
    "regularization_slidingco",
    "scaling_slidingco",
}


def make_plots(study: optuna.Study):
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)

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
                "thk_regu": t.values[2],
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
    path = os.path.join(PLOTS_DIR, "pareto_velsurf_thk.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")

    # 2. param vs each cost, with Pareto highlighted
    costs_list = ["velsurf", "thk", "thk_regu"]
    fig, axes = plt.subplots(
        len(PARAM_NAMES), len(costs_list), figsize=(13, 14), sharey="col"
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
    path = os.path.join(PLOTS_DIR, "param_vs_costs.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")

    # 3. Convergence curves (one subplot per cost term, Pareto highlighted)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
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
    path = os.path.join(PLOTS_DIR, "convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="NSGA-II DA sweep over 4 hyperparameters → 3 cost terms"
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
        default=4,
        help="Concurrent trials per GPU (default: 4).",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip running trials; just regenerate plots from the existing DB.",
    )
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]

    os.makedirs(SWEEP_DIR, exist_ok=True)

    storage = f"sqlite:///{DB_PATH}"
    sampler = NSGAIISampler(population_size=args.population, seed=args.seed)

    study = optuna.create_study(
        study_name="da_nsga2_sweep",
        sampler=sampler,
        directions=["minimize", "minimize", "minimize"],
        storage=storage,
        load_if_exists=True,
    )

    n_done = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )

    total_workers = len(gpus) * args.workers_per_gpu

    print("DA sweep — NSGA-II (velsurf, thk, thk_regu)")
    print(f"  Population size    : {args.population}")
    print(f"  Target n_trials    : {args.n_trials}")
    print(f"  Already completed  : {n_done}")
    print(f"  GPUs               : {gpus}")
    print(f"  Workers per GPU    : {args.workers_per_gpu}")
    print(f"  Concurrent trials  : {total_workers}")
    print(f"  Results directory  : {SWEEP_DIR}")
    print()

    n_remaining = max(args.n_trials - n_done, 0)
    if not args.plots_only and n_remaining > 0:
        run_parallel(
            study,
            n_trials=n_remaining,
            gpus=gpus,
            workers_per_gpu=args.workers_per_gpu,
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
                "thk_regu": t.values[2],
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
            f"{'reg_slid':>10}  {'sc_slid':>10}  {'velsurf':>9}  {'thk':>9}  {'thk_regu':>9}"
        )
        print(header)
        for r in rows:
            if r["is_pareto"]:
                print(
                    f"  {r['trial']:>5}  "
                    f"{r['regularization_thk']:>10.2e}  {r['thkobs_std']:>9.3f}  "
                    f"{r['regularization_slidingco']:>10.2e}  {r['scaling_slidingco']:>10.2e}  "
                    f"{r['velsurf']:>9.4f}  {r['thk']:>9.4f}  {r['thk_regu']:>9.4f}"
                )

        with open(os.path.join(SWEEP_DIR, "sweep_summary.json"), "w") as f:
            json.dump(
                {
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

        make_plots(study)


if __name__ == "__main__":
    main()
