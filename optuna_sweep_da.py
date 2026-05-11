#!/usr/bin/env python3
"""
Multi-objective Optuna grid search over DA hyperparameters (Argentiere).

Sweeps 3 parameters with an exhaustive grid:
    processes.iceflow.physics.init_slidingco              : [0.05, 0.1, 0.2, 0.3]
    processes.data_assimilation.regularization.thk        : [5, 10, 100]
    processes.data_assimilation.fitting.thkobs_std        : [1.0, 5.0, 10.0]

Total trials: 4 x 3 x 3 = 36

The SMB inference stage is *not* run — only data assimilation. Each trial
records the final value of three DA cost terms from costs.dat:
    1. velsurf
    2. thk
    3. thk_regu

All three are tracked as Optuna objectives (minimize), so the study returns
the Pareto front. costs.dat history per term is also stored as user attrs
for post-hoc plotting.

Outputs
-------
    sweep_results_da/
        optuna_study.db
        trial_XXXX/                 # one per combination
        sweep_summary.json
        plots/
            pareto_velsurf_thk.png
            param_vs_costs.png
            convergence.png

Usage
-----
    conda activate igm-pretrain
    python optuna_sweep_da.py
    python optuna_sweep_da.py --resume
    python optuna_sweep_da.py --plots-only
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np
import optuna
from optuna.samplers import GridSampler

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_DIR, "sweep_results_da")
DB_PATH = os.path.join(SWEEP_DIR, "optuna_study.db")
PLOTS_DIR = os.path.join(SWEEP_DIR, "plots")

# ── Search space (grid) ─────────────────────────────────────────────────────

SEARCH_SPACE = {
    "init_slidingco": [0.05, 0.1, 0.2, 0.3],
    "regularization_thk": [10, 100, 1000],
    "thkobs_std": [1.0, 5.0, 10.0],
    "regularization_slidingco": [1.0e5, 1.0e7, 1.0e9],
}

TOTAL_TRIALS = 1
for v in SEARCH_SPACE.values():
    TOTAL_TRIALS *= len(v)

# Column names (and order) in costs.dat written by IGM's DA loop. The DA
# config has cost_list = [velsurf, icemask, thk]; the printer emits one
# column per cost term plus regularization terms. We pick velsurf / thk /
# thk_regu specifically.
DA_COST_COLUMNS = ["velsurf", "thk", "thk_regu"]


# ── Metric extraction ───────────────────────────────────────────────────────


def _read_da_costs(trial_dir: str):
    """Read costs.dat and return a dict {col_name: list_of_values_per_iter}.

    Returns ({}, []) if the file is missing or unreadable.
    """
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


# ── Objective ────────────────────────────────────────────────────────────────


def objective(trial: optuna.Trial):
    params = {
        "init_slidingco": trial.suggest_categorical(
            "init_slidingco", SEARCH_SPACE["init_slidingco"]
        ),
        "regularization_thk": trial.suggest_categorical(
            "regularization_thk", SEARCH_SPACE["regularization_thk"]
        ),
        "thkobs_std": trial.suggest_categorical(
            "thkobs_std", SEARCH_SPACE["thkobs_std"]
        ),
        "regularization_slidingco": trial.suggest_categorical(
            "regularization_slidingco", SEARCH_SPACE["regularization_slidingco"]
        ),
    }

    trial_dir = os.path.join(SWEEP_DIR, f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    # Hydra overrides — params_argentiere_da selects iceflow + DA only (no SMB).
    overrides = [
        "+experiment=params_argentiere_da",
        f"processes.iceflow.physics.init_slidingco={params['init_slidingco']}",
        f"processes.data_assimilation.regularization.thk={params['regularization_thk']}",
        f"processes.data_assimilation.regularization.slidingco={params['regularization_slidingco']}",
        f"processes.data_assimilation.fitting.thkobs_std={params['thkobs_std']}",
        f"hydra.run.dir={trial_dir}",
    ]

    cmd = ["igm_run"] + overrides

    print(f"\n{'=' * 70}")
    print(f"  Trial {trial.number + 1}/{TOTAL_TRIALS}  (trial_id={trial.number})")
    print(
        f"  init_slidingco={params['init_slidingco']}  "
        f"reg_thk={params['regularization_thk']}  "
        f"thkobs_std={params['thkobs_std']}  "
        f"reg_slidingco={params['regularization_slidingco']:.0e}"
    )
    print(f"  output → {trial_dir}")
    print(f"{'=' * 70}\n")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 h safety limit per trial (DA only)
        )
    except subprocess.TimeoutExpired:
        print(f"  ✗ Trial {trial.number} TIMED OUT")
        _save_meta(trial_dir, params, "timeout", time.time() - t0)
        return float("inf"), float("inf"), float("inf")

    elapsed = time.time() - t0

    with open(os.path.join(trial_dir, "stdout.log"), "w") as f:
        f.write(result.stdout)
    with open(os.path.join(trial_dir, "stderr.log"), "w") as f:
        f.write(result.stderr)

    if result.returncode != 0:
        print(f"  ✗ Trial {trial.number} FAILED (exit {result.returncode})")
        tail = result.stderr[-500:] if result.stderr else result.stdout[-500:]
        print(f"    {tail}")
        _save_meta(trial_dir, params, "failed", elapsed)
        return float("inf"), float("inf"), float("inf")

    histories, header = _read_da_costs(trial_dir)
    missing = [c for c in DA_COST_COLUMNS if c not in histories]
    if missing:
        print(
            f"  ✗ Trial {trial.number}: costs.dat missing column(s) {missing} "
            f"(found columns: {header})"
        )
        _save_meta(trial_dir, params, "missing_output", elapsed)
        return float("inf"), float("inf"), float("inf")

    finals = {c: float(histories[c][-1]) for c in DA_COST_COLUMNS}

    # Store full history per cost term (post-hoc plots / inspection)
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
        f"  ✓ Trial {trial.number} done  "
        f"velsurf={finals['velsurf']:.4f}  thk={finals['thk']:.4f}  "
        f"thk_regu={finals['thk_regu']:.4f}  time={elapsed:.0f}s"
    )

    return finals["velsurf"], finals["thk"], finals["thk_regu"]


def _save_meta(trial_dir, params, status, elapsed, **extra):
    meta = {**params, "status": status, "elapsed_s": round(elapsed, 1), **extra}
    with open(os.path.join(trial_dir, "trial_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ── Plots ────────────────────────────────────────────────────────────────────


def make_plots(study: optuna.Study):
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)

    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        print("No completed trials — skipping plots.")
        return

    rows = []
    for t in completed:
        rows.append(
            {
                "trial": t.number,
                "init_slidingco": t.params["init_slidingco"],
                "regularization_thk": t.params["regularization_thk"],
                "thkobs_std": t.params["thkobs_std"],
                "regularization_slidingco": t.params["regularization_slidingco"],
                "velsurf": t.values[0],
                "thk": t.values[1],
                "thk_regu": t.values[2],
            }
        )
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0].keys()}

    # 1. Pairwise scatter velsurf vs thk, colored by thkobs_std, marker by reg_thk
    fig, ax = plt.subplots(figsize=(7, 5.5))
    reg_values = sorted(set(arr["regularization_thk"].tolist()))
    markers = ["o", "s", "^", "D", "P"]
    for reg, m in zip(reg_values, markers):
        sel = arr["regularization_thk"] == reg
        sc = ax.scatter(
            arr["velsurf"][sel],
            arr["thk"][sel],
            c=arr["thkobs_std"][sel],
            cmap="viridis",
            s=80,
            marker=m,
            edgecolor="k",
            linewidth=0.5,
            label=f"reg_thk={reg}",
        )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("thkobs_std")
    ax.set_xlabel("Final velsurf cost")
    ax.set_ylabel("Final thk cost")
    ax.set_title("DA Pareto: velsurf vs thk  (marker = reg_thk)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "pareto_velsurf_thk.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")

    # 2. One row per param × 3 cols (velsurf, thk, thk_regu) — boxplot/strip
    params_list = [
        "init_slidingco",
        "regularization_thk",
        "thkobs_std",
        "regularization_slidingco",
    ]
    costs_list = ["velsurf", "thk", "thk_regu"]
    fig, axes = plt.subplots(
        len(params_list), len(costs_list), figsize=(13, 12), sharey="col"
    )
    for i, p in enumerate(params_list):
        unique = sorted(set(arr[p].tolist()))
        # Use log x-scale for params spanning >2 orders of magnitude.
        log_x = max(unique) / max(min(unique), 1e-30) > 100
        for j, c in enumerate(costs_list):
            ax = axes[i, j]
            for u in unique:
                sel = arr[p] == u
                ax.scatter(
                    [u] * sel.sum(),
                    arr[c][sel],
                    s=40,
                    alpha=0.7,
                    edgecolor="k",
                    linewidth=0.3,
                )
            means = [arr[c][arr[p] == u].mean() for u in unique]
            ax.plot(unique, means, "r-", lw=1.5, label="mean")
            ax.set_xlabel(p)
            ax.set_ylabel(f"final {c}")
            ax.set_yscale("log")
            if log_x:
                ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
    fig.suptitle("Final DA cost terms vs each swept parameter")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "param_vs_costs.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")

    # 3. Convergence curves (one subplot per cost term, one line per trial)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cmap = plt.cm.viridis
    norm = plt.Normalize(
        vmin=arr["init_slidingco"].min(), vmax=arr["init_slidingco"].max()
    )
    for t in completed:
        c = cmap(norm(t.params["init_slidingco"]))
        for ax, name in zip(axes, costs_list):
            hist = t.user_attrs.get(f"history_{name}", [])
            if hist:
                ax.plot(hist, color=c, lw=0.7, alpha=0.7)
    for ax, name in zip(axes, costs_list):
        ax.set_title(f"{name} history")
        ax.set_xlabel("DA iteration")
        ax.set_ylabel(name)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="init_slidingco", fraction=0.02, pad=0.02)
    path = os.path.join(PLOTS_DIR, "convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Multi-objective DA sweep: (slidingco, reg_thk, thkobs_std) → (velsurf, thk, thk_regu)"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip running trials; just regenerate plots from the existing DB.",
    )
    args = parser.parse_args()

    os.makedirs(SWEEP_DIR, exist_ok=True)

    storage = f"sqlite:///{DB_PATH}"
    sampler = GridSampler(SEARCH_SPACE)

    study = optuna.create_study(
        study_name="da_multi_sweep",
        sampler=sampler,
        directions=["minimize", "minimize", "minimize"],
        storage=storage,
        load_if_exists=True,
    )

    n_done = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    n_remaining = TOTAL_TRIALS - n_done

    print("DA sweep — multi-objective (velsurf, thk, thk_regu)")
    print(f"  Total combinations : {TOTAL_TRIALS}")
    print(f"  Already completed  : {n_done}")
    print(f"  Remaining          : {n_remaining}")
    print(f"  Results directory  : {SWEEP_DIR}")
    print()

    if not args.plots_only and n_remaining > 0:
        study.optimize(objective, n_trials=TOTAL_TRIALS)

    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if completed:
        rows = [
            {
                "trial": t.number,
                "init_slidingco": t.params["init_slidingco"],
                "regularization_thk": t.params["regularization_thk"],
                "thkobs_std": t.params["thkobs_std"],
                "regularization_slidingco": t.params["regularization_slidingco"],
                "velsurf": t.values[0],
                "thk": t.values[1],
                "thk_regu": t.values[2],
            }
            for t in completed
        ]
        rows.sort(
            key=lambda r: (
                r["init_slidingco"],
                r["regularization_thk"],
                r["thkobs_std"],
                r["regularization_slidingco"],
            )
        )

        print(f"\n{'=' * 90}")
        print("  SWEEP RESULTS")
        print(f"{'=' * 90}")
        print(
            f"  {'slidingco':>10}  {'reg_thk':>8}  {'thkobs_std':>10}  "
            f"{'reg_slid':>10}  {'velsurf':>10}  {'thk':>10}  {'thk_regu':>10}"
        )
        for r in rows:
            print(
                f"  {r['init_slidingco']:>10.3f}  {r['regularization_thk']:>8}  "
                f"{r['thkobs_std']:>10.2f}  {r['regularization_slidingco']:>10.0e}  "
                f"{r['velsurf']:>10.4f}  {r['thk']:>10.4f}  {r['thk_regu']:>10.4f}"
            )

        pareto = study.best_trials
        print(f"\n  Pareto-optimal trials: {[t.number for t in pareto]}")

        with open(os.path.join(SWEEP_DIR, "sweep_summary.json"), "w") as f:
            json.dump(
                {
                    "results": rows,
                    "pareto_trials": [t.number for t in pareto],
                    "total_trials": TOTAL_TRIALS,
                    "completed_trials": len(completed),
                },
                f,
                indent=2,
            )

        make_plots(study)


if __name__ == "__main__":
    main()
