#!/usr/bin/env python3
"""
Multi-objective Optuna grid search over `init_slidingco` (Argentiere).

Sweeps 1 parameter with an exhaustive grid:
    processes.iceflow.physics.init_slidingco : [0.15, 0.25, 0.35, 0.45]

Objectives (both minimized):
    1. Final DA cost (sum of last row of costs.dat)
    2. Final SMB inference loss (last value of loss_history.npy)

Outputs
-------
    sweep_results_slidingco/
        optuna_study.db
        trial_XXXX/                 # one per value
        sweep_summary.json
        plots/
            pareto.png
            slidingco_vs_costs.png

Usage
-----
    conda activate igm
    python optuna_sweep_slidingco.py
    python optuna_sweep_slidingco.py --resume
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
SWEEP_DIR = os.path.join(PROJECT_DIR, "sweep_results_slidingco")
DB_PATH = os.path.join(SWEEP_DIR, "optuna_study.db")
PLOTS_DIR = os.path.join(SWEEP_DIR, "plots")

# ── Search space (grid) ─────────────────────────────────────────────────────

SEARCH_SPACE = {
    "init_slidingco": [0.05, 0.1],
}

TOTAL_TRIALS = 1
for v in SEARCH_SPACE.values():
    TOTAL_TRIALS *= len(v)


# ── Metric extraction ───────────────────────────────────────────────────────


def _read_final_da_cost(trial_dir: str):
    """Sum the last row of costs.dat (velsurf + thk_regu + glen + ...)."""
    costs_path = os.path.join(trial_dir, "costs.dat")
    if not os.path.exists(costs_path):
        return None, None
    data = np.loadtxt(costs_path, skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    final = float(data[-1].sum())
    history = data.sum(axis=1).tolist()
    return final, history


def _read_final_smb_loss(smb_outdir: str):
    """Return final and full loss history from SMB inference."""
    loss_path = os.path.join(smb_outdir, "loss_history.npy")
    if not os.path.exists(loss_path):
        return None, None
    hist = np.load(loss_path).tolist()
    return float(hist[-1]), hist


# ── Objective ────────────────────────────────────────────────────────────────


def objective(trial: optuna.Trial):
    params = {
        "init_slidingco": trial.suggest_categorical(
            "init_slidingco", SEARCH_SPACE["init_slidingco"]
        ),
    }

    trial_dir = os.path.join(SWEEP_DIR, f"trial_{trial.number:04d}")
    smb_outdir = os.path.join(trial_dir, "smb_inference")
    os.makedirs(trial_dir, exist_ok=True)

    overrides = [
        "+experiment=params_argentiere",
        f"processes.iceflow.physics.init_slidingco={params['init_slidingco']}",
        f"processes.smb_inference.output.outdir={smb_outdir}",
        f"hydra.run.dir={trial_dir}",
    ]

    cmd = ["igm_run"] + overrides

    print(f"\n{'=' * 70}")
    print(f"  Trial {trial.number + 1}/{TOTAL_TRIALS}  (trial_id={trial.number})")
    print(f"  init_slidingco = {params['init_slidingco']}")
    print(f"  output → {trial_dir}")
    print(f"{'=' * 70}\n")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=10800,  # 3 h safety limit per trial (DA + SMB)
        )
    except subprocess.TimeoutExpired:
        print(f"  ✗ Trial {trial.number} TIMED OUT")
        _save_meta(trial_dir, params, "timeout", time.time() - t0)
        return float("inf"), float("inf")

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
        return float("inf"), float("inf")

    da_final, da_history = _read_final_da_cost(trial_dir)
    smb_final, smb_history = _read_final_smb_loss(smb_outdir)

    if da_final is None or smb_final is None:
        missing = []
        if da_final is None:
            missing.append("costs.dat")
        if smb_final is None:
            missing.append("loss_history.npy")
        print(f"  ✗ Trial {trial.number}: missing {missing}")
        _save_meta(trial_dir, params, "missing_output", elapsed)
        return float("inf"), float("inf")

    # Store histories as user attrs for post-hoc plotting
    trial.set_user_attr("da_cost_history", da_history)
    trial.set_user_attr("smb_loss_history", smb_history)

    _save_meta(
        trial_dir,
        params,
        "completed",
        elapsed,
        da_final_cost=da_final,
        smb_final_loss=smb_final,
        da_n_iters=len(da_history),
        smb_n_iters=len(smb_history),
    )

    print(
        f"  ✓ Trial {trial.number} done  "
        f"da_cost={da_final:.5f}  smb_loss={smb_final:.5f}  time={elapsed:.0f}s"
    )
    return da_final, smb_final


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

    records = sorted(
        [
            (
                t.params["init_slidingco"],
                t.values[0],
                t.values[1],
                t.number,
            )
            for t in completed
        ],
        key=lambda r: r[0],
    )
    slc = np.array([r[0] for r in records])
    da = np.array([r[1] for r in records])
    smb = np.array([r[2] for r in records])

    # 1. Dual-axis: init_slidingco vs both costs
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    color1, color2 = "tab:blue", "tab:red"
    ax1.set_xlabel("init_slidingco")
    ax1.set_ylabel("Final DA cost", color=color1)
    ax1.plot(slc, da, "o-", color=color1, label="DA cost")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Final SMB inference loss", color=color2)
    ax2.plot(slc, smb, "s-", color=color2, label="SMB loss")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("init_slidingco vs final costs")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "slidingco_vs_costs.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")

    # 2. Pareto / scatter of the two objectives
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(da, smb, c=slc, cmap="viridis", s=80)
    for x, y, s in zip(da, smb, slc):
        ax.annotate(f"{s:.2f}", (x, y), textcoords="offset points", xytext=(6, 4))
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("init_slidingco")
    ax.set_xlabel("Final DA cost")
    ax.set_ylabel("Final SMB inference loss")
    ax.set_title("DA cost vs SMB loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "pareto.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")

    # 3. Convergence curves per trial
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=slc.min(), vmax=slc.max())
    for t in completed:
        s = t.params["init_slidingco"]
        c = cmap(norm(s))
        da_h = t.user_attrs.get("da_cost_history", [])
        smb_h = t.user_attrs.get("smb_loss_history", [])
        if da_h:
            axes[0].plot(da_h, color=c, label=f"{s:.2f}")
        if smb_h:
            axes[1].plot(smb_h, color=c, label=f"{s:.2f}")
    axes[0].set_title("DA cost history")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("DA cost (sum)")
    axes[0].set_yscale("log")
    axes[0].legend(title="init_slidingco", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("SMB loss history")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("SMB loss")
    axes[1].set_yscale("log")
    axes[1].legend(title="init_slidingco", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Multi-objective sweep: init_slidingco → (DA cost, SMB loss)"
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
        study_name="slidingco_multi_sweep",
        sampler=sampler,
        directions=["minimize", "minimize"],
        storage=storage,
        load_if_exists=True,
    )

    n_done = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    n_remaining = TOTAL_TRIALS - n_done

    print("init_slidingco sweep — multi-objective (DA cost, SMB loss)")
    print(f"  Total combinations : {TOTAL_TRIALS}")
    print(f"  Already completed  : {n_done}")
    print(f"  Remaining          : {n_remaining}")
    print(f"  Results directory  : {SWEEP_DIR}")
    print()

    if not args.plots_only and n_remaining > 0:
        study.optimize(objective, n_trials=TOTAL_TRIALS)

    # Summary
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if completed:
        rows = [
            {
                "trial": t.number,
                "init_slidingco": t.params["init_slidingco"],
                "da_final_cost": t.values[0],
                "smb_final_loss": t.values[1],
            }
            for t in completed
        ]
        rows.sort(key=lambda r: r["init_slidingco"])

        print(f"\n{'=' * 70}")
        print("  SWEEP RESULTS")
        print(f"{'=' * 70}")
        print(f"  {'slidingco':>10}  {'DA cost':>12}  {'SMB loss':>12}")
        for r in rows:
            print(
                f"  {r['init_slidingco']:>10.3f}  "
                f"{r['da_final_cost']:>12.5f}  {r['smb_final_loss']:>12.5f}"
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
