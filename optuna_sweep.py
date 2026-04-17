#!/usr/bin/env python3
"""
Optuna grid search over SMB inference hyperparameters (Argentiere).

Sweeps 2 parameters with an exhaustive grid:
    regularisation     : [0.001, 0.005, 0.01, 0.1]
    learning_rate      : [0.001, 0.01, 0.1]

Total trials: 4 x 3 = 12

Usage
-----
    conda activate igm
    python optuna_sweep.py          # run all 12 trials (resumable)
    python optuna_sweep.py --resume # explicitly resume a previous run
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
SWEEP_DIR = os.path.join(PROJECT_DIR, "sweep_results")
DB_PATH = os.path.join(SWEEP_DIR, "optuna_study.db")

# ── Search space (grid) ─────────────────────────────────────────────────────

SEARCH_SPACE = {
    "regularisation": [0.01, 0.1, 0.5],
    "learning_rate": [0.1],
}

TOTAL_TRIALS = 1
for v in SEARCH_SPACE.values():
    TOTAL_TRIALS *= len(v)


# ── Objective ────────────────────────────────────────────────────────────────


def objective(trial: optuna.Trial) -> float:
    """Run one IGM SMB inference trial and return the final loss."""

    params = {
        "regularisation": trial.suggest_categorical(
            "regularisation", SEARCH_SPACE["regularisation"]
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", SEARCH_SPACE["learning_rate"]
        ),
    }

    trial_dir = os.path.join(SWEEP_DIR, f"trial_{trial.number:04d}")
    smb_outdir = os.path.join(trial_dir, "smb_inference")
    os.makedirs(trial_dir, exist_ok=True)

    # Hydra overrides ---------------------------------------------------------
    overrides = [
        "+experiment=params_argentiere",
        f"processes.smb_inference.optimization.learning_rate={params['learning_rate']}",
        f"processes.smb_inference.optimization.regularisation={params['regularisation']}",
        f"processes.smb_inference.output.outdir={smb_outdir}",
        f"hydra.run.dir={trial_dir}",
    ]

    cmd = ["igm_run"] + overrides

    print(f"\n{'=' * 70}")
    print(f"  Trial {trial.number + 1}/{TOTAL_TRIALS}  (trial_id={trial.number})")
    print(
        f"  lr={params['learning_rate']}  reg={params['regularisation']}"
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
            timeout=7200,  # 2 h safety limit per trial
        )
    except subprocess.TimeoutExpired:
        print(f"  ✗ Trial {trial.number} TIMED OUT after 2 h")
        _save_meta(trial_dir, params, "timeout", time.time() - t0)
        return float("inf")

    elapsed = time.time() - t0

    # Persist logs for debugging
    with open(os.path.join(trial_dir, "stdout.log"), "w") as f:
        f.write(result.stdout)
    with open(os.path.join(trial_dir, "stderr.log"), "w") as f:
        f.write(result.stderr)

    if result.returncode != 0:
        print(f"  ✗ Trial {trial.number} FAILED (exit {result.returncode})")
        tail = result.stderr[-500:] if result.stderr else result.stdout[-500:]
        print(f"    {tail}")
        _save_meta(trial_dir, params, "failed", elapsed)
        return float("inf")

    # Load loss history -------------------------------------------------------
    loss_path = os.path.join(smb_outdir, "loss_history.npy")
    if not os.path.exists(loss_path):
        print(f"  ✗ Trial {trial.number}: loss_history.npy not found")
        _save_meta(trial_dir, params, "missing_output", elapsed)
        return float("inf")

    loss_history = np.load(loss_path).tolist()
    final_loss = loss_history[-1]
    best_loss = min(loss_history)

    # Report intermediate values so the dashboard can track convergence
    for step, lval in enumerate(loss_history):
        trial.report(lval, step)

    _save_meta(
        trial_dir,
        params,
        "completed",
        elapsed,
        final_loss=final_loss,
        best_loss=best_loss,
        n_iterations=len(loss_history),
        loss_history=loss_history,
    )

    print(
        f"  ✓ Trial {trial.number} done  "
        f"final_loss={final_loss:.5f}  best_loss={best_loss:.5f}  "
        f"iters={len(loss_history)}  time={elapsed:.0f}s"
    )
    return final_loss


def _save_meta(trial_dir, params, status, elapsed, **extra):
    meta = {
        **params,
        "status": status,
        "elapsed_s": round(elapsed, 1),
        **extra,
    }
    with open(os.path.join(trial_dir, "trial_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="SMB inference parameter sweep")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous sweep (default: auto-detect)",
    )
    parser.parse_args()

    os.makedirs(SWEEP_DIR, exist_ok=True)

    storage = f"sqlite:///{DB_PATH}"
    sampler = GridSampler(SEARCH_SPACE)

    study = optuna.create_study(
        study_name="smb_inference_sweep",
        sampler=sampler,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
    )

    n_done = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    n_remaining = TOTAL_TRIALS - n_done

    print(f"SMB Inference — Parameter Sweep (Grid Search)")
    print(f"  Total combinations : {TOTAL_TRIALS}")
    print(f"  Already completed  : {n_done}")
    print(f"  Remaining          : {n_remaining}")
    print(f"  Results directory  : {SWEEP_DIR}")
    print(f"  Optuna DB          : {DB_PATH}")
    print()

    if n_remaining <= 0:
        print("All trials already completed.")
    else:
        study.optimize(objective, n_trials=TOTAL_TRIALS)

    # Final summary
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if completed:
        print(f"\n{'=' * 70}")
        print("  SWEEP COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Best trial  : #{study.best_trial.number}")
        print(f"  Best loss   : {study.best_value:.5f}")
        print(f"  Best params :")
        for k, v in study.best_params.items():
            print(f"    {k:30s} = {v}")

        with open(os.path.join(SWEEP_DIR, "sweep_summary.json"), "w") as f:
            json.dump(
                {
                    "best_trial": study.best_trial.number,
                    "best_loss": study.best_value,
                    "best_params": study.best_params,
                    "total_trials": TOTAL_TRIALS,
                    "completed_trials": len(completed),
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
