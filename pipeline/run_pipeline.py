#!/usr/bin/env python3
"""
End-to-end SMB-inference pipeline for an OGGM glacier.

Stages:
  1   pick a glacier (registry below)
  2.1 Optuna DA sweep    -> init_slidingco, init_arrhenius   (reg.thk held fixed)
  2.2 DA reg.thk L-curve (igm_run -m) -> auto-corner reg.thk*
  2.3 Final combined run — one igm_run on the full experiment YAML
      (DA + SMB inference together) using the swept params + L-curve
      reg.thk* + a FIXED smb regularisation (= FIXED_SMB_REG below). No
      separate SMB L-curve; the constant smb reg was chosen as the
      ``smb_lcurve/2`` value from earlier experiments.

DA cost is velsurf + icemask only (thickness observations are NOT used). The
sweep is single-objective TPE on the surface-velocity misfit, tuning the two
scalar physics parameters init_slidingco and init_arrhenius.

The final stage also runs ``plot_smb_results.py`` (thickness 3-panel + 1D SMB
profile) and ``pipeline/compare_smb.py`` (Kneib-style stake-bin comparison).

Examples:
    python pipeline/run_pipeline.py --glacier rhone
    python pipeline/run_pipeline.py --glacier rhone --pause-after-da
    python pipeline/run_pipeline.py --glacier rhone --from-stage final \
        --da-reg-thk 300 --init-slidingco 0.13 --init-arrhenius 78.0
    python pipeline/run_pipeline.py --glacier rhone --dry-run
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcurve import parse_param  # noqa: E402  (re-exported for downstream tools)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name -> RGI id. experiment = params_oggm_<name>; _da derived.
# The three in-house ("insitu") glaciers used in the paper — aletsch_insitu,
# rhone_insitu, argentiere — all now assimilate on the END-of-period surface
# (the epoch the observed velocities sample) and rewind the SMB forward run to
# the start geometry on the DA-fixed bed, via
# processes.smb_inference.initial_state.usurf_variable=usurfstart.
# All three use velocity-only DA (cost_list = [velsurf, icemask]); measured
# thickness is kept for validation only and never enters the pipeline.
# The earlier DA-at-start results are archived in results/_archive_da_at_t1/.
GLACIERS = {
    "aletsch":     "RGI60-11.01450",
    # In-house Aletsch variant (swisstopo DEMs 2009-2017, DA on 2017): the "rgi"
    # here is a filename stem so input_nc resolves to data/input_aletsch.nc;
    # experiment = params_oggm_aletsch_insitu(_da).
    "aletsch_insitu": "aletsch",
    "rhone":       "RGI60-11.01238",
    # In-house Rhone variant (swisstopo DEMs 2007-2016, DA on 2016): "rgi" is a
    # filename stem so input_nc resolves to data/input_rhone_insitu.nc;
    # experiment = params_oggm_rhone_insitu(_da).
    "rhone_insitu": "rhone_insitu",
    # Hugonnet-anchored Rhone variant (Appendix A2): SAME DA epoch (2016),
    # bed and 2007-2016 window as rhone_insitu -- only the 2007 start
    # surface differs (Hugonnet-implied vs. surveyed swisstopo). "rgi" is a
    # filename stem so input_nc resolves to data/input_rhone_hugonnet2007.nc
    # (built by data/rhone/build_input_hugonnet2007.py); experiment =
    # params_oggm_rhone_hugonnet2007(_da). Replaces the earlier, WRONG
    # rhone_da2020 (mismatched ~2006-2020 window), deleted not archived.
    "rhone_hugonnet2007": "rhone_hugonnet2007",
    "findelen":    "RGI60-11.02773",
    "corbassiere": "RGI60-11.02766",
    # Glacier d'Argentière (French Alps) — IN-HOUSE variant, NOT an OGGM/GLAMOS
    # glacier. "rgi" is a filename stem so input_nc resolves to
    # data/input_argentiere.nc (prebuilt: measured bed+thk, IGE DEMs 2012/2021,
    # velocities); experiment = params_oggm_argentiere(_da). Velocity-only DA,
    # SMB inference 2012-2021 against in-house stakes. compare_smb is skipped
    # (no GLAMOS / no rawstakes CSV).
    "argentiere":  "argentiere",
}

STAGES = ["sweep", "da_lcurve", "final"]

DEFAULT_DA_REG = "50,100,300,600,1000,2000"
# Constant SMB regularisation used in the combined final run. Picked as the
# ``smb_lcurve/2`` entry from earlier experiments (≈ knee of the SMB L-curve);
# we no longer sweep it.
FIXED_SMB_REG = 0.05


# ── helpers ──────────────────────────────────────────────────────────────────

def run(cmd, dry, env=None, cwd=PROJECT_DIR):
    """Print and (unless dry) execute a command, streaming its output."""
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=full_env, check=True)


def pin_gpu(args, cmd, env):
    """Pin an igm_run to args.igm_gpu, via CUDA_VISIBLE_DEVICES + hydra override.

    CUDA_VISIBLE_DEVICES already hides every other card, so TensorFlow sees the
    chosen GPU re-indexed to 0. igm_run.py then does `gpus[i] for i in
    cfg.core.hardware.visible_gpus`, so the override must be [0] — passing the
    physical index (e.g. [1]) raises IndexError before the run starts.
    """
    if args.igm_gpu is None:
        return
    cmd.append("core.hardware.visible_gpus=[0]")
    env["CUDA_VISIBLE_DEVICES"] = str(args.igm_gpu)


def sweep_summary_path(da_experiment):
    return os.path.join(PROJECT_DIR, f"sweep_results_{da_experiment}", "sweep_summary.json")


def load_state(results_dir):
    p = os.path.join(results_dir, "pipeline_state.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_state(results_dir, state):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "pipeline_state.json"), "w") as f:
        json.dump(state, f, indent=2)


# ── stages ───────────────────────────────────────────────────────────────────

def stage_sweep(args, ctx, state):
    """2.1 Optuna DA sweep -> init_slidingco, init_arrhenius (single-obj velsurf)."""
    if args.init_slidingco is not None and args.init_arrhenius is not None:
        print(f"[sweep] using provided init_slidingco={args.init_slidingco}, "
              f"init_arrhenius={args.init_arrhenius} (skipping Optuna)")
        state["init_slidingco"] = args.init_slidingco
        state["init_arrhenius"] = args.init_arrhenius
        return

    cmd = [
        sys.executable, "optuna_sweep_da_nsga.py",
        "--experiment", ctx["da_experiment"],
        "--n-trials", args.n_trials,
        "--gpus", args.gpus,
        "--workers-per-gpu", args.workers_per_gpu,
    ]
    run(cmd, args.dry_run)

    summ = sweep_summary_path(ctx["da_experiment"])
    if args.dry_run and not os.path.exists(summ):
        print(f"[sweep] (dry-run) would read best trial from {summ}")
        state["init_slidingco"] = state.get("init_slidingco", "<best>")
        state["init_arrhenius"] = state.get("init_arrhenius", "<best>")
        return
    with open(summ) as f:
        summary = json.load(f)
    if "best_params" not in summary:
        msg = (f"[sweep] {summ} has no 'best_params' — likely a stale NSGA "
               "summary. Delete it and rerun the sweep.")
        if args.dry_run:
            print(msg + "  (continuing dry-run with placeholders)")
            state["init_slidingco"] = state.get("init_slidingco", "<best>")
            state["init_arrhenius"] = state.get("init_arrhenius", "<best>")
            return
        sys.exit(msg)
    best_params = summary["best_params"]
    state["init_slidingco"] = float(best_params["init_slidingco"])
    state["init_arrhenius"] = float(best_params["init_arrhenius"])
    print(f"[sweep] best trial {summary['best_trial']}: "
          f"velsurf={summary['best_velsurf']:.4f}  "
          f"init_slidingco={state['init_slidingco']:.4f}  "
          f"init_arrhenius={state['init_arrhenius']:.3f}")


def stage_da_lcurve(args, ctx, state):
    """2.2 DA reg.thk L-curve -> auto-corner reg.thk*."""
    init_slid = state["init_slidingco"]
    init_arrh = state["init_arrhenius"]
    sweep_dir = os.path.join(ctx["results_dir"], "da_lcurve")

    cmd = [
        "igm_run", "-m", f"+experiment={ctx['da_experiment']}",
        "processes.data_assimilation.control_list=[thk]",
        f"processes.iceflow.physics.init_slidingco={init_slid}",
        f"processes.iceflow.physics.init_arrhenius={init_arrh}",
        f"processes.data_assimilation.regularization.thk={args.da_reg}",
        "hydra/launcher=joblib", f"hydra.launcher.n_jobs={args.n_jobs}",
        f"hydra.sweep.dir={sweep_dir}",
    ]
    if args.da_nbitmax is not None:
        cmd.append(f"processes.data_assimilation.optimization.nbitmax={args.da_nbitmax}")
    env = {"TF_FORCE_GPU_ALLOW_GROWTH": "true"}
    pin_gpu(args, cmd, env)
    run(cmd, args.dry_run, env=env)

    run([sys.executable, "pipeline/analyze_da_lcurve.py", sweep_dir,
         "--input", ctx["input_nc"]], args.dry_run)

    if args.da_reg_thk is not None:
        state["reg_thk"] = float(args.da_reg_thk)
        print(f"[da_lcurve] using provided reg.thk={state['reg_thk']}")
    else:
        res = os.path.join(sweep_dir, "da_lcurve_result.json")
        if args.dry_run and not os.path.exists(res):
            state["reg_thk"] = state.get("reg_thk", "<corner>")
            return
        with open(res) as f:
            state["reg_thk"] = float(json.load(f)["best_reg"])
        print(f"[da_lcurve] auto-corner reg.thk={state['reg_thk']:g}")


def stage_final(args, ctx, state):
    """2.3 Combined DA + SMB inference run on the full experiment YAML.

    One ``igm_run`` against ``params_oggm_<glacier>.yaml`` (DA + SMB inference
    in the same pass), using the sweep's init_slidingco/init_arrhenius, the
    L-curve's reg.thk*, and a fixed smb regularisation (FIXED_SMB_REG).
    """
    final_dir = os.path.join(ctx["results_dir"], "final")
    state["reg_smb"] = FIXED_SMB_REG
    cmd = [
        "igm_run", f"+experiment={ctx['experiment']}",
        f"processes.iceflow.physics.init_slidingco={state['init_slidingco']}",
        f"processes.iceflow.physics.init_arrhenius={state['init_arrhenius']}",
        f"processes.data_assimilation.regularization.thk={state['reg_thk']}",
        f"processes.smb_inference.optimization.regularisation={FIXED_SMB_REG}",
        f"hydra.run.dir={final_dir}",
    ]
    if args.da_nbitmax is not None:
        cmd.append(f"processes.data_assimilation.optimization.nbitmax={args.da_nbitmax}")
    env = {"TF_FORCE_GPU_ALLOW_GROWTH": "true"}
    pin_gpu(args, cmd, env)
    run(cmd, args.dry_run, env=env)
    print(f"[final] combined run written under {final_dir}/  "
          f"(SMB results in smb_inference/)")

    # Diagnostic plots — thickness 3-panel + 1D profile from plot_smb_results.py,
    # plus the Kneib-style stake-bin comparison from compare_smb.py. Both
    # standalone, so a failure here doesn't invalidate the inference itself.
    smb_run_dir = os.path.join(final_dir, "smb_inference")
    run([sys.executable, "plot_smb_results.py",
         "--root", final_dir, "--input-nc", ctx["input_nc"]], args.dry_run)

    import glob, re
    stake_csvs = glob.glob(os.path.join(
        PROJECT_DIR, "data", args.glacier,
        f"SMB_{args.glacier}_*-*_rawstakes_utm32N.csv"))
    # Pick the CSV with the widest year window encoded in the filename
    # (e.g. SMB_rhone_2006-2020_… → 14 yr; SMB_aletsch_2000-2020_… → 20 yr).
    def _window(p):
        m = re.search(r"_(\d{4})-(\d{4})_rawstakes", os.path.basename(p))
        return int(m.group(2)) - int(m.group(1)) if m else -1
    stake_csv = max(stake_csvs, key=_window, default=None)
    if stake_csv:
        run([sys.executable, "pipeline/compare_smb.py",
             "--run", smb_run_dir, "--stakes", stake_csv], args.dry_run)
        # Inferred profile vs continuity (dh/dt + div(flux)) against the
        # repeat-cell anchors — compare_smb's figure with GLAMOS replaced by
        # the mass-balance-diagnostic dashed line. Auto-relaxes the repeat-cell
        # year threshold for short in-house records (e.g. aletsch_insitu, 9 yr).
        run([sys.executable, "pipeline/smb_vs_continuity.py",
             "--run", smb_run_dir, "--stakes", stake_csv,
             "--title", args.glacier], args.dry_run)
    elif not args.dry_run:
        print(f"[final] no stake CSV under data/{args.glacier}/; "
              "skipping compare_smb.py / smb_vs_continuity.py")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glacier", required=True, choices=sorted(GLACIERS))
    ap.add_argument("--da-reg", default=DEFAULT_DA_REG,
                    help=f"comma list of regularization.thk values (default {DEFAULT_DA_REG})")
    ap.add_argument("--n-trials", default="30", help="Optuna trials (default 30)")
    ap.add_argument("--n-jobs", default="4", help="joblib launcher parallel jobs for L-curves")
    ap.add_argument("--gpus", default="0,1", help="GPUs for the Optuna sweep")
    ap.add_argument("--workers-per-gpu", default="3", help="Optuna concurrent trials per GPU")
    ap.add_argument("--igm-gpu", type=int, default=None,
                    help="Pin DA L-curve + final igm_run to this single GPU "
                         "(sets CUDA_VISIBLE_DEVICES; the YAML's "
                         "core.hardware.visible_gpus is forced to [0], the "
                         "index the card gets once the others are masked). "
                         "Use this when running two pipelines in parallel to "
                         "keep them on different GPUs and avoid OOM.")
    ap.add_argument("--pause-after-da", action="store_true",
                    help="stop after the DA L-curve so you can pick reg.thk manually")
    ap.add_argument("--from-stage", choices=STAGES, default="sweep",
                    help="resume from this stage (uses pipeline_state.json + overrides)")
    # Manual overrides (also used to resume past skipped stages).
    ap.add_argument("--init-slidingco", type=float, default=None)
    ap.add_argument("--init-arrhenius", type=float, default=None)
    ap.add_argument("--da-reg-thk", type=float, default=None, help="skip DA-corner auto-pick")
    ap.add_argument("--da-nbitmax", default=None, help="override DA nbitmax (smoke tests)")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running")
    args = ap.parse_args()

    rgi = GLACIERS[args.glacier]
    rgi_us = rgi.replace("-", "_")
    experiment = f"params_oggm_{args.glacier}"
    input_nc = os.path.join("data", f"input_{rgi_us}.nc")
    results_dir = os.path.join(PROJECT_DIR, "results", args.glacier)
    ctx = {
        "rgi": rgi,
        "experiment": experiment,
        "da_experiment": f"{experiment}_da",
        "input_nc": input_nc,
        "full_yaml": os.path.join(PROJECT_DIR, "experiment", f"{experiment}.yaml"),
        "results_dir": results_dir,
    }
    os.makedirs(results_dir, exist_ok=True)

    state = load_state(results_dir)
    # CLI overrides win over persisted state.
    for k, a in (("init_slidingco", args.init_slidingco),
                 ("init_arrhenius", args.init_arrhenius),
                 ("reg_thk", args.da_reg_thk)):
        if a is not None:
            state[k] = a

    print(f"=== SMB pipeline: {args.glacier} ({rgi}) ===")
    print(f"  experiment={experiment}  input={input_nc}  results={results_dir}")
    print(f"  from-stage={args.from_stage}  dry-run={args.dry_run}")
    print(f"  fixed smb regularisation = {FIXED_SMB_REG}")

    start = STAGES.index(args.from_stage)
    plan_stages = STAGES[start:]

    # A dry-run executes nothing, so it must not touch pipeline_state.json /
    # pipeline_summary.json — otherwise it silently rewrites the recorded
    # provenance of whatever real run last wrote results/<glacier>/final, and
    # the summary ends up describing parameters no run ever used. (This is how
    # rhone_insitu's summary came to claim A=78/reg=300 for a final/ that
    # .hydra/overrides.yaml shows was actually A=55.54/reg=1000.)
    def persist(st):
        if not args.dry_run:
            save_state(results_dir, st)

    if "sweep" in plan_stages:
        stage_sweep(args, ctx, state); persist(state)
    if "da_lcurve" in plan_stages:
        stage_da_lcurve(args, ctx, state); persist(state)
        if args.pause_after_da:
            print("\n[pause] DA L-curve done. Inspect "
                  f"{os.path.join(results_dir, 'da_lcurve', 'lcurve_step1.png')} then resume:\n"
                  f"  python pipeline/run_pipeline.py --glacier {args.glacier} "
                  f"--from-stage final --da-reg-thk <value> "
                  f"--init-slidingco {state.get('init_slidingco')} "
                  f"--init-arrhenius {state.get('init_arrhenius')}")
            return
    if "final" in plan_stages:
        stage_final(args, ctx, state); persist(state)

    summary = {"glacier": args.glacier, "rgi": rgi, **state}
    if args.dry_run:
        print("\n=== dry-run: state/summary NOT written (would be) ===")
        print(json.dumps(summary, indent=2))
        return
    with open(os.path.join(results_dir, "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== done. summary: {os.path.join(results_dir, 'pipeline_summary.json')} ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
