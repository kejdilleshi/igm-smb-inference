#!/usr/bin/env python3
"""
slidingco sensitivity of the final SMB inference (in-house glaciers).

Holds init_arrhenius fixed and re-runs the combined DA + SMB-inference final
stage for a SPREAD of well-performing init_slidingco values taken from the
Optuna DA sweep, then overlays the resulting flux-divergence and SMB profiles
so the effect of basal sliding on the inferred mass balance is visible.

Slidingco selection (``--auto``, the default): from
``sweep_results_params_oggm_<glacier>_da/sweep_summary.json`` keep finite trials
with velsurf <= best*--cutoff, sort by velsurf, then greedily thin in log-space
(drop any value within --thin-ratio of an already-kept, better one) and cap at
--max-points. This is exactly "use the best-performing slidingco, but not ones
too close together". Pass ``--slidingco a,b,c`` to override the set.

arrhenius / reg.thk default to the glacier's adopted final run
(``results/<glacier>/pipeline_summary.json``); smb reg = run_pipeline.FIXED_SMB_REG.

Each value runs into  results/<glacier>/sens_slidingco/slid_<v>/  and the overlay
+ summary.csv land in  results/<glacier>/sens_slidingco/ .

Usage:
    conda activate igm-pretrain
    python pipeline/sensitivity_slidingco.py --glacier aletsch_insitu --gpu 0
    python pipeline/sensitivity_slidingco.py --glacier rhone_insitu  --gpu 1
    python pipeline/sensitivity_slidingco.py --glacier rhone_insitu --dry-run
    python pipeline/sensitivity_slidingco.py --glacier rhone_insitu --skip-run  # replot only
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "user", "code", "processes", "smb_inference"))

from run_pipeline import GLACIERS, FIXED_SMB_REG, run  # noqa: E402

INSITU = ["aletsch_insitu", "rhone_insitu"]


# ── slidingco selection ───────────────────────────────────────────────────────

def select_slidingco(glacier, cutoff, thin_ratio, max_points):
    summ = os.path.join(PROJECT_DIR,
                        f"sweep_results_params_oggm_{glacier}_da", "sweep_summary.json")
    rows = json.load(open(summ))["results"]
    rows = [r for r in rows if math.isfinite(r["velsurf"])]
    rows.sort(key=lambda r: r["velsurf"])
    best = rows[0]["velsurf"]
    good = [r for r in rows if r["velsurf"] <= best * cutoff]
    kept = []
    for r in good:                       # already velsurf-sorted (best first)
        v = r["init_slidingco"]
        if all(max(v, k["init_slidingco"]) / min(v, k["init_slidingco"]) >= thin_ratio
               for k in kept):
            kept.append(r)
        if len(kept) >= max_points:
            break
    kept.sort(key=lambda r: r["init_slidingco"])
    return kept


# ── per-run execution (mirrors run_pipeline.stage_final) ──────────────────────

def run_one(glacier, slid, arrh, reg_thk, out_dir, gpu, nbitmax, dry):
    experiment = f"params_oggm_{glacier}"
    cmd = [
        "igm_run", f"+experiment={experiment}",
        f"processes.iceflow.physics.init_slidingco={slid}",
        f"processes.iceflow.physics.init_arrhenius={arrh}",
        f"processes.data_assimilation.regularization.thk={reg_thk}",
        f"processes.smb_inference.optimization.regularisation={FIXED_SMB_REG}",
        f"hydra.run.dir={out_dir}",
    ]
    if nbitmax is not None:
        cmd.append(f"processes.data_assimilation.optimization.nbitmax={nbitmax}")
    env = {"TF_FORCE_GPU_ALLOW_GROWTH": "true"}
    if gpu is not None:
        # Pin via CUDA_VISIBLE_DEVICES, which re-indexes the chosen GPU to 0;
        # the config must then ask for device [0], not the physical id (else
        # igm_run does gpus[physical_id] -> IndexError).
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd.append("core.hardware.visible_gpus=[0]")
    run(cmd, dry, env=env)


def stake_csv_for(glacier):
    csvs = glob.glob(os.path.join(
        PROJECT_DIR, "data", glacier, f"SMB_{glacier}_*-*_rawstakes_utm32N.csv"))
    def window(p):
        m = re.search(r"_(\d{4})-(\d{4})_rawstakes", os.path.basename(p))
        return int(m.group(2)) - int(m.group(1)) if m else -1
    return max(csvs, key=window, default=None)


# ── cross-run overlay ─────────────────────────────────────────────────────────

def divflux_profile(run_dir, z_min, dz, n, rho_ice):
    """DA flux divergence binned by elevation (m w.e./yr).

    Reads div(flux) straight from the DA output (geology-optimized.nc one level
    above run_dir), matching smb_vs_continuity. Returns NaN where unavailable.
    """
    from plot_smb_results import _bin_by_elevation, _load_da_fields
    divflux, usurf, icemask = _load_da_fields(run_dir)
    if divflux is None or usurf is None:
        return np.full(n, np.nan)
    mask = (icemask > 0.5) if icemask is not None else np.isfinite(usurf)
    mask = mask & np.isfinite(divflux)
    prof = _bin_by_elevation(divflux, usurf, mask, float(z_min), float(dz), n)
    return prof * (rho_ice / 1000.0)


def final_da_velsurf(run_dir):
    """Final-iteration DA surface-velocity misfit from costs.dat (m/yr scale).

    This is the metric a stake-free glacier would use to pick slidingco (the
    Optuna sweep minimises it), here evaluated at the FIXED arrhenius. NaN if
    unavailable."""
    cp = os.path.join(run_dir, "costs.dat")
    if not os.path.exists(cp):
        return float("nan")
    hdr = open(cp).readline().split()
    if "velsurf" not in hdr:
        return float("nan")
    data = np.loadtxt(cp, skiprows=1)
    last = data[-1] if data.ndim > 1 else data
    return float(last[hdr.index("velsurf")])


def overlay(glacier, runs, rho_ice):
    """runs = list of dicts {slid, arrh, dir}. Build 2-panel overlay + summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from compare_smb import load_profile
    from smb_vs_continuity import prepare_cells
    from data.homogenize import draw_stake_overlay

    sens_dir = os.path.join(PROJECT_DIR, "results", glacier, "sens_slidingco")
    stakes = stake_csv_for(glacier)
    prep = None
    if stakes:
        prep, _ = prepare_cells(stakes, 100.0, 10, 4)

    cmap = plt.get_cmap("viridis")
    slids = [r["slid"] for r in runs]
    norm = plt.Normalize(min(slids), max(slids))

    fig, (axd, axs) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    summary = []
    for r in runs:
        rd = os.path.join(r["dir"], "smb_inference")
        if not os.path.exists(os.path.join(rd, "smb_vec.npy")):
            print(f"[overlay] skip {r['slid']:.4f}: no smb_vec.npy in {rd}")
            continue
        z, smb_we = load_profile(rd, rho_ice)
        nbin = z.shape[0]
        z_min, dz = float(z[0]), float(z[1] - z[0]) if nbin > 1 else 200.0
        div_we = divflux_profile(rd, z_min, dz, nbin, rho_ice)
        c = cmap(norm(r["slid"]))
        lbl = f"slid={r['slid']:.3f}"
        axd.plot(div_we, z, "-", color=c, lw=1.6, label=lbl)
        axs.plot(smb_we, z, "-o", color=c, ms=2.5, lw=1.4, label=lbl)

        # scalars
        fin = np.isfinite(smb_we)
        mean_smb = float(np.nanmean(smb_we))
        # ELA = lowest elevation where the profile crosses 0 (sign change)
        ela = np.nan
        zf, sf = z[fin], smb_we[fin]
        sign = np.sign(sf)
        cross = np.where(np.diff(sign) > 0)[0]
        if cross.size:
            i = cross[0]
            ela = float(np.interp(0.0, [sf[i], sf[i + 1]], [zf[i], zf[i + 1]]))
        velsurf = final_da_velsurf(r["dir"])
        summary.append({"slidingco": round(r["slid"], 4), "arrhenius": round(r["arrh"], 3),
                        "da_velsurf": None if math.isnan(velsurf) else round(velsurf, 3),
                        "mean_smb_mwe": round(mean_smb, 3),
                        "ELA_m": None if math.isnan(ela) else round(ela, 0)})

    # raw stakes (grey) + repeat-cell anchors (green) on the SMB panel
    if prep is not None:
        draw_stake_overlay(axs, prep)

    for ax, t, xl in ((axd, "Flux divergence profile", r"$\nabla\!\cdot$flux (m w.e./yr)"),
                      (axs, "Inferred SMB profile", "SMB (m w.e./yr)")):
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel(xl); ax.set_title(t); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axd.set_ylabel("Elevation (m a.s.l.)")
    arrh = runs[0]["arrh"]
    fig.suptitle(f"slidingco sensitivity — {glacier} "
                 f"(arrhenius fixed = {arrh:.1f} MPa$^{{-3}}$ yr$^{{-1}}$)",
                 fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(sens_dir, f"overlay_{glacier}.png")
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"[overlay] wrote {out_png}")

    if summary:
        out_csv = os.path.join(sens_dir, f"summary_{glacier}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader(); w.writerows(summary)
        print(f"[overlay] wrote {out_csv}")
        # echo table
        print(f"\n{'slid':>7} {'arrh':>7} {'velsurf':>8} {'meanSMB':>8} {'ELA':>6}")
        for s in summary:
            print(f"{s['slidingco']:>7.3f} {s['arrhenius']:>7.1f} "
                  f"{(s['da_velsurf'] if s['da_velsurf'] is not None else float('nan')):>8.3f} "
                  f"{s['mean_smb_mwe']:>8.3f} "
                  f"{(s['ELA_m'] or float('nan')):>6.0f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glacier", required=True, choices=INSITU)
    ap.add_argument("--slidingco", default=None,
                    help="comma list to override auto-selection")
    ap.add_argument("--arrhenius", type=float, default=None,
                    help="fixed arrhenius (default: adopted final-run value)")
    ap.add_argument("--reg-thk", type=float, default=None,
                    help="fixed DA reg.thk (default: adopted final-run value)")
    ap.add_argument("--cutoff", type=float, default=1.5,
                    help="keep trials with velsurf <= best*cutoff (default 1.5)")
    ap.add_argument("--thin-ratio", type=float, default=1.25,
                    help="min multiplicative spacing between kept slidingco (default 1.25)")
    ap.add_argument("--max-points", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--nbitmax", default=None, help="override DA nbitmax (smoke test)")
    ap.add_argument("--rho-ice", type=float, default=917.0)
    ap.add_argument("--skip-run", action="store_true", help="only (re)build the overlay")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    g = args.glacier
    summ = json.load(open(os.path.join(PROJECT_DIR, "results", g, "pipeline_summary.json")))
    arrh = args.arrhenius if args.arrhenius is not None else float(summ["init_arrhenius"])
    reg_thk = args.reg_thk if args.reg_thk is not None else float(summ["reg_thk"])

    if args.slidingco:
        vals = [float(x) for x in args.slidingco.split(",")]
        chosen = [{"init_slidingco": v, "velsurf": float("nan")} for v in vals]
    else:
        chosen = select_slidingco(g, args.cutoff, args.thin_ratio, args.max_points)

    sens_dir = os.path.join(PROJECT_DIR, "results", g, "sens_slidingco")
    os.makedirs(sens_dir, exist_ok=True)

    print(f"=== slidingco sensitivity: {g} ===")
    print(f"  fixed arrhenius = {arrh:.3f}   reg.thk = {reg_thk:g}   smb reg = {FIXED_SMB_REG}")
    print(f"  slidingco values ({len(chosen)}):")
    for r in chosen:
        vs = "" if math.isnan(r['velsurf']) else f"  (sweep velsurf={r['velsurf']:.3f})"
        print(f"    {r['init_slidingco']:.4f}{vs}")

    runs = []
    for r in chosen:
        v = r["init_slidingco"]
        out_dir = os.path.join(sens_dir, f"slid_{v:.4f}")
        runs.append({"slid": v, "arrh": arrh, "dir": out_dir})
        if not args.skip_run:
            run_one(g, v, arrh, reg_thk, out_dir, args.gpu, args.nbitmax, args.dry_run)
            stakes = stake_csv_for(g)
            run([sys.executable, "plot_smb_results.py", "--root", out_dir,
                 "--input-nc", os.path.join("data", f"input_{GLACIERS[g].replace('-', '_')}.nc")],
                args.dry_run)
            if stakes:
                run([sys.executable, "pipeline/smb_vs_continuity.py",
                     "--run", os.path.join(out_dir, "smb_inference"),
                     "--stakes", stakes, "--title", f"{g} slid={v:.3f}"], args.dry_run)

    if not args.dry_run:
        overlay(g, runs, args.rho_ice)


if __name__ == "__main__":
    main()
