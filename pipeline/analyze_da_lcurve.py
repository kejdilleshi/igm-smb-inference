#!/usr/bin/env python3
"""
DA regularization L-curve (step 2.2 of the pipeline).

Reads geology-optimized.nc from each Hydra-multirun subdir of a sweep over
`processes.data_assimilation.regularization.thk`, computes per run:
  * velocity RMSE  (velsurf_mag vs velsurfobs_mag, on icemask)  — data misfit
  * thickness RMSE (thk vs thkobs from --input)                 — validation
  * thk gradient norm                                           — smoothness
  * ice volume
plots `lcurve_step1.png`, and auto-detects the L-curve corner
(smoothness vs velocity misfit) → the chosen regularization.thk.

Generalized from ../igm-examples-invert/aletsch/analyze_step1.py (which is
hard-wired to data/input.nc); here the input grid file is an argument so any
OGGM glacier works.

Usage:
    python analyze_da_lcurve.py <results_dir> --input data/input_RGI60_11.02773.nc
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcurve import parse_param, lcurve_corner  # noqa: E402

REG_PARAM = "processes.data_assimilation.regularization.thk"


def load_thkobs(input_path, ds_out):
    """Load thkobs from the input grid, cropped to the (possibly cropped) output."""
    if not input_path or not os.path.exists(input_path):
        return None, None
    with xr.open_dataset(input_path) as ds_in:
        if "thkobs" not in ds_in:
            return None, None
        x_out, y_out = ds_out.x.values, ds_out.y.values
        thkobs = ds_in["thkobs"].sel(
            x=slice(float(x_out.min()), float(x_out.max())),
            y=slice(float(y_out.min()), float(y_out.max())),
        ).values
    mask = ~np.isnan(thkobs)
    return (thkobs, mask) if mask.any() else (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", help="Hydra sweep dir (contains <reg>/geology-optimized.nc)")
    ap.add_argument("--input", required=True, help="Input grid NetCDF (for thkobs validation)")
    ap.add_argument("--out", default=None, help="Output PNG (default <results_dir>/lcurve_step1.png)")
    args = ap.parse_args()

    nc_files = sorted(glob.glob(os.path.join(args.results_dir, "**/geology-optimized.nc"),
                                recursive=True))
    if not nc_files:
        sys.exit(f"No geology-optimized.nc under {args.results_dir}")

    with xr.open_dataset(nc_files[0]) as ds0:
        thkobs, thk_mask = load_thkobs(args.input, ds0)
    if thkobs is None:
        print("WARNING: no thkobs in --input; thickness RMSE will be NaN")

    regs, vel_rmses, thk_rmses, volumes, grad_norms = [], [], [], [], []
    for nc_path in nc_files:
        reg = parse_param(os.path.dirname(nc_path), REG_PARAM)
        if reg is None:
            print(f"  [skip] cannot parse reg from {os.path.relpath(nc_path, args.results_dir)}")
            continue
        with xr.open_dataset(nc_path) as ds:
            thk = ds["thk"].values
            vmag = ds["velsurf_mag"].values
            vobs = ds["velsurfobs_mag"].values
            icemask = ds["icemask"].values
            dx = float(ds.x.values[1] - ds.x.values[0])

        m = icemask > 0.5
        vel_rmse = float(np.sqrt(np.nanmean((vmag[m] - vobs[m]) ** 2)))
        thk_rmse = (float(np.sqrt(np.nanmean((thk[thk_mask] - thkobs[thk_mask]) ** 2)))
                    if thkobs is not None else np.nan)
        gy, gx = np.gradient(thk)
        grad_norm = float(np.sqrt(np.nanmean(gx[m] ** 2 + gy[m] ** 2)))
        vol = float(np.sum(thk) * dx ** 2 * 1e-9)

        regs.append(reg); vel_rmses.append(vel_rmse); thk_rmses.append(thk_rmse)
        volumes.append(vol); grad_norms.append(grad_norm)
        print(f"  reg={reg:>10g}: vel_rmse={vel_rmse:.2f} m/yr, thk_rmse={thk_rmse:.1f} m, "
              f"vol={vol:.1f} km3, grad_norm={grad_norm:.3f}")

    if len(regs) < 2:
        sys.exit("Not enough runs to build an L-curve.")

    idx = np.argsort(regs)
    regs = np.array(regs)[idx]
    vel_rmses = np.array(vel_rmses)[idx]
    thk_rmses = np.array(thk_rmses)[idx]
    volumes = np.array(volumes)[idx]
    grad_norms = np.array(grad_norms)[idx]

    # Corner: smoothness (grad_norm) vs data misfit (vel_rmse).
    corner = lcurve_corner(grad_norms, vel_rmses)
    best_reg = float(regs[corner])
    print(f"\nAuto-detected corner: regularization.thk = {best_reg:g} "
          f"(vel_rmse={vel_rmses[corner]:.2f} m/yr, grad_norm={grad_norms[corner]:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(grad_norms, vel_rmses, "o-", color="steelblue", markersize=8, linewidth=2)
    for i, r in enumerate(regs):
        ax.annotate(f"  {r:g}", (grad_norms[i], vel_rmses[i]), fontsize=9)
    ax.plot(grad_norms[corner], vel_rmses[corner], "*", color="red", markersize=20,
            label=f"corner reg={best_reg:g}")
    ax.set_xlabel("Thickness gradient norm (smoothness)")
    ax.set_ylabel("Velocity RMSE (m/yr)")
    ax.set_title("L-curve: velocity misfit vs smoothness")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogx(regs, vel_rmses, "o-", color="steelblue", label="Velocity RMSE (m/yr)", linewidth=2)
    ax2 = ax.twinx()
    ax2.semilogx(regs, thk_rmses, "s--", color="orangered", label="Thickness RMSE (m)", linewidth=2)
    ax.axvline(best_reg, color="red", linestyle=":", linewidth=1.5)
    ax.set_xlabel("regularization.thk")
    ax.set_ylabel("Velocity RMSE (m/yr)", color="steelblue")
    ax2.set_ylabel("Thickness RMSE (m)", color="orangered")
    ax.set_title("Misfit vs regularization")
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Step 1: DA regularization L-curve", fontsize=14, y=1.02)
    fig.tight_layout()
    out_png = args.out or os.path.join(args.results_dir, "lcurve_step1.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")

    result = {
        "param": REG_PARAM,
        "best_reg": best_reg,
        "regs": regs.tolist(),
        "vel_rmse": vel_rmses.tolist(),
        "thk_rmse": [None if np.isnan(v) else v for v in thk_rmses.tolist()],
        "grad_norm": grad_norms.tolist(),
        "plot": out_png,
    }
    out_json = os.path.join(args.results_dir, "da_lcurve_result.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
