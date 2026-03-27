#!/usr/bin/env python3
"""
Step 1: Plot L-curve from results.

Reads geology-optimized.nc from each subdirectory and plots the L-curve
(velocity misfit vs. smoothness) to help choose the regularization weight.

Works with both Hydra multirun directories and custom output directories.

Usage:
    python analyze_step1.py <results_dir>

Examples:
    python analyze_step1.py multirun/2026-03-12/10-30-00
    python analyze_step1.py results_step1
"""

import argparse
import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import re
import yaml


def parse_param(nc_path, results_dir, param_name):
    """Extract a parameter value from directory name or Hydra overrides."""
    rel_path = os.path.relpath(nc_path, results_dir)

    short_name = param_name.split(".")[-1]
    for pattern in [re.escape(param_name) + r'=([0-9.e+-]+)',
                    short_name + r'_([0-9.e+-]+)',
                    short_name + r'=([0-9.e+-]+)']:
        m = re.search(pattern, rel_path)
        if m:
            return float(m.group(1))

    nc_dir = os.path.dirname(nc_path)
    overrides_file = os.path.join(nc_dir, ".hydra", "overrides.yaml")
    if os.path.exists(overrides_file):
        with open(overrides_file) as f:
            overrides = yaml.safe_load(f)
        for override in overrides:
            m = re.search(re.escape(param_name) + r'=([0-9.e+-]+)', override)
            if m:
                return float(m.group(1))

    return None


def load_thk_ref(input_path, ds_out):
    """Load true thk from input.nc, cropped to match output grid."""
    input_ds = xr.open_dataset(input_path)
    if "thk" not in input_ds:
        input_ds.close()
        return None, None

    # Crop input thk to output extent
    x_out, y_out = ds_out.x.values, ds_out.y.values
    thk_full = input_ds["thk"].sel(
        x=slice(x_out.min(), x_out.max()),
        y=slice(y_out.min(), y_out.max()),
    ).values
    input_ds.close()

    thk_mask = ds_out["icemask"].values > 0.5
    if not thk_mask.any():
        return None, None
    return thk_full, thk_mask


def main():
    parser = argparse.ArgumentParser(description="Step 1: Plot L-curve from results")
    parser.add_argument("results_dir", help="Path to results directory")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "data", "input.nc")

    nc_files = sorted(glob.glob(os.path.join(args.results_dir, "**/geology-optimized.nc"), recursive=True))
    if not nc_files:
        print(f"No geology-optimized.nc found in {args.results_dir}")
        return

    # Load true thk once using first output to determine crop
    ds0 = xr.open_dataset(nc_files[0])
    thk_ref, thk_mask = load_thk_ref(input_path, ds0)
    ds0.close()

    regs, vel_rmses, thk_rmses, volumes, grad_norms = [], [], [], [], []

    for nc_path in nc_files:
        reg = parse_param(nc_path, args.results_dir,
                          "processes.data_assimilation.regularization.thk")
        if reg is None:
            rel = os.path.relpath(nc_path, args.results_dir)
            print(f"  [skip] Cannot parse reg weight from: {rel}")
            continue

        ds = xr.open_dataset(nc_path)
        thk = ds["thk"].values
        velsurf_mag = ds["velsurf_mag"].values
        velsurfobs_mag = ds["velsurfobs_mag"].values
        icemask = ds["icemask"].values
        dx = float(ds.x.values[1] - ds.x.values[0])
        ds.close()

        mask_vel = icemask > 0.5
        vel_rmse = np.sqrt(np.nanmean((velsurf_mag[mask_vel] - velsurfobs_mag[mask_vel]) ** 2))

        if thk_ref is not None:
            thk_rmse = np.sqrt(np.nanmean((thk[thk_mask] - thk_ref[thk_mask]) ** 2))
        else:
            thk_rmse = np.nan

        dy, dxx = np.gradient(thk)
        grad_norm = np.sqrt(np.nanmean(dxx[mask_vel] ** 2 + dy[mask_vel] ** 2))

        vol = np.sum(thk) * dx ** 2 * 1e-9

        regs.append(reg)
        vel_rmses.append(vel_rmse)
        thk_rmses.append(thk_rmse)
        volumes.append(vol)
        grad_norms.append(grad_norm)

        print(f"  reg={reg:>8g}: vel_rmse={vel_rmse:.2f} m/yr, thk_rmse={thk_rmse:.1f} m, "
              f"vol={vol:.1f} km3, grad_norm={grad_norm:.2f}")

    if len(regs) < 2:
        print("Not enough results to plot.")
        return

    idx = np.argsort(regs)
    regs = np.array(regs)[idx]
    vel_rmses = np.array(vel_rmses)[idx]
    thk_rmses = np.array(thk_rmses)[idx]
    volumes = np.array(volumes)[idx]
    grad_norms = np.array(grad_norms)[idx]

    # --- L-curve plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(grad_norms, vel_rmses, "o-", color="steelblue", markersize=8, linewidth=2)
    for i, r in enumerate(regs):
        ax.annotate(f"  {r:g}", (grad_norms[i], vel_rmses[i]), fontsize=9)
    ax.set_xlabel("Gradient norm (smoothness)", fontsize=12)
    ax.set_ylabel("Velocity RMSE (m/yr)", fontsize=12)
    ax.set_title("L-curve: velocity misfit vs. smoothness", fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogx(regs, vel_rmses, "o-", color="steelblue", label="Velocity RMSE (m/yr)", linewidth=2)
    ax2 = ax.twinx()
    ax2.semilogx(regs, thk_rmses, "s--", color="orangered", label="Thickness RMSE vs input.nc thk (m)", linewidth=2)
    ax.set_xlabel("Regularization weight", fontsize=12)
    ax.set_ylabel("Velocity RMSE (m/yr)", color="steelblue", fontsize=12)
    ax2.set_ylabel("Thickness RMSE (m)", color="orangered", fontsize=12)
    ax.set_title("Misfit vs. regularization", fontsize=13)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Step 1: L-curve analysis (velocity-only inversion)", fontsize=14, y=1.02)
    fig.tight_layout()
    out_png = os.path.join(args.results_dir, "lcurve_step1.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
