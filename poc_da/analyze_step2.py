#!/usr/bin/env python3
"""
Step 2: Plot L-curve from step_size sweep results.

Reads geology-optimized.nc from each subdirectory and plots how velocity
misfit and thickness error vary with the optimizer step size.

Usage:
    python analyze_step2.py <results_dir>

Examples:
    python analyze_step2.py results_step2
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
    parser = argparse.ArgumentParser(description="Step 2: L-curve for step_size sweep")
    parser.add_argument("results_dir", help="Path to results directory")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "data", "input.nc")

    nc_files = sorted(glob.glob(os.path.join(args.results_dir, "**/geology-optimized.nc"), recursive=True))
    if not nc_files:
        print(f"No geology-optimized.nc found in {args.results_dir}")
        return

    ds0 = xr.open_dataset(nc_files[0])
    thk_ref, thk_mask = load_thk_ref(input_path, ds0)
    ds0.close()

    step_sizes, vel_rmses, thk_rmses, volumes, grad_norms = [], [], [], [], []

    for nc_path in nc_files:
        step_size = parse_param(nc_path, args.results_dir,
                                "processes.data_assimilation.optimization.step_size")
        if step_size is None:
            rel = os.path.relpath(nc_path, args.results_dir)
            print(f"  [skip] Cannot parse step_size from: {rel}")
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

        thk_rmse = np.sqrt(np.nanmean((thk[thk_mask] - thk_ref[thk_mask]) ** 2)) if thk_ref is not None else np.nan

        dy, dxx = np.gradient(thk)
        grad_norm = np.sqrt(np.nanmean(dxx[mask_vel] ** 2 + dy[mask_vel] ** 2))

        vol = np.sum(thk) * dx ** 2 * 1e-9

        step_sizes.append(step_size)
        vel_rmses.append(vel_rmse)
        thk_rmses.append(thk_rmse)
        volumes.append(vol)
        grad_norms.append(grad_norm)

        print(f"  step_size={step_size:>8g}: vel_rmse={vel_rmse:.2f} m/yr, thk_rmse={thk_rmse:.1f} m, "
              f"vol={vol:.1f} km3, grad_norm={grad_norm:.2f}")

    if len(step_sizes) < 2:
        print("Not enough results to plot.")
        return

    idx = np.argsort(step_sizes)
    step_sizes = np.array(step_sizes)[idx]
    vel_rmses = np.array(vel_rmses)[idx]
    thk_rmses = np.array(thk_rmses)[idx]
    volumes = np.array(volumes)[idx]
    grad_norms = np.array(grad_norms)[idx]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: velocity RMSE vs step_size ---
    ax = axes[0]
    ax.semilogx(step_sizes, vel_rmses, "o-", color="steelblue", markersize=8, linewidth=2)
    for i, s in enumerate(step_sizes):
        ax.annotate(f"  {s:g}", (step_sizes[i], vel_rmses[i]), fontsize=9)
    ax.set_xlabel("Step size", fontsize=12)
    ax.set_ylabel("Velocity RMSE (m/yr)", fontsize=12)
    ax.set_title("Velocity misfit vs. step size", fontsize=13)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: thickness RMSE vs step_size ---
    ax = axes[1]
    ax.semilogx(step_sizes, thk_rmses, "s-", color="orangered", markersize=8, linewidth=2)
    for i, s in enumerate(step_sizes):
        ax.annotate(f"  {s:g}", (step_sizes[i], thk_rmses[i]), fontsize=9)
    ax.set_xlabel("Step size", fontsize=12)
    ax.set_ylabel("Thickness RMSE vs input thk (m)", fontsize=12)
    ax.set_title("Thickness error vs. step size", fontsize=13)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: L-curve (vel RMSE vs thk RMSE) ---
    ax = axes[2]
    sc = ax.scatter(vel_rmses, thk_rmses, c=np.log10(np.where(step_sizes > 0, step_sizes, np.nan)),
                    cmap="viridis", s=80, zorder=3)
    ax.plot(vel_rmses, thk_rmses, "-", color="gray", linewidth=1, zorder=2)
    for i, s in enumerate(step_sizes):
        ax.annotate(f"  {s:g}", (vel_rmses[i], thk_rmses[i]), fontsize=9)
    plt.colorbar(sc, ax=ax, label="log10(step_size)")
    ax.set_xlabel("Velocity RMSE (m/yr)", fontsize=12)
    ax.set_ylabel("Thickness RMSE vs input thk (m)", fontsize=12)
    ax.set_title("L-curve: vel misfit vs. thk error", fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Step 2: step_size sweep analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    out_png = os.path.join(args.results_dir, "lcurve_step2.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
