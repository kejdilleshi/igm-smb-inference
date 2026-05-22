#!/usr/bin/env python3
"""
Plot SMB inference validation figures for all runs in outputs/.

For each run containing smb_inference results, generates:
  1. Thickness comparison (observed / optimized / residual) - 3 panels
  2. SMB elevation profile (elevation vs SMB)

Usage:
    python plot_smb_results.py                               # default: outputs/
    python plot_smb_results.py --root sweep_results_slidingco
    python plot_smb_results.py --root sweep_results_slidingco --input-nc data/input_argentiere.nc
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import yaml

DEFAULT_INPUT_NC = "data/input_argentiere.nc"
DZ_DEFAULT = 200.0


def _get_dz_from_hydra(run_dir):
    """Read dz from the .hydra/config.yaml two levels above smb_inference/."""
    hydra_cfg = os.path.join(run_dir, "..", ".hydra", "config.yaml")
    hydra_cfg = os.path.normpath(hydra_cfg)
    if not os.path.exists(hydra_cfg):
        return None
    with open(hydra_cfg) as f:
        cfg = yaml.safe_load(f)
    try:
        return float(cfg["processes"]["smb_inference"]["inversion"]["dz"])
    except (KeyError, TypeError):
        return None


def _get_z_min_from_input(thk_obs, input_nc):
    """Compute z_min = min(topg + thk) from input.nc, using observed thk."""
    if not os.path.exists(input_nc):
        return None
    try:
        import xarray as xr
        ds = xr.open_dataset(input_nc)
        usurf = ds["usurf"].values
        thk = ds["thk"].values
        topg = usurf - thk
        z_surf = topg + thk_obs
        return float(z_surf.min())
    except Exception:
        return None


def find_smb_runs(root="outputs"):
    """Find all directories containing smb_inference results."""
    pattern = os.path.join(root, "**", "smb_inference", "thk_optimized.npy")
    matches = sorted(glob.glob(pattern, recursive=True))
    return [os.path.dirname(m) for m in matches]


def _run_label(run_dir, root):
    try:
        return os.path.relpath(run_dir, root)
    except ValueError:
        return run_dir


def load_npy(path):
    """Load a .npy file, return None if missing."""
    if os.path.exists(path):
        return np.load(path)
    return None


def plot_thickness(run_dir, thk_obs, thk_opt, root):
    """3-panel figure: observed, optimized, residual thickness."""
    residual = thk_opt - thk_obs

    vmax_thk = max(np.nanmax(thk_obs), np.nanmax(thk_opt))
    res_abs = 50#np.nanmax(np.abs(residual))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(thk_obs, origin="lower", cmap="viridis", vmin=0, vmax=vmax_thk)
    axes[0].set_title("Observed thickness")
    plt.colorbar(im0, ax=axes[0], label="m")

    im1 = axes[1].imshow(thk_opt, origin="lower", cmap="viridis", vmin=0, vmax=vmax_thk)
    axes[1].set_title("Optimized thickness")
    plt.colorbar(im1, ax=axes[1], label="m")

    im2 = axes[2].imshow(residual, origin="lower", cmap="RdBu_r", vmin=-res_abs, vmax=res_abs)
    axes[2].set_title("Residual (opt - obs)")
    plt.colorbar(im2, ax=axes[2], label="m")

    for ax in axes:
        ax.set_xlabel("x (grid)")
        ax.set_ylabel("y (grid)")

    run_label = _run_label(run_dir, root)
    fig.suptitle(f"Thickness comparison  [{run_label}]", fontsize=13)
    fig.tight_layout()

    out_path = os.path.join(run_dir, "thickness_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_smb_profile(run_dir, smb_vec, z_min, dz, root):
    """SMB profile: elevation (y) vs SMB (x)."""
    n = len(smb_vec)

    if z_min is not None and dz is not None:
        elevations = float(z_min) + np.arange(n) * float(dz)
        ylabel = "Elevation (m a.s.l.)"
    else:
        print(f"  WARNING: z_min.npy or dz.npy missing in {run_dir}, using bin index")
        elevations = np.arange(n)
        ylabel = "Bin index"

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(smb_vec, elevations, "o-", color="steelblue", linewidth=2, markersize=5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("SMB (m ice eq / yr)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    run_label = _run_label(run_dir, root)
    ax.set_title(f"SMB elevation profile  [{run_label}]")
    fig.tight_layout()

    out_path = os.path.join(run_dir, "smb_profile.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--root",
        default="outputs",
        help="Directory to search recursively for smb_inference results (default: outputs)",
    )
    parser.add_argument(
        "--input-nc",
        default=DEFAULT_INPUT_NC,
        help=f"Input .nc file used for z_min fallback (default: {DEFAULT_INPUT_NC})",
    )
    args = parser.parse_args()

    runs = find_smb_runs(args.root)
    if not runs:
        print(f"No smb_inference results found in {args.root}/")
        return

    print(f"Found {len(runs)} SMB inference run(s)\n")

    for run_dir in runs:
        print(f"[{_run_label(run_dir, args.root)}]")

        thk_obs = load_npy(os.path.join(run_dir, "thk_observed.npy"))
        thk_opt = load_npy(os.path.join(run_dir, "thk_optimized.npy"))
        smb_vec = load_npy(os.path.join(run_dir, "smb_vec.npy"))

        z_min = load_npy(os.path.join(run_dir, "z_min.npy"))
        if z_min is None and thk_obs is not None:
            z_min = _get_z_min_from_input(thk_obs, args.input_nc)

        dz = load_npy(os.path.join(run_dir, "dz.npy"))
        if dz is None:
            dz = _get_dz_from_hydra(run_dir)
        if dz is None:
            dz = DZ_DEFAULT

        if thk_obs is not None and thk_opt is not None:
            plot_thickness(run_dir, thk_obs, thk_opt, args.root)
        else:
            missing = []
            if thk_obs is None:
                missing.append("thk_observed.npy")
            if thk_opt is None:
                missing.append("thk_optimized.npy")
            print(f"  Skipping thickness plot: missing {', '.join(missing)}")

        if smb_vec is not None:
            plot_smb_profile(run_dir, smb_vec, z_min, dz, args.root)
        else:
            print(f"  Skipping SMB profile plot: missing smb_vec.npy")

        print()


if __name__ == "__main__":
    main()
