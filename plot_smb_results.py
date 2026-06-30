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


def _bin_by_elevation(field, z_surf, mask, z_min, dz, n_bins):
    """Average a 2D field into elevation bins centered at z_min + i*dz.

    Bins match the SMB-profile elevation axis (centers z_min + i*dz, width dz).
    Returns a 1D array of length n_bins; empty bins are NaN.
    """
    profile = np.full(n_bins, np.nan, dtype=np.float64)
    valid = mask & np.isfinite(field) & np.isfinite(z_surf)
    if not np.any(valid):
        return profile

    idx = np.round((z_surf[valid] - z_min) / dz).astype(int)
    vals = field[valid].astype(np.float64)
    in_range = (idx >= 0) & (idx < n_bins)
    idx, vals = idx[in_range], vals[in_range]
    if idx.size == 0:
        return profile

    sums = np.bincount(idx, weights=vals, minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    nonempty = counts > 0
    profile[nonempty] = sums[nonempty] / counts[nonempty]
    return profile


def plot_mass_balance_diagnostic(run_dir, smb_vec, z_min, dz, usurf, thk_opt,
                                 dhdt, divflux, root):
    """Mass-balance diagnostic: SMB, dh/dt (Hugonnet) and the DA-derived flux
    divergence as elevation profiles on a common axis.

    The continuity equation governing glacier evolution is
        dh/dt + div(flux) = SMB,
    so the dashed (dh/dt + div(flux)) curve should track the inferred SMB
    where the model is consistent with the observed thinning. The flux
    divergence is the steady value from the data assimilation, not a
    time-mean of the SMB forward replay.
    """
    n = len(smb_vec)
    elevations = float(z_min) + np.arange(n) * float(dz)
    mask = thk_opt > 0.5  # restrict binning to on-ice pixels

    dhdt_prof = (
        _bin_by_elevation(dhdt, usurf, mask, float(z_min), float(dz), n)
        if dhdt is not None else None
    )
    div_prof = (
        _bin_by_elevation(divflux, usurf, mask, float(z_min), float(dz), n)
        if divflux is not None else None
    )

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(smb_vec, elevations, "o-", color="steelblue", linewidth=2,
            markersize=4, label="SMB (inferred)")
    if dhdt_prof is not None:
        ax.plot(dhdt_prof, elevations, "s-", color="firebrick", linewidth=2,
                markersize=4, label="dh/dt (obs)")
    if div_prof is not None:
        ax.plot(div_prof, elevations, "^-", color="seagreen", linewidth=2,
                markersize=4, label=r"$\nabla\!\cdot$flux (DA)")
    if dhdt_prof is not None and div_prof is not None:
        ax.plot(dhdt_prof + div_prof, elevations, "--", color="k", linewidth=1.5,
                alpha=0.7, label=r"dh/dt + $\nabla\!\cdot$flux")

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("rate (m / yr, ice eq.)")
    ax.set_ylabel("Elevation (m a.s.l.)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    run_label = _run_label(run_dir, root)
    ax.set_title(f"Mass-balance diagnostic  [{run_label}]\n"
                 r"dh/dt + $\nabla\!\cdot$flux = SMB")
    fig.tight_layout()

    out_path = os.path.join(run_dir, "mass_balance_diagnostic.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _load_da_fields(run_dir):
    """Load divflux / usurf / icemask from the DA output geology-optimized.nc.

    The combined DA+SMB run writes geology-optimized.nc one level above the
    smb_inference/ dir. Returns (divflux, usurf, icemask) as numpy arrays, or
    (None, None, None) if the file or divflux is missing.
    """
    nc_path = os.path.normpath(os.path.join(run_dir, "..", "geology-optimized.nc"))
    if not os.path.exists(nc_path):
        return None, None, None
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)
        divflux = ds["divflux"].values if "divflux" in ds else None
        usurf = ds["usurf"].values if "usurf" in ds else None
        icemask = ds["icemask"].values if "icemask" in ds else None
        return divflux, usurf, icemask
    except Exception as e:
        print(f"  Could not read {nc_path}: {e}")
        return None, None, None


def plot_smb_from_fluxdiv(run_dir, smb_vec, z_min, dz, dhdt, root):
    """Flux-divergence (geodetic) SMB reconstruction, run-free.

    Combines the DA-derived steady flux divergence (assumed constant over the
    inversion period) with Hugonnet dh/dt via the continuity equation
        SMB = dh/dt + div(flux),
    saves a 3-panel map (dh/dt, div(flux), reconstructed SMB), then bins the
    reconstructed SMB by elevation and overlays it on the inferred profile.
    """
    divflux, usurf, icemask = _load_da_fields(run_dir)
    if divflux is None or usurf is None or dhdt is None:
        missing = []
        if divflux is None:
            missing.append("geology-optimized.nc:divflux")
        if usurf is None:
            missing.append("geology-optimized.nc:usurf")
        if dhdt is None:
            missing.append("dhdt_obs.npy")
        print(f"  Skipping flux-divergence SMB: missing {', '.join(missing)}")
        return

    if icemask is not None:
        mask = icemask > 0.5
    else:
        mask = np.isfinite(usurf)
    mask = mask & np.isfinite(divflux) & np.isfinite(dhdt)

    smb_recon = dhdt + divflux  # continuity: SMB = dh/dt + div(flux)

    run_label = _run_label(run_dir, root)

    # ── 3-panel map (on-ice only) ────────────────────────────────────────────
    def _masked(field):
        return np.where(mask, field, np.nan)

    fields = [
        (_masked(dhdt), "dh/dt (obs)"),
        (_masked(divflux), r"$\nabla\!\cdot$flux (DA, steady)"),
        (_masked(smb_recon), r"SMB = dh/dt + $\nabla\!\cdot$flux"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (field, title) in zip(axes, fields):
        finite = field[np.isfinite(field)]
        vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        im = ax.imshow(field, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x (grid)")
        ax.set_ylabel("y (grid)")
        plt.colorbar(im, ax=ax, label="m / yr", fraction=0.046, pad=0.04)
    fig.suptitle(f"Flux-divergence SMB reconstruction  [{run_label}]", fontsize=13)
    fig.tight_layout()
    map_path = os.path.join(run_dir, "smb_from_fluxdiv_maps.png")
    fig.savefig(map_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {map_path}")

    # ── profile comparison vs inferred SMB ────────────────────────────────────
    n = len(smb_vec)
    elevations = float(z_min) + np.arange(n) * float(dz)
    recon_prof = _bin_by_elevation(smb_recon, usurf, mask, float(z_min), float(dz), n)

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(smb_vec, elevations, "o-", color="steelblue", linewidth=2,
            markersize=4, label="SMB (inferred)")
    ax.plot(recon_prof, elevations, "D-", color="darkorange", linewidth=2,
            markersize=4, label=r"SMB = dh/dt + $\nabla\!\cdot$flux (DA)")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("SMB (m / yr, ice eq.)")
    ax.set_ylabel("Elevation (m a.s.l.)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"Inferred vs flux-divergence SMB  [{run_label}]")
    fig.tight_layout()
    prof_path = os.path.join(run_dir, "smb_from_fluxdiv_profile.png")
    fig.savefig(prof_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {prof_path}")


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

        # Mass-balance diagnostic: SMB / dh/dt / div(flux) elevation profiles.
        # div(flux) is the steady value from the DA (geology-optimized.nc),
        # not a time-mean of the SMB forward replay.
        usurf = load_npy(os.path.join(run_dir, "usurf_modeled.npy"))
        dhdt = load_npy(os.path.join(run_dir, "dhdt_obs.npy"))
        divflux, _, _ = _load_da_fields(run_dir)

        if (smb_vec is not None and usurf is not None and thk_opt is not None
                and z_min is not None and (dhdt is not None or divflux is not None)):
            plot_mass_balance_diagnostic(
                run_dir, smb_vec, z_min, dz, usurf, thk_opt,
                dhdt, divflux, args.root,
            )
        else:
            missing = []
            if usurf is None:
                missing.append("usurf_modeled.npy")
            if dhdt is None and divflux is None:
                missing.append("dhdt_obs.npy/geology-optimized.nc:divflux")
            if z_min is None:
                missing.append("z_min")
            if missing:
                print(f"  Skipping mass-balance diagnostic: missing "
                      f"{', '.join(missing)}")

        # Flux-divergence SMB reconstruction (run-free): DA divflux + Hugonnet
        # dh/dt -> SMB map + profile, compared to the inferred profile.
        if smb_vec is not None and z_min is not None:
            plot_smb_from_fluxdiv(run_dir, smb_vec, z_min, dz, dhdt, args.root)

        print()


if __name__ == "__main__":
    main()
