"""
Validate Data Assimilation results.

Compares true vs reconstructed:
  - Surface velocity magnitude
  - Ice thickness

Each comparison shows: true | reconstructed | residual (reconstructed - true)

Usage:
    python validate_da.py \
        --input  ../data/input.nc \
        --output outputs/2026-03-26/14-16-00/geology-optimized.nc
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import netCDF4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_var(ds, *names):
    """Return the first variable name found in ds, squeezed to 2-D."""
    for name in names:
        if name in ds.variables:
            return np.squeeze(ds.variables[name][:])
    raise KeyError(f"None of {names} found in {ds.filepath()}")


def mask_noice(arr, icemask, fill=np.nan):
    """Set pixels outside the ice mask to fill value."""
    out = arr.astype(float).copy()
    out[icemask == 0] = fill
    return out


def plot_trio(ax_true, ax_recon, ax_res,
              true, recon, residual,
              title_true, title_recon, title_res,
              cmap_main="viridis", cmap_res="RdBu_r",
              vmin=None, vmax=None):
    """Plot true / reconstructed / residual into three axes."""
    vmin = vmin if vmin is not None else np.nanmin(true)
    vmax = vmax if vmax is not None else np.nanmax(true)
    res_abs = np.nanmax(np.abs(residual))

    im0 = ax_true.imshow(true,  origin="lower", cmap=cmap_main, vmin=vmin, vmax=vmax)
    im1 = ax_recon.imshow(recon, origin="lower", cmap=cmap_main, vmin=vmin, vmax=vmax)
    im2 = ax_res.imshow(residual, origin="lower", cmap=cmap_res,
                        vmin=-res_abs, vmax=res_abs)

    for ax, im, title in [(ax_true, im0, title_true),
                          (ax_recon, im1, title_recon),
                          (ax_res, im2, title_res)]:
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate DA results")
    script_dir = Path(__file__).parent
    parser.add_argument("--input",  default=str(script_dir / "data/input.nc"),
                        help="Path to input NetCDF (contains true thk, obs velocities)")
    parser.add_argument("--output", required=True,
                        help="Path to geology-optimized.nc (DA result)")
    parser.add_argument("--save",   default="da_validation.png",
                        help="Output figure path (default: da_validation.png)")
    args = parser.parse_args()

    # --- Load data ----------------------------------------------------------
    ds_in  = netCDF4.Dataset(args.input)
    ds_out = netCDF4.Dataset(args.output)

    # True thickness from input
    thk_true = load_var(ds_in, "thk")

    # True velocity magnitude computed from obs components
    u_obs = load_var(ds_in, "uvelsurfobs")
    v_obs = load_var(ds_in, "vvelsurfobs")
    vel_true = np.sqrt(u_obs**2 + v_obs**2)

    # Reconstructed fields from DA output
    thk_recon = load_var(ds_out, "thk")
    vel_recon = load_var(ds_out, "velsurf_mag")

    # Ice mask (use output mask as reference domain)
    icemask = load_var(ds_out, "icemask")

    ds_in.close()
    ds_out.close()

    # --- Apply ice mask -----------------------------------------------------
    thk_true_m  = mask_noice(thk_true,  icemask)
    thk_recon_m = mask_noice(thk_recon, icemask)
    vel_true_m  = mask_noice(vel_true,  icemask)
    vel_recon_m = mask_noice(vel_recon, icemask)

    thk_res = thk_recon_m - thk_true_m
    vel_res = vel_recon_m - vel_true_m

    # --- Stats --------------------------------------------------------------
    def stats(res, name):
        valid = res[~np.isnan(res)]
        print(f"{name:30s}  MAE={np.mean(np.abs(valid)):7.2f}  "
              f"RMSE={np.sqrt(np.mean(valid**2)):7.2f}  "
              f"bias={np.mean(valid):+7.2f}  "
              f"max|err|={np.max(np.abs(valid)):7.2f}")

    print("\n--- DA Validation ---")
    print(f"{'Field':<30}  {'MAE':>8}  {'RMSE':>8}  {'bias':>8}  {'max|err|':>10}")
    stats(thk_res, "Thickness  [m]")
    stats(vel_res, "Velocity   [m/yr]")

    # --- Figure -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Data Assimilation Validation", fontsize=13, fontweight="bold")

    # Row 0: velocity
    plot_trio(axes[0, 0], axes[0, 1], axes[0, 2],
              vel_true_m, vel_recon_m, vel_res,
              "Velocity — true (obs) [m/yr]",
              "Velocity — DA reconstructed [m/yr]",
              "Velocity residual [m/yr]\n(reconstructed − true)",
              cmap_main="plasma")

    # Row 1: thickness
    plot_trio(axes[1, 0], axes[1, 1], axes[1, 2],
              thk_true_m, thk_recon_m, thk_res,
              "Thickness — true [m]",
              "Thickness — DA reconstructed [m]",
              "Thickness residual [m]\n(reconstructed − true)",
              cmap_main="Blues")

    plt.tight_layout()
    out_path = Path(args.save)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
