#!/usr/bin/env python3
"""Compare a finished SMB-inference profile against the continuity estimate.

This is the ``compare_smb`` figure with the GLAMOS product replaced by the
``dh/dt + div(flux)`` profile from ``mass_balance_diagnostic``.  The continuity
equation governing glacier evolution is

    dh/dt + div(flux) = SMB,

so the dashed (dh/dt + div(flux)) curve is an independent, geometry-based SMB
estimate that should track the inferred profile.

Curves (all in m w.e./yr so they are directly comparable to the stakes):
  * black ●–line  — inferred SMB profile (our method)
  * faint grey    — raw single-year stakes
  * green ◆ ± bar — repeat cells observed in many years (continuous-record
                    anchors; mean ± interannual std; assumption-free)
  * dashed line   — dh/dt + div(flux), binned by elevation, where div(flux)
                    is the steady flux divergence from the data assimilation
                    (geology-optimized.nc)

RMSE is reported against the green repeat-cell *means* for both
(a) the inferred profile and (b) the dh/dt + div(flux) estimate.

Example:
    python pipeline/smb_vs_continuity.py \
        --run results/rhone_insitu/final/smb_inference \
        --stakes data/rhone_insitu/SMB_rhone_insitu_2007-2016_rawstakes_utm32N.csv
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "user", "code", "processes", "smb_inference"))
from data.homogenize import prepare_stake_overlay, draw_stake_overlay  # noqa: E402
from plot_smb_results import _bin_by_elevation, _load_da_fields  # noqa: E402
from compare_smb import load_profile  # noqa: E402


def continuity_profile(run_dir, z_min, dz, n, rho_ice):
    """Bin (dh/dt + div(flux)) by elevation onto the profile axis (m w.e./yr).

    Mirrors ``plot_smb_results.plot_smb_from_fluxdiv``: dh/dt (obs) and the
    DA-derived steady flux divergence (read from ``geology-optimized.nc`` one
    level above ``run_dir``) are averaged into the same elevation bins as the
    inferred profile, summed via the continuity equation, then converted from
    ice eq to water eq.  Returns (continuity_we, n_bins_used) or (None, 0) if
    the required fields are missing.
    """
    dhdt = _load(run_dir, "dhdt_obs.npy")
    divflux, usurf, icemask = _load_da_fields(run_dir)
    if dhdt is None or divflux is None or usurf is None:
        return None, 0
    mask = (icemask > 0.5) if icemask is not None else np.isfinite(usurf)
    mask = mask & np.isfinite(divflux) & np.isfinite(dhdt)
    dhdt_prof = _bin_by_elevation(dhdt, usurf, mask, float(z_min), float(dz), n)
    div_prof = _bin_by_elevation(divflux, usurf, mask, float(z_min), float(dz), n)
    cont = dhdt_prof + div_prof              # m ice eq / yr
    cont_we = cont * (rho_ice / 1000.0)      # -> m w.e. / yr
    return cont_we, int(np.isfinite(cont_we).sum())


def _load(run_dir, name):
    p = os.path.join(run_dir, name)
    return np.load(p) if os.path.exists(p) else None


def _interp_at(curve_z, curve_v, alt):
    """Interpolate a (possibly NaN-gapped) profile to the cell altitudes.

    Cells outside the curve's finite range get NaN (no extrapolation)."""
    fin = np.isfinite(curve_v)
    if fin.sum() < 2 or alt.size == 0:
        return np.full(alt.shape, np.nan)
    return np.interp(alt, curve_z[fin], curve_v[fin], left=np.nan, right=np.nan)


def _rmse(values_at, clu_mean, mask):
    """RMSE of one interpolated curve against the cell means over ``mask``."""
    if not np.any(mask):
        return None
    return float(np.sqrt(np.mean((clu_mean[mask] - values_at[mask]) ** 2)))


def prepare_cells(stakes_path, bin_m, cluster_min_years, relax_floor):
    """Repeat-cell overlay, auto-relaxing min_years until >=2 cells exist.

    Short records (e.g. Aletsch in-house 2009-2017, 9 yr) have no cell observed
    in the default 10 distinct years, so the threshold is lowered toward
    ``relax_floor`` until at least two continuous-record anchors appear.
    Returns (prep, used_min_years).
    """
    used = cluster_min_years
    prep = prepare_stake_overlay(stakes_path, bin_m=bin_m, cluster_min_years=used)
    while (prep is None or prep["clu_alt"].size < 2) and used > relax_floor:
        used -= 1
        prep = prepare_stake_overlay(stakes_path, bin_m=bin_m, cluster_min_years=used)
    return prep, used


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="smb_inference dir (smb_vec.npy, z_min.npy, dz.npy, "
                         "dhdt_obs.npy); div(flux) is read from the DA output "
                         "../geology-optimized.nc")
    ap.add_argument("--stakes", required=True, help="raw-stakes CSV")
    ap.add_argument("--out", default=None,
                    help="output PNG (default <run>/smb_vs_continuity.png)")
    ap.add_argument("--rho-ice", type=float, default=917.0, help="ice density kg m-3")
    ap.add_argument("--bin-m", type=float, default=100.0, help="stake elevation bin width (m)")
    ap.add_argument("--cluster-min-years", type=int, default=10,
                    help="min distinct years for a repeat-cell anchor")
    ap.add_argument("--cluster-min-years-floor", type=int, default=4,
                    help="lowest min-years to auto-relax to if too few cells")
    ap.add_argument("--title", default=None, help="figure title suffix (e.g. glacier name)")
    args = ap.parse_args()

    z, smb_we = load_profile(args.run, args.rho_ice)        # inferred, m w.e./yr
    n = z.shape[0]
    z_min, dz = float(z[0]), float(z[1] - z[0]) if n > 1 else 200.0

    prep, used_my = prepare_cells(args.stakes, args.bin_m,
                                  args.cluster_min_years, args.cluster_min_years_floor)
    if prep is None:
        sys.exit(f"[smb_vs_continuity] no usable stakes in {args.stakes}")
    if used_my != args.cluster_min_years:
        print(f"[smb_vs_continuity] relaxed cluster-min-years "
              f"{args.cluster_min_years} -> {used_my} to get "
              f"{prep['clu_alt'].size} repeat cells")
    if prep["clu_alt"].size == 0:
        print("[smb_vs_continuity] warning: no repeat cells; RMSE unavailable")

    cont_we, n_cont_bins = continuity_profile(args.run, z_min, dz, n, args.rho_ice)

    fig, ax = plt.subplots(figsize=(6, 9))
    ax.plot(smb_we, z, "-o", color="k", markersize=3, label="inferred profile (our method)")
    draw_stake_overlay(ax, prep)  # faded grey raw stakes + green repeat cells
    if cont_we is not None:
        ax.plot(cont_we, z, "--", color="firebrick", linewidth=2.0, zorder=4,
                label=r"dh/dt + $\nabla\!\cdot$flux")
    else:
        print("[smb_vs_continuity] continuity fields missing; dashed line skipped")

    # Both RMSEs are scored against the SAME repeat cells — those where both
    # curves are defined — so the two numbers are directly comparable.
    clu_alt, clu_mean = prep["clu_alt"], prep["clu_mean"]
    inf_at = _interp_at(z, smb_we, clu_alt)
    cont_at = _interp_at(z, cont_we, clu_alt) if cont_we is not None \
        else np.full(clu_alt.shape, np.nan)
    common = np.isfinite(inf_at)
    if cont_we is not None:
        common &= np.isfinite(cont_at)
    n_cells = int(common.sum())

    rmse_inf = _rmse(inf_at, clu_mean, common)
    rmse_cont = _rmse(cont_at, clu_mean, common) if cont_we is not None else None
    if rmse_inf is not None:
        ax.plot([], [], " ",
                label=f"RMSE={rmse_inf:.3f} inferred vs cells ({n_cells} cells)")
    if rmse_cont is not None:
        ax.plot([], [], " ",
                label=f"RMSE={rmse_cont:.3f} (dh/dt+$\\nabla\\!\\cdot$flux) vs cells")

    ax.axvline(0.0, color="k", linewidth=0.5)
    ax.set_xlabel("SMB (m w.e./yr)")
    ax.set_ylabel("Elevation (m a.s.l.)")
    title = "Inferred SMB vs continuity (dh/dt + " + r"$\nabla\!\cdot$flux)"
    if args.title:
        title += f"\n{args.title}"
    ax.set_title(title, fontsize=11)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = args.out or os.path.join(args.run, "smb_vs_continuity.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)

    if rmse_inf is not None:
        print(f"[smb_vs_continuity] RMSE inferred vs repeat cells = "
              f"{rmse_inf:.3f} m w.e./yr ({n_cells} cells)")
    if rmse_cont is not None:
        print(f"[smb_vs_continuity] RMSE (dh/dt+div) vs repeat cells = "
              f"{rmse_cont:.3f} m w.e./yr ({n_cells} cells)")
    print(f"[smb_vs_continuity] wrote {out}")


if __name__ == "__main__":
    main()
