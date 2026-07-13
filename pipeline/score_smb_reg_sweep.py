#!/usr/bin/env python3
"""Score an SMB-regularisation sweep and overlay the continuity estimate.

Iterates the ``reg_*`` (or numbered) subdirs of a sweep dir. For each run it
draws the inferred profile (solid) and the continuity equivalent
``SMB = dh/dt + div(flux)`` (dashed, same color), and reports RMSE — overall
and restricted to the ablation zone — against the reference observations:

  * ``--mode cells``  (Rhône): repeat stake cells (>=N distinct years) from a
    rawstakes CSV, as in ``smb_vs_continuity.py``. Raw stakes drawn faint.
  * ``--mode points`` (Argentière): plain stake CSV (X, Y, Alt, Annual_SMB).

The best (lowest ablation RMSE) legend entry is bold. Writes
``smb_reg_sweep.png`` + ``smb_reg_sweep_summary.csv`` into the sweep dir.

Examples:
    python pipeline/score_smb_reg_sweep.py --sweep results/gpr/rhone_smbreg \
        --mode cells --stakes data/rhone/SMB_rhone_2006-2020_rawstakes_utm32N.csv
    python pipeline/score_smb_reg_sweep.py --sweep results/gpr/argentiere_smbreg \
        --mode points --stakes data/argentiere/SMB_argentiere_2012-2021_utm32N.csv \
        --dhdt-from-input data/input_argentiere.nc --dhdt-years 9
"""
import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(PROJECT_DIR, "pipeline")
for p in (PROJECT_DIR, HERE,
          os.path.join(PROJECT_DIR, "user", "code", "processes", "smb_inference")):
    sys.path.insert(0, p)
from data.homogenize import prepare_stake_overlay, draw_stake_overlay  # noqa: E402
from plot_smb_results import _bin_by_elevation, _load_da_fields  # noqa: E402
from compare_smb import load_profile  # noqa: E402


def read_reg(subdir):
    p = os.path.join(subdir, ".hydra", "overrides.yaml")
    if not os.path.exists(p):
        return None
    for line in open(p):
        m = re.search(r"regularisation=([0-9.eE+-]+)", line)
        if m:
            return float(m.group(1))
    return None


def continuity_profile(run_dir, z_min, dz, n, rho_ice, dhdt_override=None):
    """(dh/dt + div(flux)) binned onto the profile axis, in m w.e./yr."""
    dhdt = dhdt_override
    if dhdt is None:
        p = os.path.join(run_dir, "dhdt_obs.npy")
        dhdt = np.load(p) if os.path.exists(p) else None
    divflux, usurf, icemask = _load_da_fields(run_dir)
    if dhdt is None or divflux is None or usurf is None:
        return None
    mask = (icemask > 0.5) if icemask is not None else np.isfinite(usurf)
    mask = mask & np.isfinite(divflux) & np.isfinite(dhdt)
    cont = (_bin_by_elevation(dhdt, usurf, mask, z_min, dz, n)
            + _bin_by_elevation(divflux, usurf, mask, z_min, dz, n))
    return cont * (rho_ice / 1000.0)


def dhdt_from_input(input_nc, run_dir, years):
    """(usurfinfer - usurfobs)/years from the input NC, on the run's cropped grid."""
    import xarray as xr
    ds = xr.open_dataset(input_nc)
    geo = xr.open_dataset(os.path.join(run_dir, "..", "geology-optimized.nc"))
    sub = ds.sel(x=geo.x, y=geo.y, method="nearest")
    return ((sub["usurfinfer"].values - sub["usurfobs"].values) / float(years))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--mode", required=True, choices=["cells", "points"])
    ap.add_argument("--stakes", required=True)
    ap.add_argument("--rho-ice", type=float, default=917.0)
    ap.add_argument("--bin-m", type=float, default=100.0)
    ap.add_argument("--cluster-min-years", type=int, default=10)
    ap.add_argument("--dhdt-from-input", default=None,
                    help="input NC to derive dhdt=(usurfinfer-usurfobs)/years "
                         "when runs lack dhdt_obs.npy (Argentière)")
    ap.add_argument("--dhdt-years", type=float, default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    # Reference observations: altitudes + multi-year mean SMB (m w.e./yr).
    prep = None
    if args.mode == "cells":
        prep = prepare_stake_overlay(args.stakes, bin_m=args.bin_m,
                                     cluster_min_years=args.cluster_min_years)
        if prep is None or not prep["clu_alt"].size:
            sys.exit("no repeat cells found")
        ref_alt, ref_smb = prep["clu_alt"], prep["clu_mean"]
        ref_label = f"repeat cells (≥{args.cluster_min_years} yr)"
    else:
        import pandas as pd
        df = pd.read_csv(args.stakes)
        smb_col = next(c for c in df.columns if "SMB" in c)
        ref_alt = df["Alt"].values.astype(float)
        ref_smb = df[smb_col].values.astype(float)
        ref_label = "stakes"
    abl = ref_smb < 0
    print(f"reference: {ref_alt.size} {ref_label}, {int(abl.sum())} in ablation zone")

    def rmse(z, p, m):
        pr = np.interp(ref_alt[m], z, p, left=np.nan, right=np.nan)
        v = np.isfinite(pr)
        return (float(np.sqrt(np.mean((ref_smb[m][v] - pr[v]) ** 2)))
                if v.any() else np.nan)

    subdirs = sorted(
        (d for d in os.listdir(args.sweep)
         if os.path.exists(os.path.join(args.sweep, d, ".hydra", "overrides.yaml"))),
        key=lambda d: read_reg(os.path.join(args.sweep, d)) or 0)

    fig, ax = plt.subplots(figsize=(7, 9))
    cmap = plt.get_cmap("viridis")
    rows, handles = [], []
    for k, sd in enumerate(subdirs):
        run = os.path.join(args.sweep, sd, "smb_inference")
        if not os.path.exists(os.path.join(run, "smb_vec.npy")):
            print(f"  [{sd}] missing smb_vec.npy — skipped")
            continue
        reg = read_reg(os.path.join(args.sweep, sd))
        z, smb_we = load_profile(run, args.rho_ice)
        c = cmap(k / max(len(subdirs) - 1, 1))
        r_abl, r_all = rmse(z, smb_we, abl), rmse(z, smb_we, np.ones_like(abl))
        rows.append((reg, r_abl, r_all, sd))
        (h,) = ax.plot(smb_we, z, "-o", ms=3, color=c,
                       label=f"reg={reg:g}  abl-RMSE={r_abl:.2f}")
        handles.append(h)

        dhdt_ov = None
        if args.dhdt_from_input:
            dhdt_ov = dhdt_from_input(args.dhdt_from_input, run, args.dhdt_years)
        cont = continuity_profile(run, z[0], z[1] - z[0], len(z),
                                  args.rho_ice, dhdt_override=dhdt_ov)
        if cont is not None:
            ax.plot(cont, z, "--", color=c, lw=1.2, alpha=0.9)

    ax.plot([], [], "--", color="grey", label="dh/dt + div(flux) (continuity)")
    if args.mode == "cells":
        draw_stake_overlay(ax, prep)
    else:
        ax.plot(ref_smb, ref_alt, ".", color="grey", ms=6, alpha=0.7, label=ref_label)

    # Bold the best (lowest ablation RMSE) sweep entry in the legend.
    best = min((r for r in rows if np.isfinite(r[1])), key=lambda r: r[1])
    ax.axvline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3)
    ax.set_xlabel("SMB (m w.e./yr)")
    ax.set_ylabel("Elevation (m a.s.l.)")
    ax.set_title(args.title or f"SMB-reg sweep: {os.path.basename(args.sweep)}")
    leg = ax.legend(fontsize=8)
    for txt in leg.get_texts():
        if txt.get_text().startswith(f"reg={best[0]:g} "):
            txt.set_fontweight("bold")

    fig.tight_layout()
    png = os.path.join(args.sweep, "smb_reg_sweep.png")
    fig.savefig(png, dpi=130)

    rows.sort(key=lambda r: (not np.isfinite(r[1]), r[1]))
    csv = os.path.join(args.sweep, "smb_reg_sweep_summary.csv")
    with open(csv, "w") as f:
        f.write("reg,rmse_ablation,rmse_all,subdir\n")
        for r in rows:
            f.write(f"{r[0]:g},{r[1]:.4f},{r[2]:.4f},{r[3]}\n")
    print("\n reg     ablRMSE  allRMSE")
    for reg, r_abl, r_all, sd in rows:
        star = " <== best" if reg == best[0] else ""
        print(f" {reg:<6g}  {r_abl:6.3f}   {r_all:6.3f}{star}")
    print(f"\nwrote {csv}\nwrote {png}")


if __name__ == "__main__":
    main()
