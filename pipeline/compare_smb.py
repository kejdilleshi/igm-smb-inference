#!/usr/bin/env python3
"""Compare a finished SMB-inference profile against time-homogenized stakes.

The inferred profile is a multi-year mean; raw stakes are single-year point
balances. This regenerates the comparison plot from a saved run without rerunning
inference, using the temporal homogenization in
``user/code/processes/smb_inference/data/homogenize.py``:

  * faint grey   — raw single-year stakes
  * blue ● ± bar — stakes homogenized to the observed-year mean, binned by elevation
  * green ◆ ± bar — repeat cells observed in many years (assumption-free anchor)

Example:
    python pipeline/compare_smb.py \
        --run results/rhone/final/smb_inference \
        --stakes data/rhone/SMB_rhone_2006-2020_rawstakes_utm32N.csv \
        --out results/rhone/smb_comparison.png
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
sys.path.insert(0, os.path.join(PROJECT_DIR, "user", "code", "processes", "smb_inference"))
from data.homogenize import prepare_stake_overlay, draw_stake_overlay  # noqa: E402

# GLAMOS published "observation-period elevation bins" product.
GLAMOS_DEFAULT_CSV = os.path.join(PROJECT_DIR, "data",
                                  "massbalance_observation_elevationbins.csv")
# GLAMOS glacier id (Swiss Glacier Inventory) keyed by our glacier name.
GLAMOS_IDS = {"rhone": "B43-03", "aletsch": "B36-26"}


def infer_glamos_meta(stakes_path):
    """(glamos_id, start_year, end_year) from a SMB_<g>_<y0>-<y1>_rawstakes name."""
    m = re.search(r"SMB_(.+?)_(\d{4})-(\d{4})_rawstakes", os.path.basename(stakes_path))
    if not m:
        return None, None, None
    name = m.group(1).replace("_insitu", "")
    return GLAMOS_IDS.get(name), int(m.group(2)), int(m.group(3))


def load_glamos_elevationbins(csv_path, glacier_id, start_year, end_year):
    """GLAMOS published per-elevation-bin annual balance for one glacier/period.

    Returns a DataFrame with columns alt, Ba_mwe, Ba_std_mwe (m w.e./yr), or
    None if nothing matches. Ported from data/oggm/plot_smb_vs_glamos.py.
    """
    import pandas as pd
    skip = 0
    with open(csv_path) as f:
        for i, line in enumerate(f):
            if line.startswith("glacier name,"):
                skip = i
                break
    df = pd.read_csv(csv_path, skiprows=skip, low_memory=False)
    df = df[df["glacier id"].astype(str) == glacier_id].copy()
    if df.empty:
        return None
    df.rename(columns={"start date of observation": "date_start",
                       "end date of observation": "date_end",
                       "annual mass balance": "Ba",
                       "lower elevation of bin": "h_min",
                       "upper elevation of bin": "h_max"}, inplace=True)
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce")
    for c in ("Ba", "h_min", "h_max"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    mask = ((df["date_start"] >= pd.Timestamp(f"{start_year - 1}-09-15"))
            & (df["date_end"] <= pd.Timestamp(f"{end_year}-10-15")))
    df = df[mask].dropna(subset=["Ba", "h_min", "h_max"])
    if df.empty:
        return None
    grp = df.groupby(["h_min", "h_max"], as_index=False).agg(
        Ba=("Ba", "mean"), Ba_std=("Ba", "std"))
    grp["alt"] = 0.5 * (grp["h_min"] + grp["h_max"])
    grp["Ba_mwe"] = grp["Ba"] / 1000.0                      # mm w.e. -> m w.e.
    grp["Ba_std_mwe"] = grp["Ba_std"].fillna(0.0) / 1000.0
    return grp.sort_values("alt").reset_index(drop=True)


def load_profile(run_dir, rho_ice):
    """Load smb_vec (m ice eq) + z grid from a run dir, return (z, smb m w.e.)."""
    smb_vec = np.load(os.path.join(run_dir, "smb_vec.npy"))
    z_min = float(np.load(os.path.join(run_dir, "z_min.npy")))
    dz = float(np.load(os.path.join(run_dir, "dz.npy")))
    z = z_min + dz * np.arange(smb_vec.shape[0])
    smb_we = smb_vec * (rho_ice / 1000.0)  # m ice eq -> m w.e.
    return z, smb_we


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="dir with smb_vec.npy, z_min.npy, dz.npy")
    ap.add_argument("--stakes", required=True, help="raw-stakes CSV")
    ap.add_argument("--out", default=None, help="output PNG (default <run>/smb_comparison.png)")
    ap.add_argument("--rho-ice", type=float, default=917.0, help="ice density kg m-3")
    ap.add_argument("--bin-m", type=float, default=100.0, help="elevation bin width (m)")
    ap.add_argument("--cluster-min-years", type=int, default=10,
                    help="min distinct years for a repeat-cell anchor")
    ap.add_argument("--glamos-csv", default=GLAMOS_DEFAULT_CSV,
                    help="GLAMOS elevation-bins product CSV")
    ap.add_argument("--glamos-id", default=None,
                    help="GLAMOS glacier id (e.g. B43-03); inferred from --stakes if omitted")
    ap.add_argument("--glamos-start", type=int, default=None,
                    help="GLAMOS period start year (inferred from --stakes if omitted)")
    ap.add_argument("--glamos-end", type=int, default=None,
                    help="GLAMOS period end year (inferred from --stakes if omitted)")
    args = ap.parse_args()

    z, smb_we = load_profile(args.run, args.rho_ice)
    prep = prepare_stake_overlay(args.stakes, bin_m=args.bin_m,
                                 cluster_min_years=args.cluster_min_years)
    if prep is None:
        sys.exit(f"[compare_smb] no usable stakes in {args.stakes}")

    fig, ax = plt.subplots(figsize=(6, 9))
    ax.plot(smb_we, z, "-o", color="k", markersize=3, label="inferred profile")
    draw_stake_overlay(ax, prep)  # faded grey raw stakes + green anchors

    # GLAMOS published elevation-band balance (blue dots ± interannual std).
    gid, y0, y1 = args.glamos_id, args.glamos_start, args.glamos_end
    if gid is None or y0 is None or y1 is None:
        i_gid, i_y0, i_y1 = infer_glamos_meta(args.stakes)
        gid = gid or i_gid
        y0 = y0 or i_y0
        y1 = y1 or i_y1
    glamos_rmse = None
    if gid and y0 and y1 and os.path.exists(args.glamos_csv):
        obs = load_glamos_elevationbins(args.glamos_csv, gid, y0, y1)
        if obs is not None and len(obs):
            ax.errorbar(obs["Ba_mwe"], obs["alt"], xerr=obs["Ba_std_mwe"],
                        fmt="o", color="tab:blue", ecolor="tab:blue",
                        elinewidth=1.0, capsize=2, markersize=5, zorder=5,
                        label=f"GLAMOS {gid} elev-bins ({y0}–{y1})")
            prof_at = np.interp(obs["alt"].values, z, smb_we,
                                left=np.nan, right=np.nan)
            v = np.isfinite(prof_at)
            if v.any():
                glamos_rmse = float(np.sqrt(np.mean(
                    (obs["Ba_mwe"].values[v] - prof_at[v]) ** 2)))
                ax.plot([], [], " ",
                        label=f"RMSE={glamos_rmse:.3f} vs GLAMOS ({int(v.sum())} bins)")
        else:
            print(f"[compare_smb] no GLAMOS bins for {gid} {y0}-{y1}")
    elif gid is None:
        print(f"[compare_smb] could not infer GLAMOS id from {args.stakes}; "
              "skipping product overlay")

    ax.axvline(0.0, color="k", linewidth=0.5)
    ax.set_xlabel("SMB (m w.e./yr)")
    ax.set_ylabel("Elevation (m a.s.l.)")
    ax.set_title("Inferred SMB profile vs GLAMOS elevation-band balance")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = args.out or os.path.join(args.run, "smb_comparison.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)

    if glamos_rmse is not None:
        print(f"[compare_smb] RMSE vs GLAMOS product ({gid} {y0}-{y1}) = "
              f"{glamos_rmse:.3f} m w.e./yr")
    print(f"[compare_smb] wrote {out}")


if __name__ == "__main__":
    main()
