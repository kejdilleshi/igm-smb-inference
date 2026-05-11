#!/usr/bin/env python3
"""
Plot inferred SMB(z) profile (200 m bins from smb_inference) against
GLAMOS observed mean annual mass balance per 100 m elevation bin for
Grosser Aletschgletscher (B36-26) over 2000-01-01 → 2020-01-01.

The inferred profile is in m ice eq./yr (the trainable variable lives in
ice-eq units); we convert to m w.e./yr (× ρ_ice/ρ_w ≈ 0.917) so it lines
up with GLAMOS's mm w.e. → m w.e.

Usage:
    python plot_smb_vs_glamos.py <smb_inference_output_dir> [out_png]

    smb_inference_output_dir is the directory containing
    `smb_inference_result.npz` (or a folder of smb_vec_iter_*.npy / the
    final smb_vec). Defaults are detected automatically below.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GLAMOS_CSV = os.path.normpath(os.path.join(HERE, "..", "massbalance_observation_elevationbins.csv"))
GLACIER_ID = "B36-26"
START_YEAR = 2000
END_YEAR = 2020
RHO_ICE_OVER_W = 0.917   # m ice eq. → m w.e.


def load_glamos_profile():
    skip = 0
    with open(GLAMOS_CSV) as f:
        for i, line in enumerate(f):
            if line.startswith("glacier name,"):
                skip = i
                break
    df = pd.read_csv(GLAMOS_CSV, skiprows=skip, low_memory=False)
    df = df[df["glacier id"].astype(str).str.match(r"^[A-Z]")].copy()
    df.rename(columns={
        "start date of observation": "date_start",
        "end date of observation": "date_end",
        "annual mass balance": "Ba",
        "lower elevation of bin": "h_min",
        "upper elevation of bin": "h_max",
        "glacier id": "glacier_id",
        "area of elevation bin": "area",
    }, inplace=True)
    df = df[df["glacier_id"] == GLACIER_ID]
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce")
    for c in ("Ba", "h_min", "h_max", "area"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mask = ((df["date_start"] >= pd.Timestamp(f"{START_YEAR-1}-09-15"))
            & (df["date_end"]   <= pd.Timestamp(f"{END_YEAR}-10-15")))
    df = df[mask].dropna(subset=["Ba", "h_min", "h_max"])
    n_years = df["date_start"].dt.year.nunique()
    print(f"GLAMOS {GLACIER_ID}: {len(df)} rows, {n_years} hydro years")

    # Average Ba per 100 m bin, convert mm w.e. → m w.e.
    grp = df.groupby(["h_min", "h_max"], as_index=False).agg(
        Ba=("Ba", "mean"),
        Ba_std=("Ba", "std"),
        area=("area", "mean"),
        n_years=("Ba", "count"),
    )
    grp["alt"] = 0.5 * (grp["h_min"] + grp["h_max"])
    grp["Ba_mwe"] = grp["Ba"] / 1000.0
    grp["Ba_std_mwe"] = grp["Ba_std"] / 1000.0
    grp = grp.sort_values("alt").reset_index(drop=True)
    return grp


def load_inferred_profile(smb_inference_dir):
    """Load the final SMB vector and (z_min, dz) from smb_inference outputs.

    Expected files written by user/code/processes/smb_inference/smb_inference.py
    when save_results=true:
        smb_vec.npy   — final SMB profile (m ice eq./yr, length = #bins)
        z_min.npy     — minimum elevation of the binning (scalar)
        dz.npy        — bin width (scalar, e.g. 200.0)
    """
    smb_vec_path = os.path.join(smb_inference_dir, "smb_vec.npy")
    z_min_path   = os.path.join(smb_inference_dir, "z_min.npy")
    dz_path      = os.path.join(smb_inference_dir, "dz.npy")
    if not os.path.exists(smb_vec_path):
        raise FileNotFoundError(f"missing {smb_vec_path}")
    smb = np.load(smb_vec_path)
    z_min = float(np.load(z_min_path))
    dz = float(np.load(dz_path))
    z_axis = z_min + dz * np.arange(len(smb))
    smb_we = smb * RHO_ICE_OVER_W                # ice → water equivalent
    return z_axis, smb_we, z_min, dz


def main(input_path, out_png=None):
    obs = load_glamos_profile()
    smb_inference_dir = (input_path if os.path.isdir(input_path)
                         else os.path.dirname(input_path))
    print(f"Loading inferred SMB from: {smb_inference_dir}")
    z_inf, smb_we_inf, z_min, dz = load_inferred_profile(smb_inference_dir)

    # Stats: RMSE between inferred (200 m) profile interpolated to each
    # GLAMOS 100 m bin centroid.
    smb_at_obs = np.interp(obs["alt"].values, z_inf, smb_we_inf,
                           left=np.nan, right=np.nan)
    valid = np.isfinite(smb_at_obs)
    resid = obs["Ba_mwe"].values[valid] - smb_at_obs[valid]
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    bias = float(np.mean(resid))
    print(f"Comparison stats over {valid.sum()} GLAMOS bins: "
          f"bias={bias:+.3f} m w.e./yr, RMSE={rmse:.3f} m w.e./yr")

    # Plot — elevation on y axis (standard SMB profile convention).
    fig, ax = plt.subplots(figsize=(6.5, 8.5))
    ax.errorbar(obs["Ba_mwe"], obs["alt"],
                xerr=obs["Ba_std_mwe"],
                fmt="o", color="tab:red", markersize=5, capsize=2,
                ecolor="lightcoral", elinewidth=1, zorder=3,
                label=f"GLAMOS B36-26 {START_YEAR}-{END_YEAR}\n(100 m bins, mean ± std)")
    ax.plot(smb_we_inf, z_inf, "-o", color="tab:blue", markersize=5,
            linewidth=1.7, zorder=4,
            label=f"Inferred SMB(z) (200 m bins, dz={dz:g})")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Annual SMB (m w.e./yr)")
    ax.set_ylabel("Elevation (m a.s.l.)")
    ax.set_title(f"Aletsch — SMB profile {START_YEAR}-{END_YEAR}\n"
                 f"Inferred from Hugonnet dhdt + OGGM dynamics vs GLAMOS in-situ\n"
                 f"bias={bias:+.3f}, RMSE={rmse:.3f} m w.e./yr "
                 f"({int(valid.sum())} bins)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    if out_png is None:
        out_png = os.path.join(smb_inference_dir,
                               f"smb_profile_vs_glamos_{START_YEAR}-{END_YEAR}.png")
    fig.savefig(out_png, dpi=130)
    print(f"\nWrote: {out_png}")

    # Also save the comparison table as CSV.
    csv_out = out_png.replace(".png", ".csv")
    table = obs.copy()
    table["inferred_SMB_mwe"] = smb_at_obs
    table["residual_mwe"] = table["Ba_mwe"] - table["inferred_SMB_mwe"]
    table.to_csv(csv_out, index=False)
    print(f"Wrote: {csv_out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: plot_smb_vs_glamos.py <smb_inference_output_dir> [out_png]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
