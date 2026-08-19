#!/usr/bin/env python3
"""
Initialisation-shock check for the transient SMB-inference pipeline.

Quantifies how far the glacier drifts from mechanical equilibrium when
integrated forward for one year under zero SMB, for the two candidate
starting states of Sect. 2 of the paper:

  t1 : the DERIVED start-epoch thickness H_{t1} = max(usurfstart - z_b, 0) *
       icemask (the state the forward SMB-inference run actually starts
       from, now that the data assimilation is anchored at t2).
  t2 : the DA-calibrated thickness H_{t2} itself (the state that WAS the
       forward run's start under the old t1-anchored DA, still shock-free
       by construction since it is the DA's own fit to the velocity data).

Both one-year replays are produced by two ``igm_run`` invocations of the
SAME production experiment YAML, with the SMB-inference stage reduced to a
pure one-year zero-SMB replay (smb_init_low=high=0, optimizer=adam,
learning_rate=0, nbitmax=1 -- the optimizer variable never moves off its
zero initial value, so ``thk_optimized.npy`` is exactly H after one forward
step of Eq. (1)-(3) under smb=0) -- see shock_check_run.sh for the exact
overrides used. This script does the (cheap) analysis+figure only; it does
not run IGM itself.

Usage:
    python pipeline/shock_check.py --glacier argentiere \\
        --t1-dir results/argentiere/shock_check/t1 \\
        --t2-dir results/argentiere/shock_check/t2 \\
        --input-nc data/input_argentiere.nc \\
        --out results/argentiere/shock_check
"""
import argparse
import json
import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _on_ice_stats(drift, mask):
    on = mask > 0.5
    d = drift[on]
    return {
        "n_cells": int(on.sum()),
        "mean_m": float(np.mean(d)),
        "rms_m": float(np.sqrt(np.mean(d ** 2))),
        "p95_abs_m": float(np.percentile(np.abs(d), 95)),
        "max_abs_m": float(np.max(np.abs(d))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glacier", required=True)
    ap.add_argument("--t1-dir", required=True,
                     help="igm_run dir for the H_t1 (derived-state) 1yr zero-smb replay")
    ap.add_argument("--t2-dir", required=True,
                     help="igm_run dir for the H_t2 (DA-state) 1yr zero-smb replay")
    ap.add_argument("--input-nc", required=True,
                     help="original input .nc, for the usurfstart (t1 DEM) field")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--display-name", default="", help="glacier name for the figure title")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- t1 (derived) state: geometry + 1yr replay -------------------------
    geo1 = xr.open_dataset(os.path.join(args.t1_dir, "geology-optimized.nc"))
    inp = xr.open_dataset(args.input_nc)
    # NB: the raw input .nc stores y descending while IGM's own outputs
    # (geology-optimized.nc) store y ascending -- reindex onto geo1's axes
    # rather than combine .values arrays directly, or usurfstart ends up
    # vertically flipped relative to topg/icemask.
    inp = inp.reindex(y=geo1["y"])
    topg = (geo1["usurf"] - geo1["thk"]).values  # z_b = usurf(t2) - H_t2
    icemask = geo1["icemask"].values
    usurfstart = inp["usurfstart"].values
    on_ice = (icemask > 0.5).astype(np.float32)
    H_t1 = np.maximum(usurfstart - topg, 0.0) * on_ice
    H_t1_after = np.load(os.path.join(args.t1_dir, "smb_inference", "thk_optimized.npy"))
    drift_t1 = H_t1_after - H_t1

    # --- t2 (DA) state: geometry + 1yr replay -------------------------------
    geo2 = xr.open_dataset(os.path.join(args.t2_dir, "geology-optimized.nc"))
    H_t2 = geo2["thk"].values
    icemask2 = geo2["icemask"].values
    H_t2_after = np.load(os.path.join(args.t2_dir, "smb_inference", "thk_optimized.npy"))
    drift_t2 = H_t2_after - H_t2

    stats = {
        "glacier": args.glacier,
        "H_t1_derived": _on_ice_stats(drift_t1, on_ice),
        "H_t2_DA": _on_ice_stats(drift_t2, icemask2),
    }
    with open(os.path.join(args.out, "shock_check_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    # --- figure --------------------------------------------------------------
    x = geo1["x"].values
    dx = float(x[1] - x[0])
    y = geo1["y"].values
    dy = float(y[1] - y[0])
    extent = [0, len(x) * dx / 1000, 0, len(y) * dy / 1000]

    drift_t1_m = np.where(on_ice > 0.5, drift_t1, np.nan)
    drift_t2_m = np.where(icemask2 > 0.5, drift_t2, np.nan)
    # Clip the colour scale to the 98th percentile of |drift| pooled over both
    # maps, so the bulk of the on-ice pattern stays visible; a handful of
    # margin outliers (see the histogram tails in panel c) then saturate the
    # colourbar rather than washing out the whole map.
    pooled = np.abs(np.concatenate([
        drift_t1_m[~np.isnan(drift_t1_m)], drift_t2_m[~np.isnan(drift_t2_m)]]))
    vmax = max(float(np.percentile(pooled, 98)), 1e-3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=140)

    im0 = axes[0].imshow(drift_t1_m, cmap="RdBu_r", origin="lower", extent=extent,
                          vmin=-vmax, vmax=vmax)
    axes[0].set_title(r"(a) $H_{t_1}$ (derived) drift after 1 yr, $smb$=0")
    axes[0].set_xlabel("x (km)")
    axes[0].set_ylabel("y (km)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="$\\Delta H$ (m)",
                 extend="both")

    im1 = axes[1].imshow(drift_t2_m, cmap="RdBu_r", origin="lower", extent=extent,
                          vmin=-vmax, vmax=vmax)
    axes[1].set_title(r"(b) $H_{t_2}$ (DA state) drift after 1 yr, $smb$=0")
    axes[1].set_xlabel("x (km)")
    axes[1].set_ylabel("y (km)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="$\\Delta H$ (m)",
                 extend="both")

    ax2 = axes[2]
    d1 = drift_t1[on_ice > 0.5]
    d2 = drift_t2[icemask2 > 0.5]
    bins = np.linspace(-vmax, vmax, 41)
    ax2.hist(d1, bins=bins, histtype="step", lw=2, color="#3a7ca8",
              label=fr"$H_{{t_1}}$: RMS={stats['H_t1_derived']['rms_m']:.2f} m")
    ax2.hist(d2, bins=bins, histtype="step", lw=2, color="#c8553d",
              label=fr"$H_{{t_2}}$: RMS={stats['H_t2_DA']['rms_m']:.2f} m")
    ax2.axvline(0, color="k", lw=0.7)
    ax2.set_xlabel(r"$\Delta H$ after 1 yr, $smb=0$ (m)")
    ax2.set_ylabel("on-ice grid cells")
    ax2.set_title("(c) On-ice thickness drift")
    ax2.legend(fontsize=9, loc="upper right")

    display_name = args.display_name or args.glacier.capitalize()
    fig.suptitle(f"Mechanical-equilibrium check — {display_name}", y=1.02)
    fig.tight_layout()
    out_path = os.path.join(args.out, "shock_check.png")
    fig.savefig(out_path, bbox_inches="tight")
    out_pdf = os.path.join(args.out, "shock_check.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[shock_check] wrote {out_path} and {out_pdf}")


if __name__ == "__main__":
    main()
