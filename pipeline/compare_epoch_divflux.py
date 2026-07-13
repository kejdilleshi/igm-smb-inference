#!/usr/bin/env python3
"""Compare DA-inferred fields between the 2012 and 2021 Argentiere surfaces.

Both runs are the *same* velocity-only DA (slidingco=0.2689, arrhenius=78,
reg.thk=328.17, control=[thk], cost=[velsurf,icemask]); the only difference is
the target surface DEM (usurf): 2012 vs 2021 (usurfinfer). This quantifies how
much the flux-divergence (and thickness / surface velocity) fields change
between the two epochs.

Outputs (into --out-dir):
  epoch_divflux_maps.png   3x3 maps: divflux / thk / velsurf_mag, cols 2012|2021|delta
  epoch_divflux_profile.png elevation-binned divflux 2012 vs 2021 (+ their delta)
  epoch_metrics.csv         per-field on-ice metrics
"""
import argparse, os, sys, csv
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from plot_smb_results import _bin_by_elevation  # noqa: E402

RHO = 917.0  # ice density; divflux(ice eq) -> w.e. via RHO/1000


def load(nc):
    ds = xr.open_dataset(nc)
    return {k: ds[k].values for k in ds.data_vars}, ds


def _stats(a, b, mask):
    """on-ice stats for field b(2021) vs a(2012)."""
    a, b = a[mask], b[mask]
    fin = np.isfinite(a) & np.isfinite(b)
    a, b = a[fin], b[fin]
    d = b - a
    r = float(np.corrcoef(a, b)[0, 1]) if a.size > 2 else float("nan")
    return {
        "mean_2012": float(a.mean()), "mean_2021": float(b.mean()),
        "mean_delta": float(d.mean()), "rms_delta": float(np.sqrt(np.mean(d**2))),
        "median_absdelta": float(np.median(np.abs(d))),
        "p95_absdelta": float(np.percentile(np.abs(d), 95)),
        "corr_2012_2021": r,
        "frac_change_meanmag": float(abs(d.mean()) / (abs(a.mean()) + 1e-9)),
    }


def _map(ax, fld, mask, title, cmap, vmin=None, vmax=None, sym=False):
    m = np.where(mask, fld, np.nan)
    if sym:
        lim = np.nanpercentile(np.abs(m), 98) if np.isfinite(m).any() else 1.0
        vmin, vmax = -lim, lim
    im = ax.imshow(m, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--da2012", default="results/argentiere/sens_slidingco/slid_0.2689/geology-optimized.nc")
    ap.add_argument("--da2021", default="results/argentiere/epoch2021/da_2021_slid0.2689/geology-optimized.nc")
    ap.add_argument("--out-dir", default="results/argentiere/epoch2021")
    args = ap.parse_args()

    d12, _ = load(os.path.join(PROJECT_DIR, args.da2012) if not os.path.isabs(args.da2012) else args.da2012)
    d21, _ = load(os.path.join(PROJECT_DIR, args.da2021) if not os.path.isabs(args.da2021) else args.da2021)
    outdir = os.path.join(PROJECT_DIR, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(outdir, exist_ok=True)

    # common on-ice domain: ice in BOTH runs
    mask = (d12["icemask"] > 0.5) & (d21["icemask"] > 0.5)

    # divflux to w.e.
    for d in (d12, d21):
        d["divflux_we"] = d["divflux"] * (RHO / 1000.0)

    # ── metrics ───────────────────────────────────────────────────────────────
    fields = [("divflux_we", "div(flux)  [m w.e./yr]"),
              ("thk", "thickness  [m]"),
              ("velsurf_mag", "surf velocity  [m/yr]"),
              ("usurf", "surface elev  [m]")]
    rows = []
    for key, lbl in fields:
        s = _stats(d12[key], d21[key], mask); s["field"] = lbl; rows.append(s)
    csv_path = os.path.join(outdir, "epoch_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["field", "mean_2012", "mean_2021", "mean_delta",
                                          "rms_delta", "median_absdelta", "p95_absdelta",
                                          "corr_2012_2021", "frac_change_meanmag"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in w.fieldnames})
    print(f"[epochs] wrote {csv_path}")
    print(f"\n{'field':26s} {'mean12':>9} {'mean21':>9} {'Δmean':>9} {'RMSΔ':>8} {'r':>6}")
    for r in rows:
        print(f"{r['field']:26s} {r['mean_2012']:9.3f} {r['mean_2021']:9.3f} "
              f"{r['mean_delta']:9.3f} {r['rms_delta']:8.3f} {r['corr_2012_2021']:6.3f}")

    # ── 3x3 maps ──────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(3, 3, figsize=(12, 11))
    rowspec = [("divflux_we", r"div(flux) [m w.e./yr]", "RdBu_r"),
               ("thk", "thickness [m]", "viridis"),
               ("velsurf_mag", "surf velocity [m/yr]", "magma")]
    for r, (key, lbl, cmap) in enumerate(rowspec):
        both = np.concatenate([d12[key][mask], d21[key][mask]])
        vmin, vmax = np.nanpercentile(both, 2), np.nanpercentile(both, 98)
        if key == "divflux_we":
            lim = max(abs(vmin), abs(vmax)); vmin, vmax = -lim, lim
        _map(axs[r, 0], d12[key], mask, f"2012 — {lbl}", cmap, vmin, vmax)
        _map(axs[r, 1], d21[key], mask, f"2021 — {lbl}", cmap, vmin, vmax)
        _map(axs[r, 2], d21[key] - d12[key], mask, f"Δ (2021−2012) — {lbl}", "RdBu_r", sym=True)
    fig.suptitle("Argentiere DA fields: 2012 vs 2021 surface (same velocity-only DA)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = os.path.join(outdir, "epoch_divflux_maps.png"); fig.savefig(p, dpi=140); plt.close(fig)
    print(f"[epochs] wrote {p}")

    # ── elevation-binned divflux profile 2012 vs 2021 ─────────────────────────
    z_min = float(np.nanmin(d12["usurf"][mask])); dz = 50.0
    n = int((np.nanmax(d21["usurf"][mask]) - z_min) / dz) + 2
    z = z_min + np.arange(n) * dz
    p12 = _bin_by_elevation(d12["divflux_we"], d12["usurf"], mask, z_min, dz, n)
    p21 = _bin_by_elevation(d21["divflux_we"], d21["usurf"], mask, z_min, dz, n)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    a1.plot(p12, z, "-", color="steelblue", lw=2, label="2012 surface")
    a1.plot(p21, z, "--", color="firebrick", lw=2, label="2021 surface")
    a1.axvline(0, color="k", lw=0.5); a1.set_xlabel("div(flux) [m w.e./yr]")
    a1.set_ylabel("Elevation [m a.s.l.]"); a1.set_title("Flux-divergence profile"); a1.grid(alpha=0.3); a1.legend()
    a2.plot(p21 - p12, z, "-", color="k", lw=2)
    a2.axvline(0, color="k", lw=0.5); a2.set_xlabel("Δ div(flux) [m w.e./yr]")
    a2.set_title("Change (2021 − 2012)"); a2.grid(alpha=0.3)
    fig.suptitle("Argentiere flux divergence vs elevation — 2012 vs 2021 DA", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(outdir, "epoch_divflux_profile.png"); fig.savefig(p, dpi=140); plt.close(fig)
    print(f"[epochs] wrote {p}")


if __name__ == "__main__":
    main()
