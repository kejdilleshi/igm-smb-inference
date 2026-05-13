#!/usr/bin/env python3
"""
Plot GLAMOS observed annual mass balance per 100 m elevation bin for
Grosser Aletschgletscher (B36-26) — one curve per hydrological year, colored
by year, overlaid with the multi-year mean.

Useful for spotting anomalous years (e.g. very negative 2003, 2018, 2022 heat
waves) inside the 2000-2020 averaging window used by the SMB-inference
comparison plot.

Usage:
    python plot_glamos_yearly_profiles.py [out_png]
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

HERE = os.path.dirname(os.path.abspath(__file__))
GLAMOS_CSV = os.path.normpath(os.path.join(HERE, "..", "massbalance_observation_elevationbins.csv"))
GLACIER_ID = "B36-26"
START_YEAR = 2000
END_YEAR = 2020


def load_glamos_year_profiles():
    skip = 0
    with open(GLAMOS_CSV) as f:
        for i, line in enumerate(f):
            if line.startswith("glacier name,"):
                skip = i
                break
    df = pd.read_csv(GLAMOS_CSV, skiprows=skip, low_memory=False)
    df = df[df["glacier id"].astype(str).str.match(r"^[A-Z]")].copy()
    df = df.rename(columns={
        "glacier id": "id",
        "start date of observation": "date_start",
        "end date of observation": "date_end",
        "annual mass balance": "Ba",
        "lower elevation of bin": "h_min",
        "upper elevation of bin": "h_max",
    })
    df = df[df["id"] == GLACIER_ID].copy()
    df["date_start"] = pd.to_datetime(df["date_start"])
    df["date_end"]   = pd.to_datetime(df["date_end"])
    for c in ("Ba", "h_min", "h_max"):
        df[c] = pd.to_numeric(df[c])

    # Same window as plot_smb_vs_glamos.py: include the 1999/2000 hydro year
    # because it overlaps the Hugonnet 2000-01-01 → 2020-01-01 reference for
    # 9 of 12 months.
    mask = ((df["date_start"] >= pd.Timestamp(f"{START_YEAR-1}-09-15"))
            & (df["date_end"]   <= pd.Timestamp(f"{END_YEAR}-10-15")))
    df = df[mask].dropna(subset=["Ba", "h_min", "h_max"]).copy()

    # Label each row by hydrological year (= calendar year of date_end).
    df["hydro_year"] = df["date_end"].dt.year
    df["alt"] = 0.5 * (df["h_min"] + df["h_max"])
    df["Ba_mwe"] = df["Ba"] / 1000.0
    return df


def main(out_png=None):
    df = load_glamos_year_profiles()
    years = sorted(df["hydro_year"].unique())
    print(f"GLAMOS {GLACIER_ID}: {len(df)} rows over {len(years)} hydrological years "
          f"({years[0]}–{years[-1]})")

    # 21-year mean per elevation bin.
    bin_mean = (df.groupby("alt", as_index=False)["Ba_mwe"]
                  .mean()
                  .sort_values("alt"))

    fig, ax = plt.subplots(figsize=(7.5, 9.5))
    cmap = cm.get_cmap("viridis", len(years))

    # One curve per year; sorted by ascending altitude for each curve.
    extremes = []
    for i, y in enumerate(years):
        sub = df[df["hydro_year"] == y].sort_values("alt")
        color = cmap(i / max(1, len(years) - 1))
        ax.plot(sub["Ba_mwe"], sub["alt"], "-", color=color, alpha=0.85,
                linewidth=1.2)
        # Track each year's glacier-wide mean (area-weighted not needed for
        # spotting anomalies — simple mean across bins is enough).
        extremes.append((y, sub["Ba_mwe"].mean()))

    # Multi-year mean profile overlay.
    ax.plot(bin_mean["Ba_mwe"], bin_mean["alt"], "-",
            color="black", linewidth=2.6, zorder=10,
            label=f"mean {years[0]}–{years[-1]}  (N={len(years)})")

    # Highlight the 3 most-negative and 3 most-positive years by simple
    # across-bin mean — these are the most likely "anomalous" curves.
    extremes_sorted = sorted(extremes, key=lambda kv: kv[1])
    top_neg = extremes_sorted[:3]
    top_pos = extremes_sorted[-3:]
    note = (f"most-negative years (mean across bins): "
            + ", ".join(f"{y} ({m:+.2f})" for y, m in top_neg)
            + f"\nmost-positive years: "
            + ", ".join(f"{y} ({m:+.2f})" for y, m in top_pos))
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="lightgray", alpha=0.85))

    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Annual SMB (m w.e./yr)")
    ax.set_ylabel("Elevation (m a.s.l.)")
    ax.set_title(f"Grosser Aletschgletscher (B36-26) — GLAMOS annual SMB profiles\n"
                 f"per hydrological year, 1999/2000 – 2019/2020")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=9)

    # Year colorbar (legend-replacement so we don't get 21 line entries).
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=years[0], vmax=years[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("Hydrological year (= year of date_end)")

    fig.tight_layout()
    if out_png is None:
        out_png = os.path.join(HERE, f"glamos_yearly_profiles_{START_YEAR}-{END_YEAR}.png")
    fig.savefig(out_png, dpi=140)
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
