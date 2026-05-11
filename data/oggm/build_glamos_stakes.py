#!/usr/bin/env python3
"""
Build a stakes CSV from the GLAMOS elevation-band file for the period
2000-2020 (hydro years 2000/01 → 2019/20). Output matches the column
schema of SMB_argentiere_2012-2021_utm32N.csv:

    X, Y, Alt, Annual_SMB (m w.e.)

For each 100 m GLAMOS elevation bin spanned by the on-glacier pixels of
the start-of-period DEM (usurf in input_<RGI_ID>.nc), the mean annual Ba
is converted from mm to m w.e. and a synthetic stake is placed at the
(X, Y) centroid of those pixels.

The CSV is used only for diagnostic overlay during smb_inference (it does
not enter the inversion cost). It also serves as the ground-truth source
for the final SMB profile comparison plot.

Usage:
    python build_glamos_stakes.py [RGI_ID]
"""

import os
import sys
import numpy as np
import pandas as pd
import netCDF4 as nc

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RGI_ID = "RGI60-11.01450"
GLACIER_ID = "B36-26"            # Grosser Aletschgletscher in GLAMOS

GLAMOS_CSV = os.path.normpath(os.path.join(HERE, "..", "massbalance_observation_elevationbins.csv"))

START_YEAR = 2000                # hydro year 2000/01 starts 2000-10-01
END_YEAR = 2020                  # hydro year 2019/20 ends 2020-09-30


def load_glamos(path):
    """The GLAMOS CSV has 6 lines of preamble, then the human-readable
    header (line 7), a snake-case alias row (line 8), a units row (line 9),
    and finally the data. We read from line 7, drop the alias + units rows,
    and rename to snake_case for convenience."""
    skip = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if line.startswith("glacier name,"):
                skip = i
                break
    df = pd.read_csv(path, skiprows=skip, low_memory=False)
    df = df[df["glacier id"].astype(str).str.match(r"^[A-Z]")].copy()
    df.rename(columns={
        "start date of observation": "date_start",
        "end date of winter observation": "date_end_winter",
        "end date of observation": "date_end",
        "winter mass balance": "Bw",
        "summer mass balance": "Bs",
        "annual mass balance": "Ba",
        "area of elevation bin": "area",
        "lower elevation of bin": "h_min",
        "upper elevation of bin": "h_max",
        "glacier id": "glacier_id",
    }, inplace=True)
    return df


def main(rgi_id):
    input_nc = os.path.normpath(os.path.join(HERE, "..",
                                              f"input_{rgi_id.replace('-', '_')}.nc"))
    out_csv = os.path.join(HERE, f"stakes_glamos_{rgi_id}_{START_YEAR}-{END_YEAR}.csv")
    print(f"== Building GLAMOS-derived stakes CSV for {rgi_id}")
    print(f"   GLAMOS: {GLAMOS_CSV}")
    print(f"   grid  : {input_nc}")
    print(f"   out   : {out_csv}")

    df = load_glamos(GLAMOS_CSV)
    df = df[df["glacier_id"] == GLACIER_ID].copy()
    for c in ("date_start", "date_end"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("Bw", "Bs", "Ba", "h_min", "h_max", "area"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Hydrological year filter: include rows whose period falls inside
    # [START_YEAR-09-15, END_YEAR-10-15] (looseness absorbs ±2 weeks).
    mask = ((df["date_start"] >= pd.Timestamp(f"{START_YEAR-1}-09-15"))
            & (df["date_end"]   <= pd.Timestamp(f"{END_YEAR}-10-15")))
    df = df[mask].dropna(subset=["Ba", "h_min", "h_max"])
    if df.empty:
        sys.exit(f"no GLAMOS rows for {GLACIER_ID} in {START_YEAR}-{END_YEAR}")

    n_years = df["date_start"].dt.year.nunique()
    print(f"   {len(df)} GLAMOS rows spanning {n_years} hydro years "
          f"({START_YEAR-1}/{START_YEAR} … {END_YEAR-1}/{END_YEAR})")

    bin_means = (df.groupby(["h_min", "h_max"])["Ba"]
                   .mean() / 1000.0).rename("Ba_mwe").reset_index()
    bin_means["alt_centroid"] = 0.5 * (bin_means["h_min"] + bin_means["h_max"])
    print(f"   {len(bin_means)} elevation bins, Ba range "
          f"[{bin_means['Ba_mwe'].min():.2f}, {bin_means['Ba_mwe'].max():.2f}] "
          f"m w.e./yr")

    with nc.Dataset(input_nc) as ds:
        x = np.asarray(ds.variables["x"][:], dtype=np.float64)
        y = np.asarray(ds.variables["y"][:], dtype=np.float64)
        usurf = np.asarray(ds.variables["usurf"][:], dtype=np.float32)
        icemask = np.asarray(ds.variables["icemask"][:], dtype=np.float32)

    X, Y = np.meshgrid(x, y)
    on_ice = icemask > 0.5

    rows = []
    skipped = []
    for _, r in bin_means.iterrows():
        hmin, hmax, ba = r["h_min"], r["h_max"], r["Ba_mwe"]
        sel = on_ice & (usurf >= hmin) & (usurf < hmax)
        n = int(sel.sum())
        if n == 0:
            skipped.append((hmin, hmax, ba))
            continue
        rows.append({
            "X": float(X[sel].mean()),
            "Y": float(Y[sel].mean()),
            "Alt": float(usurf[sel].mean()),
            "Annual_SMB (m w.e.)": ba,
        })

    out = pd.DataFrame(rows).sort_values("Alt").reset_index(drop=True)
    out.to_csv(out_csv, index=False)

    print(f"\n   Wrote {len(out)} synthetic stakes → {out_csv}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
    if skipped:
        print(f"\n   Skipped {len(skipped)} GLAMOS bins with no on-glacier pixels: "
              f"{skipped}")


if __name__ == "__main__":
    rgi_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RGI_ID
    main(rgi_id)
