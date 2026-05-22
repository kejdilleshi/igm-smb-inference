#!/usr/bin/env python3
"""
Build data/aletsch/SMB_aletsch_<start>-<end>_utm32N.csv from the GLAMOS
elevation-band mass balance file (data/massbalance_observation_elevationbins.csv).

For each 100 m GLAMOS elevation bin within the hydrological years
[START_YEAR-10-01, END_YEAR-09-30] inclusive of the START year and exclusive
of END+1:

  1. Average annual mass balance Ba (mm w.e.) across all bin-years in window.
  2. Convert mm w.e. → m w.e.
  3. Place ONE synthetic stake at the (X, Y) centroid of the on-glacier pixels
     of input_aletsch.nc whose start-of-period (e.g. 1999) surface elevation
     falls inside the bin.
  4. Record Alt = pixel-centroid elevation (mean of usurf over the same set).

Output columns (matching SMB_argentiere_2012-2021_utm32N.csv):
  X, Y, Alt, Annual_SMB (m w.e.)

Rationale: the SMB inversion runs in `profile` mode with dz=200 m, so it fits a
1-D SMB(z) curve binned in altitude. GLAMOS elevation-band Ba is exactly that
signal already aggregated, so one synthetic stake per GLAMOS bin is sufficient.
"""

import os
import sys
import numpy as np
import pandas as pd
import netCDF4 as nc

HERE = os.path.dirname(os.path.abspath(__file__))
GLAMOS_CSV  = os.path.normpath(os.path.join(HERE, "..", "massbalance_observation_elevationbins.csv"))
INPUT_NC    = os.path.normpath(os.path.join(HERE, "..", "input_aletsch.nc"))
GLACIER_ID  = "B36-26"        # Grosser Aletschgletscher (Swiss Glacier Inventory)

START_YEAR  = 2009            # hydrological year start: START_YEAR-10-01
END_YEAR    = 2017            # last hydro year end: END_YEAR-09-30 (exclusive of new year)

OUT_CSV     = os.path.join(HERE, f"SMB_aletsch_{START_YEAR}-{END_YEAR}_utm32N.csv")


def load_glamos(path):
    """The GLAMOS CSV has 6 lines of preamble, then a human-readable header
    line (line 7), a snake-case alias row (line 8), a units row (line 9),
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


def main():
    if not os.path.exists(GLAMOS_CSV):
        sys.exit(f"missing GLAMOS file: {GLAMOS_CSV}")
    if not os.path.exists(INPUT_NC):
        sys.exit(f"missing input NetCDF (run build_input_nc.py first): {INPUT_NC}")

    df = load_glamos(GLAMOS_CSV)
    df = df[df["glacier_id"] == GLACIER_ID].copy()
    for c in ("date_start", "date_end"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("Bw", "Bs", "Ba", "h_min", "h_max", "area"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter to the requested hydrological years.
    mask = (df["date_start"] >= pd.Timestamp(f"{START_YEAR}-09-15")) & \
           (df["date_end"]   <= pd.Timestamp(f"{END_YEAR}-10-15"))
    df = df[mask].dropna(subset=["Ba", "h_min", "h_max"])
    if df.empty:
        sys.exit(f"no GLAMOS rows for {GLACIER_ID} in {START_YEAR}-{END_YEAR}")

    n_years = df["date_start"].dt.year.nunique()
    print(f"{len(df)} GLAMOS rows for {GLACIER_ID} spanning {n_years} hydro years "
          f"({START_YEAR}/{START_YEAR+1} … {END_YEAR-1}/{END_YEAR})")

    # Mean annual Ba per elevation bin (mm w.e.) → m w.e.
    bin_means = (df.groupby(["h_min", "h_max"])["Ba"].mean() / 1000.0).rename("Ba_mwe").reset_index()
    bin_means["alt_centroid"] = 0.5 * (bin_means["h_min"] + bin_means["h_max"])
    print(f"{len(bin_means)} elevation bins, Ba range "
          f"[{bin_means['Ba_mwe'].min():.2f}, {bin_means['Ba_mwe'].max():.2f}] m w.e./yr")

    # Locate each bin on the glacier grid using usurf (= start-of-period DEM).
    with nc.Dataset(INPUT_NC) as ds:
        x = np.asarray(ds.variables["x"][:], dtype=np.float64)
        y = np.asarray(ds.variables["y"][:], dtype=np.float64)
        usurf   = np.asarray(ds.variables["usurf"][:], dtype=np.float32)
        icemask = np.asarray(ds.variables["icemask"][:], dtype=np.float32)

    X, Y = np.meshgrid(x, y)  # both (ny, nx)
    glacier = icemask > 0.5

    rows = []
    skipped = []
    for _, r in bin_means.iterrows():
        hmin, hmax, ba, ac = r["h_min"], r["h_max"], r["Ba_mwe"], r["alt_centroid"]
        sel = glacier & (usurf >= hmin) & (usurf < hmax)
        n = int(sel.sum())
        if n == 0:
            skipped.append((hmin, hmax, ba))
            continue
        x_c = float(X[sel].mean())
        y_c = float(Y[sel].mean())
        alt_c = float(usurf[sel].mean())
        rows.append({"X": x_c, "Y": y_c, "Alt": alt_c, "Annual_SMB (m w.e.)": ba,
                     "h_min": hmin, "h_max": hmax, "n_pixels": n})

    out = pd.DataFrame(rows).sort_values("Alt").reset_index(drop=True)
    out_csv = out[["X", "Y", "Alt", "Annual_SMB (m w.e.)"]]
    out_csv.to_csv(OUT_CSV, index=False)

    print(f"\nWrote {len(out_csv)} synthetic stakes → {OUT_CSV}")
    print(out[["h_min", "h_max", "Alt", "Annual_SMB (m w.e.)", "n_pixels"]].to_string(index=False))
    if skipped:
        print(f"\nSkipped {len(skipped)} GLAMOS bins with no on-glacier pixels "
              f"at start-of-period DEM elevation (likely outside the modelled "
              f"icemask): {skipped}")


if __name__ == "__main__":
    main()
