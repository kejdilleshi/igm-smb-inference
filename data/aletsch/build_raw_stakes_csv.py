#!/usr/bin/env python3
"""
Build data/aletsch/SMB_aletsch_<START>-<END>_rawstakes_utm32N.csv from the
GLAMOS Swiss Glacier Point Mass Balance archive (raw stake / snow-pit
readings, NOT the interpolated elevation curve).

Source (release 2023, CC-BY 4.0):
  https://doi.glamos.ch/data/massbalance_point/massbalance_point_2023_r2023.zip
  DOI 10.18750/massbalance.point.2023.r2023

A subset of that archive lives in raw_glamos/aletsch_annual.dat.

For each annual stake reading whose hydrological year start lies in
[START, END]:
  1. Annualise mb_we (mm w.e. over the exact period) to mm w.e./yr by
     dividing by period_days / 365.25, then convert to m w.e./yr.
  2. Reproject (x_pos, y_pos) from CH1903 / LV03 (EPSG:21781) to
     UTM 32N (EPSG:32632) to match the project grid.
  3. Keep z_pos (m a.s.l.) as Alt.

Output columns (matching build_stake_csv.py / SMB_argentiere_*.csv):
  X, Y, Alt, Annual_SMB (m w.e.), Year (hydrological year start)

The smb_inference plotting overlay (smb_vec_iter_*.png) reads this CSV via
load_smb_point_obs and scatters (SMB, Alt) on top of the inferred profile.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
DAT_FILE = os.path.join(HERE, "raw_glamos", "aletsch_annual.dat")

START_YEAR = 2000
END_YEAR = 2020

COLUMNS = [
    "name", "date0", "time0", "date1", "time1", "period", "date_quality",
    "x_pos", "y_pos", "z_pos", "position_quality",
    "mb_raw", "density", "density_quality",
    "mb_we", "measurement_quality", "measurement_type",
    "mb_error", "reading_error", "density_error",
    "error_evaluation_method", "source",
]


def read_glamos_dat(path):
    rows = []
    # latin-1 tolerates the non-ASCII glacier-name bytes some GLAMOS files
    # carry in their comment header (e.g. "Corbassière"); the numeric data is
    # pure ASCII so values are unaffected.
    with open(path, encoding="latin-1") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) != len(COLUMNS):
                continue
            rows.append(parts)
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("date0", "date1"):
        df[c] = pd.to_datetime(df[c], format="%Y%m%d", errors="coerce")
    for c in ("period", "x_pos", "y_pos", "z_pos", "mb_we"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=START_YEAR,
                        help=f"first hydrological year (default {START_YEAR})")
    parser.add_argument("--end", type=int, default=END_YEAR,
                        help=f"last hydrological year (default {END_YEAR})")
    args = parser.parse_args()
    start_year, end_year = args.start, args.end
    out_csv = os.path.join(
        HERE, f"SMB_aletsch_{start_year}-{end_year}_rawstakes_utm32N.csv")

    if not os.path.exists(DAT_FILE):
        sys.exit(f"missing raw stake file: {DAT_FILE}")

    df = read_glamos_dat(DAT_FILE)
    df = df.dropna(subset=["date0", "date1", "x_pos", "y_pos", "z_pos", "mb_we", "period"])

    hy = df["date0"].dt.year
    df = df[(hy >= start_year) & (hy <= end_year)].copy()
    if df.empty:
        sys.exit(f"no annual stake rows in {start_year}-{end_year}")

    df["smb_mwe_per_yr"] = (df["mb_we"] / df["period"]) * 365.25 / 1000.0

    transformer = Transformer.from_crs("EPSG:21781", "EPSG:32632", always_xy=True)
    X, Y = transformer.transform(df["x_pos"].values, df["y_pos"].values)

    out = pd.DataFrame({
        "X": X,
        "Y": Y,
        "Alt": df["z_pos"].values,
        "Annual_SMB (m w.e.)": df["smb_mwe_per_yr"].values,
        "Year": df["date0"].dt.year.values,
    }).sort_values("Alt").reset_index(drop=True)

    out.to_csv(out_csv, index=False)

    print(f"Read {len(df)} annual stake measurements for hydro years "
          f"{start_year}-{end_year} from {os.path.relpath(DAT_FILE, HERE)}")
    print(f"Elevation range: {out['Alt'].min():.0f} - {out['Alt'].max():.0f} m a.s.l.")
    print(f"SMB range:       {out['Annual_SMB (m w.e.)'].min():.2f} - "
          f"{out['Annual_SMB (m w.e.)'].max():.2f} m w.e./yr")
    print(f"\nWrote {len(out)} stakes -> {out_csv}")

    yearly = df.groupby(df["date0"].dt.year).size().rename("n_stakes")
    print("\nStakes per hydrological year:")
    print(yearly.to_string())


if __name__ == "__main__":
    main()
