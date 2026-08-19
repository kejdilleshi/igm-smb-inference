"""Shared access to the slidingco sensitivity sweeps the paper figures plot.

Every figure that fans over basal sliding reads
``results/<glacier>/sens_slidingco/`` , where ``sensitivity_slidingco.py`` writes
one ``slid_<tau>/`` run directory per value plus a ``summary_<glacier>.csv`` with

    slidingco,arrhenius,da_velsurf,mean_smb_mwe,ELA_m

The figures used to carry that table as a hand-copied dict literal. That silently
rots the moment a sweep is re-run: the tau keys no longer match the directories
on disk, the velsurf lookup returns NaN, and the affected points vanish from the
plot without any error. Read the CSV instead — it is written by the same run that
produced the profiles, so the two cannot disagree.
"""

import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Stake observations per glacier, and whether the CSV carries a Year column
# (repeat-cell anchors) or is a single-snapshot list of raw stakes.
STAKES = {
    "argentiere":     (f"{ROOT}/data/argentiere/SMB_argentiere_2012-2021_utm32N.csv", False),
    "aletsch_insitu": (f"{ROOT}/data/aletsch_insitu/SMB_aletsch_insitu_2009-2017_rawstakes_utm32N.csv", True),
    "rhone_insitu":   (f"{ROOT}/data/rhone_insitu/SMB_rhone_insitu_2007-2016_rawstakes_utm32N.csv", True),
}

# A run is a candidate for adoption only if its DA velocity misfit is within
# CUTOFF x the best of that glacier's sweep. Relative, not a per-glacier absolute
# threshold: the three glaciers sit at completely different velsurf scales
# (Argentiere's best is 0.87, Aletsch's 8.86), and the high-velsurf arm added for
# the correlation figure reaches 140 on Aletsch but only 37 on Rhone. A bare
# {"aletsch_insitu": 15.0} left Rhone and Argentiere with no cap at all, so a
# badly-fitting run was eligible to be adopted on stake RMSE alone.
# 1.5 matches sensitivity_slidingco.py --cutoff, which selected the sweep values.
CUTOFF = 1.5

RHO = 917.0


def sens_dir(glacier):
    return os.path.join(ROOT, "results", glacier, "sens_slidingco")


def summary_path(glacier):
    return os.path.join(sens_dir(glacier), f"summary_{glacier}.csv")


def da_velsurf(glacier):
    """{tau (rounded to 4dp): da_velsurf} for one glacier, from its summary CSV."""
    p = summary_path(glacier)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found — run  python pipeline/sensitivity_slidingco.py "
            f"--glacier {glacier}  before regenerating the figures.")
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                out[round(float(row["slidingco"]), 4)] = float(row["da_velsurf"])
            except (TypeError, ValueError):
                continue
    return out


def taus(glacier, require_smb=True):
    """Sorted tau values that have both a summary row and a finished SMB run.

    A finished run with no summary row is reported on stderr rather than
    dropped quietly: that is exactly how the old hand-maintained velsurf dicts
    diverged from disk. (The archived DA-at-t1 Argentiere sweep has slid_0.0747
    and slid_0.2689 directories absent from its summary CSV — and 0.2689 was the
    run the figures adopted.)
    """
    vel = da_velsurf(glacier)
    out, orphan = [], []
    for d in sorted(glob.glob(os.path.join(sens_dir(glacier), "slid_*"))):
        tau = round(float(os.path.basename(d)[5:]), 4)
        finished = os.path.exists(os.path.join(d, "smb_inference", "smb_vec.npy"))
        if require_smb and not finished:
            continue
        if tau not in vel:
            orphan.append(tau)
            continue
        out.append(tau)
    if orphan:
        print(f"WARNING [{glacier}]: {len(orphan)} finished run(s) missing from "
              f"{os.path.basename(summary_path(glacier))} and therefore NOT plotted: "
              f"{['%.4f' % t for t in orphan]}", file=sys.stderr)
    return sorted(out)


def run_dir(glacier, tau):
    return os.path.join(sens_dir(glacier), f"slid_{tau:.4f}")


def _helpers():
    sys.path.insert(0, os.path.join(ROOT, "pipeline"))
    sys.path.insert(0, os.path.join(ROOT, "user", "code", "processes", "smb_inference"))
    from smb_vs_continuity import prepare_cells, _interp_at, _rmse
    return prepare_cells, _interp_at, _rmse


def stake_rmse(glacier, tau):
    """Inferred SMB profile vs the stake observations, m w.e./yr."""
    import numpy as np
    prepare_cells, _interp_at, _rmse = _helpers()
    run = os.path.join(run_dir(glacier, tau), "smb_inference")
    smb = np.load(os.path.join(run, "smb_vec.npy")).ravel() * (RHO / 1000.0)
    z0 = float(np.load(os.path.join(run, "z_min.npy")))
    dz = float(np.load(os.path.join(run, "dz.npy")))
    z = z0 + dz * np.arange(smb.size)

    csv_path, has_year = STAKES[glacier]
    if has_year:
        prep, _ = prepare_cells(csv_path, 100.0, 10, 4)
        alt, obs = prep["clu_alt"], prep["clu_mean"]
    else:
        import pandas as pd
        df = pd.read_csv(csv_path)
        alt = df["Alt"].to_numpy(float)
        obs = df["Annual_SMB (m w.e.)"].to_numpy(float)
    at = _interp_at(z, smb, alt)
    return float(_rmse(at, obs, np.isfinite(at)))


def adopted_tau(glacier):
    """The tau the paper adopts: best stake RMSE among runs that fit velocities.

    Single definition shared by every figure — the fan panels, the continuity
    3-panel and the DA-state maps must all depict the same run, and they used to
    hardcode it separately (and inconsistently) as a literal path.
    """
    cand = [t for t in taus(glacier) if not excluded(glacier, t)]
    if not cand:
        raise RuntimeError(f"no candidate run for {glacier}")
    return min(cand, key=lambda t: stake_rmse(glacier, t))


def vel_cap(glacier):
    """Velocity-misfit ceiling for adoption: CUTOFF x this glacier's best."""
    vel = da_velsurf(glacier)
    finite = [v for v in vel.values() if v == v]
    return CUTOFF * min(finite)


def excluded(glacier, tau):
    """True if tau fits velocities too poorly to be a candidate."""
    return da_velsurf(glacier).get(tau, float("inf")) > vel_cap(glacier)
