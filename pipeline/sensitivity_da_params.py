#!/usr/bin/env python3
"""
Sensitivity of the DA velocity misfit (velsurf) to the two scalar physics
parameters tuned in pipeline stage 2.1:

    init_slidingco   (basal sliding;  search range [0.05, 2.0], log)
    init_arrhenius   (flow-law A;      search range [30, 200],  log)

Question answered: which of the two is MORE influential on the surface-velocity
misfit that the DA sweep minimises?

We reuse the existing Optuna TPE studies (``sweep_results_*/sweep_summary.json``)
— no GPU needed — and report three complementary importance metrics per glacier:

  1. PedANOVA importance (optuna.importance.PedAnovaImportanceEvaluator)
     -> density-based importance (good vs all trials). Normalised, sums to 1.
        (PedANOVA would need sklearn, which is absent from the igm envs.)
  2. |standardised beta| from a linear fit  velsurf ~ b1*z(log slid) + b2*z(log arrh)
     -> linear effect size in log-parameter space (both standardised).
  3. |Spearman rho|      -> monotonic rank association of each param with velsurf.

Trials with non-finite velsurf (failed/timed-out DA runs) are dropped.

Usage:
    conda run -n igm-pretrain python pipeline/sensitivity_da_params.py
    python pipeline/sensitivity_da_params.py --glaciers rhone aletsch
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import optuna
from optuna.distributions import FloatDistribution
from optuna.importance import PedAnovaImportanceEvaluator, get_param_importances
from optuna.trial import TrialState, create_trial

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = ["init_slidingco", "init_arrhenius"]
DISTS = {
    "init_slidingco": FloatDistribution(0.05, 2.0, log=True),
    "init_arrhenius": FloatDistribution(30.0, 200.0, log=True),
}


def _spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    return float((rx * ry).mean())


def analyse(name, summary_path):
    rows = json.load(open(summary_path))["results"]
    rows = [r for r in rows if math.isfinite(r["velsurf"])]
    if len(rows) < 4:
        return None
    slid = np.array([r["init_slidingco"] for r in rows])
    arrh = np.array([r["init_arrhenius"] for r in rows])
    vel = np.array([r["velsurf"] for r in rows])

    # 1. PedANOVA importance via an in-memory study rebuilt from finite trials.
    study = optuna.create_study(direction="minimize")
    study.add_trials([
        create_trial(
            state=TrialState.COMPLETE,
            value=float(v),
            params={"init_slidingco": float(s), "init_arrhenius": float(a)},
            distributions=DISTS,
        )
        for s, a, v in zip(slid, arrh, vel)
    ])
    imp = get_param_importances(study, evaluator=PedAnovaImportanceEvaluator())
    imp = {p: float(imp.get(p, 0.0)) for p in PARAMS}

    # 2. Standardised linear regression in log-parameter space.
    z = lambda u: (u - u.mean()) / u.std()
    X = np.column_stack([z(np.log(slid)), z(np.log(arrh))])
    yz = z(vel)
    beta, *_ = np.linalg.lstsq(X, yz, rcond=None)
    beta = {"init_slidingco": float(beta[0]), "init_arrhenius": float(beta[1])}

    # 3. Spearman rank correlation of each param with velsurf.
    rho = {"init_slidingco": _spearman(slid, vel),
           "init_arrhenius": _spearman(arrh, vel)}

    return {"name": name, "n": len(rows),
            "fanova": imp, "beta": beta, "rho": rho}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glaciers", nargs="*", default=None,
                    help="subset of glacier names; default = all sweeps found")
    args = ap.parse_args()

    summaries = sorted(glob.glob(
        os.path.join(PROJECT_DIR, "sweep_results_*", "sweep_summary.json")))
    results = []
    for s in summaries:
        name = (os.path.basename(os.path.dirname(s))
                .replace("sweep_results_params_oggm_", "").replace("_da", ""))
        if args.glaciers and name not in args.glaciers:
            continue
        r = analyse(name, s)
        if r:
            results.append(r)

    if not results:
        print("No usable sweeps found.")
        return

    print(f"\nDA velsurf sensitivity to init_slidingco vs init_arrhenius")
    print(f"(higher = more influential; PedANOVA normalised, |beta| & |rho| absolute)\n")
    h = f"{'glacier':16s} {'n':>3s} | {'PedANOVA slid':>11s} {'arrh':>6s} | " \
        f"{'|beta| slid':>11s} {'arrh':>6s} | {'|rho| slid':>10s} {'arrh':>6s} | winner"
    print(h); print("-" * len(h))

    tally = {"init_slidingco": 0, "init_arrhenius": 0}
    for r in results:
        f_s, f_a = r["fanova"]["init_slidingco"], r["fanova"]["init_arrhenius"]
        b_s, b_a = abs(r["beta"]["init_slidingco"]), abs(r["beta"]["init_arrhenius"])
        p_s, p_a = abs(r["rho"]["init_slidingco"]), abs(r["rho"]["init_arrhenius"])
        # majority vote of the three metrics
        votes = sum([f_s > f_a, b_s > b_a, p_s > p_a])
        win = "slidingco" if votes >= 2 else "arrhenius"
        tally["init_slidingco" if votes >= 2 else "init_arrhenius"] += 1
        print(f"{r['name']:16s} {r['n']:>3d} | {f_s:>11.2f} {f_a:>6.2f} | "
              f"{b_s:>11.2f} {b_a:>6.2f} | {p_s:>10.2f} {p_a:>6.2f} | {win}")

    print("-" * len(h))
    # pooled means
    def m(metric, key):
        return np.mean([
            (r["fanova"][key] if metric == "fanova"
             else abs(r["beta"][key]) if metric == "beta"
             else abs(r["rho"][key]))
            for r in results])
    print(f"{'MEAN':16s} {len(results):>3d} | "
          f"{m('fanova','init_slidingco'):>11.2f} {m('fanova','init_arrhenius'):>6.2f} | "
          f"{m('beta','init_slidingco'):>11.2f} {m('beta','init_arrhenius'):>6.2f} | "
          f"{m('rho','init_slidingco'):>10.2f} {m('rho','init_arrhenius'):>6.2f} |")
    print(f"\nper-glacier winner tally: slidingco={tally['init_slidingco']}  "
          f"arrhenius={tally['init_arrhenius']}")


if __name__ == "__main__":
    main()
