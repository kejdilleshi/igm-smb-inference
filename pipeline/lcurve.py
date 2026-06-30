#!/usr/bin/env python3
"""
Shared helpers for the end-to-end SMB-inference pipeline:

  * parse_param         — read a Hydra-swept parameter value from a run dir
  * lcurve_corner       — auto-detect the L-curve corner (max curvature)
  * knee_select         — balanced-knee pick from an Optuna Pareto front

These back analyze_da_lcurve.py, analyze_smb_lcurve.py and run_pipeline.py.
Numpy-only; no heavy deps.
"""

import os
import re
import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - yaml ships with the igm env
    yaml = None


def parse_param(run_dir, param_name):
    """Return the value of a Hydra-swept parameter for a single run directory.

    Tries the directory name first (multirun dirs are named e.g.
    `processes.data_assimilation.regularization.thk=100`), then the canonical
    `.hydra/overrides.yaml`. Returns float or None.
    """
    short = param_name.split(".")[-1]
    base = os.path.basename(os.path.normpath(run_dir))
    for pattern in (re.escape(param_name) + r"=([0-9.eE+-]+)",
                    re.escape(short) + r"=([0-9.eE+-]+)",
                    re.escape(short) + r"_([0-9.eE+-]+)"):
        m = re.search(pattern, base)
        if m:
            return float(m.group(1))

    overrides_file = os.path.join(run_dir, ".hydra", "overrides.yaml")
    if yaml is not None and os.path.exists(overrides_file):
        with open(overrides_file) as f:
            overrides = yaml.safe_load(f) or []
        for ov in overrides:
            m = re.search(re.escape(param_name) + r"=([0-9.eE+-]+)", str(ov))
            if m:
                return float(m.group(1))
    return None


def lcurve_corner(x, y, log=True):
    """Index of the L-curve corner (point of maximum curvature).

    `x`, `y` are the two trade-off axes (e.g. smoothness vs misfit), each
    paired with one regularization value. Points are sorted by `x`. Curvature
    is the Menger curvature of consecutive triplets, computed in log space by
    default (L-curves span orders of magnitude). The returned index is always
    one of the *sampled* points, so the caller can reuse a run that actually
    happened — no interpolated regularization value.

    With < 3 points there is no interior corner; returns the lower-misfit end.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    xs, ys = x[order], y[order]

    if log:
        # Guard against non-positive values before taking logs.
        eps = 1e-12
        xs = np.log10(np.clip(xs, eps, None))
        ys = np.log10(np.clip(ys, eps, None))

    n = len(xs)
    if n < 3:
        # Fall back to the point with the smallest misfit (y).
        return int(order[int(np.argmin(y))])

    curv = np.zeros(n)
    for i in range(1, n - 1):
        p1 = np.array([xs[i - 1], ys[i - 1]])
        p2 = np.array([xs[i], ys[i]])
        p3 = np.array([xs[i + 1], ys[i + 1]])
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        # Menger curvature = 4 * triangle_area / (a*b*c).
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1])
                         - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        denom = a * b * c
        curv[i] = (4.0 * area / denom) if denom > 0 else 0.0

    best_sorted = int(np.argmax(curv))
    return int(order[best_sorted])


def knee_select(rows, x_key="velsurf", y_key="thk", pareto_only=True):
    """Balanced-knee pick from a list of trial dicts (Optuna sweep_summary rows).

    Min-max normalizes both objectives across the candidate set and returns the
    row closest to the (0, 0) ideal point. By default only Pareto trials
    (`is_pareto`) are considered. Returns the chosen row dict.
    """
    cand = [r for r in rows if r.get("is_pareto")] if pareto_only else list(rows)
    if not cand:
        cand = list(rows)
    if not cand:
        raise ValueError("knee_select: no candidate rows")
    if len(cand) == 1:
        return cand[0]

    xs = np.array([float(r[x_key]) for r in cand])
    ys = np.array([float(r[y_key]) for r in cand])

    def _norm(v):
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)

    dist = np.hypot(_norm(xs), _norm(ys))
    return cand[int(np.argmin(dist))]
