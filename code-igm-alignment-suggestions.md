# Code–IGM Alignment Suggestions for `smb_inference`

A review of the `smb_inference` user module against IGM's built-in utilities.
Each section identifies duplicated or suboptimal code and proposes a concrete replacement using IGM's API.

---

## 1. Flux Divergence — replace custom upwind with IGM's slope-limited version

**Current** (`utils/emulator_tools.py:compute_divflux`):
Hand-rolled upwind scheme (~30 lines) with manual staggered-grid averaging.

**IGM equivalent** (`igm.utils.grad.compute_divflux_slope_limiter`):
```python
from igm.utils.grad import compute_divflux_slope_limiter
divflux = compute_divflux_slope_limiter(ubar, vbar, thk, dx, dy, dt, slope_type="superbee")
```
- More accurate (superbee/minmod slope limiters vs basic upwind).
- Already used by IGM's `thk` process — consistency with IGM's own thickness evolution.
- Also available: `igm.utils.grad.compute_divflux` for simpler upwind if preferred.

**Files to change**: `utils/emulator_tools.py`, `core/forward_schemes/pinn_emulator_step.py`, `core/forward_schemes/pretrained_cnn_step.py`

---

## 2. Spatial Gradients — replace custom `compute_gradient` with IGM's `grad_xy`

**Current** (`utils/emulator_tools.py:compute_gradient`):
Central differences with manual boundary extrapolation (~20 lines).

**IGM equivalent** (`igm.utils.grad.grad_xy`):
```python
from igm.utils.grad import grad_xy
dsdx, dsdy = grad_xy(s, dx, dy, staggered_grid=False, mode="extrapolate")
```
- Supports staggered/unstaggered grids and multiple padding modes (`periodic`, `symmetric`, `extrapolate`).

**Files to change**: `utils/emulator_tools.py`

---

## 3. Staggered Grid Averaging — use IGM's `stag` utilities

**Current** (`utils/emulator_tools.py:compute_divflux`):
Manual averaging `0.5 * (u[:, :-1, :] + u[:, 1:, :])` etc.

**IGM equivalent** (`igm.utils.stag`):
```python
from igm.utils.stag import stag2x, stag2y
u_stag = stag2y(u)
v_stag = stag2x(v)
```

**Files to change**: `utils/emulator_tools.py`

---

## 4. Emulator Loading — reuse IGM's artifact infrastructure

**Current** (`core/load_pinn.py:load_pinn_emulator`, ~140 lines):
Manually parses `manifest.yaml`, builds OmegaConf config, constructs FNO2 from `Architectures` registry, loads weights, creates normalizer, and computes `V_bar` via `LagrangeDiscr`.

**IGM equivalent** (`igm.processes.iceflow.emulate`):
IGM already has `get_emulator_path()`, `save_iceflow_model()`, and the `Architectures` + `NormalizationsDict` registries. The vertical discretization classes (`LagrangeDiscr`, `MOLHODiscr`) are available directly.

**Suggestion**: Factor the loading into a thin wrapper around IGM's existing emulator setup rather than reimplementing manifest parsing and architecture construction. At minimum, replace the manual `LagrangeDiscr` construction:

```python
from igm.processes.iceflow.vertical import LagrangeDiscr
vd = LagrangeDiscr(cfg)   # builds V_bar, zeta, weights
V_bar = vd.V_bar
```

Also consider reusing `_load_pretrained_artifact` from `smb_inference.py` which already calls into IGM — consolidate the two loading paths.

**Files to change**: `core/load_pinn.py`, `smb_inference.py:_load_pretrained_artifact`

---

## 5. Config / Argparse — remove dead code

**Current** (`config/read_config.py`):
`parse_arguments()` and `Config` class (~97 lines) for standalone CLI usage.

**Status**: Now unused — all parameters come from Hydra via `cfg.processes.smb_inference`. This file is never imported at runtime (only `Config` was imported in `smb_inference.py` but is no longer needed since the module runs inside IGM).

**Suggestion**: Delete `config/read_config.py` and remove the import from `smb_inference.py`. If standalone testing is needed, write a small test script that builds a DictConfig manually.

**Files to change**: `smb_inference.py` (remove `from smb_inference.config.read_config import Config`), delete `config/read_config.py`

---

## 6. Vector Magnitude — use IGM's `getmag`

**Current**: Inline `tf.sqrt(ubar**2 + vbar**2)` in forward schemes.

**IGM equivalent** (`igm.utils.math.getmag`):
```python
from igm.utils.math import getmag
vel_mag = getmag(ubar, vbar)
```

**Files to change**: `core/forward_schemes/pinn_emulator_step.py`, `core/forward_schemes/pretrained_cnn_step.py`

---

## 7. 1D Interpolation — use IGM's `interp1d_tf`

**Current** (`core/smb.py:update_smb_profile`):
Custom hat-basis interpolation (~15 lines) to map a 1D SMB profile to 2D topography.

**IGM equivalent** (`igm.utils.math.interp1d_tf`):
```python
from igm.utils.math import interp1d_tf
elevations = tf.range(z_min, z_min + len(smb_vec) * dz, dz)
smb_2d = interp1d_tf(elevations, smb_vec, tf.reshape(Z_topo, [-1]))
smb_2d = tf.reshape(smb_2d, Z_topo.shape)
```
- The current hat-basis implementation is functionally equivalent to piecewise-linear interpolation. `interp1d_tf` does the same thing and is already tested/maintained.

**Caveat**: Verify that `interp1d_tf` handles extrapolation at boundaries the same way. If the current hat-basis clamping behavior is important for gradient flow, keep the custom version and add a comment explaining why.

**Files to change**: `core/smb.py`

---

## 8. Thickness Update Pattern — align with IGM's `thk` process

**Current** (in both forward scheme factories):
```python
H_new = H_ice + dt * (smb - divflux)
H_new = tf.maximum(H_new, 0.0)
```

**IGM's `thk.update`** does the same but also computes `lsurf` and `usurf`:
```python
state.divflux = compute_divflux_slope_limiter(...)
state.thk = tf.maximum(state.thk + state.dt * (state.smb - state.divflux), 0)
state.lsurf = tf.maximum(state.topg, -ratio_density * state.thk + sealevel)
state.usurf = state.lsurf + state.thk
```

**Suggestion**: Ensure `usurf` is consistently updated after thickness changes (it currently is, but scattered across the forward scheme). Consider extracting a shared `_thickness_step(H, smb, divflux, dt, topg)` that returns `(H_new, usurf_new)` following IGM's pattern.

**Files to change**: `core/forward_schemes/pinn_emulator_step.py`, `core/forward_schemes/pretrained_cnn_step.py`

---

## 9. Boundary Conditions — consolidate two implementations

**Current** (`utils/emulator_tools.py`):
Two functions doing the same thing: `apply_boundary_condition` (NumPy) and `apply_boundary_condition_tf` (TensorFlow).

**Suggestion**: Keep only `apply_boundary_condition_tf` since everything runs in TF. The NumPy version is unused in the forward loop and risks silent CPU round-trips.

**Files to change**: `utils/emulator_tools.py`

---

## 10. Plotting — consider using IGM's output modules

**Current** (`visualization/plots.py`, ~480 lines):
Rich custom plotting for loss curves, thickness, velocities, extents.

**IGM has**:
- `igm.outputs.plot2d` — 2D field plotting with basemap
- `igm.processes.data_assimilation.outputs.plots` — cost function plotting, inversion diagnostics

**Suggestion**: For standard field plots (thickness, velocity magnitude), delegate to IGM's `plot2d` output module by writing results to `state` and letting IGM's output pipeline handle visualization. Keep only inversion-specific plots (loss curves, SMB profile evolution, sim-vs-obs extent comparison) as custom code.

---

## 11. `IPython.display.clear_output` — already handled but worth noting

The `try/except ImportError` guard is correct. Consider also removing the `matplotlib.use('Agg')` hardcoding and letting it inherit from the environment — IGM's `plot2d` does not force a backend.

---

## Summary — Impact Ranking

| # | Change | Lines Removed | Benefit |
|---|--------|:---:|---|
| 5 | Delete `config/read_config.py` | ~100 | Remove dead code |
| 1 | Use `compute_divflux_slope_limiter` | ~30 | Better numerics + consistency |
| 2 | Use `grad_xy` | ~20 | Consistency |
| 3 | Use `stag2x/stag2y` | ~10 | Readability |
| 4 | Simplify emulator loading | ~50 | Less duplication, easier maintenance |
| 9 | Remove NumPy boundary condition | ~15 | Remove dead code path |
| 6 | Use `getmag` | ~5 | Minor clarity |
| 7 | Use `interp1d_tf` | ~15 | Reuse, but verify boundary behavior first |
| 8 | Align thickness step | ~0 | Consistency (refactor, not removal) |
| 10 | Delegate standard plots | ~100+ | Long-term maintainability |
