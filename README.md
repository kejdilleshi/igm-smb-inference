# igm-smb-inference

A user module for [IGM (Instructed Glacier Model)](https://igm-model.org/) that performs **Surface Mass Balance (SMB) inversion** from glacier thickness observations using gradient-based optimization through a differentiable ice flow emulator.

## How it works

1. **Data assimilation** runs first (built-in IGM process) — optimizes ice thickness from observed surface velocities and trains an ice flow emulator.
2. **`smb_inference`** then takes the DA-optimized state (thickness + trained emulator) and runs a second inversion:
   - Parameterizes SMB as a 1D elevation-dependent profile (hat basis functions).
   - Runs a differentiable forward glacier simulation using `tf.GradientTape` + `tf.while_loop` with gradient checkpointing.
   - Minimizes MAE between simulated and observed thickness plus a curvature regularization term.
   - Uses Adam optimizer; supports early stopping and periodic emulator retraining.

## Usage

```bash
igm_run +experiment=params
```

See `experiment/params.yaml` for the full configuration. An alternative experiment using a pretrained CNN emulator is available:

```bash
igm_run +experiment=params_pretrained
```

## Configuration

All parameters are in `user/conf/processes/smb_inference.yaml` and accessed via `cfg.processes.smb_inference`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `ttot` | 2017.0 | End time of forward simulation (yr) |
| `t_start` | 1700.0 | Start time (yr) |
| `emulator.from_state` | false | If true, inherit emulator from `state.iceflow_model` |
| `inversion.method` | profile | Inversion method (1D elevation-dependent profile) |
| `inversion.dz` | 100.0 | Elevation bin spacing (m) |
| `optimization.nbitmax` | 300 | Maximum optimization iterations |
| `optimization.learning_rate` | 0.05 | Adam learning rate |
| `optimization.regularisation` | 0.05 | Curvature regularization weight |

## SMB inference for any glacier via OGGM

The repo ships a generic, RGI-ID-parameterised pipeline that pulls all the
geophysical inputs from the [OGGM](https://oggm.org) shop and produces an
IGM-ready NetCDF — so applying the method to a new glacier is essentially a
one-line invocation plus a YAML clone.

### Data sources pulled automatically

| Field | Source | Period |
|---|---|---|
| Reference DEM (`topo`) | OGGM L3 prepro (SRTM v4 in the Alps) | Feb 2000 |
| Elevation-change rate (`hugonnet_dhdt`) | Hugonnet et al. 2021 | 2000-01-01 → 2020-01-01 |
| Surface velocity (`millan_vx/vy`) | Millan et al. 2022 | 2017–18 composite |
| Thickness prior (`millan_ice_thickness`) | Millan et al. 2022 | — |
| Point thickness (`thkobs`) | GlaThiDa (rasterised) | varies per glacier |
| Glacier outline (`icemask`) | RGI v6 | — |

The end-of-period observation `usurfinfer` is the COP-DEM 30 (≈2020 anchor —
fetched separately by `data/oggm/fetch_copdem.py`), and the start surface is
*derived* from it: `usurfobs = usurfinfer − dhdt × N_YEARS`. `N_YEARS` is a
CLI flag on `build_input_nc.py` (default 20 → start ≈ Jan 2000; pass
`--n-years 14` for a Jan 2006 start, etc.). The SMB inversion fits against
`usurfinfer`.

### Worked example — Argentière

1. **Look up the RGI ID** (Argentière is `RGI60-11.03638`).

2. **Fetch the OGGM bundle** (one HTTP download, ~30 MB, a few minutes):
   ```bash
   conda activate igm-pretrain
   python data/oggm/fetch_oggm.py RGI60-11.03638
   ```
   Writes `data/oggm/RGI60-11.03638/{gridded_data.nc, glathida_data.csv, …}`.

3. **Build the IGM-ready NetCDF**:
   ```bash
   python data/oggm/build_input_nc.py RGI60-11.03638
   ```
   Produces `data/input_RGI60_11.03638.nc` (OGGM local TMerc grid, 137 m
   resolution, all variables in IGM's expected layout). Runs a sanity check
   that on-glacier ⟨dh/dt⟩ falls within ±3 m yr⁻¹.

4. **Compute the crop bbox** (tight bounding box + 500 m margin around the
   icemask):
   ```bash
   python -c "
   import netCDF4 as nc, numpy as np
   ds = nc.Dataset('data/input_RGI60_11.03638.nc')
   x, y = ds['x'][:].astype(float), ds['y'][:].astype(float)
   m = ds['icemask'][:] > 0.5
   c = np.where(np.any(m,axis=0))[0]; r = np.where(np.any(m,axis=1))[0]
   print(f'  xmin: {x[c[0]]-500:.1f}')
   print(f'  xmax: {x[c[-1]]+500:.1f}')
   print(f'  ymin: {min(y[r[0]],y[r[-1]])-500:.1f}')
   print(f'  ymax: {max(y[r[0]],y[r[-1]])+500:.1f}')"
   ```

5. **Clone the experiment YAML**:
   ```bash
   cp experiment/params_oggm_aletsch.yaml experiment/params_oggm_argentiere.yaml
   ```
   Edit only the three glacier-specific lines:
   ```yaml
   inputs:
     local:
       filename: input_RGI60_11.03638.nc      # ← new NC from step 3
       crop:                                   # ← values from step 4
         xmin: …
         xmax: …
         ymin: …
         ymax: …
   ```
   The pretrained `SIADecompNet_18x80_dil1-32/` emulator is grid-agnostic
   (it consumes `dX` as an input feature) so it works unchanged. DA
   hyperparameters (`reg_thk`, `reg_slidingco`, `init_slidingco`, etc.) are
   glacier-dependent and almost certainly need re-tuning — clone a
   DA-only variant of the YAML and run `optuna_sweep_da_nsga.py`.

6. **(Optional) Reproject existing stake data.** If you already have a stake
   CSV in a non-OGGM CRS (e.g. the existing
   `data/argentiere/SMB_argentiere_2012-2021_utm32N.csv` is in UTM32N, but
   the OGGM grid is local TMerc), reproject the `(X, Y)` columns once:
   ```python
   from pyproj import Transformer
   import xarray as xr, pandas as pd
   ds = xr.open_dataset("data/oggm/RGI60-11.03638/gridded_data.nc")
   tr = Transformer.from_crs("EPSG:32632", ds.attrs["pyproj_srs"], always_xy=True)
   df = pd.read_csv("data/argentiere/SMB_argentiere_2012-2021_utm32N.csv")
   df["X"], df["Y"] = tr.transform(df["X"].values, df["Y"].values)
   df.to_csv("data/oggm/stakes_RGI60-11.03638.csv", index=False)
   ```
   Then point `processes.smb_inference.observations.smb_points_file` at the
   reprojected file. The stake CSV is used only for the per-iteration
   diagnostic overlay — it does not enter the inversion cost, so omitting
   it is also harmless.

7. **Run end-to-end** (DA + SMB inference):
   ```bash
   igm_run +experiment=params_oggm_argentiere
   ```
   Output lands in `outputs/<date>/<time>/smb_inference/` with `smb_vec.npy`,
   `z_min.npy`, `dz.npy`, the per-iteration profile PNGs, and a final loss
   curve.

### Validation against in-situ stake observations

The inferred SMB profile is, by construction, the **multi-year mean** balance
that carries the glacier from its start-of-period geometry to the end-of-period
geometry. A raw stake reading is a *single-year* point balance, so a direct
scatter mixes the elevation signal with year-to-year (climate) variability and
biases the comparison toward the years that happen to be over-sampled.

The validation follows Kneib et al., *The Cryosphere* **18**, 5965–6005 (2024,
§2.8): rather than comparing the profile to individual stakes, we group stakes
by horizontal position into **stake bins** and compare against each bin's
multi-year mean. Each bin = one physical stake observed across many years;
its mean is therefore a multi-year point balance directly comparable to the
inferred profile, and its standard deviation is the interannual scatter at that
location (shown as a thin error bar).

This logic lives in
[`user/code/processes/smb_inference/data/homogenize.py`](user/code/processes/smb_inference/data/homogenize.py).
It is invoked automatically in two places:

- **During inference** — the per-iteration profile PNG
  (`smb_inference/smb_vec_iters/smb_vec_iter_*.png`) overlays the qualifying
  bins (≥10 distinct years) with mean ± std.
- **After inference** — [`pipeline/compare_smb.py`](pipeline/compare_smb.py)
  regenerates the same comparison from saved `smb_vec.npy` + `z_min.npy` +
  `dz.npy` without re-running inference, and prints the per-stake-bin RMSE.

#### Test glaciers

| Glacier | Window | Stake source | Qualifying bins | Span coverage |
|---|---|---|--:|---|
| Aletsch (RGI60-11.01450) | 2000–2020 (20 yr) | GLAMOS annual | 2 (terminus 1956 m, head 3341 m) | full window |
| Rhône (RGI60-11.01238)   | **2006–2020 (14 yr)** | GLAMOS annual | 6–7 between 2246 and 3235 m | full window |

The Rhône window is intentionally **14 years**, not 20: the GLAMOS stake
network on Rhonegletscher starts in 2006, so matching the inversion window to
the stake record makes the profile and the bins represent the *same* period.
The 14-yr window is wired through:

```bash
# regenerate the IGM-ready NC with usurfobs ≈ Jan 2006
python data/oggm/build_input_nc.py RGI60-11.01238 --n-years 14
# regenerate the stake CSV in the same window (script default)
python data/rhone/build_raw_stakes_csv.py
# run the pipeline (DA sweep + L-curves + final SMB inference)
python pipeline/run_pipeline.py --glacier rhone
```

For Aletsch the default `--n-years 20` and `--start 2000 --end 2020` are
correct, so:

```bash
python pipeline/run_pipeline.py --glacier aletsch
```

#### Adapting to a new glacier

Drop a raw-stake CSV at `data/<glacier>/SMB_<glacier>_<start>-<end>_rawstakes_utm32N.csv`
with columns `X, Y, Alt, Annual_SMB (m w.e.), Year` (one row per stake-year),
point `observations.smb_points_file` at it in the experiment YAML, and
`prepare_stake_overlay` will pick it up. Choose the inversion window so it
either matches the stake-record span or is fully contained in it — otherwise
the bins represent a different period than the profile, which the overlay will
report in its RMSE label but cannot correct.

### Caveats when generalising beyond the Alps

- **Outside the SRTM swath (lat > 60° N/S):** OGGM serves a different DEM
  source (ArcticDEM, COPDEM30, etc.), whose epoch is not 2000.
  `build_input_nc.py` would then need to back-correct `topo` to the year
  2000 using the same Hugonnet rate before assembling `usurfinfer`.
- **Polar / cold glaciers:** SRTM has a known radar penetration bias of a
  few metres in dry firn, which would propagate into `usurfobs` and bias
  the inferred SMB at high elevations.
- **Velocity epoch:** Millan22 is a 2017–18 mean — for fast-evolving
  glaciers this may mis-time the dynamics used in the 20-year forward
  integration. ITS_LIVE annual fields can be averaged manually for a
  truer 2000-epoch state, but coverage varies per glacier.

## Directory structure

```
igm-smb-inference/
├── experiment/
│   ├── params.yaml              # DA + SMB inference experiment
│   └── params_pretrained.yaml   # Pretrained CNN experiment
├── user/
│   ├── code/processes/smb_inference/   # Python package
│   │   ├── __init__.py                 # Exports initialize/update/finalize
│   │   ├── smb_inference.py            # Main entry point
│   │   ├── core/                       # Glacier model, SMB, inversion, climate
│   │   ├── data/                       # Data loading (NetCDF)
│   │   ├── utils/                      # Numerical tools (flux divergence, etc.)
│   │   ├── visualization/              # Plotting
│   │   └── config/                     # Config parsing
│   └── conf/processes/
│       └── smb_inference.yaml          # Hydra config (default parameters)
├── data/
│   └── input.nc
└── README.md
```


## License

GNU GPL v3 (same as IGM)
