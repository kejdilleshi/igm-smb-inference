# End-to-end SMB-inference pipeline

`run_pipeline.py` drives the whole DA → SMB-inference workflow for one OGGM
glacier and auto-selects the regularization weights from L-curve corners.

## Glaciers

| name        | RGI id          | experiment configs (in `experiment/`)            |
|-------------|-----------------|--------------------------------------------------|
| aletsch     | RGI60-11.01450  | `params_oggm_aletsch{,_da}.yaml`                 |
| rhone       | RGI60-11.01238  | `params_oggm_rhone{,_da}.yaml`                   |
| findelen    | RGI60-11.02773  | `params_oggm_findelen{,_da}.yaml`                |
| corbassiere | RGI60-11.02766  | `params_oggm_corbassiere{,_da}.yaml`             |

## Stages

1. **Optuna DA sweep** (`optuna_sweep_da_nsga.py --fix-reg-thk`) over `thkobs_std`
   and `init_slidingco`, with `regularization.thk` held fixed. The balanced knee
   of the velsurf/thk Pareto front is chosen.
2. **DA reg.thk L-curve** — `igm_run -m` over `regularization.thk`; `analyze_da_lcurve.py`
   plots `lcurve_step1.png` and auto-detects the corner (velocity misfit vs thickness
   smoothness).
3. **Bake** the chosen DA run's geometry (`thk`+`usurf`) into `input_<RGI>_daopt.nc`
   and generate `params_oggm_<glacier>_smbonly.yaml` (iceflow + smb_inference only;
   `init_slidingco` fixed). SMB inference then runs on a *fixed* DA geometry rather
   than re-running DA per regularization value.
4. **SMB reg L-curve** — `igm_run -m` over `smb_inference.optimization.regularisation`;
   `analyze_smb_lcurve.py` plots `lcurve_smb.png` (data misfit vs profile roughness)
   and auto-detects the corner.
5. **Final SMB run** at the chosen regularisation → `results/<glacier>/final/`.

Regularization weights are auto-selected; pass `--pause-after-da` to stop after
stage 2 and pick `regularization.thk` by hand from `lcurve_step1.png`.

## Usage

```bash
conda activate igm-pretrain   # keras 3.x — required; the keras-2 `igm` env can't load the emulator
# Full run (auto-select both regularizations):
python pipeline/run_pipeline.py --glacier findelen

# Pause after the DA L-curve for a manual reg.thk pick, then resume:
python pipeline/run_pipeline.py --glacier findelen --pause-after-da
python pipeline/run_pipeline.py --glacier findelen --from-stage smb_lcurve \
    --da-reg-thk 300 --thkobs-std 4.0 --init-slidingco 0.13

# Print every command without executing (no GPU needed):
python pipeline/run_pipeline.py --glacier findelen --dry-run
```

Key flags: `--da-reg`, `--smb-reg` (comma grids), `--n-trials`, `--n-jobs`
(joblib parallelism for the L-curve multiruns), `--gpus`/`--workers-per-gpu`
(Optuna), `--da-nbitmax`/`--smb-nbitmax` (cut iterations for smoke tests),
`--from-stage {sweep,da_lcurve,smb_lcurve,final}`. Per-run state is saved to
`results/<glacier>/pipeline_state.json` so resuming reuses earlier picks.

## Outputs (`results/<glacier>/`)

```
da_lcurve/   lcurve_step1.png, da_lcurve_result.json, <reg>/geology-optimized.nc
smb_lcurve/  lcurve_smb.png,   smb_lcurve_result.json, <reg>/smb_inference/*
final/       smb_inference/{smb_vec.npy, smb_vec_iters/*.png, ...}
pipeline_summary.json   # chosen thkobs_std, init_slidingco, reg.thk, reg_smb
```

## Prerequisite: run under the `igm-pretrain` env (keras 3.x)

Every `igm_run` here loads the pretrained iceflow emulator named in the configs'
`processes.iceflow.unified.network.pretrained_path`. That artifact was exported
with **keras 3** and only loads under the `igm-pretrain` conda env (keras 3.12 /
tf 2.19). Running under the `igm` env (keras 2.15) fails with:

```
ValueError: Layer 'slide_head_out' expected 2 variables, but received 0 ...
  rebuild_emulator_from_manifest → model.load_weights(...)
```

That is purely a wrong-environment symptom, not a code or artifact problem —
`conda activate igm-pretrain` and it loads ("✅ Emulator loaded successfully").
The pipeline scripts themselves are env-agnostic; `--dry-run` and the analysis
steps work anywhere.

## Files

- `run_pipeline.py`     — orchestrator + glacier registry
- `lcurve.py`           — `parse_param`, `lcurve_corner` (Menger curvature), `knee_select`
- `analyze_da_lcurve.py`  — DA L-curve plot + corner
- `analyze_smb_lcurve.py` — SMB L-curve plot + corner
- `bake_da_geometry.py`   — bake DA-optimized geometry into `input_<RGI>_daopt.nc`

`experiment/*_smbonly.yaml` and `data/*_daopt.nc` are regenerated each run and
are git-ignored.
