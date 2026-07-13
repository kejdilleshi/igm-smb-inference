#!/usr/bin/env bash
# Argentiere DA retargeted to the 2021 DEM — identical to
# results/argentiere/sens_slidingco/slid_0.2689 (velocity-only, slidingco=0.2689,
# arrhenius=78, reg.thk=328.17) EXCEPT the surface is usurfinfer(2021) instead of
# usurf(2012). SMB inference is dropped (~processes.smb_inference); DA only.
# Output geology-optimized.nc is the 2021-epoch flux-divergence field.
set -o pipefail
export PATH="$HOME/miniconda3/envs/igm-pretrain/bin:$PATH"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=1
cd /home/klleshi/Documents/igm-smb-inference || exit 1
OUT=results/argentiere/epoch2021/da_2021_slid0.2689
mkdir -p logs/sens_slidingco

echo "[$(date '+%F %T')] DA @ 2021 DEM -> $OUT"
igm_run +experiment=params_oggm_argentiere \
  ~processes.smb_inference \
  inputs.local.filename=input_argentiere_2021.nc \
  processes.iceflow.physics.init_slidingco=0.2689189948444277 \
  processes.iceflow.physics.init_arrhenius=78.0 \
  processes.data_assimilation.regularization.thk=328.17251227955273 \
  processes.data_assimilation.control_list=[thk] \
  processes.data_assimilation.cost_list=[velsurf,icemask] \
  hydra.run.dir="$OUT" \
  core.hardware.visible_gpus=[0] \
  2>&1 | tee logs/sens_slidingco/argentiere_da2021.log
rc=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] DONE exit=$rc"
[ "$rc" -eq 0 ] && echo "OK -> $OUT/geology-optimized.nc" || echo "FAILED exit=$rc"
