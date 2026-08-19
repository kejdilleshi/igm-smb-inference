#!/usr/bin/env bash
# Initialisation-shock check — igm_run commands used to produce the two 1-year
# zero-smb replays that pipeline/shock_check.py then diffs against their
# starting geometry. See that script's docstring for the rationale.
#
# Both runs redo the SAME data assimilation as the glacier's adopted run
# (identical init_slidingco / init_arrhenius / reg.thk overrides — copy them
# from that run's .hydra/overrides.yaml), then replace the SMB-inference
# stage with a 1-year replay under a smb profile pinned at exactly zero
# (smb_init_low=high=0, optimizer=adam, learning_rate=0, nbitmax=1: the
# optimizer's first "step" has zero magnitude, so state.thk after the run is
# exactly H after one forward year of Eq. (1)-(3) with smb=0). The only
# difference between the two is initial_state.usurf_variable, which selects
# which of the two candidate starting geometries (Sect. 2.1) the replay
# starts from.
#
# Usage: shock_check_run.sh <glacier> <experiment> <init_slidingco> \
#          <init_arrhenius> <reg_thk> <t_start> <ttot> <outdir>
set -euo pipefail
GLACIER=$1; EXPERIMENT=$2; SLID=$3; ARRH=$4; REG=$5; T0=$6; T1=$7; OUT=$8

COMMON=(
  igm_run "+experiment=${EXPERIMENT}"
  "processes.iceflow.physics.init_slidingco=${SLID}"
  "processes.iceflow.physics.init_arrhenius=${ARRH}"
  "processes.data_assimilation.regularization.thk=${REG}"
  "processes.smb_inference.optimization.regularisation=0.05"
  "processes.smb_inference.t_start=${T0}"
  "processes.smb_inference.ttot=${T1}"
  "processes.smb_inference.inversion.smb_init_low=0.0"
  "processes.smb_inference.inversion.smb_init_high=0.0"
  "processes.smb_inference.optimization.optimizer=adam"
  "processes.smb_inference.optimization.learning_rate=0.0"
  "processes.smb_inference.optimization.nbitmax=1"
  "core.hardware.visible_gpus=[0]"
)

TF_FORCE_GPU_ALLOW_GROWTH=true "${COMMON[@]}" \
  "hydra.run.dir=${OUT}/t1"
# H_t2 (DA-state) replay: disable the re-anchoring, so H_init = state.thk
# (the raw DA output) instead of the derived start-epoch geometry.
TF_FORCE_GPU_ALLOW_GROWTH=true "${COMMON[@]}" \
  "processes.smb_inference.initial_state.usurf_variable=" \
  "hydra.run.dir=${OUT}/t2"

python pipeline/shock_check.py --glacier "${GLACIER}" \
  --t1-dir "${OUT}/t1" --t2-dir "${OUT}/t2" \
  --input-nc "data/input_${GLACIER}.nc" --out "${OUT}"
