#!/usr/bin/env bash
# Broader argentiere slidingco sensitivity (velocity-only DA, NO thk in cost).
#
#   reuse (top-3 performers, already computed): 0.1077, 0.1353, 0.1991
#   run fresh (3 sweep trials):                 0.4592, 0.7442, 1.0779
#
# Config reproduces the Jul-6 velocity-only runs exactly:
#   arrhenius=78 (fixed), reg.thk=328.17 (from summary), smb-reg=0.1,
#   control_list=[thk], cost_list=[velsurf,icemask]  (thk NOT a cost term).
# GPU 1, sequential.
set -o pipefail

export PATH="$HOME/miniconda3/envs/igm-pretrain/bin:$PATH"
cd /home/klleshi/Documents/igm-smb-inference || exit 1
LOGDIR="logs/sens_slidingco"
mkdir -p "$LOGDIR"

GPU=1
COMMON=( --glacier argentiere --arrhenius 78 --smb-reg 0.1
         --extra processes.data_assimilation.control_list=[thk]
         --extra processes.data_assimilation.cost_list=[velsurf,icemask] )

echo "==================================================================="
echo "[$(date '+%F %T')] STEP 1/2  run fresh: 0.4592, 0.7442, 1.0779"
echo "==================================================================="
python pipeline/sensitivity_slidingco.py "${COMMON[@]}" \
    --slidingco 0.4592,0.7442,1.0779 --gpu "$GPU" \
    2>&1 | tee "$LOGDIR/argentiere_broad_new.log"
rc=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] STEP 1 done  exit=$rc"

if [ "$rc" -ne 0 ]; then
  echo "STEP 1 FAILED (exit=$rc) — skipping combined overlay so a partial run is obvious."
  exit "$rc"
fi

echo
echo "==================================================================="
echo "[$(date '+%F %T')] STEP 2/2  rebuild combined 6-point overlay"
echo "  reuse 0.1077 0.1353 0.1991  +  new 0.4592 0.7442 1.0779"
echo "==================================================================="
python pipeline/sensitivity_slidingco.py "${COMMON[@]}" \
    --slidingco 0.1077,0.1353,0.1991,0.4592,0.7442,1.0779 \
    --skip-run --gpu "$GPU" \
    2>&1 | tee "$LOGDIR/argentiere_broad_overlay.log"
rc=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] STEP 2 done  exit=$rc"

echo
echo "===================== SUMMARY ====================="
echo "overlay + summary written to results/argentiere/sens_slidingco/"
echo "  overlay_argentiere.png   summary_argentiere.csv"
[ "$rc" -eq 0 ] && echo "ALL OK" || echo "OVERLAY STEP FAILED (exit=$rc)"
