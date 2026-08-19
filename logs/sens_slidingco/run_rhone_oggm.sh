#!/usr/bin/env bash
# slidingco sensitivity for the OGGM/Hugonnet rhone (RGI60-11.01238, 2006-2020)
# to quantify how the Hugonnet dh/dt performs vs the in-house DEMs (rhone_insitu).
# Same 6 slidingco values as rhone_insitu so it's a direct comparison; only the
# DEM/dh/dt source differs. arrhenius=78 (unified), reg.thk=1000, smb-reg=0.1
# (the sweep-optimal: results/gpr/rhone_smbreg shows reg=0.1 gives the lowest
# stake RMSE, 1.01 vs 1.09 for 0.05), velocity-only base DA (no --extra). GPU 1.
#
# All 6 run fresh (no prior sens runs for OGGM rhone). Some high-sliding trials
# were 'inf' in the sweep (at other arrhenius) so may fit poorly at 78 — if the
# single 6-point run aborts, rebuild the overlay from whatever completed.
set -o pipefail
export PATH="$HOME/miniconda3/envs/igm-pretrain/bin:$PATH"
cd /home/klleshi/Documents/igm-smb-inference || exit 1
LOGDIR="logs/sens_slidingco"; mkdir -p "$LOGDIR"

VALS="0.1205,0.1536,0.1991,0.4592,0.7442,1.0779"
GPU=1

echo "[$(date '+%F %T')] OGGM rhone sensitivity — $VALS  (arrh=78, reg.thk=1000, smb-reg=0.1)"
python pipeline/sensitivity_slidingco.py --glacier rhone \
    --slidingco "$VALS" --arrhenius 78 --smb-reg 0.1 --gpu "$GPU" \
    2>&1 | tee "$LOGDIR/rhone_oggm.log"
rc=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] main run exit=$rc"

if [ "$rc" -ne 0 ]; then
  echo "main run aborted (a point likely diverged) — rebuilding overlay from completed points"
  python pipeline/sensitivity_slidingco.py --glacier rhone \
      --slidingco "$VALS" --arrhenius 78 --smb-reg 0.1 --skip-run --gpu "$GPU" \
      2>&1 | tee "$LOGDIR/rhone_oggm_overlay.log"
fi

echo "overlay + summary -> results/rhone/sens_slidingco/"
[ "$rc" -eq 0 ] && echo "ALL OK" || echo "PARTIAL (exit=$rc) — see logs"
