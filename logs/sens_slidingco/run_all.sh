#!/usr/bin/env bash
# slidingco sensitivity — run all three in-house glaciers sequentially on GPU 1.
# Sequential (not parallel) because all three target the same GPU; running them
# at once would compete for GPU-1 memory. Continues even if one glacier fails,
# then prints a summary of exit codes.
set -o pipefail

export PATH="$HOME/miniconda3/envs/igm-pretrain/bin:$PATH"
cd /home/klleshi/Documents/igm-smb-inference || exit 1

LOGDIR="logs/sens_slidingco"
mkdir -p "$LOGDIR"

declare -A ARRH=( [argentiere]=78 [aletsch_insitu]=182.1 [rhone_insitu]=78 )
ORDER=( argentiere aletsch_insitu rhone_insitu )
declare -A STATUS

SLID="0.01,0.03,0.40,0.60,1.00"
GPU=1

for g in "${ORDER[@]}"; do
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "==================================================================="
  echo "[$ts] START $g  (slidingco=$SLID  arrhenius=${ARRH[$g]}  gpu=$GPU)"
  echo "==================================================================="
  python pipeline/sensitivity_slidingco.py --glacier "$g" \
      --slidingco "$SLID" --arrhenius "${ARRH[$g]}" --gpu "$GPU" \
      2>&1 | tee "$LOGDIR/${g}.log"
  rc=${PIPESTATUS[0]}
  STATUS[$g]=$rc
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$ts] DONE  $g  exit=$rc"
done

echo
echo "===================== SUMMARY ====================="
fail=0
for g in "${ORDER[@]}"; do
  printf '  %-16s exit=%s\n' "$g" "${STATUS[$g]}"
  [ "${STATUS[$g]}" -ne 0 ] && fail=1
done
echo "==================================================="
[ "$fail" -eq 0 ] && echo "ALL GLACIERS OK" || echo "SOME GLACIERS FAILED (see logs above)"
