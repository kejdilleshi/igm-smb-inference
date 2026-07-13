#!/usr/bin/env bash
# Broader slidingco sensitivity for rhone_insitu + aletsch_insitu.
# Mirrors the argentiere broadening: reuse top-3 performers, add 3 sweep trials
# (0.4592, 0.7442, 1.0779 — same grid points as argentiere, so figures compare).
#
# These two are OGGM/insitu glaciers whose BASE DA is already velocity-only
# (control_list=[thk], cost_list=[velsurf,icemask]) => NO --extra needed, and
# smb-reg stays at the 0.05 default. arrhenius/reg.thk come from each
# pipeline_summary.json (rhone 78.0/300, aletsch 182.147/1000) so the fresh runs
# match the reused ones exactly.
#
#   rhone_insitu :  reuse 0.1205 0.1536 0.1991   +  new 0.4592 0.7442 1.0779
#   aletsch_insitu: reuse 0.1536 0.1991 0.2689   +  new 0.4592 0.7442 1.0779
#
# Runs on GPU 1, sequential, AFTER the argentiere driver finishes.
set -o pipefail

export PATH="$HOME/miniconda3/envs/igm-pretrain/bin:$PATH"
cd /home/klleshi/Documents/igm-smb-inference || exit 1
LOGDIR="logs/sens_slidingco"
mkdir -p "$LOGDIR"
GPU=1

# ── wait for the argentiere sensitivity driver to finish (free GPU 1) ──────────
echo "[$(date '+%F %T')] waiting for argentiere sensitivity driver to finish..."
empty=0
while true; do
  if pgrep -f 'sensitivity_slidingco.py' >/dev/null 2>&1; then
    empty=0
  else
    empty=$((empty+1))
    [ "$empty" -ge 2 ] && break     # 2 consecutive clear checks => truly done
  fi
  sleep 20
done
echo "[$(date '+%F %T')] GPU 1 free — starting rhone + aletsch."
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n '2p'

# ── per-glacier: run the 3 fresh values, then rebuild the combined 6-pt overlay ─
run_glacier () {
  local g="$1"; local reuse="$2"     # reuse = comma list of top-3 existing values
  local new="0.4592,0.7442,1.0779"
  echo
  echo "==================================================================="
  echo "[$(date '+%F %T')] $g  STEP 1/2  run fresh: $new"
  echo "==================================================================="
  python pipeline/sensitivity_slidingco.py --glacier "$g" \
      --slidingco "$new" --gpu "$GPU" \
      2>&1 | tee "$LOGDIR/${g}_broad_new.log"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] $g STEP 1 done exit=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "$g STEP 1 FAILED (exit=$rc) — skipping its overlay."
    return "$rc"
  fi
  echo
  echo "==================================================================="
  echo "[$(date '+%F %T')] $g  STEP 2/2  combined overlay: $reuse,$new"
  echo "==================================================================="
  python pipeline/sensitivity_slidingco.py --glacier "$g" \
      --slidingco "$reuse,$new" --skip-run --gpu "$GPU" \
      2>&1 | tee "$LOGDIR/${g}_broad_overlay.log"
  local rc2=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] $g STEP 2 done exit=$rc2"
  return "$rc2"
}

run_glacier rhone_insitu   "0.1205,0.1536,0.1991"
rc_rhone=$?
run_glacier aletsch_insitu "0.1536,0.1991,0.2689"
rc_aletsch=$?

echo
echo "===================== SUMMARY ====================="
printf '  %-16s exit=%s\n' rhone_insitu   "$rc_rhone"
printf '  %-16s exit=%s\n' aletsch_insitu "$rc_aletsch"
echo "overlays -> results/<glacier>/sens_slidingco/overlay_<glacier>.png + summary_<glacier>.csv"
{ [ "$rc_rhone" -eq 0 ] && [ "$rc_aletsch" -eq 0 ]; } && echo "ALL OK" || echo "SOME STEPS FAILED (see logs)"
