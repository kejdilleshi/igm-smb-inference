#!/usr/bin/env bash
# Redo ALL aletsch_insitu slidingco-sensitivity points at arrhenius=78
# (was 182.147). Replaces the 6 figure runs; low sensitivity to arrhenius is
# expected. Single invocation runs all 6 (DA+SMB each) and auto-builds the
# combined overlay + summary. reg.thk=1000 / smb-reg=0.05 / velocity-only base
# DA (no --extra), unchanged. GPU 1.
set -o pipefail
export PATH="$HOME/miniconda3/envs/igm-pretrain/bin:$PATH"
cd /home/klleshi/Documents/igm-smb-inference || exit 1
LOGDIR="logs/sens_slidingco"; mkdir -p "$LOGDIR"

echo "[$(date '+%F %T')] aletsch_insitu — all 6 points at arrhenius=78"
python pipeline/sensitivity_slidingco.py --glacier aletsch_insitu \
    --slidingco 0.1536,0.1991,0.2689,0.4592,0.7442,1.0779 \
    --arrhenius 78 --gpu 1 \
    2>&1 | tee "$LOGDIR/aletsch_arrh78.log"
rc=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] DONE exit=$rc"
echo "overlay + summary -> results/aletsch_insitu/sens_slidingco/"
[ "$rc" -eq 0 ] && echo "ALL OK" || echo "FAILED (exit=$rc)"
