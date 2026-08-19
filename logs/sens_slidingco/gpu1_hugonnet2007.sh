#!/bin/bash
# Rhone Hugonnet-2007 sweep, GPU1 queue: 0.1317 & 0.7442 concurrent.
set -x
cd /home/klleshi/Documents/igm-smb-inference
source ~/miniconda3/etc/profile.d/conda.sh
conda activate igm-pretrain
export MPLBACKEND=Agg
ARRH=66.7998857041393
REGTHK=1000.0
GPU=1
LOGDIR=logs/sens_slidingco

run_one() {
  v=$1
  python pipeline/sensitivity_slidingco.py --glacier rhone_hugonnet2007 \
    --slidingco "$v" --arrhenius $ARRH --reg-thk $REGTHK --gpu $GPU \
    > "$LOGDIR/rhone_hugonnet2007_gpu${GPU}_${v}.log" 2>&1
  echo "DONE $v exit=$?" >> "$LOGDIR/gpu1_hugonnet2007_driver.log"
}

run_one 0.1317 &
run_one 0.7442 &
wait

echo "GPU1 QUEUE COMPLETE" >> "$LOGDIR/gpu1_hugonnet2007_driver.log"
