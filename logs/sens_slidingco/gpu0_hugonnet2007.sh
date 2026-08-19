#!/bin/bash
# Rhone Hugonnet-2007 sweep, GPU0 queue: 0.0978 & 0.1991 concurrent, then 1.0779.
set -x
cd /home/klleshi/Documents/igm-smb-inference
source ~/miniconda3/etc/profile.d/conda.sh
conda activate igm-pretrain
export MPLBACKEND=Agg
ARRH=66.7998857041393
REGTHK=1000.0
GPU=0
LOGDIR=logs/sens_slidingco

run_one() {
  v=$1
  python pipeline/sensitivity_slidingco.py --glacier rhone_hugonnet2007 \
    --slidingco "$v" --arrhenius $ARRH --reg-thk $REGTHK --gpu $GPU \
    > "$LOGDIR/rhone_hugonnet2007_gpu${GPU}_${v}.log" 2>&1
  echo "DONE $v exit=$?" >> "$LOGDIR/gpu0_hugonnet2007_driver.log"
}

run_one 0.0978 &
run_one 0.1991 &
wait

run_one 1.0779

echo "GPU0 QUEUE COMPLETE" >> "$LOGDIR/gpu0_hugonnet2007_driver.log"
