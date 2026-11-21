#!/bin/bash
# Phase 1: baseline + learning-rate sweep (20 runs, seed 0). Safe to re-run: finished runs
# are skipped. Run inside tmux/screen or with nohup:  bash run_sweep.sh
source "$(dirname "$0")/env.sh"
mkdir -p logs
LRS="1e-06 3e-06 1e-05 3e-05 1e-04"

[ -f results/baseline.json ] || python train.py --baseline 2>&1 | tee logs/baseline.log

for SETTING in narrow broad; do       # narrow first: fast, so you see results early
  for LR in $LRS; do
    for OPT in adamw muon; do
      NAME=${SETTING}_${OPT}_lr${LR}_s0
      echo "=== $(date +%H:%M) starting $NAME"
      python train.py --setting $SETTING --opt $OPT --lr $LR --seed 0 2>&1 | tee logs/$NAME.log \
        || echo "!!! $NAME failed (see logs/$NAME.log) - continuing"
    done
  done
done
python pick_best.py
echo "=== sweep finished $(date)"
