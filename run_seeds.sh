#!/bin/bash
# Phase 2: two extra seeds at each optimizer's best learning rate (8 runs).
# Run after run_sweep.sh has finished:  bash run_seeds.sh
source "$(dirname "$0")/env.sh"
mkdir -p logs
[ -f seeds_todo.txt ] || python pick_best.py
while read SETTING OPT LR SEED; do
  [ -z "$SETTING" ] && continue
  NAME=${SETTING}_${OPT}_lr${LR}_s${SEED}
  echo "=== $(date +%H:%M) starting $NAME"
  python train.py --setting $SETTING --opt $OPT --lr $LR --seed $SEED 2>&1 | tee logs/$NAME.log \
    || echo "!!! $NAME failed - continuing"
done < seeds_todo.txt
python analyze.py
echo "=== seeds finished $(date)"
