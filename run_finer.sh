#!/bin/bash
# Phase 1b: two extra learning rates per setting around the optimum (as in the paper),
# then re-pick the best rates. Run after run_sweep.sh and before run_seeds.sh.
source "$(dirname "$0")/env.sh"
mkdir -p logs
for O in adamw muon; do
  for LR in 5e-06 2e-05; do python train.py --setting narrow --opt $O --lr $LR --seed 0 2>&1 | tee logs/narrow_${O}_lr${LR}_s0.log; done
  for LR in 2e-06 5e-06; do python train.py --setting broad  --opt $O --lr $LR --seed 0 2>&1 | tee logs/broad_${O}_lr${LR}_s0.log; done
done
python pick_best.py
