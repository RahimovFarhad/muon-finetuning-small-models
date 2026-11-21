#!/bin/bash
# Quick check that everything works (~10 min). Run:  bash smoke.sh
source "$(dirname "$0")/env.sh"
set -e
python train.py --setting narrow --opt muon  --lr 1e-5 --max-steps 30 --tag smoke
python train.py --setting broad  --opt adamw --lr 1e-5 --max-steps 30 --tag smoke
python estimate.py
