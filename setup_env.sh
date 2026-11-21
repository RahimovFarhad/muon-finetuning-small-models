#!/bin/bash
# One-time setup: installs Miniforge + Python 3.11 + packages into $STUDY_BASE (default ./.work).
# Skip this if you already have Python >= 3.10 with a CUDA build of PyTorch >= 2.9:
#   pip install -r requirements.txt
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
BASE="${STUDY_BASE:-$REPO/.work}"
mkdir -p "$BASE"
if [ ! -d "$BASE/miniforge3" ]; then
  curl -L -o "$BASE/miniforge.sh" \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$BASE/miniforge.sh" -b -p "$BASE/miniforge3"
  rm "$BASE/miniforge.sh"
fi
source "$BASE/miniforge3/etc/profile.d/conda.sh"
[ -d "$BASE/env" ] || conda create -y -p "$BASE/env" python=3.11
conda activate "$BASE/env"
source "$REPO/env.sh"
pip install --upgrade pip
pip install -r requirements.txt
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
assert hasattr(torch.optim, "Muon"), "torch.optim.Muon missing - need torch >= 2.9"
PY
echo "Setup done."
