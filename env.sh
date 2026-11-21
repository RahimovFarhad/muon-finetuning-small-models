# Source this at the start of every session:  source env.sh
# Code and results live in this repo folder; Python, model weights and caches go in $STUDY_BASE
# (default: ./.work inside the repo). Point STUDY_BASE at a fast local disk if you have one.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${STUDY_BASE:-$REPO/.work}"
if [ -f "$BASE/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$BASE/miniforge3/etc/profile.d/conda.sh"
  [ -d "$BASE/env" ] && conda activate "$BASE/env"
fi
export STUDY_ROOT="${STUDY_ROOT:-$REPO}"
export HF_HOME="${HF_HOME:-$BASE/hf_cache}"
export PIP_CACHE_DIR="$BASE/pip_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
