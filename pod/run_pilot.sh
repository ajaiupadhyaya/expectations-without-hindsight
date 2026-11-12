#!/usr/bin/env bash
# GPU pilot (plan Task 3.8), run ON THE POD inside tmux. Non-instruct base model only.
set -euo pipefail
set -o pipefail
cd /workspace/ewh
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60
bash pod/setup.sh 2>&1 | tee pilot_setup.log
export PATH=/workspace/uvenv/bin:$PATH
uv run python pod/pilot_gpu.py 2>&1 | tee pilot_gpu.log
echo PILOT_DONE
