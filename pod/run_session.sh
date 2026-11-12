#!/usr/bin/env bash
# Run ON THE POD inside tmux, from /workspace/ewh (a checkout of tag pap-v1 including .git).
# Outputs land on the pod's /workspace volume so they survive a stop.
set -uo pipefail
cd /workspace/ewh
export PATH=/workspace/uvenv/bin:$PATH HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 \
       HF_HUB_DOWNLOAD_TIMEOUT=60 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash pod/setup.sh 2>&1 | tee -a session_setup.log
uv run python pod/run_session.py 2>&1 | tee -a session.log
echo "SESSION_EXIT $?" | tee -a session.log
