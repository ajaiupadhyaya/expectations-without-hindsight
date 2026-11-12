#!/usr/bin/env bash
# Run ON THE POD after the repo has been rsynced to /workspace/ewh (no data/ directory).
# Installs uv (pinned, from PyPI) and the locked environment. Weights go to /workspace/hf.
set -euo pipefail
cd /workspace/ewh
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60
python3 -m pip install --quiet "uv==0.10.4"
uv sync --frozen --quiet
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"
