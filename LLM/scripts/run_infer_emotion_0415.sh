#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env_hf_cache.sh"

MODEL="${1:-qwen2}"
GPU="${2:-0}"

python "${SCRIPT_DIR}/../inference/infer_emotion_0415.py" \
  --model "${MODEL}" \
  --gpu "${GPU}"
