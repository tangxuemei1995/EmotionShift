#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=env_hf_cache.sh
source "${SCRIPT_DIR}/env_hf_cache.sh"

LLAMA_FACTORY="${LLAMA_FACTORY:-/mnt/SSD_2T/tangxuemei/LLaMA-Factory}"
CONFIG="${SCRIPT_DIR}/../configs/train/qwen2_lora_sft_emotion_0415.yaml"

cd "${LLAMA_FACTORY}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" llamafactory-cli train "${CONFIG}"
