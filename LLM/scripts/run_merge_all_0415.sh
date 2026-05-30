#!/usr/bin/env bash
# Merge LoRA adapters into full models under LLaMA-Factory/output/*_lora/emotion_0415/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env_hf_cache.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LLAMA_FACTORY="${LLAMA_FACTORY:-/mnt/SSD_2T/tangxuemei/LLaMA-Factory}"
cd "${LLAMA_FACTORY}"

MODEL="${1:-qwen2}"  # qwen2 | glm4 | llama3 | all

merge_one() {
  local name="$1"
  llamafactory-cli export "${SCRIPT_DIR}/../configs/merge/${name}_lora_merge_0415.yaml"
}

case "${MODEL}" in
  qwen2) merge_one qwen2 ;;
  glm4) merge_one glm4 ;;
  llama3) merge_one llama3 ;;
  all)
    merge_one qwen2
    merge_one glm4
    merge_one llama3
    ;;
  *) echo "Usage: $0 [qwen2|glm4|llama3|all]"; exit 1 ;;
esac
