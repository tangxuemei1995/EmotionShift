#!/usr/bin/env bash
# Link emotion_github FT data into LLaMA-Factory/data/ for dataset train_emotion_0415.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EMOTION_GITHUB="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LLAMA_FACTORY="${LLAMA_FACTORY:-/mnt/SSD_2T/tangxuemei/LLaMA-Factory}"
FT_DIR="${EMOTION_GITHUB}/data/ft"

mkdir -p "${LLAMA_FACTORY}/data"
ln -sfn "${FT_DIR}/train_emotion_0415.json" "${LLAMA_FACTORY}/data/train_emotion_0415.json"
ln -sfn "${FT_DIR}/test_emotion_0415.json" "${LLAMA_FACTORY}/data/test_emotion_0415.json"

echo "Linked:"
ls -la "${LLAMA_FACTORY}/data/train_emotion_0415.json" "${LLAMA_FACTORY}/data/test_emotion_0415.json"
echo ""
echo "Ensure dataset_info.json contains train_emotion_0415 (see ../dataset_info_train_emotion_0415.json)."
