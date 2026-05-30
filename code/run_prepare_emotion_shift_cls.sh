#!/bin/bash
set -e
cd "$(dirname "$0")"
python ../../prepare_emotion_shift_cls_data.py \
  --train-input ../data/splits/train_updated.jsonl \
  --test-input ../data/splits/test_updated.jsonl \
  --out-dir ../data/splits/shift_cls
