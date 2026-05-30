#!/bin/bash
set -e
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=0 python train_test_emotion_shift_cls.py \
  --train-path ../data/splits/trigger_cls/train_reversal_cls_10_1_train.jsonl \
  --dev-path ../data/splits/trigger_cls/train_reversal_cls_10_1_dev.jsonl \
  --test-path ../data/splits/trigger_cls/test_reversal_cls.jsonl \
  --label-map-path ../data/splits/trigger_cls/reversal_label_map.json \
  --model-name hfl/chinese-roberta-wwm-ext \
  --output-dir ../../output/reversal_cls_roberta \
  --epochs 30 \
  --batch-size 16 \
  --lr 2e-5 \
  --max-len 192
