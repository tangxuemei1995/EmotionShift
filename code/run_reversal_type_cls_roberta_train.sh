#!/bin/bash
set -e
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=0 python train_test_emotion_shift_cls.py \
  --train-path ../data/splits/mechanism_cls/train_reversal_type_cls_10_1_train.jsonl \
  --dev-path ../data/splits/mechanism_cls/train_reversal_type_cls_10_1_dev.jsonl \
  --test-path ../data/splits/mechanism_cls/test_reversal_type_cls.jsonl \
  --label-map-path ../data/splits/mechanism_cls/reversal_type_label_map.json \
  --model-name hfl/chinese-roberta-wwm-ext \
  --output-dir ../../output/reversal_type_cls_roberta \
  --epochs 30 \
  --batch-size 16 \
  --lr 2e-5 \
  --max-len 192
