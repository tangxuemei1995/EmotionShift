#!/bin/bash
set -e
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=2 python train_test_emotion_shift_cls.py \
  --train-path ../data/splits/shift_cls/train_emotion_shift_cls_10_1_train.jsonl \
  --dev-path ../data/splits/shift_cls/train_emotion_shift_cls_10_1_dev.jsonl \
  --test-path ../data/splits/shift_cls/test_emotion_shift_cls.jsonl \
  --label-map-path ../data/splits/shift_cls/emotion_shift_label_map.json \
  --model-name bert-base-chinese \
  --output-dir ../../output/emotion_shift_cls_bert \
  --epochs 50 \
  --batch-size 16 \
  --lr 2e-5 \
  --max-len 200
