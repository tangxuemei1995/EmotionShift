#!/bin/bash
set -e
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=0 python train_test_bert_crf.py \
  --train-path ../data/splits/conll_bieo/train_bieo_10_1_train.conll \
  --dev-path ../data/splits/conll_bieo/train_bieo_10_1_dev.conll \
  --test-path ../data/splits/conll_bieo/test_bieo.conll \
  --model-name bert-base-chinese \
  --output-dir ../../output/bert_crf \
  --epochs 50 \
  --patience 0 \
  --batch-size 16 \
  --lr 1e-5 \
  --max-len 192
