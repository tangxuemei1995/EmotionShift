#!/bin/bash
set -e
cd "$(dirname "$0")"
set -e

echo "[1/2] Export reversal test predictions to qwen-eval JSONL..."
/mnt/SSD_2T/tangxuemei/anaconda3/envs/llama-fact/bin/python \
  export_pair_cls_preds_for_qwen_eval.py \
  --model-dir ../../output/reversal_cls_bert/best_model \
  --test-cls-jsonl ../data/splits/trigger_cls/test_reversal_cls.jsonl \
  --target-field reversal \
  --out-jsonl ../../output/reversal_cls_bert/test_predictions_for_qwen_eval.jsonl

echo "[2/2] Evaluate with evaluate_qwen_test_responses_span_f1.py..."
/mnt/SSD_2T/tangxuemei/anaconda3/envs/llama-fact/bin/python \
  evaluate_qwen_test_responses_span_f1.py \
  --input_path ../../output/reversal_cls_bert/test_predictions_for_qwen_eval.jsonl \
  --model_name reversal_cls_bert_bridge \
  --results_dir ../../results
