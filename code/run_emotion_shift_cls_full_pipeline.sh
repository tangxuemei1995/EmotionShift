#!/bin/bash
set -e
cd "$(dirname "$0")"
set -e

# echo "[1/3] Train emotion_shift classification model..."
# bash run_emotion_shift_cls_bert.sh

echo "[2/3] Export test predictions to qwen-eval JSONL..."
/mnt/SSD_2T/tangxuemei/anaconda3/envs/llama-fact/bin/python \
  export_emotion_shift_preds_for_qwen_eval.py \
  --model-dir ../../output/emotion_shift_cls_bert/best_model \
  --test-cls-jsonl ../data/splits/shift_cls/test_emotion_shift_cls.jsonl \
  --out-jsonl ../../output/emotion_shift_cls_bert/test_predictions_for_qwen_eval.jsonl

echo "[3/3] Evaluate with qwen span-f1 script..."
/mnt/SSD_2T/tangxuemei/anaconda3/envs/llama-fact/bin/python \
  evaluate_qwen_test_responses_span_f1.py \
  --input_path ../../output/emotion_shift_cls_bert/test_predictions_for_qwen_eval.jsonl \
  --model_name emotion_shift_cls_bert_bridge \
  --results_dir ../../results

echo "Done. Check results:"
echo "../../output/emotion_shift_cls_bert/metrics.json"
echo "results/emotion_shift_cls_bert_bridge.json"
