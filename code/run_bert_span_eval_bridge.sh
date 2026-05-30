#!/bin/bash
set -e
cd "$(dirname "$0")"
python ../../convert_conll_pred_to_eval_jsonl.py \
  --conll-path ../../output/bert_crf/test_predictions.conll \
  --gold-jsonl-path ../data/splits/test_updated.jsonl \
  --out-path ../../output/bert_crf/test_predictions_for_roberta_eval.jsonl

python evaluate_qwen_test_responses_span_f1.py \
  --input_path ../../output/bert_crf/test_predictions_for_roberta_eval.jsonl \
  --model_name bert_crf_span_eval_bridge \
  --results_dir ../../results
