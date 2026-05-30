# Emotion Shift

[中文说明](README.md) · **English**

A consolidated repository for **data**, **BERT / RoBERTa**, and **LLM (LLaMA-Factory)** experiments on Chinese **emotion shift** annotation.  
For LLM-specific configs, see [`LLM/README.md`](LLM/README.md).

---

## Repository layout

```text
emotion_github/
├── README.md          # Chinese documentation
├── README_EN.md       # This file (English)
├── LLM/               # LLaMA-Factory emotion_0415 train / merge / infer (see LLM/README.md)
├── code/              # BERT/RoBERTa training, export, evaluation scripts and run_*.sh
└── data/
    ├── splits/        # Human-annotated splits, classification, and sequence labeling data
    └── ft/            # LLaMA-Factory instruction-tuning JSON
```

---

## How to run

Execute all `code/run_*.sh` scripts from **`emotion_github/code/`** (each script `cd`s into its own directory):

```bash
cd emotion_github/code
conda activate llama-fact   # or your PyTorch + transformers environment
bash run_bert_crf.sh
```

Path conventions (relative to `code/`):

| Purpose | Path |
|---------|------|
| Data | `../data/splits/...` |
| Model outputs | `../../output/<experiment_name>/` |

---

## `code/` — Script index

### Core Python modules

| File | Description |
|------|-------------|
| `train_test_bert_crf.py` | **Emotion Span**: BERT/RoBERTa + CRF, BIEO CoNLL sequence labeling |
| `train_test_emotion_shift_cls.py` | **Trigger / Mechanism / Emotion_shift**: sentence-pair classification (BERT or RoBERTa) |
| `export_pair_cls_preds_for_qwen_eval.py` | Export Trigger/Mechanism predictions to unified JSONL for span-F1 evaluation |
| `export_emotion_shift_preds_for_qwen_eval.py` | Export Emotion_shift predictions to evaluation JSONL |
| `evaluate_qwen_test_responses_span_f1.py` | Span matching + per-field F1 (same protocol as LLM evaluation) |
| `evaluate_pair_cls_model.py` | Classification accuracy on the test set only (no span alignment) |

### Task × backbone coverage

Field mapping in the annotation schema:

| Annotation field | Data directory (`data/splits/`) | Training script |
|------------------|-----------------------------------|-----------------|
| **Emotion_span** | `conll_bieo/` | `train_test_bert_crf.py` |
| **Trigger** (formerly Reversal) | `trigger_cls/` | `train_test_emotion_shift_cls.py` |
| **Mechanism** (formerly Reversal_type) | `mechanism_cls/` | `train_test_emotion_shift_cls.py` |
| **Emotion_shift** | `shift_cls/` | `train_test_emotion_shift_cls.py` |

| Task | BERT training | RoBERTa training | Export + span-F1 eval |
|------|---------------|------------------|------------------------|
| Emotion Span (CRF) | `run_bert_crf.sh` | `run_roberta_crf.sh` | `run_bert_span_eval_bridge.sh` |
| Trigger classification | `run_reversal_cls_bert.sh` | `run_reversal_cls_roberta_train.sh` | `run_reversal_cls_roberta_test.sh` (**BERT weights only**) |
| Mechanism classification | `run_reversal_type_cls_bert.sh` | `run_reversal_type_cls_roberta_train.sh` | `run_reversal_type_cls_roberta_test.sh` (**BERT weights only**) |
| Emotion_shift classification | `run_emotion_shift_cls_bert.sh` (**misleading filename in some scripts; uses BERT**) | **No dedicated `run_*_roberta.sh`** | `run_emotion_shift_cls_full_pipeline.sh` (BERT) |

**Completeness notes:**

- **Training**: BERT + RoBERTa entry points exist for all four task types (two `train_test_*.py` modules, switch via `--model-name`).
- **Gaps / caveats:**
  1. **RoBERTa span-F1 bridge**: only `run_bert_span_eval_bridge.sh` for BERT-CRF; for RoBERTa-CRF, adapt paths to `../../output/roberta_crf/`.
  2. **`run_*_roberta_test.sh` for Trigger/Mechanism**: filenames say “roberta” but they load `output/reversal_cls_bert` and `output/reversal_type_cls_bert`; copy scripts and point to `*_roberta/best_model` for RoBERTa checkpoints.
  3. **Emotion_shift**: `run_emotion_shift_cls_roberta.sh` uses `bert-base-chinese`; for RoBERTa, add a script with `hfl/chinese-roberta-wwm-ext` and `output/emotion_shift_cls_roberta` (that output dir may already exist from a prior manual run).
  4. **Data prep**: `run_prepare_emotion_shift_cls.sh` only builds shift data; run `prepare_reversal_field_cls_data.py` in the parent repo for Trigger/Mechanism (see table above).

---

## `data/` — Data guide

### `data/splits/` — Structured annotations

| File / directory | Description |
|------------------|-------------|
| `train_updated.jsonl` | Training set: raw text + multi-span annotations (~1997 examples) |
| `test_updated.jsonl` | Test set (~487 examples) |
| `split_summary.json` | Split metadata |
| `conll_bieo/` | Emotion Span in BIEO CoNLL; `train_bieo_10_1_{train,dev}.conll`, `test_bieo.conll` |
| `trigger_cls/` | Trigger labels (opposite / negation / irony), sentence-pair format; 9:1 train/dev |
| `mechanism_cls/` | Mechanism labels (contingent / frustration / satiation) |
| `shift_cls/` | Emotion_shift path labels (e.g. `happiness-->sadness`) |
| `added_na50_entry_ids.txt` | IDs of NA-only samples merged from legacy LLaMA-Factory data |

Each `*_cls/*.jsonl` file typically has `text_a` (context), `text_b` (emotion span), and `label`. See `*_label_map.json` and `*_data_summary.json` in each folder.

### `data/ft/` — LLM instruction tuning

| File | Description |
|------|-------------|
| `instruction_cn.txt` | Main Chinese instruction (Trigger / Mechanism / Emotion_shift; no Emotion_label) |
| `instruction.txt` | English / legacy instruction backup |
| `train_emotion_0415.json` | LLaMA-Factory training set (converted + NA samples, ~2047 examples) |
| `test_emotion_0415.json` | LLaMA-Factory test set (~487 examples) |

Generated by `Emotion_reversal/process_emotion_reversal_for_llama_factory.py` (see that script for working directory; outputs go to `emotion_github/data/ft/`).

---

## LLM fine-tuning (emotion_0415)

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for LoRA SFT on **`train_emotion_0415.json`**, predicting Emotion_span / Trigger / Mechanism / Emotion_shift end-to-end (prompt: `data/ft/instruction_cn.txt`).  
Configs and launch scripts live under **[`LLM/`](LLM/)** (curated from the main LLaMA-Factory tree). See [`LLM/README.md`](LLM/README.md) for file-level mapping.

### Environment

- Conda env with **LLaMA-Factory** installed (e.g. `llama-fact`)
- Default LLaMA-Factory root on our machine: `/mnt/SSD_2T/tangxuemei/LLaMA-Factory`
- GPU; set `CUDA_VISIBLE_DEVICES` in each `run_*.sh`

### Train → merge → infer

| Step | Command | Main outputs (under `LLaMA-Factory/`) |
|------|---------|----------------------------------------|
| LoRA training | `bash emotion_github/LLM/scripts/run_train_qwen2_0415.sh` | `saves/emotion_0415/qwen2-14b/lora/sft/` |
| | `bash .../run_train_glm4_0415.sh` | `saves/emotion_0415/glm4-9b/lora/sft/` |
| | `bash .../run_train_llama3_0415.sh` | `saves/emotion_0415/llama3-8b/lora/sft/` |
| Merge LoRA | `bash .../run_merge_all_0415.sh qwen2` (or `glm4` / `llama3` / `all`) | `output/{qwen2,glm4,llama3}_lora/emotion_0415/` |
| Test inference | `bash .../run_infer_emotion_0415.sh qwen2 0` | `test_responses.json` in the same output dir |

Update `adapter_name_or_path` in `LLM/configs/merge/*_0415.yaml` to match your actual checkpoint before merging.

### Evaluation

After inference, evaluate with the same span-F1 script as the BERT pipeline:

```bash
python evaluate_qwen_test_responses_span_f1.py \
  --input_path /mnt/SSD_2T/tangxuemei/LLaMA-Factory/output/qwen2_lora/emotion_0415/test_responses.json \
  --model_name qwen2_lora_emotion_0415 \
  --results_dir ../../results
```

(A copy of the same evaluator may exist at the `Emotion_reversal/` repo root with identical arguments.)

### `LLM/` layout

```text
LLM/
├── configs/train/     # LoRA training yaml for qwen2 / glm4 / llama3
├── configs/merge/     # llamafactory-cli export configs
├── scripts/           # run_train_*, run_merge_*, run_infer_*, setup_llamafactory_data.sh
└── inference/         # infer_emotion_0415.py
```

---

## Recommended experiment order

```bash
cd Emotion_reversal/emotion_github/code

# 1) Span (optional: generate CoNLL from repo root first)
# python ../../convert_jsonl_to_bieo_conll.py ...

# 2) Sequence labeling
bash run_bert_crf.sh
bash run_roberta_crf.sh

# 3) Three classification heads (BERT)
bash run_reversal_cls_bert.sh              # Trigger
bash run_reversal_type_cls_bert.sh         # Mechanism
bash run_emotion_shift_cls_bert.sh         # Emotion_shift

# 4) Three classification heads (RoBERTa)
bash run_reversal_cls_roberta_train.sh
bash run_reversal_type_cls_roberta_train.sh
# Emotion_shift RoBERTa: add a script; use hfl/chinese-roberta-wwm-ext

# 5) Span-F1 evaluation aligned with LLM protocol
bash run_bert_span_eval_bridge.sh
bash run_reversal_cls_roberta_test.sh
bash run_reversal_type_cls_roberta_test.sh
bash run_emotion_shift_cls_full_pipeline.sh
```

**LLM end-to-end (independent of the table above; can run in parallel):**

```bash
# Prepare data and LLaMA-Factory symlinks (see “LLM fine-tuning” above)
bash /mnt/SSD_2T/tangxuemei/emotion_github/LLM/scripts/setup_llamafactory_data.sh

bash emotion_github/LLM/scripts/run_train_qwen2_0415.sh
bash emotion_github/LLM/scripts/run_merge_all_0415.sh qwen2
bash emotion_github/LLM/scripts/run_infer_emotion_0415.sh qwen2 0
# Then run code/evaluate_qwen_test_responses_span_f1.py on test_responses.json
```

---

## Dependencies

- Python 3.8+
- **BERT / RoBERTa**: `torch`, `transformers`, `seqeval` (CRF training)
- **LLM**: LLaMA-Factory, `llamafactory-cli`
- GPU recommended; adjust `CUDA_VISIBLE_DEVICES` in each `run_*.sh`
