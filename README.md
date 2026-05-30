# Emotion Shift

[English](README_EN.md) · **中文**

中文情绪转变（Emotion shift）任务的**数据**、**BERT / RoBERTa** 与 **LLM（LLaMA-Factory）** 实验资源汇总目录。  
LLM 相关配置见 [`LLM/README.md`](LLM/README.md)。

---

## 目录结构

```text
emotion_github/
├── README.md          # 中文说明
├── README_EN.md       # English documentation
├── LLM/               # LLaMA-Factory emotion_0415 训练 / 合并 / 推理（见 LLM/README.md）
├── code/              # BERT/RoBERTa 训练、导出、评测脚本与 run_*.sh
└── data/
    ├── splits/        # 人工标注拆分与分类/序列标注数据
    └── ft/            # LLaMA-Factory 指令微调 JSON
```


---

## 运行方式

所有 `code/run_*.sh` 均在 **`emotion_github/code/`** 下执行（脚本已 `cd` 到自身目录）：

```bash
cd emotion_github/code
conda activate llama-fact   # 或你的 PyTorch + transformers 环境
bash run_bert_crf.sh
```

路径约定（相对 `code/`）：

| 用途 | 路径 |
|------|------|
| 数据 | `../data/splits/...` |
| 模型输出 | `../../output/<实验名>/` |
---

## `code/` — 脚本清单

### 核心 Python

| 文件 | 作用 |
|------|------|
| `train_test_bert_crf.py` | **Emotion Span**：BERT/RoBERTa + CRF，BIEO CoNLL 序列标注 |
| `train_test_emotion_shift_cls.py` | **Trigger / Mechanism / Emotion_shift**：句对分类（BERT 或 RoBERTa） |
| `export_pair_cls_preds_for_qwen_eval.py` | 将 Trigger/Mechanism 分类结果导出为统一 JSONL，供 span-F1 评测 |
| `export_emotion_shift_preds_for_qwen_eval.py` | 将 Emotion_shift 分类结果导出为评测 JSONL |
| `evaluate_qwen_test_responses_span_f1.py` | 与 LLM 相同的 span 匹配 + 字段 F1 评测 |
| `evaluate_pair_cls_model.py` | 单独在分类测试集上算准确率（不经过 span 对齐） |

### 任务 ×  backbone 覆盖情况

标注规范中字段对应关系：

| 标注字段 | 任务目录（`data/splits/`） | 模型脚本 |
|----------|---------------------------|----------|
| **Emotion_span** | `conll_bieo/` | `train_test_bert_crf.py` |
| **Trigger**（原 Reversal） | `trigger_cls/` | `train_test_emotion_shift_cls.py` |
| **Mechanism**（原 Reversal_type） | `mechanism_cls/` | `train_test_emotion_shift_cls.py` |
| **Emotion_shift** | `shift_cls/` | `train_test_emotion_shift_cls.py` |

| 任务 | BERT 训练 | RoBERTa 训练 | 导出 + span-F1 评测 |
|------|-----------|--------------|---------------------|
| Emotion Span (CRF) | `run_bert_crf.sh` | `run_roberta_crf.sh` | `run_bert_span_eval_bridge.sh` |
| Trigger 分类 | `run_reversal_cls_bert.sh` | `run_reversal_cls_roberta_train.sh` | `run_reversal_cls_roberta_test.sh`（**仅用 BERT 权重**） |
| Mechanism 分类 | `run_reversal_type_cls_bert.sh` | `run_reversal_type_cls_roberta_train.sh` | `run_reversal_type_cls_roberta_test.sh`（**仅用 BERT 权重**） |
| Emotion_shift 分类 | `run_emotion_shift_cls_bert.sh`（**文件名误导，实为 BERT**） | **缺少独立 `run_*_roberta.sh`** | `run_emotion_shift_cls_full_pipeline.sh`（BERT） |

**结论（完整性检查）：**

- **训练脚本**：四类任务的 BERT + RoBERTa **训练入口均已具备**（共 2 个 `train_test_*.py`，通过 `--model-name` 切换）。
- **仍不完整 / 需注意：**
  1. **RoBERTa 的 span-F1 桥接评测**：仅有 BERT-CRF 的 `run_bert_span_eval_bridge.sh`；RoBERTa-CRF 需仿照该脚本改 `../../output/roberta_crf/` 路径后自行运行。
  2. **Trigger / Mechanism 的 `run_*_roberta_test.sh`**：名称含 roberta，实际加载的是 `output/reversal_cls_bert` 与 `output/reversal_type_cls_bert`；RoBERTa 模型需复制脚本并改为 `*_roberta/best_model`。
  3. **Emotion_shift**：`run_emotion_shift_cls_roberta.sh` 使用 `bert-base-chinese`；RoBERTa 版需另建脚本并指定 `hfl/chinese-roberta-wwm-ext` 与 `output/emotion_shift_cls_roberta`（仓库中已有该输出目录，说明曾手动训练过）。
  4. **数据准备**：`run_prepare_emotion_shift_cls.sh` 仅生成 shift 数据；Trigger/Mechanism 数据需在上层运行 `prepare_reversal_field_cls_data.py`（见下表）。


---

## `data/` — 数据说明

### `data/splits/` — 结构化标注

| 文件 / 目录 | 说明 |
|-------------|------|
| `train_updated.jsonl` | 训练集原文 + 多 span 标注（约 1997 条） |
| `test_updated.jsonl` | 测试集（约 487 条） |
| `split_summary.json` | 划分元信息 |
| `conll_bieo/` | Emotion Span 的 BIEO CoNLL；`train_bieo_10_1_{train,dev}.conll`、`test_bieo.conll` |
| `trigger_cls/` | Trigger（opposite / negation / irony）句对分类；含 9:1 train/dev |
| `mechanism_cls/` | Mechanism（contingent / frustration / satiation）句对分类 |
| `shift_cls/` | Emotion_shift 路径分类（如 `happiness-->sadness`） |
| `added_na50_entry_ids.txt` | 从旧 LLaMA-Factory 数据并入的 NA 样本 id |

每条 `*_cls/*.jsonl` 一般为：`text_a`（上下文）、`text_b`（情绪 span）、`label`；详见各目录下 `*_label_map.json`、`*_data_summary.json`。

### `data/ft/` — LLM 指令微调

| 文件 | 说明 |
|------|------|
| `instruction_cn.txt` | 当前主指令（Trigger / Mechanism / Emotion_shift，无 Emotion_label） |
| `instruction.txt` | 英文/旧版指令备份 |
| `train_emotion_0415.json` | LLaMA-Factory 训练集（转换集 + NA 样本，约 2047 条） |
| `test_emotion_0415.json` | LLaMA-Factory 测试集（约 487 条） |

由 `Emotion_reversal/process_emotion_reversal_for_llama_factory.py` 生成（工作目录见该脚本说明；数据落在 `emotion_github/data/ft/`）。

---

## LLM 微调（emotion_0415）

基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 对 **`train_emotion_0415.json`** 做 LoRA SFT，端到端预测 Emotion_span / Trigger / Mechanism / Emotion_shift（指令见 `data/ft/instruction_cn.txt`）。  
配置与脚本集中在 **[`LLM/`](LLM/)**（从 `LLaMA-Factory` 筛选整理），详细对照表见 [`LLM/README.md`](LLM/README.md)。

### 依赖与环境

- 已安装 **LLaMA-Factory** 的 conda 环境（如 `llama-fact`）
- 本机 LLaMA-Factory 路径默认：`/mnt/SSD_2T/tangxuemei/LLaMA-Factory`
- GPU；训练脚本内可改 `CUDA_VISIBLE_DEVICES`

### 训练 → 合并 → 推理

| 步骤 | 命令 | 主要输出（均在 `LLaMA-Factory/` 下） |
|------|------|--------------------------------------|
| LoRA 训练 | `bash emotion_github/LLM/scripts/run_train_qwen2_0415.sh` | `saves/emotion_0415/qwen2-14b/lora/sft/` |
| | `bash .../run_train_glm4_0415.sh` | `saves/emotion_0415/glm4-9b/lora/sft/` |
| | `bash .../run_train_llama3_0415.sh` | `saves/emotion_0415/llama3-8b/lora/sft/` |
| 合并 LoRA | `bash .../run_merge_all_0415.sh qwen2`（或 `glm4` / `llama3` / `all`） | `output/{qwen2,glm4,llama3}_lora/emotion_0415/` |
| 测试集推理 | `bash .../run_infer_emotion_0415.sh qwen2 0` | 同上目录下的 `test_responses.json` |

合并前请根据实际 checkpoint 修改 `LLM/configs/merge/*_0415.yaml` 中的 `adapter_name_or_path`。

### 评测

合并模型推理得到 `test_responses.json` 后，用与 BERT 管线相同的 span-F1 脚本评测：

```bash
python evaluate_qwen_test_responses_span_f1.py \
  --input_path /mnt/SSD_2T/tangxuemei/LLaMA-Factory/output/qwen2_lora/emotion_0415/test_responses.json \
  --model_name qwen2_lora_emotion_0415 \
  --results_dir ../../results
```

（`Emotion_reversal/` 根目录下也有同名评测脚本，参数一致。）

### `LLM/` 目录速览

```text
LLM/
├── configs/train/     # qwen2 / glm4 / llama3 的 LoRA 训练 yaml
├── configs/merge/     # llamafactory-cli export 配置
├── scripts/           # run_train_*、run_merge_*、run_infer_*、setup_llamafactory_data.sh
└── inference/         # infer_emotion_0415.py
```

---

## 推荐实验顺序

```bash
cd Emotion_reversal/emotion_github/code

# 1) Span（可选：先在仓库根目录生成 CoNLL）
# python ../../convert_jsonl_to_bieo_conll.py ...

# 2) 序列标注
bash run_bert_crf.sh
bash run_roberta_crf.sh

# 3) 三类分类（BERT）
bash run_reversal_cls_bert.sh              # Trigger
bash run_reversal_type_cls_bert.sh         # Mechanism
bash run_emotion_shift_cls_bert.sh         # Emotion_shift

# 4) 三类分类（RoBERTa）
bash run_reversal_cls_roberta_train.sh
bash run_reversal_type_cls_roberta_train.sh
# Emotion_shift RoBERTa：需自建脚本，参考上表将 model-name 改为 hfl/chinese-roberta-wwm-ext

# 5) 与 LLM 对齐的 span-F1 评测
bash run_bert_span_eval_bridge.sh
bash run_reversal_cls_roberta_test.sh
bash run_reversal_type_cls_roberta_test.sh
bash run_emotion_shift_cls_full_pipeline.sh
```

**LLM 端到端（与上表独立，可并行实验）：**

```bash
# 准备数据与 LLaMA-Factory 链接（见上文「LLM 微调」）
bash /mnt/SSD_2T/tangxuemei/emotion_github/LLM/scripts/setup_llamafactory_data.sh

bash emotion_github/LLM/scripts/run_train_qwen2_0415.sh
bash emotion_github/LLM/scripts/run_merge_all_0415.sh qwen2
bash emotion_github/LLM/scripts/run_infer_emotion_0415.sh qwen2 0
# 再用 code/evaluate_qwen_test_responses_span_f1.py 评测 test_responses.json
```

---

## 依赖

- Python 3.8+
- **BERT / RoBERTa**：`torch`, `transformers`, `seqeval`（CRF 训练）
- **LLM**：LLaMA-Factory、`llamafactory-cli`
- GPU 推荐；各 `run_*.sh` 内可自行修改 `CUDA_VISIBLE_DEVICES`


