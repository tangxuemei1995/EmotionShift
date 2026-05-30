# LLM 微调（emotion_0415）

从 [`LLaMA-Factory`](../../LLaMA-Factory) 中筛选的、用于 **`train_emotion_0415`** 数据集的 LoRA 训练、合并与推理脚本。  
训练数据本体在 [`../data/ft/train_emotion_0415.json`](../data/ft/train_emotion_0415.json)，测试集为 [`../data/ft/test_emotion_0415.json`](../data/ft/test_emotion_0415.json)。

---

## 目录结构

```text
LLM/
├── README.md
├── dataset_info_train_emotion_0415.json   # 合并进 LLaMA-Factory/data/dataset_info.json
├── configs/
│   ├── train/                             # llamafactory-cli train
│   │   ├── qwen2_lora_sft_emotion_0415.yaml    # Qwen2.5-14B-Instruct
│   │   ├── glm4_lora_emotion_0415.yaml          # GLM-4-9B
│   │   └── llama3_lora_sft_emotion_0415.yaml    # Llama-3.1-8B
│   └── merge/                             # llamafactory-cli export
│       ├── qwen2_lora_merge_0415.yaml
│       ├── glm4_lora_merge_0415.yaml
│       └── llama3_lora_merge_0415.yaml
├── scripts/
│   ├── setup_llamafactory_data.sh         # 软链数据到 LLaMA-Factory/data/
│   ├── env_hf_cache.sh
│   ├── run_train_{qwen2,glm4,llama3}_0415.sh
│   ├── run_merge_all_0415.sh
│   └── run_infer_emotion_0415.sh
└── inference/
    └── infer_emotion_0415.py              # 合并后模型在 test 上生成
```

---

## 与 LLaMA-Factory 的对应关系

| 本目录文件 | 原 LLaMA-Factory 路径 | 说明 |
|-----------|----------------------|------|
| `configs/train/qwen2_lora_sft_emotion_0415.yaml` | `examples/train_lora/qwen3_lora_sft_emotion.yaml` | `dataset: train_emotion_0415` |
| `configs/train/glm4_lora_emotion_0415.yaml` | `examples/train_lora/glm4_lora_emotion_0325.yaml` | 文件名含 0325，内容已是 0415 |
| `configs/train/llama3_lora_sft_emotion_0415.yaml` | `examples/train_lora/llama3_lora_sft_emotion_0325.yaml` | 同上 |
| `configs/merge/qwen2_lora_merge_0415.yaml` | `examples/merge_lora/qwen3_lora_sft.yaml` | checkpoint-415 → `emotion_0415` |
| `configs/merge/glm4_lora_merge_0415.yaml` | `examples/merge_lora/glm4_lora_sft.yaml` | checkpoint-1245 |
| `configs/merge/llama3_lora_merge_0415.yaml` | `examples/merge_lora/llama3_lora_sft.yaml` | checkpoint-1245 |
| `scripts/run_train_qwen2_0415.sh` | `run_qwen_emotion.sh` | |
| `scripts/run_train_glm4_0415.sh` | `run_glm.sh` | |
| `scripts/run_train_llama3_0415.sh` | `run_llama_emotion_0325.sh` | |
| `scripts/run_merge_all_0415.sh` | `model_merge.sh` | |
| `inference/infer_emotion_0415.py` | `vllm_inference.py` → `qwen_emotion()` | 仅保留 0415 推理逻辑 |

**未纳入本目录（非 emotion_0415 或日志）：**

- `data/train_emotion.json`、`train_emotion_0325.json` 及对应训练配置
- `run_llama_emotion_0325.sh` 以外的 ched/duie/mix 等任务脚本
- `qwen.txt` / `glm.txt` / `llama.txt`（训练日志，非可执行脚本）
- `wen_emotion.txt`、`glm_emotion_0325.txt` 等旧说明

---

## 使用前准备

### 1. 环境

```bash
conda activate llama-fact   # 或已安装 llamafactory 的环境
```

### 2. 注册数据集

将 [`dataset_info_train_emotion_0415.json`](dataset_info_train_emotion_0415.json) 中的 `train_emotion_0415` 条目合并进：

`/mnt/SSD_2T/tangxuemei/LLaMA-Factory/data/dataset_info.json`

### 3. 链接训练/测试 JSON

```bash
bash /mnt/SSD_2T/tangxuemei/emotion_github/LLM/scripts/setup_llamafactory_data.sh
```

会把 `emotion_github/data/ft/*.json` 软链到 `LLaMA-Factory/data/`。

### 4. 可选：HF 缓存

```bash
export CACHE_DIR=/mnt/external_8t/huggingface_cache   # 与原先 run_qwen_emotion.sh 一致
```

---

## 训练（LoRA）

在任意目录执行（脚本内会 `cd` 到 LLaMA-Factory）：

```bash
# Qwen2.5-14B
bash emotion_github/LLM/scripts/run_train_qwen2_0415.sh

# GLM-4-9B
bash emotion_github/LLM/scripts/run_train_glm4_0415.sh

# Llama-3.1-8B
bash emotion_github/LLM/scripts/run_train_llama3_0415.sh
```

默认输出目录（在 LLaMA-Factory 下）：

| 模型 | `output_dir` |
|------|----------------|
| Qwen2-14B | `saves/emotion_0415/qwen2-14b/lora/sft` |
| GLM-4-9B | `saves/emotion_0415/glm4-9b/lora/sft` |
| Llama-3-8B | `saves/emotion_0415/llama3-8b/lora/sft` |

合并前请根据实际 **last checkpoint** 修改 `configs/merge/*_0415.yaml` 中的 `adapter_name_or_path`。

---

## 合并 LoRA

```bash
bash emotion_github/LLM/scripts/run_merge_all_0415.sh qwen2
bash emotion_github/LLM/scripts/run_merge_all_0415.sh all
```

合并结果：

- `LLaMA-Factory/output/qwen2_lora/emotion_0415/`
- `LLaMA-Factory/output/glm4_lora/emotion_0415/`
- `LLaMA-Factory/output/llama3_lora/emotion_0415/`

---

## 测试集推理

需先完成 merge。生成 `test_responses.json`：

```bash
bash emotion_github/LLM/scripts/run_infer_emotion_0415.sh qwen2 0
bash emotion_github/LLM/scripts/run_infer_emotion_0415.sh glm4 2
bash emotion_github/LLM/scripts/run_infer_emotion_0415.sh llama3 0
```

输出路径示例：  
`LLaMA-Factory/output/qwen2_lora/emotion_0415/test_responses.json`

可用 [`../code/evaluate_qwen_test_responses_span_f1.py`](../code/evaluate_qwen_test_responses_span_f1.py) 或仓库根目录同名脚本做 span-F1 评测。

---

## 完整性检查（BERT / RoBERTa 见 [`../code/README.md`](../README.md)）

| 能力 | emotion_0415 LLM 本目录 |
|------|-------------------------|
| Qwen2 LoRA 训练 | 有 |
| GLM4 LoRA 训练 | 有 |
| Llama3 LoRA 训练 | 有 |
| 三模型 merge | 有 |
| 三模型 test 推理 | 有（`infer_emotion_0415.py`） |
| vLLM 批量推理 | 未单独拷贝（可用 LLaMA-Factory 原 `vllm_inference.py`） |

---

## 更新训练数据后

1. 运行 `Emotion_reversal/process_emotion_reversal_for_llama_factory.py` 重新生成 `data/ft/train_emotion_0415.json`。
2. 重新执行 `setup_llamafactory_data.sh`。
3. 按需重新训练 / 合并 / 推理。
