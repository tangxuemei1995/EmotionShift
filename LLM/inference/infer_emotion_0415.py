#!/usr/bin/env python3
"""
Run merged LoRA emotion_0415 models on test_emotion_0415.json.

Extracted from LLaMA-Factory/vllm_inference.py :: qwen_emotion (0415 only).

Example:
  python infer_emotion_0415.py --model qwen2 --gpu 0
  python infer_emotion_0415.py --model glm4 --gpu 2
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_LLAMA_FACTORY = Path("/mnt/SSD_2T/tangxuemei/LLaMA-Factory")
DEFAULT_TEST = Path("/mnt/SSD_2T/tangxuemei/emotion_github/data/ft/test_emotion_0415.json")
DATASET_TAG = "emotion_0415"

MODEL_PRESETS = {
    "qwen2": {
        "tokenizer": "Qwen/Qwen2.5-14B-Instruct",
        "export_subdir": "qwen2_lora",
    },
    "glm4": {
        "tokenizer": "THUDM/glm-4-9b-chat-hf",
        "export_subdir": "glm4_lora",
    },
    "llama3": {
        "tokenizer": "meta-llama/Llama-3.1-8B-Instruct",
        "export_subdir": "llama3_lora",
    },
    "qwen3": {
        "tokenizer": "Qwen/Qwen3-4B-Instruct-2507",
        "export_subdir": "qwen3_lora",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=sorted(MODEL_PRESETS.keys()),
        default="qwen2",
        help="Which merged model under output/<name>_lora/emotion_0415/",
    )
    p.add_argument("--llama-factory-root", type=Path, default=DEFAULT_LLAMA_FACTORY)
    p.add_argument("--test-path", type=Path, default=DEFAULT_TEST)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    preset = MODEL_PRESETS[args.model]
    model_dir = (
        args.llama_factory_root
        / "output"
        / preset["export_subdir"]
        / DATASET_TAG
    )
    out_path = model_dir / "test_responses.json"

    tokenizer = AutoTokenizer.from_pretrained(preset["tokenizer"])
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype="auto",
        device_map="auto",
    )

    with args.test_path.open(encoding="utf-8") as f:
        test_rows = json.load(f)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in test_rows:
            messages = [
                {
                    "role": "user",
                    "content": row["instruction"] + row["input"],
                }
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            content = tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")
            f.write(
                json.dumps(
                    {"answer": row["output"], "response": content},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
