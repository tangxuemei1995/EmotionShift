#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_val(s: str) -> str:
    if s is None:
        return "NA"
    s = str(s).strip()
    if not s:
        return "NA"
    return s


def make_span_obj(span_text: str, target_field: str, value: str) -> Dict[str, str]:
    obj = {
        "Emotion_span": span_text,
        "Emotion_label": "NA",
        "Reversal": "NA",
        "Reversal_type": "NA",
        "Emotion_shift": "NA",
        "Marker": "NA",
    }
    val = norm_val(value)
    if target_field == "emotion_shift":
        obj["Emotion_shift"] = val.replace("->", "-->")
    elif target_field == "reversal":
        obj["Reversal"] = val
    elif target_field == "reversal_type":
        obj["Reversal_type"] = val
    else:
        raise ValueError(f"Unsupported target_field: {target_field}")
    return obj


def main():
    parser = argparse.ArgumentParser(description="Export pair-classification predictions to qwen-eval JSONL format.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-cls-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--target-field", type=str, required=True, choices=["emotion_shift", "reversal", "reversal_type"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=192)
    args = parser.parse_args()

    rows = read_jsonl(args.test_cls_jsonl)
    if not rows:
        raise ValueError("Empty test file")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    grouped = defaultdict(list)

    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]
            text_a = [x["text_a"] for x in batch]
            text_b = [x["text_b"] for x in batch]
            enc = tokenizer(
                text_a,
                text_b,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

            for r, pid in zip(batch, pred_ids):
                id2label = model.config.id2label
                pred_label = id2label[str(pid)] if isinstance(next(iter(id2label.keys())), str) else id2label[pid]
                entry_id = r["entry_id"]
                span_text = r["text_b"]
                gold_label = r["label"]
                grouped[entry_id].append(
                    {
                        "answer_item": make_span_obj(span_text, args.target_field, gold_label),
                        "response_item": make_span_obj(span_text, args.target_field, pred_label),
                    }
                )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for entry_id in sorted(grouped.keys()):
            answer_arr = [x["answer_item"] for x in grouped[entry_id]]
            response_arr = [x["response_item"] for x in grouped[entry_id]]
            f.write(
                json.dumps(
                    {
                        "entry_id": entry_id,
                        "answer": json.dumps(answer_arr, ensure_ascii=False),
                        "response": json.dumps(response_arr, ensure_ascii=False),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"records={len(grouped)} out={args.out_jsonl}")


if __name__ == "__main__":
    main()
