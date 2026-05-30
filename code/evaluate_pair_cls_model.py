#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class PairClsDataset(Dataset):
    def __init__(self, rows: List[dict], label2id: Dict[str, int]):
        self.samples = []
        for r in rows:
            lab = r.get("label", "na")
            if lab not in label2id:
                continue
            self.samples.append(
                {"text_a": r["text_a"], "text_b": r["text_b"], "label_id": label2id[lab], "label": lab}
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_collate_fn(tokenizer, max_len: int):
    def collate(batch):
        a = [x["text_a"] for x in batch]
        b = [x["text_b"] for x in batch]
        y = torch.tensor([x["label_id"] for x in batch], dtype=torch.long)
        enc = tokenizer(a, b, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc["labels"] = y
        return enc

    return collate


def compute_metrics(preds: List[int], golds: List[int]):
    n = len(golds)
    acc = sum(int(p == g) for p, g in zip(preds, golds)) / n if n else 0.0
    labels = sorted(set(preds) | set(golds))
    f1s = []
    for lab in labels:
        tp = sum(1 for p, g in zip(preds, golds) if p == lab and g == lab)
        fp = sum(1 for p, g in zip(preds, golds) if p == lab and g != lab)
        fn = sum(1 for p, g in zip(preds, golds) if p != lab and g == lab)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {"accuracy": acc, "macro_f1": macro_f1}


def main():
    parser = argparse.ArgumentParser("Evaluate text-pair classifier on test set")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--label-map-path", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-path", type=Path, required=True)
    args = parser.parse_args()

    label_map = json.loads(args.label_map_path.read_text(encoding="utf-8"))
    label2id = label_map["label2id"]
    rows = read_jsonl(args.test_path)
    ds = PairClsDataset(rows, label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=build_collate_fn(tokenizer, args.max_len))

    preds, golds = [], []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()
            n_batches += 1
            pred = torch.argmax(out.logits, dim=-1)
            preds.extend(pred.cpu().tolist())
            golds.extend(batch["labels"].cpu().tolist())

    metrics = compute_metrics(preds, golds)
    metrics["loss"] = total_loss / max(n_batches, 1)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
