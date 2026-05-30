#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class PairClsDataset(Dataset):
    def __init__(self, rows: List[dict], label2id: Dict[str, int], split_name: str):
        self.samples = []
        skipped = 0
        for r in rows:
            label = r.get("label")
            if label not in label2id:
                skipped += 1
                continue
            self.samples.append(
                {
                    "text_a": r["text_a"],
                    "text_b": r["text_b"],
                    "label_id": label2id[label],
                    "label": label,
                }
            )
        if skipped > 0:
            print(f"[{split_name}] skipped {skipped} rows with unknown labels.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_collate_fn(tokenizer, max_len: int):
    def collate_fn(batch):
        text_a = [x["text_a"] for x in batch]
        text_b = [x["text_b"] for x in batch]
        labels = torch.tensor([x["label_id"] for x in batch], dtype=torch.long)
        enc = tokenizer(
            text_a,
            text_b,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc

    return collate_fn


def compute_metrics(preds: List[int], golds: List[int]) -> Dict[str, float]:
    assert len(preds) == len(golds)
    n = len(golds)
    if n == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    acc = sum(int(p == g) for p, g in zip(preds, golds)) / n

    label_set = sorted(set(golds) | set(preds))
    f1s = []
    for lab in label_set:
        tp = sum(1 for p, g in zip(preds, golds) if p == lab and g == lab)
        fp = sum(1 for p, g in zip(preds, golds) if p == lab and g != lab)
        fn = sum(1 for p, g in zip(preds, golds) if p != lab and g == lab)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {"accuracy": acc, "macro_f1": macro_f1}


def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    preds, golds = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()
            n_batches += 1
            logits = out.logits
            pred = torch.argmax(logits, dim=-1)
            preds.extend(pred.cpu().tolist())
            golds.extend(batch["labels"].cpu().tolist())

    metrics = compute_metrics(preds, golds)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser("Train/Test emotion-shift classification with BERT/RoBERTa + Softmax")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--dev-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--label-map-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.label_map_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()} if isinstance(next(iter(label_map["id2label"].keys())), str) else label_map["id2label"]

    train_rows = read_jsonl(args.train_path)
    dev_rows = read_jsonl(args.dev_path)
    test_rows = read_jsonl(args.test_path)
    train_ds = PairClsDataset(train_rows, label2id, "train")
    dev_ds = PairClsDataset(dev_rows, label2id, "dev")
    test_ds = PairClsDataset(test_rows, label2id, "test")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    collate_fn = build_collate_fn(tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label={int(k): v for k, v in id2label.items()},
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = []
    best_dev_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / max(len(train_loader), 1)
        dev_metrics = evaluate(model, dev_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "dev": dev_metrics}
        history.append(row)
        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} "
            f"dev_loss={dev_metrics['loss']:.4f} "
            f"dev_acc={dev_metrics['accuracy']:.4f} "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f}"
        )

        if dev_metrics["accuracy"] > best_dev_acc:
            best_dev_acc = dev_metrics["accuracy"]
            model.save_pretrained(args.output_dir / "best_model")
            tokenizer.save_pretrained(args.output_dir / "best_model")

    # Evaluate best dev model on test set.
    best_model = AutoModelForSequenceClassification.from_pretrained(args.output_dir / "best_model").to(device)
    best_test_metrics = evaluate(best_model, test_loader, device)
    print(
        f"[Best@Dev] test_loss={best_test_metrics['loss']:.4f} "
        f"test_acc={best_test_metrics['accuracy']:.4f} "
        f"test_macro_f1={best_test_metrics['macro_f1']:.4f}"
    )

    model.save_pretrained(args.output_dir / "last_model")
    tokenizer.save_pretrained(args.output_dir / "last_model")

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_dev_accuracy": best_dev_acc,
                "best_model_test_metrics": best_test_metrics,
                "history": history,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
