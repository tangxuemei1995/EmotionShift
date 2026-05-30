#!/usr/bin/env python3
import argparse
import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from torchcrf import CRF
except ImportError as exc:
    try:
        from TorchCRF import CRF
    except ImportError:
        raise ImportError(
            "Please install a CRF package first: pip install torchcrf "
            "(or pip install TorchCRF)"
        ) from exc


LABELS = ["O", "B-SPAN", "I-SPAN", "E-SPAN", "S-SPAN"]
LABEL2ID = {x: i for i, x in enumerate(LABELS)}
ID2LABEL = {i: x for x, i in LABEL2ID.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_conll(path: Path) -> List[Tuple[List[str], List[str]]]:
    samples = []
    chars, tags = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if chars:
                    samples.append((chars, tags))
                    chars, tags = [], []
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            ch, tag = parts[0], parts[1]
            if tag not in LABEL2ID:
                raise ValueError(f"Unknown tag {tag} in {path}")
            chars.append(ch)
            tags.append(tag)
    if chars:
        samples.append((chars, tags))
    return samples


class ConllDataset(Dataset):
    def __init__(self, samples: List[Tuple[List[str], List[str]]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    raw_chars: List[List[str]]
    raw_tags: List[List[str]]


def build_collate_fn(tokenizer, max_len: int):
    def collate_fn(batch):
        chars_batch = [x[0] for x in batch]
        tags_batch = [x[1] for x in batch]

        enc = tokenizer(
            chars_batch,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        )

        labels = torch.full_like(enc["input_ids"], fill_value=-100)
        for i, tags in enumerate(tags_batch):
            if not hasattr(enc, "word_ids"):
                raise RuntimeError(
                    "Current tokenizer does not support word_ids(). "
                    "Please use a fast tokenizer (use_fast=True)."
                )
            word_ids = enc.word_ids(batch_index=i)
            prev = None
            for j, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if wid != prev:
                    if wid < len(tags):
                        labels[i, j] = LABEL2ID[tags[wid]]
                prev = wid
        return Batch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
            raw_chars=chars_batch,
            raw_tags=tags_batch,
        )

    return collate_fn


class BertCrf(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        # torchcrf supports batch_first; TorchCRF does not.
        self.crf_mode = "torchcrf"
        try:
            self.crf = CRF(num_labels, batch_first=True)
        except TypeError:
            self.crf = CRF(num_labels)
            self.crf_mode = "TorchCRF"

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))

        valid_mask = attention_mask.bool()
        if labels is not None:
            # CRF does not support -100 labels, so mask invalid positions out.
            crf_labels = labels.clone()
            crf_mask = valid_mask & (labels != -100)
            crf_labels[labels == -100] = 0
            if self.crf_mode == "torchcrf":
                llh = self.crf(emissions, crf_labels, mask=crf_mask, reduction="mean")
            else:
                llh = self.crf(emissions, crf_labels, crf_mask)
                if isinstance(llh, torch.Tensor) and llh.dim() > 0:
                    llh = llh.mean()
            loss = -llh
            return loss, emissions, crf_mask
        return emissions, valid_mask

    def decode(self, emissions, mask):
        if hasattr(self.crf, "decode"):
            return self.crf.decode(emissions, mask=mask)
        if hasattr(self.crf, "viterbi_decode"):
            return self.crf.viterbi_decode(emissions, mask)
        raise RuntimeError("CRF backend does not provide decode/viterbi_decode.")


def extract_spans(tags: List[str]) -> List[Tuple[int, int]]:
    """
    Tolerant BIOES span extraction.
    It accepts mildly malformed sequences (common in early training), e.g.:
    - B I I O  -> span(B..last I)
    - I I E    -> span(first I..E)
    """
    spans = []
    i = 0
    n = len(tags)
    while i < n:
        t = tags[i]
        if t == "B-SPAN":
            start = i
            i += 1
            while i < n and tags[i] == "I-SPAN":
                i += 1
            if i < n and tags[i] == "E-SPAN":
                spans.append((start, i))
                i += 1
            else:
                # Recover incomplete B-I* span.
                end = i - 1
                spans.append((start, end))
        elif t == "I-SPAN":
            # Recover orphan I-SPAN as a span start.
            start = i
            i += 1
            while i < n and tags[i] == "I-SPAN":
                i += 1
            if i < n and tags[i] == "E-SPAN":
                spans.append((start, i))
                i += 1
            else:
                end = i - 1
                spans.append((start, end))
        elif t in ("S-SPAN", "E-SPAN"):
            spans.append((i, i))
            i += 1
        else:
            i += 1
    return spans


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    gold_total, pred_total, hit_total = 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)

            loss, emissions, crf_mask = model(input_ids, attention_mask, labels=labels)
            total_loss += loss.item()
            n_batches += 1

            pred_ids = model.decode(emissions, crf_mask)
            for b in range(input_ids.size(0)):
                valid_positions = (batch.labels[b] != -100).nonzero(as_tuple=False).view(-1).tolist()
                gold_tags = [ID2LABEL[batch.labels[b, pos].item()] for pos in valid_positions]
                pred_tags = [ID2LABEL[x] for x in pred_ids[b][: len(valid_positions)]]

                gold_spans = set(extract_spans(gold_tags))
                pred_spans = set(extract_spans(pred_tags))
                gold_total += len(gold_spans)
                pred_total += len(pred_spans)
                hit_total += len(gold_spans & pred_spans)

    p = hit_total / pred_total if pred_total else 0.0
    r = hit_total / gold_total if gold_total else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "loss": total_loss / max(n_batches, 1),
        "precision": p,
        "recall": r,
        "f1": f1,
        "gold_spans": gold_total,
        "pred_spans": pred_total,
        "hit_spans": hit_total,
    }


def predict_and_save(model, loader, device, out_path: Path) -> None:
    model.eval()
    with out_path.open("w", encoding="utf-8") as f, torch.no_grad():
        for batch in loader:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            emissions, valid_mask = model(input_ids, attention_mask, labels=None)
            pred_ids = model.decode(emissions, valid_mask)

            for b in range(input_ids.size(0)):
                valid_positions = (batch.labels[b] != -100).nonzero(as_tuple=False).view(-1).tolist()
                chars = [batch.raw_chars[b][i] for i in range(len(valid_positions))]
                gold_tags = [batch.raw_tags[b][i] for i in range(len(valid_positions))]
                pred_tags = [ID2LABEL[x] for x in pred_ids[b][: len(valid_positions)]]

                for ch, g, p in zip(chars, gold_tags, pred_tags):
                    f.write(f"{ch}\t{g}\t{p}\n")
                f.write("\n")


def main():
    parser = argparse.ArgumentParser("Train/Test BERT+CRF for BIEO CoNLL")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--dev-path", type=Path, required=False, default=None)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="bert-base-chinese")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience on dev F1; <=0 disables early stopping.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = read_conll(args.train_path)
    dev_samples = read_conll(args.dev_path) if args.dev_path else None
    test_samples = read_conll(args.test_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    collate_fn = build_collate_fn(tokenizer, args.max_len)
    train_loader = DataLoader(
        ConllDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        ConllDataset(test_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    dev_loader = None
    if dev_samples is not None:
        dev_loader = DataLoader(
            ConllDataset(dev_samples),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertCrf(args.model_name, len(LABELS)).to(device)
    print(f"Using CRF backend: {model.crf_mode}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = -1.0
    best_ckpt = args.output_dir / "best_model.pt"
    history = []
    no_improve = 0
    seen_positive_f1 = False
    serializable_args = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)

            optimizer.zero_grad()
            loss, _, _ = model(input_ids, attention_mask, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()

        train_loss = running / max(len(train_loader), 1)
        eval_loader = dev_loader if dev_loader is not None else test_loader
        eval_name = "dev" if dev_loader is not None else "test"
        metrics = evaluate(model, eval_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        metrics["eval_set"] = eval_name
        history.append(metrics)

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} "
            f"{eval_name}_loss={metrics['loss']:.4f} P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            no_improve = 0
            if metrics["f1"] > 0:
                seen_positive_f1 = True
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model_name,
                    "labels": LABELS,
                    "args": serializable_args,
                    "best_metrics": metrics,
                },
                best_ckpt,
            )
        else:
            # Do not early-stop while F1 is still always zero.
            if seen_positive_f1:
                no_improve += 1
            else:
                no_improve = 0

            if args.patience > 0 and seen_positive_f1 and no_improve >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}: "
                    f"{eval_name} F1 not improved for {args.patience} epochs."
                )
                break

    print(f"Best {('dev' if dev_loader is not None else 'test')} F1: {best_f1:.4f}, checkpoint: {best_ckpt}")

    # Load best checkpoint and evaluate on final test set.
    state = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    final_metrics = evaluate(model, test_loader, device)
    predict_and_save(model, test_loader, device, args.output_dir / "test_predictions.conll")

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_f1": best_f1,
                "final_test_metrics": final_metrics,
                "history": history,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (args.output_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, ensure_ascii=False, indent=2)

    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
