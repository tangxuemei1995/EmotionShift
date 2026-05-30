#!/usr/bin/env python3
"""
Span-aware evaluation for LLM structured emotion-reversal outputs.

Goal:
  1) Treat Emotion_span as the matching key.
  2) First decide whether each predicted span is correctly extracted from gold.
     Allow small boundary shifts using fuzzy similarity.
  3) Only for matched spans, compare other fields (Emotion_label/Reversal/Reversal_type/Emotion_shift/Marker).

Metrics:
  - Span extraction micro-F1 (based on span matching counts across all records).
  - Conditional micro-F1 for other categorical fields on matched span pairs:
      Emotion_label (multi-label micro-F1 over 5 emotions)
      Reversal (micro-F1 over {opposite,negation,irony})
      Reversal_type (micro-F1 over {contingent,frustration,satiation})
      Emotion_shift (micro-F1 over exact normalized shift strings, excluding 'na')
      Marker_present (binary: marker != 'na')

Input:
  LLaMA-Factory/output/qwen3_lora/emotion_0325/test_responses.json
  which is JSONL: each line has { "answer": ..., "response": ... }.
  answer/response are strings containing JSON arrays of span dicts.
"""

from __future__ import annotations

import difflib
import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


VALID_EMOTIONS = {"happiness", "anger", "sadness", "fear", "surprise"}
VALID_REVERSAL = {"opposite", "negation", "irony"}
VALID_REVERSAL_TYPE = {"contingent", "frustration", "satiation"}

SPAN_PUNCT_RE = re.compile(r"[ \t\n\r]+")
TRAILING_PUNCT_RE = re.compile(r"[。！？!?,.;:：…\u2026]+$")


def _to_na_str(x: Any) -> str:
    if x is None:
        return "na"
    s = str(x).strip()
    if not s:
        return "na"
    low = s.lower()
    if low in {"na", "n/a", "null", "none"}:
        return "na"
    return s


def _normalize_label_set(label_field: Any, valid_set: Set[str]) -> Set[str]:
    s = _to_na_str(label_field).lower()
    if s == "na":
        return set()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return {p for p in parts if p in valid_set}


def _normalize_label_canonical(label_field: Any, valid_set: Set[str]) -> str:
    """Treat a (possibly composite) label as an atomic unit.
    Filters to valid labels, sorts alphabetically, joins with comma.
    Returns '' when no valid labels found (equivalent to 'na').
    e.g. 'happiness,anger' -> 'anger,happiness'
    """
    s = _to_na_str(label_field).lower()
    if s == "na":
        return ""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    valid_parts = sorted(p for p in parts if p in valid_set)
    return ",".join(valid_parts)


def _normalize_shift(shift_field: Any) -> str:
    """Normalize an emotion shift chain (possibly multi-hop: A-->B-->C-->...).
    All parts must be valid emotions; the full chain is returned as an atomic string.
    Returns 'na' if any part is invalid or the chain has fewer than 2 nodes.
    """
    s = _to_na_str(shift_field)
    if s == "na":
        return "na"
    s = str(s).strip().lower()
    if "-->" not in s:
        return "na"
    parts = [p.strip() for p in s.split("-->")]
    if len(parts) < 2:
        return "na"
    if any(p not in VALID_EMOTIONS for p in parts):
        return "na"
    return "-->".join(parts)


def _normalize_marker_present(marker_field: Any) -> bool:
    return _to_na_str(marker_field).lower() != "na"


def _normalize_span_text(span_text: Any) -> str:
    s = _to_na_str(span_text)
    if s == "na":
        return ""
    s = s.lower()
    s = SPAN_PUNCT_RE.sub("", s)  # remove whitespace
    s = TRAILING_PUNCT_RE.sub("", s)
    return s


def _span_similarity(gold: str, pred: str) -> float:
    """
    Similarity in [0,1].
    Uses normalized substring overlap or SequenceMatcher ratio.
    """
    g = _normalize_span_text(gold)
    p = _normalize_span_text(pred)
    if not g or not p:
        return 0.0

    if g == p:
        return 1.0

    # If one contains the other, use length overlap as score.
    if g in p or p in g:
        short = g if len(g) <= len(p) else p
        long = p if short is g else g
        return len(short) / max(1, len(long))

    # Otherwise use fuzzy ratio.
    return difflib.SequenceMatcher(None, g, p).ratio()


def _try_parse_json_array(s: Any) -> List[Dict[str, Any]]:
    if not isinstance(s, str) or not s:
        return []
    if "[" not in s:
        return []
    decoder = json.JSONDecoder()
    best: Optional[List[Any]] = None
    # Try to parse from each '['; take the longest successful list.
    for m in re.finditer(r"\[", s):
        start = m.start()
        sub = s[start:]
        try:
            obj, _end = decoder.raw_decode(sub)
        except Exception:
            continue
        if isinstance(obj, list) and len(obj) > 0:
            if best is None or len(obj) > len(best):
                best = obj
    if not best:
        return []
    return [x for x in best if isinstance(x, dict)]


def extract_spans_array(text: Any) -> List[Dict[str, Any]]:
    return _try_parse_json_array(text)


@dataclass
class PRFCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def greedy_span_matching(
    gold_spans: Sequence[Dict[str, Any]],
    pred_spans: Sequence[Dict[str, Any]],
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Returns:
      matches: list of (gold_idx, pred_idx) for matched spans
      unmatched_gold: indices not matched
      unmatched_pred: indices not matched
    """
    gold_used = [False] * len(gold_spans)
    pred_used = [False] * len(pred_spans)

    candidates: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gold_spans):
        g_text = g.get("Emotion_span", "")
        for pi, p in enumerate(pred_spans):
            p_text = p.get("Emotion_span", "")
            sim = _span_similarity(g_text, p_text)
            if sim >= threshold:
                candidates.append((sim, gi, pi))
    candidates.sort(reverse=True, key=lambda x: x[0])

    matches: List[Tuple[int, int]] = []
    for _sim, gi, pi in candidates:
        if gold_used[gi] or pred_used[pi]:
            continue
        gold_used[gi] = True
        pred_used[pi] = True
        matches.append((gi, pi))

    matched_gold = {gi for gi, _ in matches}
    matched_pred = {pi for _, pi in matches}
    unmatched_gold = [i for i in range(len(gold_spans)) if i not in matched_gold]
    unmatched_pred = [i for i in range(len(pred_spans)) if i not in matched_pred]

    return matches, unmatched_gold, unmatched_pred


def micro_f1_from_sets(gold_set: Set[str], pred_set: Set[str]) -> PRFCounts:
    counts = PRFCounts()
    counts.tp = len(gold_set & pred_set)
    counts.fp = len(pred_set - gold_set)
    counts.fn = len(gold_set - pred_set)
    return counts


def _as_metric_dict(c: PRFCounts) -> Dict[str, float]:
    return {
        "tp": c.tp,
        "fp": c.fp,
        "fn": c.fn,
        "precision": c.precision(),
        "recall": c.recall(),
        "f1": c.f1(),
    }


def _macro_from_per_class(per_class_counts: Dict[str, PRFCounts]) -> Dict[str, float]:
    if not per_class_counts:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    ps = []
    rs = []
    f1s = []
    for c in per_class_counts.values():
        ps.append(c.precision())
        rs.append(c.recall())
        f1s.append(c.f1())
    n = len(f1s)
    return {
        "precision": sum(ps) / n,
        "recall": sum(rs) / n,
        "f1": sum(f1s) / n,
    }


def _update_binary_label_counts(gold_set: Set[str], pred_set: Set[str], label: str, bucket: Dict[str, PRFCounts]) -> None:
    c = bucket[label]
    gold_pos = label in gold_set
    pred_pos = label in pred_set
    if gold_pos and pred_pos:
        c.tp += 1
    elif (not gold_pos) and pred_pos:
        c.fp += 1
    elif gold_pos and (not pred_pos):
        c.fn += 1


def _infer_model_name(input_path: Path) -> str:
    """
    Try to infer model name from paths like:
    .../LLaMA-Factory/output/<model_name>/<run_name>/test_responses.json
    Fallback to parent folder name.
    """
    parts = list(input_path.parts)
    if "output" in parts:
        idx = parts.index("output")
        if idx + 1 < len(parts):
            candidate = parts[idx + 1].strip()
            if candidate:
                return candidate
    parent = input_path.parent.name.strip()
    return parent if parent else "model"


def main() -> None:
    parser = argparse.ArgumentParser(description="Span-aware evaluation (F1) with span-boundary fuzzy matching.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to test_responses.json (JSONL).",
    )
    parser.add_argument("--span_match_threshold", type=float, default=0.75, help="Similarity threshold for Emotion_span match.")
    parser.add_argument("--max_records", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--model_name", type=str, default="", help="Model name used for output json filename.")
    parser.add_argument(
        "--save_case_study",
        action="store_true",
        help="Save case study into two JSONL files: missed spans and label errors under matched spans.",
    )
    parser.add_argument(
        "--case_study_path",
        type=str,
        default="",
        help=(
            "Optional base JSONL path for case study (only used with --save_case_study). "
            "If it ends with '.jsonl', the script will write two files: <base>_span_missed.jsonl and <base>_label_errors.jsonl. "
            "Otherwise it will be treated as an output directory."
        ),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="Emotion_reversal/results",
        help="Directory to save metrics json.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    model_name = args.model_name.strip() or _infer_model_name(input_path)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{model_name}.json"

    if args.save_case_study:
        if args.case_study_path.strip():
            raw_path = Path(args.case_study_path).expanduser()
            if str(raw_path).endswith(".jsonl"):
                base = raw_path.with_suffix("")
                case_missed_path = base.with_name(base.name + "_span_missed.jsonl")
                case_label_errors_path = base.with_name(base.name + "_label_errors.jsonl")
            else:
                Path(raw_path).mkdir(parents=True, exist_ok=True)
                case_missed_path = Path(raw_path) / f"{model_name}_span_missed.jsonl"
                case_label_errors_path = Path(raw_path) / f"{model_name}_label_errors.jsonl"
        else:
            case_missed_path = results_dir / f"{model_name}_span_missed.jsonl"
            case_label_errors_path = results_dir / f"{model_name}_label_errors.jsonl"

        case_missed_fw = case_missed_path.open("w", encoding="utf-8")
        case_label_errors_fw = case_label_errors_path.open("w", encoding="utf-8")

    total_gold_spans = 0
    total_pred_spans = 0
    total_matched_spans = 0

    # Conditional metrics on matched pairs only.
    counts_emotion = PRFCounts()
    counts_reversal = PRFCounts()
    counts_rev_type = PRFCounts()
    counts_shift = PRFCounts()
    counts_marker_present = PRFCounts()
    matched_pairs = 0

    # Per-class counts on matched spans.
    # All label fields use composite-as-atomic: keys are canonical composite strings.
    per_class_emotion: Dict[str, PRFCounts] = defaultdict(PRFCounts)
    per_class_reversal: Dict[str, PRFCounts] = defaultdict(PRFCounts)
    per_class_rev_type: Dict[str, PRFCounts] = defaultdict(PRFCounts)
    per_class_shift: Dict[str, PRFCounts] = defaultdict(PRFCounts)

    with input_path.open("r", encoding="utf-8") as f:
        for rec_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if args.max_records and rec_idx >= args.max_records:
                break

            gold_text = rec.get("answer", "")
            pred_text = rec.get("response", "")
            gold_spans = extract_spans_array(gold_text)
            pred_spans = extract_spans_array(pred_text)

            total_gold_spans += len(gold_spans)
            total_pred_spans += len(pred_spans)

            matches, _unmatched_gold, _unmatched_pred = greedy_span_matching(
                gold_spans, pred_spans, threshold=args.span_match_threshold
            )
            total_matched_spans += len(matches)

            # Missed spans: gold 有，但在 prediction 中找不到匹配的那部分（用于 case study）。
            unmatched_gold_spans_for_case: List[Dict[str, Any]] = []
            matched_span_label_errors_for_case: List[Dict[str, Any]] = []
            if args.save_case_study and _unmatched_gold:
                for gi in _unmatched_gold:
                    g = gold_spans[gi]
                    g_text_raw = g.get("Emotion_span", "")
                    g_text_norm = _normalize_span_text(g_text_raw)

                    # For context, record the best similarity to any predicted span.
                    best_sim = -1.0
                    best_pi: Optional[int] = None
                    for pi, p in enumerate(pred_spans):
                        p_text_raw = p.get("Emotion_span", "")
                        sim = _span_similarity(g_text_raw, p_text_raw)
                        if sim > best_sim:
                            best_sim = sim
                            best_pi = pi

                    unmatched_gold_spans_for_case.append(
                        {
                            "gold_span_index": gi,
                            "gold_emotion_span_raw": g_text_raw,
                            "gold_emotion_span_norm": g_text_norm,
                            "best_pred_span_index": best_pi,
                            "best_pred_span_similarity": best_sim if best_sim >= 0 else None,
                            "best_pred_span": pred_spans[best_pi] if best_pi is not None else None,
                            "gold_span": g,
                        }
                    )

            # Update conditional metrics only for matched pairs.
            for gi, pi in matches:
                matched_pairs += 1
                g = gold_spans[gi]
                p = pred_spans[pi]

                # Composite label treated as atomic: exact canonical string match.
                gold_emotion_str = _normalize_label_canonical(g.get("Emotion_label", "na"), VALID_EMOTIONS)
                pred_emotion_str = _normalize_label_canonical(p.get("Emotion_label", "na"), VALID_EMOTIONS)
                if gold_emotion_str == pred_emotion_str and gold_emotion_str:
                    counts_emotion.tp += 1
                    per_class_emotion[gold_emotion_str].tp += 1
                else:
                    if pred_emotion_str:
                        counts_emotion.fp += 1
                        per_class_emotion[pred_emotion_str].fp += 1
                    if gold_emotion_str:
                        counts_emotion.fn += 1
                        per_class_emotion[gold_emotion_str].fn += 1

                # Composite label treated as atomic: exact canonical string match.
                gold_rev_str = _normalize_label_canonical(g.get("Reversal", "na"), VALID_REVERSAL)
                pred_rev_str = _normalize_label_canonical(p.get("Reversal", "na"), VALID_REVERSAL)
                if gold_rev_str == pred_rev_str and gold_rev_str:
                    counts_reversal.tp += 1
                    per_class_reversal[gold_rev_str].tp += 1
                else:
                    if pred_rev_str:
                        counts_reversal.fp += 1
                        per_class_reversal[pred_rev_str].fp += 1
                    if gold_rev_str:
                        counts_reversal.fn += 1
                        per_class_reversal[gold_rev_str].fn += 1

                # Composite label treated as atomic: exact canonical string match.
                gold_rev_type_str = _normalize_label_canonical(g.get("Reversal_type", "na"), VALID_REVERSAL_TYPE)
                pred_rev_type_str = _normalize_label_canonical(p.get("Reversal_type", "na"), VALID_REVERSAL_TYPE)
                if gold_rev_type_str == pred_rev_type_str and gold_rev_type_str:
                    counts_rev_type.tp += 1
                    per_class_rev_type[gold_rev_type_str].tp += 1
                else:
                    if pred_rev_type_str:
                        counts_rev_type.fp += 1
                        per_class_rev_type[pred_rev_type_str].fp += 1
                    if gold_rev_type_str:
                        counts_rev_type.fn += 1
                        per_class_rev_type[gold_rev_type_str].fn += 1

                # Emotion_shift: already a single atomic A-->B string, direct comparison.
                g_shift = _normalize_shift(g.get("Emotion_shift", "na"))
                p_shift = _normalize_shift(p.get("Emotion_shift", "na"))
                if g_shift == p_shift and g_shift != "na":
                    counts_shift.tp += 1
                    per_class_shift[g_shift].tp += 1
                else:
                    if p_shift != "na":
                        counts_shift.fp += 1
                        per_class_shift[p_shift].fp += 1
                    if g_shift != "na":
                        counts_shift.fn += 1
                        per_class_shift[g_shift].fn += 1

                g_marker_present = _normalize_marker_present(g.get("Marker", "na"))
                p_marker_present = _normalize_marker_present(p.get("Marker", "na"))
                gold_marker_set = {"present"} if g_marker_present else set()
                pred_marker_set = {"present"} if p_marker_present else set()
                c = micro_f1_from_sets(gold_marker_set, pred_marker_set)
                counts_marker_present.tp += c.tp
                counts_marker_present.fp += c.fp
                counts_marker_present.fn += c.fn

                # Additionally save conditional label errors under correctly matched spans.
                if args.save_case_study:
                    field_errors: Dict[str, Any] = {}
                    if gold_emotion_str != pred_emotion_str:
                        field_errors["Emotion_label"] = {
                            "gold": gold_emotion_str,
                            "pred": pred_emotion_str,
                        }
                    if gold_rev_str != pred_rev_str:
                        field_errors["Reversal"] = {
                            "gold": gold_rev_str,
                            "pred": pred_rev_str,
                        }
                    if gold_rev_type_str != pred_rev_type_str:
                        field_errors["Reversal_type"] = {
                            "gold": gold_rev_type_str,
                            "pred": pred_rev_type_str,
                        }
                    if g_shift != p_shift:
                        field_errors["Emotion_shift"] = {"gold": g_shift, "pred": p_shift}
                    if g_marker_present != p_marker_present:
                        field_errors["Marker_present"] = {"gold": g_marker_present, "pred": p_marker_present}

                    if field_errors:
                        span_sim = _span_similarity(g.get("Emotion_span", ""), p.get("Emotion_span", ""))
                        matched_span_label_errors_for_case.append(
                            {
                                "gold_span_index": gi,
                                "pred_span_index": pi,
                                "span_similarity": span_sim,
                                "gold_span": g,
                                "pred_span": p,
                                "field_errors": field_errors,
                            }
                        )

            if args.save_case_study and unmatched_gold_spans_for_case:
                case_missed_fw.write(
                    json.dumps(
                        {
                            "record_index": rec_idx,
                            "span_match_threshold": args.span_match_threshold,
                            "unmatched_gold_spans": unmatched_gold_spans_for_case,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if args.save_case_study and matched_span_label_errors_for_case:
                case_label_errors_fw.write(
                    json.dumps(
                        {
                            "record_index": rec_idx,
                            "span_match_threshold": args.span_match_threshold,
                            "matched_span_label_errors": matched_span_label_errors_for_case,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    if args.save_case_study:
        case_missed_fw.close()
        case_label_errors_fw.close()

    span_prf = PRFCounts(tp=total_matched_spans, fp=total_pred_spans - total_matched_spans, fn=total_gold_spans - total_matched_spans)
    print(f"Input: {input_path}")
    print(f"Records approx: (not tracked) gold_spans={total_gold_spans} pred_spans={total_pred_spans} matched_spans={total_matched_spans}")
    print("== Span extraction (Emotion_span) micro-F1 ==")
    print(f"Precision={span_prf.precision():.6f}")
    print(f"Recall   ={span_prf.recall():.6f}")
    print(f"F1       ={span_prf.f1():.6f}")

    print("\n== Conditional micro-F1 on matched spans ==")
    print("-- Emotion_label --")
    print(f"Precision={counts_emotion.precision():.6f} Recall={counts_emotion.recall():.6f} F1={counts_emotion.f1():.6f}")
    print("-- Reversal --")
    print(f"Precision={counts_reversal.precision():.6f} Recall={counts_reversal.recall():.6f} F1={counts_reversal.f1():.6f}")
    print("-- Reversal_type --")
    print(f"Precision={counts_rev_type.precision():.6f} Recall={counts_rev_type.recall():.6f} F1={counts_rev_type.f1():.6f}")
    print("-- Emotion_shift --")
    print(f"Precision={counts_shift.precision():.6f} Recall={counts_shift.recall():.6f} F1={counts_shift.f1():.6f}")
    print("-- Marker_present (marker!='na') --")
    print(f"Precision={counts_marker_present.precision():.6f} Recall={counts_marker_present.recall():.6f} F1={counts_marker_present.f1():.6f}")

    emotion_macro = _macro_from_per_class(per_class_emotion)
    reversal_macro = _macro_from_per_class(per_class_reversal)
    rev_type_macro = _macro_from_per_class(per_class_rev_type)
    shift_macro = _macro_from_per_class(per_class_shift)

    print("\n== Conditional macro-F1 on matched spans ==")
    print(f"Emotion_label macro-F1={emotion_macro['f1']:.6f}")
    print(f"Reversal macro-F1     ={reversal_macro['f1']:.6f}")
    print(f"Reversal_type macro-F1={rev_type_macro['f1']:.6f}")
    print(f"Emotion_shift macro-F1={shift_macro['f1']:.6f}")

    result_obj = {
        "meta": {
            "model_name": model_name,
            "input_path": str(input_path),
            "span_match_threshold": args.span_match_threshold,
            "max_records": args.max_records,
            "matched_pairs": matched_pairs,
            "total_gold_spans": total_gold_spans,
            "total_pred_spans": total_pred_spans,
            "total_matched_spans": total_matched_spans,
        },
        "span_extraction_micro": _as_metric_dict(span_prf),
        "conditional_micro_on_matched_spans": {
            "Emotion_label": _as_metric_dict(counts_emotion),
            "Reversal": _as_metric_dict(counts_reversal),
            "Reversal_type": _as_metric_dict(counts_rev_type),
            "Emotion_shift": _as_metric_dict(counts_shift),
            "Marker_present": _as_metric_dict(counts_marker_present),
        },
        "conditional_macro_on_matched_spans": {
            "Emotion_label": emotion_macro,
            "Reversal": reversal_macro,
            "Reversal_type": rev_type_macro,
            "Emotion_shift": shift_macro,
        },
        "per_class_on_matched_spans": {
            "Emotion_label": {k: _as_metric_dict(v) for k, v in per_class_emotion.items()},
            "Reversal": {k: _as_metric_dict(v) for k, v in per_class_reversal.items()},
            "Reversal_type": {k: _as_metric_dict(v) for k, v in per_class_rev_type.items()},
            "Emotion_shift": {k: _as_metric_dict(v) for k, v in per_class_shift.items()},
        },
    }
    with result_path.open("w", encoding="utf-8") as fw:
        json.dump(result_obj, fw, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics json: {result_path}")


if __name__ == "__main__":
    main()

