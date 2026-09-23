#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ENTRY_RE = re.compile(r"^####([^#]+)####")
FIELD_RE = re.compile(r"^\s*([A-Za-z][A-Za-z ]*[A-Za-z])\s*:\s*(.*)\s*$")


@dataclass
class SpanAnno:
    span: str
    reversal: str
    reversal_type: str
    emotion_shift: str


def norm_text(x: str) -> str:
    s = (x or "").strip().lower()
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def norm_span(x: str) -> str:
    s = norm_text(x)
    if s in {"", "na", "n/a", "none", "null"}:
        return "na"
    return s


def norm_label(x: str) -> str:
    s = norm_text(x)
    if s in {"", "na", "n/a", "none", "null"}:
        return "na"
    return s


def norm_shift(x: str) -> str:
    s = norm_label(x)
    s = s.replace("->", "-->")
    s = s.replace("--->", "-->")
    s = re.sub(r"\s*-->\s*", "-->", s)
    return s


def parse_annotation_md(path: Path) -> Dict[str, List[SpanAnno]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    data: Dict[str, List[SpanAnno]] = {}

    cur_entry: Optional[str] = None
    cur_span: Optional[str] = None
    cur_rev: str = "na"
    cur_rev_type: str = "na"
    cur_shift: str = "na"

    def flush_span() -> None:
        nonlocal cur_span, cur_rev, cur_rev_type, cur_shift, cur_entry
        if cur_entry is None or cur_span is None:
            return
        data.setdefault(cur_entry, []).append(
            SpanAnno(
                span=norm_span(cur_span),
                reversal=norm_label(cur_rev),
                reversal_type=norm_label(cur_rev_type),
                emotion_shift=norm_shift(cur_shift),
            )
        )
        cur_span = None
        cur_rev = "na"
        cur_rev_type = "na"
        cur_shift = "na"

    def flush_entry() -> None:
        flush_span()

    for raw in lines:
        line = raw.rstrip("\n")

        m = ENTRY_RE.match(line.strip())
        if m:
            flush_entry()
            cur_entry = m.group(1).strip()
            continue

        fm = FIELD_RE.match(line)
        if not fm or cur_entry is None:
            continue
        field = fm.group(1).strip().lower()
        val = fm.group(2).strip()

        if field == "emotion span":
            flush_span()
            cur_span = val
        elif field == "reversal":
            cur_rev = val
        elif field == "reversal type":
            cur_rev_type = val
        elif field == "emotion shift":
            cur_shift = val

    flush_entry()
    return data


def tuple_category(spans: Sequence[str]) -> str:
    # Use sorted unique span tuple as one categorical label for kappa.
    uniq = sorted(set(spans))
    return " || ".join(uniq) if uniq else "na"


def non_na_spans(spans: Sequence[SpanAnno]) -> List[SpanAnno]:
    return [s for s in spans if s.span != "na"]


def fleiss_kappa(items: Sequence[Sequence[str]]) -> Optional[float]:
    if not items:
        return None
    n = len(items[0])
    if n < 2:
        return None
    if any(len(x) != n for x in items):
        return None

    cats = sorted({lab for row in items for lab in row})
    if not cats:
        return None
    idx = {c: i for i, c in enumerate(cats)}

    N = len(items)
    k = len(cats)
    mat = [[0] * k for _ in range(N)]
    for i, row in enumerate(items):
        for lab in row:
            mat[i][idx[lab]] += 1

    p_j = [0.0] * k
    for j in range(k):
        p_j[j] = sum(mat[i][j] for i in range(N)) / (N * n)

    P_i = []
    for i in range(N):
        sq = sum(mat[i][j] ** 2 for j in range(k))
        P_i.append((sq - n) / (n * (n - 1)))
    P_bar = sum(P_i) / N
    P_e = sum(p * p for p in p_j)

    if abs(1 - P_e) < 1e-12:
        return None
    return (P_bar - P_e) / (1 - P_e)


def cohen_kappa(labels1: Sequence[str], labels2: Sequence[str]) -> Optional[float]:
    if len(labels1) != len(labels2) or not labels1:
        return None
    n = len(labels1)
    cats = sorted(set(labels1) | set(labels2))
    p1 = {c: 0 for c in cats}
    p2 = {c: 0 for c in cats}
    agree = 0
    for a, b in zip(labels1, labels2):
        p1[a] += 1
        p2[b] += 1
        if a == b:
            agree += 1
    po = agree / n
    pe = sum((p1[c] / n) * (p2[c] / n) for c in cats)
    if abs(1 - pe) < 1e-12:
        return None
    return (po - pe) / (1 - pe)


def evaluate_group(
    group_name: str,
    annotator_maps: Dict[str, Dict[str, List[SpanAnno]]],
) -> Dict[str, object]:
    annotators = sorted(annotator_maps.keys())
    if len(annotators) < 2:
        raise ValueError(f"{group_name}: need >=2 annotators")

    common_entries = set.intersection(*(set(annotator_maps[a].keys()) for a in annotators))
    common_entries = set(sorted(common_entries))

    # Emotion Span kappa (all common entries)
    span_items: List[List[str]] = []
    for eid in sorted(common_entries):
        row = []
        for a in annotators:
            spans = [x.span for x in non_na_spans(annotator_maps[a][eid])]
            row.append(tuple_category(spans))
        # Skip entries where all annotators only marked NA spans.
        if not all(v == "na" for v in row):
            span_items.append(row)

    result: Dict[str, object] = {
        "group": group_name,
        "annotators": annotators,
        "common_entry_count": len(common_entries),
    }

    if len(annotators) == 2:
        span_kappa = cohen_kappa([r[0] for r in span_items], [r[1] for r in span_items])
        span_method = "cohen"
    else:
        span_kappa = fleiss_kappa(span_items)
        span_method = "fleiss"

    result["emotion_span"] = {
        "method": span_method,
        "kappa": span_kappa,
        "item_count": len(span_items),
    }

    # Conditional: only entries where all annotators have identical span tuple.
    span_equal_entries: List[str] = []
    for eid in sorted(common_entries):
        cats = []
        for a in annotators:
            spans = [x.span for x in non_na_spans(annotator_maps[a][eid])]
            cats.append(tuple_category(spans))
        # Only keep entries that have at least one non-NA span and are identical across annotators.
        if len(set(cats)) == 1 and cats[0] != "na":
            span_equal_entries.append(eid)

    result["span_equal_entry_count"] = len(span_equal_entries)

    def field_items(field_name: str) -> List[List[str]]:
        rows: List[List[str]] = []
        for eid in span_equal_entries:
            per_ann: List[Dict[str, SpanAnno]] = []
            ok = True
            for a in annotators:
                d: Dict[str, SpanAnno] = {}
                for sa in non_na_spans(annotator_maps[a][eid]):
                    if sa.span not in d:
                        d[sa.span] = sa
                per_ann.append(d)
            span_keys = sorted(per_ann[0].keys())
            for key in span_keys:
                row: List[str] = []
                for d in per_ann:
                    if key not in d:
                        ok = False
                        break
                    val = getattr(d[key], field_name)
                    row.append(val)
                if ok:
                    rows.append(row)
        return rows

    for field_name, out_key in [
        ("reversal", "reversal"),
        ("reversal_type", "reversal_type"),
        ("emotion_shift", "emotion_shift"),
    ]:
        rows = field_items(field_name)
        if len(annotators) == 2:
            kappa = cohen_kappa([r[0] for r in rows], [r[1] for r in rows]) if rows else None
            method = "cohen"
        else:
            kappa = fleiss_kappa(rows) if rows else None
            method = "fleiss"
        result[out_key] = {
            "method": method,
            "kappa": kappa,
            "item_count": len(rows),
            "condition": "computed only on spans from entries where all annotators have identical Emotion Span sets",
        }

    return result


def collect_groups(annotation_dir: Path) -> Dict[str, Dict[str, Path]]:
    files = sorted(annotation_dir.glob("new_samples_V*_*.md"))
    groups: Dict[str, Dict[str, Path]] = {}
    pat = re.compile(r"new_samples_(V[0-9]+)_([^/\\]+)\.md$")
    for p in files:
        m = pat.search(p.name)
        if not m:
            continue
        ver = m.group(1)
        ann = m.group(2)
        groups.setdefault(ver, {})[ann] = p
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute inter-annotator kappa for emotion reversal annotation files.")
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default="/Users/tangtang/Desktop/Reseaerch/EMOTION/EmotionReversal/human_annotation/download_from_server/annotation_results",
        help="Directory containing new_samples_V*_*.md files.",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default="",
        help="Optional path to save full result JSON.",
    )
    args = parser.parse_args()

    annotation_dir = Path(args.annotation_dir).expanduser()
    groups = collect_groups(annotation_dir)

    outputs: Dict[str, object] = {}
    parsed_cache: Dict[Path, Dict[str, List[SpanAnno]]] = {}

    for ver in sorted(groups.keys()):
        amap: Dict[str, Dict[str, List[SpanAnno]]] = {}
        for ann, path in sorted(groups[ver].items()):
            parsed_cache[path] = parse_annotation_md(path)
            amap[ann] = parsed_cache[path]
        outputs[ver] = evaluate_group(ver, amap)

    # Additional combined report for V2+V3 (same two annotators).
    if "V2" in groups and "V3" in groups:
        common_anns = sorted(set(groups["V2"].keys()) & set(groups["V3"].keys()))
        if len(common_anns) >= 2:
            merged: Dict[str, Dict[str, List[SpanAnno]]] = {}
            for ann in common_anns:
                p2 = groups["V2"][ann]
                p3 = groups["V3"][ann]
                d2 = parsed_cache.get(p2) or parse_annotation_md(p2)
                d3 = parsed_cache.get(p3) or parse_annotation_md(p3)
                merged_map = {}
                merged_map.update(d2)
                merged_map.update(d3)
                merged[ann] = merged_map
            outputs["V2_V3_combined"] = evaluate_group("V2_V3_combined", merged)

    print(json.dumps(outputs, ensure_ascii=False, indent=2))

    if args.save_json:
        out = Path(args.save_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
