#!/usr/bin/env python3
"""Build the cleaned train_v3 dataset from train.jsonl and train_v2.jsonl.

The transformation is deterministic and preserves the original text and hashes.
It removes only demonstrable duplicate conflicts and external boundary punctuation;
context-dependent surface labels are retained and audited rather than overwritten.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LABELS = {"ORG", "NAME", "GEO"}
# Characters explicitly treated as external delimiters by LABELING_GUIDE.md.
LEADING = set('«“„‟\"\'“”‘’([{#@')
TRAILING = set('.,!?;:)]}»”\"\'’')


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_no}: empty line")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def validate_row(row: dict[str, Any], source: str) -> None:
    text = row.get("text")
    entities = row.get("entities")
    if not isinstance(row.get("hash"), str) or not row["hash"] or not isinstance(text, str):
        raise ValueError(f"{source}: invalid hash/text")
    if not isinstance(entities, list):
        raise ValueError(f"{source}: entities must be a list")
    previous_end = -1
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict) or entity.get("label") not in LABELS:
            raise ValueError(f"{source}/entities[{index}]: invalid entity")
        start, end = entity.get("start"), entity.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
            raise ValueError(f"{source}/entities[{index}]: invalid offsets")
        if start < previous_end:
            raise ValueError(f"{source}: overlapping entities")
        previous_end = end


def clean_entities(text: str, entities: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    cleaned: list[dict[str, Any]] = []
    changes: Counter[str] = Counter()
    for entity in entities:
        start, end = entity["start"], entity["end"]
        original = (start, end)
        while start < end and (text[start].isspace() or text[start] in LEADING):
            start += 1
            changes["leading_trim"] += 1
        while end > start and (text[end - 1].isspace() or text[end - 1] in TRAILING):
            end -= 1
            changes["trailing_trim"] += 1
        if start == end:
            changes["dropped_empty_after_trim"] += 1
            continue
        if (start, end) != original:
            changes["spans_changed"] += 1
        cleaned.append({"label": entity["label"], "start": start, "end": end})

    # Boundary cleanup can make two annotations identical; retain one.
    unique: dict[tuple[str, int, int], dict[str, Any]] = {}
    for entity in cleaned:
        key = (entity["label"], entity["start"], entity["end"])
        if key in unique:
            changes["duplicate_entities_after_trim"] += 1
        unique[key] = entity
    result = sorted(unique.values(), key=lambda x: (x["start"], x["end"], x["label"]))
    for left, right in zip(result, result[1:]):
        if right["start"] < left["end"]:
            raise ValueError(f"boundary cleanup created overlap: {text!r}")
    return result, changes


def signature(row: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    return tuple(sorted((e["label"], e["start"], e["end"]) for e in row["entities"]))


def surface(text: str, entity: dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", text[entity["start"]:entity["end"]].casefold().strip())


def choose_duplicate(group: list[dict[str, Any]], original_hashes: set[str]) -> tuple[dict[str, Any], bool]:
    """Choose the supported annotation; prefer original train on ties."""
    by_sig: dict[tuple[tuple[str, int, int], ...], list[dict[str, Any]]] = defaultdict(list)
    for row in group:
        by_sig[signature(row)].append(row)
    ranked = sorted(
        by_sig.items(),
        key=lambda item: (
            len(item[1]),
            sum(row["hash"] in original_hashes for row in item[1]),
            tuple(sorted(row["hash"] for row in item[1])),
        ),
        reverse=True,
    )
    chosen = sorted(ranked[0][1], key=lambda row: (row["hash"] not in original_hashes, row["hash"]))[0]
    return chosen, len(by_sig) > 1


def build(train_path: Path, v2_path: Path, output_path: Path, audit_path: Path) -> dict[str, Any]:
    original = read_jsonl(train_path)
    v2 = read_jsonl(v2_path)
    for index, row in enumerate(original, 1):
        validate_row(row, f"{train_path}:{index}")
    for index, row in enumerate(v2, 1):
        validate_row(row, f"{v2_path}:{index}")
    original_hashes = {row["hash"] for row in original}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in v2:
        cleaned, changes = clean_entities(row["text"], row["entities"])
        row = {"hash": row["hash"], "text": row["text"], "entities": cleaned}
        row["_changes"] = dict(changes)
        groups[row["text"]].append(row)

    selected: list[dict[str, Any]] = []
    duplicate_groups = conflicting_groups = dropped_duplicates = 0
    changes_total: Counter[str] = Counter()
    conflict_details: list[dict[str, Any]] = []
    for text, group in groups.items():
        if len(group) > 1:
            duplicate_groups += 1
            chosen, conflicting = choose_duplicate(group, original_hashes)
            if conflicting:
                conflicting_groups += 1
                conflict_details.append({
                    "text_prefix": text[:240],
                    "hashes": [r["hash"] for r in group],
                    "chosen_hash": chosen["hash"],
                    "signatures": [list(signature(r)) for r in group],
                })
            dropped_duplicates += len(group) - 1
        else:
            chosen = group[0]
        changes_total.update(chosen.pop("_changes", {}))
        selected.append(chosen)

    selected.sort(key=lambda row: row["hash"])
    surface_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selected:
        for entity in row["entities"]:
            surface_labels[surface(row["text"], entity)][entity["label"]] += 1
    ambiguous = [
        {"surface": s, "counts": dict(counts), "total": sum(counts.values())}
        for s, counts in surface_labels.items() if len(counts) > 1
    ]
    ambiguous.sort(key=lambda item: (-item["total"], item["surface"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    audit = {
        "schema_version": 1,
        "source": {"train": str(train_path), "train_v2": str(v2_path)},
        "output": str(output_path),
        "input_records": len(v2),
        "output_records": len(selected),
        "duplicate_text_groups": duplicate_groups,
        "conflicting_duplicate_groups": conflicting_groups,
        "dropped_duplicate_records": dropped_duplicates,
        "boundary_changes": dict(changes_total),
        "ambiguous_surface_count": len(ambiguous),
        "ambiguous_surface_occurrences": sum(item["total"] for item in ambiguous),
        "ambiguous_surfaces": ambiguous,
        "conflict_details": conflict_details,
        "policy": "Context-dependent surface labels are retained; only exact-text duplicate conflicts are resolved.",
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--train-v2", type=Path, default=Path("data/train_v2.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/train_v3.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("data/train_v3_audit.json"))
    args = parser.parse_args()
    print(json.dumps(build(args.train, args.train_v2, args.output, args.audit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
