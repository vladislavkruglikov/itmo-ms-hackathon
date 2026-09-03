from __future__ import annotations

import argparse
import json
from pathlib import Path


Entity = tuple[str, int, int]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def entity_set(entities: list[dict]) -> set[Entity]:
    return {(e["label"], int(e["start"]), int(e["end"])) for e in entities}


def overlaps(a: Entity, b: Entity) -> bool:
    return a[1] < b[2] and b[1] < a[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an auditable, conservative active-learning relabeling."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    candidates = {row["hash"]: row for row in read_jsonl(args.candidates)}
    changed = 0
    added = 0
    skipped_overlap = 0
    audit_rows = []
    reviewed = []

    for record in records:
        row = candidates.get(record["hash"])
        old = list(record.get("entities", []))
        old_set = entity_set(old)
        additions: set[Entity] = set()
        if row is not None and len(row.get("predictions", [])) >= 2:
            predictions = [entity_set(value) for value in row["predictions"]]
            shared = set.intersection(*predictions)
            for candidate in sorted(shared - old_set):
                if any(overlaps(candidate, existing) for existing in old_set):
                    skipped_overlap += 1
                else:
                    additions.add(candidate)

        new_set = old_set | additions
        if additions:
            changed += 1
            added += len(additions)
        new_entities = [
            {"label": label, "start": start, "end": end}
            for label, start, end in sorted(new_set, key=lambda item: (item[1], item[2], item[0]))
        ]
        updated = dict(record)
        updated["entities"] = new_entities
        reviewed.append(updated)
        if row is not None:
            gold = old_set
            union = set.union(*(entity_set(value) for value in row["predictions"]))
            audit_rows.append({
                "hash": record["hash"],
                "review_score": row["review_score"],
                "policy": "add_exact_two_model_consensus_nonoverlap",
                "original_entity_count": len(old_set),
                "added_entities": [
                    {"label": label, "start": start, "end": end,
                     "text": record["text"][start:end]}
                    for label, start, end in sorted(additions, key=lambda item: (item[1], item[2], item[0]))
                ],
                "gold_entities_missed_by_all_models": [
                    {"label": label, "start": start, "end": end,
                     "text": record["text"][start:end]}
                    for label, start, end in sorted(gold - union, key=lambda item: (item[1], item[2], item[0]))
                ],
                "action": "add_consensus_only" if additions else "keep_original",
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in reviewed:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with args.audit.open("w", encoding="utf-8") as stream:
        for row in audit_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {len(reviewed)} records; changed {changed}; added {added} entities")
    print(f"Skipped overlapping consensus candidates: {skipped_overlap}")
    print(f"Audit: {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
