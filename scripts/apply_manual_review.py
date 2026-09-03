from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def all_occurrences(text: str, value: str):
    start = 0
    while True:
        start = text.find(value, start)
        if start < 0:
            return
        yield start, start + len(value)
        start += len(value)


def non_overlapping(entities):
    selected = []
    for entity in sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"]), e["label"])):
        if any(entity["start"] < other["end"] and other["start"] < entity["end"] for other in selected):
            continue
        selected.append(entity)
    return sorted(selected, key=lambda e: (e["start"], e["end"], e["label"]))


def main():
    p = argparse.ArgumentParser(description="Apply explicit human review decisions.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--keep-excluded",
        action="store_true",
        help="retain records marked exclude_record; useful for comparing data-policy variants",
    )
    args = p.parse_args()
    decisions = {row["hash"]: row for row in read(args.decisions)}
    output = []
    changed = 0
    for row in read(args.input):
        decision = decisions.get(row["hash"])
        if decision and decision["action"] == "exclude_record" and not args.keep_excluded:
            changed += 1
            continue
        updated = dict(row)
        if decision and decision["action"] in {"replace_entities", "add_entities"}:
            entities = [] if decision["action"] == "replace_entities" else list(updated.get("entities", []))
            existing = {(e["label"], e["start"], e["end"]) for e in entities}
            for item in decision["entities"]:
                for start, end in all_occurrences(row["text"], item["text"]):
                    value = (item["label"], start, end)
                    if value not in existing:
                        entities.append({"label": item["label"], "start": start, "end": end})
                        existing.add(value)
            updated["entities"] = non_overlapping(entities)
            changed += 1
        if decision and decision["action"] == "remove_entities":
            removals = {(item["label"], item["text"]) for item in decision.get("entities", [])}
            updated["entities"] = [
                entity for entity in updated.get("entities", [])
                if (entity["label"], row["text"][entity["start"]:entity["end"]]) not in removals
            ]
            changed += 1
        if decision and decision["action"] == "relabel_entities":
            relabel = {
                (item["from_label"], item["text"]): item["to_label"]
                for item in decision.get("entities", [])
            }
            for entity in updated.get("entities", []):
                value = row["text"][entity["start"]:entity["end"]]
                target = relabel.get((entity["label"], value))
                if target:
                    entity["label"] = target
            changed += 1
        output.append(updated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {len(output)} records; manually replaced {changed}")


if __name__ == "__main__":
    main()
