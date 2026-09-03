from __future__ import annotations

import argparse
import json
from pathlib import Path


def entities(value):
    return {(e["label"], int(e["start"]), int(e["end"])) for e in value}


def show(text, entity):
    label, start, end = entity
    left = max(0, start - 70)
    right = min(len(text), end + 70)
    return f"{label} [{start}:{end}] {text[left:right]!r}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--count", type=int, default=10)
    args = p.parse_args()
    rows = [json.loads(x) for x in args.candidates.open(encoding="utf-8") if x.strip()]
    for number, row in enumerate(rows[args.start:args.start + args.count], args.start + 1):
        gold = entities(row["gold_entities"])
        predictions = [entities(x) for x in row["predictions"]]
        union = set.union(*predictions)
        intersection = set.intersection(*predictions)
        print(f"\n### {number} score={row['review_score']} script={row['script']} length={row['text_length']} hash={row['hash']}")
        print(row["text"])
        print("GOLD_ONLY")
        for entity in sorted(gold - union, key=lambda x: (x[1], x[2], x[0])):
            print(" ", show(row["text"], entity))
        print("CONSENSUS_NOT_GOLD")
        for entity in sorted(intersection - gold, key=lambda x: (x[1], x[2], x[0])):
            print(" ", show(row["text"], entity))
        print("PREDICTION_DISAGREEMENTS")
        for entity in sorted(union - intersection, key=lambda x: (x[1], x[2], x[0])):
            print(" ", show(row["text"], entity))


if __name__ == "__main__":
    main()
