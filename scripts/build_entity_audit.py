from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path):
    return {row["hash"]: row for row in map(json.loads, path.open(encoding="utf-8"))}


def overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    gold = read(args.gold)
    pred = read(args.pred)
    rows = []
    for key, source in gold.items():
        gs = source.get("entities", [])
        ps = pred.get(key, {}).get("entities", [])
        for entity in gs:
            same = [x for x in ps if x["label"] == entity["label"] and overlap(entity, x)]
            if same:
                best = max(same, key=lambda x: min(entity["end"], x["end"]) - max(entity["start"], x["start"]))
                if (best["start"], best["end"]) != (entity["start"], entity["end"]):
                    rows.append({"kind": "boundary", "hash": key, "text": source["text"], "gold": entity, "pred": best})
            else:
                rows.append({"kind": "missed", "hash": key, "text": source["text"], "gold": entity, "pred": None})
        for entity in ps:
            if not any(x["label"] == entity["label"] and overlap(entity, x) for x in gs):
                rows.append({"kind": "false_positive", "hash": key, "text": source["text"], "gold": None, "pred": entity})
    rows.sort(key=lambda x: (x["hash"], x["kind"], (x["gold"] or x["pred"])["start"]))
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    from collections import Counter
    print(f"Wrote {len(rows)} entity candidates: {Counter(x['kind'] for x in rows)}")


if __name__ == "__main__":
    main()
