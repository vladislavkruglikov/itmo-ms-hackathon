from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read(path: Path) -> dict[str, dict]:
    return {row["hash"]: row for row in map(json.loads, path.open(encoding="utf-8")) if row}


def script(text: str) -> str:
    cyr = any("А" <= char.upper() <= "Я" for char in text)
    lat = any("a" <= char.lower() <= "z" for char in text)
    return "mixed" if cyr and lat else "cyrillic" if cyr else "latin"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank labeled examples for active-learning review.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    gold = read(args.gold)
    predictions = [read(Path(path)) for path in args.predictions]
    rows = []
    for key, record in gold.items():
        gold_entities = {(e["label"], e["start"], e["end"]) for e in record["entities"]}
        pred_sets = [{(e["label"], e["start"], e["end"]) for e in pred.get(key, {}).get("entities", [])} for pred in predictions]
        union = set().union(*pred_sets) if pred_sets else set()
        intersection = set.intersection(*pred_sets) if pred_sets else set()
        fn = len(gold_entities - union)
        fp = len(union - gold_entities)
        disagreement = len(union - intersection)
        score = 3 * disagreement + 2 * fn + fp + (2 if len(record["text"]) > 1000 else 0)
        rows.append({
            "hash": key,
            "text": record["text"],
            "script": script(record["text"]),
            "text_length": len(record["text"]),
            "gold_entities": record["entities"],
            "predictions": [pred.get(key, {}).get("entities", []) for pred in predictions],
            "false_negatives": fn,
            "false_positives": fp,
            "model_disagreement": disagreement,
            "review_score": score,
        })
    rows.sort(key=lambda row: (-row["review_score"], -row["model_disagreement"], row["hash"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows[: args.limit]:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {min(args.limit, len(rows))} review candidates to {args.output}")
    print("scripts", Counter(row["script"] for row in rows[: args.limit]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
