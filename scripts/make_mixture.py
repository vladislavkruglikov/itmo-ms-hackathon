from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def is_cyrillic(text: str) -> bool:
    return any("А" <= char.upper() <= "Я" for char in text)


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic weighted NER mixture.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", action="append", required=True, help="PATH:REPEATS")
    parser.add_argument("--cyrillic-boost", type=float, default=1.0)
    parser.add_argument("--mixed-boost", type=float, default=1.0)
    args = parser.parse_args()
    if args.cyrillic_boost < 1 or args.mixed_boost < 1:
        raise ValueError("script boosts must be >= 1")
    rng = random.Random(args.seed)
    rows: list[dict] = []
    copy_index = 0
    for spec in args.source:
        path_text, repeats_text = spec.rsplit(":", 1)
        source_rows = load(Path(path_text))
        repeats = int(repeats_text)
        if repeats < 1:
            raise ValueError("source repeats must be positive")
        for row in source_rows:
            script_weight = 1.0
            cyr = is_cyrillic(row["text"])
            lat = any("a" <= char.lower() <= "z" for char in row["text"])
            if cyr and not lat:
                script_weight = args.cyrillic_boost
            elif cyr and lat:
                script_weight = args.mixed_boost
            copies = repeats * script_weight
            whole = int(copies)
            if rng.random() < copies - whole:
                whole += 1
            for _ in range(whole):
                copied = dict(row)
                copied["hash"] = f"{row['hash']}#mix-{copy_index}"
                copy_index += 1
                rows.append(copied)
    rng.shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {len(rows)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
