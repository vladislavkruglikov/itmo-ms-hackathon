from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a reproducible training set excluding reviewed active-learning candidates.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-top", type=int, required=True)
    args = parser.parse_args()
    candidates = list(rows(args.candidates))[: args.exclude_top]
    excluded = {row["hash"] for row in candidates}
    kept = [row for row in rows(args.input) if row["hash"] not in excluded]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in kept:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Excluded {len(excluded)} records; wrote {len(kept)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
