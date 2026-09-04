#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Concatenate JSONL NER datasets with hash deduplication.")
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    rows: list[dict] = []
    seen: dict[str, dict] = {}
    inputs = []
    skipped = 0
    labels: Counter[str] = Counter()
    for path in args.input:
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                count += 1
                row_hash = row["hash"]
                if row_hash in seen:
                    if row != seen[row_hash]:
                        raise ValueError(f"conflicting duplicate hash {row_hash} at {path}:{line_number}")
                    skipped += 1
                    continue
                seen[row_hash] = row
                rows.append(row)
                labels.update(entity["label"] for entity in row["entities"])
        inputs.append({"path": str(path), "records": count})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "scripts/combine_datasets.py",
        "method": "ordered concatenation with exact-row hash deduplication",
        "inputs": inputs,
        "duplicates_skipped": skipped,
        "records_out": len(rows),
        "entities_by_label": dict(sorted(labels.items())),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
