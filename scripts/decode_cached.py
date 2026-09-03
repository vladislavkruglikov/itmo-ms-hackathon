from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.common import decode_bio_tokens, read_records


def viterbi(logits: torch.Tensor, labels: dict[int, str]) -> list[int]:
    n, classes = logits.shape
    neg_inf = -1e9
    dp = torch.full((n, classes), neg_inf)
    back = torch.zeros((n, classes), dtype=torch.long)
    dp[0] = logits[0]
    for pos in range(1, n):
        for current in range(classes):
            tag = labels[current]
            allowed = []
            for previous in range(classes):
                previous_tag = labels[previous]
                valid = not tag.startswith("I-") or previous_tag in {"B-" + tag[2:], tag}
                if valid:
                    allowed.append(previous)
            values = dp[pos - 1, allowed]
            best = int(values.argmax())
            back[pos, current] = allowed[best]
            dp[pos, current] = values[best] + logits[pos, current]
    current = int(dp[-1].argmax())
    result = [current]
    for pos in range(n - 1, 0, -1):
        current = int(back[pos, current])
        result.append(current)
    return result[::-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode cached NER logits with optional BIO constraints.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", nargs="+", required=True, help="CACHE[:WEIGHT]")
    parser.add_argument("--constrained", action="store_true")
    args = parser.parse_args()
    records = read_records(args.input, require_entities=False)
    combined = [{} for _ in records]
    id2label = None
    total = 0.0
    for spec in args.cache:
        path_text, weight_text = spec.rsplit(":", 1) if ":" in spec else (spec, "1")
        weight = float(weight_text)
        payload = torch.load(path_text, map_location="cpu", weights_only=False)
        if id2label is None:
            id2label = payload["id2label"]
        for index, row in enumerate(payload["scores"]):
            for key, value in row.items():
                combined[index][key] = combined[index].get(key, torch.zeros_like(value)) + weight * value
        total += weight
    assert id2label is not None and total > 0
    predictions = []
    for record, row in zip(records, combined, strict=True):
        ordered = sorted(row.items())
        logits = torch.stack([value / total for _, value in ordered])
        ids = viterbi(logits, id2label) if args.constrained else [int(value.argmax()) for value in logits]
        tagged = [(start, end, id2label[label_id]) for (start, end), label_id in zip((key for key, _ in ordered), ids, strict=True)]
        predictions.append({"hash": record["hash"], "entities": decode_bio_tokens(tagged)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Predictions: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
