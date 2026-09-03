from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.common import load_fast_tokenizer, read_records, resolve_device, validate_window
from baseline.predict import _build_windows
from transformers import AutoModelForTokenClassification


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache per-token NER logits for decoder sweeps.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    records = read_records(args.input, require_entities=False)
    tokenizer = load_fast_tokenizer(args.model)
    validate_window(tokenizer, args.max_length, args.stride)
    windows = _build_windows(records, tokenizer, max_length=args.max_length, stride=args.stride)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()
    scores = [{} for _ in records]
    with torch.inference_mode():
        for start in range(0, len(windows), args.batch_size):
            batch_windows = windows[start : start + args.batch_size]
            batch = tokenizer.pad([feature for _, feature, _ in batch_windows], padding=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.float().cpu()
            for row_index, (record_index, _, offsets) in enumerate(batch_windows):
                for token_index, (offset_start, offset_end) in enumerate(offsets):
                    if offset_start == offset_end:
                        continue
                    key = (offset_start, offset_end)
                    value = logits[row_index, token_index]
                    if key in scores[record_index]:
                        previous, count = scores[record_index][key]
                        scores[record_index][key] = (previous + value, count + 1)
                    else:
                        scores[record_index][key] = (value.clone(), 1)
    normalized = [{key: value / count for key, (value, count) in row.items()} for row in scores]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"id2label": {int(k): str(v) for k, v in model.config.id2label.items()}, "scores": normalized}, args.output)
    print(f"Cached {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
