from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.common import decode_bio_tokens, load_fast_tokenizer, read_records, resolve_device, validate_window
from baseline.predict import _build_windows


def predict_model(model, tokenizer, windows, record_count, batch_size, device):
    scores = [{} for _ in range(record_count)]
    model.eval()
    with torch.inference_mode():
        for start in tqdm(range(0, len(windows), batch_size), desc="Predict", leave=False):
            batch_windows = windows[start : start + batch_size]
            batch = tokenizer.pad([feature for _, feature, _ in batch_windows], padding=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.float().cpu()
            for row_index, (record_index, _, offsets) in enumerate(batch_windows):
                for token_index, (offset_start, offset_end) in enumerate(offsets):
                    if offset_start == offset_end:
                        continue
                    key = (offset_start, offset_end)
                    value = logits[row_index, token_index]
                    previous = scores[record_index].get(key)
                    if previous is None:
                        scores[record_index][key] = (value.clone(), 1)
                    else:
                        scores[record_index][key] = (previous[0] + value, previous[1] + 1)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Weighted logit ensemble for exact-span NER.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True, help="MODEL[:WEIGHT]")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    records = read_records(args.input, require_entities=False)
    model_specs = [item.rsplit(":", 1) if ":" in item else (item, "1") for item in args.models]
    tokenizer = load_fast_tokenizer(model_specs[0][0])
    validate_window(tokenizer, args.max_length, args.stride)
    windows = _build_windows(records, tokenizer, max_length=args.max_length, stride=args.stride)
    combined = [{} for _ in records]
    total_weight = 0.0
    id2label = None
    for model_path, weight_text in model_specs:
        weight = float(weight_text)
        model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
        if id2label is None:
            id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        scores = predict_model(model, tokenizer, windows, len(records), args.batch_size, device)
        for record_index, record_scores in enumerate(scores):
            for key, (value, count) in record_scores.items():
                value = value / count
                combined[record_index][key] = combined[record_index].get(key, torch.zeros_like(value)) + weight * value
        total_weight += weight
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert id2label is not None and total_weight > 0
    predictions = []
    for record, record_scores in zip(records, combined, strict=True):
        tagged = [(start, end, id2label[int((value / total_weight).argmax())]) for (start, end), value in sorted(record_scores.items())]
        predictions.append({"hash": record["hash"], "entities": decode_bio_tokens(tagged)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Predictions: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
