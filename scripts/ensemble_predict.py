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


def constrained_ids(logits: torch.Tensor, labels: dict[int, str]) -> list[int]:
    neg_inf = -1e9
    dp = torch.full_like(logits, neg_inf)
    back = torch.zeros(logits.shape, dtype=torch.long)
    dp[0] = logits[0]
    for label_id, tag in labels.items():
        if tag.startswith("I-"):
            dp[0, label_id] = neg_inf
    for pos in range(1, logits.shape[0]):
        for current, tag in labels.items():
            allowed = [previous for previous, previous_tag in labels.items()
                       if not tag.startswith("I-") or previous_tag in {"B-" + tag[2:], tag}]
            values = dp[pos - 1, allowed]
            best = int(values.argmax())
            back[pos, current] = allowed[best]
            dp[pos, current] = values[best] + logits[pos, current]
    current = int(dp[-1].argmax())
    result = [current]
    for pos in range(logits.shape[0] - 1, 0, -1):
        current = int(back[pos, current])
        result.append(current)
    return result[::-1]


def script(text: str) -> str:
    cyrillic = any("А" <= char.upper() <= "Я" for char in text)
    latin = any("a" <= char.lower() <= "z" for char in text)
    return "mixed" if cyrillic and latin else "cyrillic" if cyrillic else "latin"


def main() -> int:
    parser = argparse.ArgumentParser(description="Weighted logit ensemble for exact-span NER.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True, help="MODEL[:WEIGHT]")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--label-bias", nargs="*", default=[], metavar="LABEL:BIAS",
                        help="Add a constant to selected label logits before decoding.")
    parser.add_argument(
        "--script-label-bias",
        nargs="*",
        default=[],
        metavar="SCRIPT:LABEL:BIAS",
        help="Add a label-logit bias only for latin, cyrillic, or mixed documents.",
    )
    parser.add_argument(
        "--min-model-support",
        nargs="*",
        default=[],
        metavar="SCRIPT:LABEL:COUNT",
        help="Drop decoded spans not independently produced by at least COUNT component models.",
    )
    args = parser.parse_args()
    label_bias = {}
    for item in args.label_bias:
        name, value = item.rsplit(":", 1)
        label_bias[name] = float(value)
    script_label_bias = {}
    for item in args.script_label_bias:
        script_name, name, value = item.split(":", 2)
        if script_name not in {"latin", "cyrillic", "mixed"}:
            raise ValueError(f"unknown script: {script_name}")
        script_label_bias[(script_name, name)] = float(value)
    min_model_support = {}
    for item in args.min_model_support:
        script_name, name, value = item.split(":", 2)
        if script_name not in {"latin", "cyrillic", "mixed"}:
            raise ValueError(f"unknown script: {script_name}")
        count = int(value)
        if count < 0:
            raise ValueError("model support must be non-negative")
        min_model_support[(script_name, name)] = count
    device = resolve_device(args.device)
    records = read_records(args.input, require_entities=False)
    model_specs = [item.rsplit(":", 1) if ":" in item else (item, "1") for item in args.models]
    tokenizer = load_fast_tokenizer(model_specs[0][0])
    validate_window(tokenizer, args.max_length, args.stride)
    windows = _build_windows(records, tokenizer, max_length=args.max_length, stride=args.stride)
    combined = [{} for _ in records]
    total_weight = 0.0
    id2label = None
    component_scores = []
    for model_path, weight_text in model_specs:
        weight = float(weight_text)
        model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
        if id2label is None:
            id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        scores = predict_model(model, tokenizer, windows, len(records), args.batch_size, device)
        if min_model_support:
            component_scores.append(scores)
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
        ordered = sorted(record_scores.items())
        logits = torch.stack([value / total_weight for _, value in ordered])
        record_script = script(record["text"])
        for label_id, label in id2label.items():
            if label in label_bias:
                logits[:, label_id] += label_bias[label]
            logits[:, label_id] += script_label_bias.get((record_script, label), 0.0)
        ids = constrained_ids(logits, id2label) if args.constrained else [int(value.argmax()) for value in logits]
        tagged = [(start, end, id2label[label_id]) for (start, end), label_id in zip((key for key, _ in ordered), ids, strict=True)]
        entities = decode_bio_tokens(tagged)
        if min_model_support:
            support_sets = []
            for scores in component_scores:
                component_ordered = sorted(scores[len(predictions)].items())
                component_logits = torch.stack([value / count for _, (value, count) in component_ordered])
                component_ids = (
                    constrained_ids(component_logits, id2label)
                    if args.constrained
                    else [int(value.argmax()) for value in component_logits]
                )
                component_tagged = [
                    (start, end, id2label[label_id])
                    for (start, end), label_id in zip(
                        (key for key, _ in component_ordered), component_ids, strict=True
                    )
                ]
                support_sets.append({
                    (entity["label"], entity["start"], entity["end"])
                    for entity in decode_bio_tokens(component_tagged)
                })
            entities = [
                entity
                for entity in entities
                if sum(
                    (entity["label"], entity["start"], entity["end"]) in support
                    for support in support_sets
                )
                >= min_model_support.get((record_script, entity["label"]), 0)
            ]
        predictions.append({"hash": record["hash"], "entities": entities})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Predictions: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
