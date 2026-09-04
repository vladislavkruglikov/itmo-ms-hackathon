from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.common import decode_bio_tokens, read_records


SCRIPTS = ("latin", "cyrillic", "mixed")
ENTITY_LABELS = ("ORG", "NAME", "GEO")
RECORDS = []
ID2LABEL = {}
TRANSITION = None


def script(text: str) -> str:
    cyrillic = any("А" <= char.upper() <= "Я" for char in text)
    latin = any("a" <= char.lower() <= "z" for char in text)
    return "mixed" if cyrillic and latin else "cyrillic" if cyrillic else "latin"


def metric(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def viterbi(logits: torch.Tensor, bias: torch.Tensor) -> list[int]:
    adjusted = logits + bias
    n, classes = adjusted.shape
    back = torch.zeros((n, classes), dtype=torch.long)
    dp = adjusted[0].clone()
    for label_id, tag in ID2LABEL.items():
        if tag.startswith("I-"):
            dp[label_id] = -1e9
    for pos in range(1, n):
        values, previous = (dp[:, None] + TRANSITION).max(dim=0)
        back[pos] = previous
        dp = values + adjusted[pos]
    current = int(dp.argmax())
    result = [current]
    for pos in range(n - 1, 0, -1):
        current = int(back[pos, current])
        result.append(current)
    return result[::-1]


def bias_tensor(params: dict[str, float], record_script: str) -> torch.Tensor:
    values = torch.zeros(len(ID2LABEL))
    for label_id, tag in ID2LABEL.items():
        values[label_id] = params.get(f"{record_script}:{tag}", 0.0)
    return values


def decode_record(record, params: dict[str, float]):
    record_script, offsets, logits, _ = record
    ids = viterbi(logits, bias_tensor(params, record_script))
    tagged = [
        (start, end, ID2LABEL[label_id])
        for (start, end), label_id in zip(offsets, ids, strict=True)
    ]
    return {(entity["label"], entity["start"], entity["end"]) for entity in decode_bio_tokens(tagged)}


def evaluate_subset(params: dict[str, float], selected_script: str | None = None, fold: int | None = None):
    tp = fp = fn = 0
    for index, record in enumerate(RECORDS):
        record_script, _, _, gold = record
        if selected_script is not None and record_script != selected_script:
            continue
        if fold is not None and index % 5 != fold:
            continue
        predicted = decode_record(record, params)
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    return metric(tp, fp, fn)


def worker(job):
    selected_script, params = job
    return params, evaluate_subset(params, selected_script)


def candidate_values(start: float, stop: float, step: float) -> list[float]:
    count = round((stop - start) / step)
    return [round(start + index * step, 6) for index in range(count + 1)]


def choose(results, current: dict[str, float]):
    def rank(item):
        params, values = item
        magnitude = sum(abs(value) for value in params.values())
        movement = sum(abs(value - current.get(key, 0.0)) for key, value in params.items())
        return values["f1"], values["precision"], -movement, -magnitude
    return max(results, key=rank)


def set_pair(params: dict[str, float], selected_script: str, label: str, value: float):
    result = dict(params)
    result[f"{selected_script}:B-{label}"] = value
    result[f"{selected_script}:I-{label}"] = value
    return result


def set_tag(params: dict[str, float], selected_script: str, tag: str, value: float):
    result = dict(params)
    result[f"{selected_script}:{tag}"] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune script-aware BIO logit biases from cached ensemble logits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache", nargs="+", required=True, help="CACHE[:WEIGHT]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    torch.set_num_threads(1)
    rows = read_records(args.input, require_entities=True)
    combined = [{} for _ in rows]
    total_weight = 0.0
    global ID2LABEL, TRANSITION, RECORDS
    for spec in args.cache:
        path_text, weight_text = spec.rsplit(":", 1) if ":" in spec else (spec, "1")
        weight = float(weight_text)
        payload = torch.load(path_text, map_location="cpu", weights_only=False)
        labels = {int(key): str(value) for key, value in payload["id2label"].items()}
        if not ID2LABEL:
            ID2LABEL = labels
        elif labels != ID2LABEL:
            raise ValueError(f"label map mismatch in {path_text}")
        for index, scores in enumerate(payload["scores"]):
            for offset, value in scores.items():
                combined[index][offset] = combined[index].get(offset, torch.zeros_like(value)) + weight * value
        total_weight += weight

    classes = len(ID2LABEL)
    TRANSITION = torch.full((classes, classes), -1e9)
    for previous, previous_tag in ID2LABEL.items():
        for current, tag in ID2LABEL.items():
            if not tag.startswith("I-") or previous_tag in {"B-" + tag[2:], tag}:
                TRANSITION[previous, current] = 0.0

    for row, scores in zip(rows, combined, strict=True):
        ordered = sorted(scores.items())
        offsets = [offset for offset, _ in ordered]
        logits = torch.stack([value / total_weight for _, value in ordered])
        gold = {(entity["label"], entity["start"], entity["end"]) for entity in row["entities"]}
        RECORDS.append((script(row["text"]), offsets, logits, gold))

    params: dict[str, float] = {}
    baseline = evaluate_subset(params)
    print("baseline", json.dumps(baseline, sort_keys=True))
    coarse = candidate_values(-0.4, 0.4, 0.05)
    fine_delta = candidate_values(-0.15, 0.15, 0.025)
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers) as pool:
        for selected_script in SCRIPTS:
            print(f"tuning {selected_script}")
            for label in ENTITY_LABELS:
                candidates = [set_pair(params, selected_script, label, value) for value in coarse]
                params, values = choose(pool.map(worker, [(selected_script, value) for value in candidates]), params)
                print(selected_script, label, "joint", params[f"{selected_script}:B-{label}"], values["f1"])
            for fine_round in range(2):
                for label in ENTITY_LABELS:
                    for prefix in ("B", "I"):
                        tag = f"{prefix}-{label}"
                        center = params[f"{selected_script}:{tag}"]
                        candidates = [set_tag(params, selected_script, tag, round(center + delta, 6)) for delta in fine_delta]
                        params, values = choose(pool.map(worker, [(selected_script, value) for value in candidates]), params)
                        print(selected_script, tag, f"fine{fine_round + 1}", params[f"{selected_script}:{tag}"], values["f1"])

    final = evaluate_subset(params)
    folds = []
    for fold in range(5):
        before = evaluate_subset({}, fold=fold)
        after = evaluate_subset(params, fold=fold)
        folds.append({"fold": fold, "baseline": before, "tuned": after, "delta_f1": after["f1"] - before["f1"]})
    report = {"baseline": baseline, "tuned": final, "params": params, "folds": folds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    predictions = []
    for row, record in zip(rows, RECORDS, strict=True):
        entities = [
            {"label": label, "start": start, "end": end}
            for label, start, end in sorted(decode_record(record, params), key=lambda item: (item[1], item[2], item[0]))
        ]
        predictions.append({"hash": row["hash"], "entities": entities})
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
