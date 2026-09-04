#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def script(text):
    cyr = any("А" <= char.upper() <= "Я" for char in text)
    lat = any("a" <= char.lower() <= "z" for char in text)
    return "mixed" if cyr and lat else "cyrillic" if cyr else "latin"


def spans(row):
    return {(e["label"], e["start"], e["end"]) for e in row["entities"]}


def main():
    parser = argparse.ArgumentParser(description="Reject decoded spans with selected component-support masks.")
    parser.add_argument("--input", required=True, help="JSONL containing source text")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--components", nargs="+", required=True)
    parser.add_argument("--reject", nargs="+", required=True, metavar="SCRIPT:LABEL:MASK")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = read(args.input)
    predictions = read(args.predictions)
    components = [read(path) for path in args.components]
    rejected = {tuple(item.split(":")) for item in args.reject}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for index, (record, prediction) in enumerate(zip(source, predictions, strict=True)):
            component_spans = [spans(rows[index]) for rows in components]
            row_script = script(record["text"])
            entities = []
            for entity in prediction["entities"]:
                span = (entity["label"], entity["start"], entity["end"])
                mask = "".join("1" if span in values else "0" for values in component_spans)
                if (row_script, entity["label"], mask) not in rejected:
                    entities.append(entity)
            stream.write(json.dumps({"hash": prediction["hash"], "entities": entities}, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
