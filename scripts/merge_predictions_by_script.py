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


def main():
    parser = argparse.ArgumentParser(description="Select prediction files by document script.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--latin", required=True)
    parser.add_argument("--cyrillic", required=True)
    parser.add_argument("--mixed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = read(args.input)
    choices = {name: read(getattr(args, name)) for name in ("latin", "cyrillic", "mixed")}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for index, record in enumerate(source):
            prediction = choices[script(record["text"])][index]
            if prediction["hash"] != record["hash"]:
                raise ValueError(f"hash mismatch at record {index}")
            stream.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
