#!/usr/bin/env python3
"""Validate prepared SRR data before running LLaMA-Factory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def verify(data_dir: Path, expected_count: int | None = 6169) -> tuple[int, int]:
    info_path = data_dir / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    name = "srr_train"
    assert name in info, f"Missing {name} in {info_path}"
    dataset_path = data_dir / info[name]["file_name"]
    n = 0
    unique_images = set()
    for lineno, line in enumerate(dataset_path.open(encoding="utf-8"), 1):
        example = json.loads(line)
        images = example.get("images")
        messages = example.get("messages")
        assert isinstance(messages, list) and isinstance(images, list), f"Invalid row {lineno}"
        assert len(images) == 1, f"Expected 1 image at row {lineno}"
        assert sum(m["content"].count("<image>") for m in messages) == 1, f"Image token mismatch at row {lineno}"
        for name in images:
            assert not Path(name).is_absolute(), f"Absolute image path at row {lineno}: {name}"
            assert ".." not in Path(name).parts, f"Unsafe relative image path at row {lineno}"
            assert (data_dir / name).is_file(), f"Missing image at row {lineno}: {name}"
            unique_images.add(name)
        n += 1
    if expected_count is not None:
        assert n == expected_count, f"Expected {expected_count} rows, found {n}"
    print(f"PASS: {n} valid rows, {len(unique_images)} unique images, no absolute image paths")
    return n, len(unique_images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--expected-count", type=int, default=6169)
    args = parser.parse_args()
    verify(args.data_dir, args.expected_count)

if __name__ == "__main__":
    main()
