#!/usr/bin/env python3
"""Export public HF SRR image+messages rows into local LLaMA-Factory JSONL.

Usage (from SRR repository root):
    pip install datasets pillow
    python scripts/prepare_srr.py

Notes:
- No personal paths are embedded in the published script/config.
- Output dataset rows contain relative `images/<sha256>.png` image paths.
- Image duplicates are deduplicated by their decoded pixel content.
- Decoded pixels are stored losslessly as PNG.
- The original dataset rows are expected to contain `image` and `messages`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


def get_image(value: Any):
    """Return a Pillow Image from a dataset Image feature (PIL or dict)."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, Mapping):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    if isinstance(value, (str, os.PathLike)):
        return Image.open(value)
    raise TypeError(f"Unsupported HF image type: {type(value)!r}")


def export_records(records: Iterable[Mapping[str, Any]], output_dir: Path, limit: int | None = None) -> tuple[int, int]:
    """Export records; returns (number of samples, number of unique images)."""
    output_dir = output_dir.resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "srr_train.jsonl"
    temp_file = output_dir / "srr_train.jsonl.tmp"
    n = 0
    seen_images: set[str] = set()
    try:
        with temp_file.open("w", encoding="utf-8") as writer:
            for row in records:
                if limit is not None and n >= limit:
                    break

                messages = row.get("messages")
                if not isinstance(messages, list):
                    raise ValueError(f"Row {n}: missing or invalid `messages` list")
                for msg in messages:
                    if not isinstance(msg, dict) or not isinstance(msg.get("content"), str):
                        raise ValueError(f"Row {n}: unexpected message structure")
                token_count = sum(m["content"].count("<image>") for m in messages)
                if token_count != 1:
                    raise ValueError(f"Row {n}: expected one <image> token; found {token_count}")

                if "image" not in row:
                    raise ValueError(f"Row {n}: missing `image` column in HF SRR dataset")

                im = get_image(row["image"])
                im.load()
                # PNG preserves pixel values while removing dependencies on original local filenames.
                digest = hashlib.sha256(
                    f"{im.mode}|{im.width}x{im.height}|".encode("utf-8") + im.tobytes()
                ).hexdigest()
                image_name = f"{digest}.png"
                image_path = images_dir / image_name
                if not image_path.exists():
                    im.save(image_path, format="PNG")
                seen_images.add(image_name)

                item = {"messages": messages, "images": [f"images/{image_name}"]}
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
                if n % 500 == 0:
                    print(f"Processed {n} samples, {len(seen_images)} unique images", flush=True)

        if n == 0:
            raise ValueError("Dataset is empty")
        temp_file.replace(output_file)
    finally:
        if temp_file.exists():
            temp_file.unlink()

    print(f"Wrote {n} samples -> {output_file}")
    print(f"Wrote/used {len(seen_images)} unique image files -> {images_dir}")
    return n, len(seen_images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Anonymous-SRR/SRR", help="Hugging Face dataset repository")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=None, help="For a small smoke test only")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    from datasets import load_dataset

    print(f"Loading {args.dataset} [{args.split}] from Hugging Face...")
    ds = load_dataset(args.dataset, split=args.split)
    export_records(ds, args.output_dir, limit=args.limit)


if __name__ == "__main__":
    main()
