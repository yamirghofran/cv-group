from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


WORKSPACE = "roboflow-jvuqo"
PROJECT = "basketball-jersey-numbers-ocr"
VERSION = 7

SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def download(api_key: str, raw_dir: Path) -> Path:
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    dataset = version.download("jsonl", location=str(raw_dir))
    return raw_dir


def convert(raw_dir: Path, out_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}

    for raw_split, out_split in SPLIT_MAP.items():
        split_dir = raw_dir / raw_split
        if not split_dir.exists():
            continue

        jsonl_file = _find_jsonl(split_dir)
        if jsonl_file is None:
            print(f"  No JSONL file found in {split_dir}, skipping.")
            continue

        images_dir = split_dir / "images" if (split_dir / "images").exists() else split_dir

        print(f"  Converting {raw_split} -> {out_split} using {jsonl_file.name}")

        n = 0
        with jsonl_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                label = _extract_label(entry)
                image_name = _extract_image_name(entry)

                if label is None or image_name is None:
                    continue

                src = images_dir / image_name
                if not src.exists():
                    src = split_dir / image_name
                if not src.exists():
                    continue

                dst_dir = out_dir / out_split / label
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst_dir / image_name)
                n += 1

        counts[out_split] = n
        print(f"    Copied {n} images.")

    return counts


def _find_jsonl(split_dir: Path) -> Path | None:
    for name in ["_annotations.jsonl", "annotations.jsonl"]:
        p = split_dir / name
        if p.exists():
            return p
    matches = list(split_dir.glob("*.jsonl"))
    return matches[0] if matches else None


def _extract_label(entry: dict) -> str | None:
    for key in ["label", "class", "suffix", "answer"]:
        if key in entry:
            return str(entry[key]).strip()

    for msg in reversed(entry.get("messages", [])):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "").strip()

    return None


def _extract_image_name(entry: dict) -> str | None:
    for key in ["image", "file_name", "filename"]:
        if key in entry:
            return Path(str(entry[key])).name

    for msg in entry.get("messages", []):
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url and not url.startswith("data:"):
                    return Path(url).name
            if part.get("type") == "image":
                return Path(str(part.get("image", ""))).name

    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and convert jersey number dataset.")
    parser.add_argument("--output", default="numbers/data/jersey_numbers", help="Output folder for converted dataset.")
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"), help="Roboflow API key (default: ROBOFLOW_API_KEY env var).")
    parser.add_argument("--keep-raw", action="store_true", help="Keep the raw downloaded files after conversion.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.api_key:
        raise SystemExit("No API key. Set ROBOFLOW_API_KEY in .env or pass --api-key.")

    out_dir = Path(args.output)
    raw_dir = out_dir.parent / "jersey_numbers_raw"

    print(f"Downloading {WORKSPACE}/{PROJECT} v{VERSION} ...")
    download(args.api_key, raw_dir)

    print("\nConverting JSONL -> ImageFolder structure ...")
    counts = convert(raw_dir, out_dir)

    print(f"\nDone. Dataset at: {out_dir}")
    for split, n in counts.items():
        print(f"  {split}: {n} images")

    if not args.keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)
        print(f"Removed raw download: {raw_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
