from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from scripts.augment_dataset import augment
from jersey_numbers.ocr.smolvlm2 import PROMPT


def merge_splits(src_dir: Path, dst_dir: Path) -> list[dict]:
    entries = []
    for split in ["val", "test"]:
        jsonl = src_dir / f"{split}.jsonl"
        if not jsonl.exists():
            continue
        with jsonl.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

    dst_dir.mkdir(parents=True, exist_ok=True)
    out = dst_dir / "eval.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"  {len(entries)} images -> {out}")
    return entries


def build_augmented(entries: list[dict], dst_dir: Path, copies: int, seed: int = 42) -> None:
    from PIL import Image as PILImage

    rng = random.Random(seed)
    images_dir = dst_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    aug_entries = []
    for entry in entries:
        img_path = Path(entry["image"])
        label = entry["answer"]
        img = PILImage.open(img_path).convert("RGB")

        dst_orig = images_dir / img_path.name
        shutil.copy2(img_path, dst_orig)
        aug_entries.append({"image": str(dst_orig.resolve()), "prompt": PROMPT, "answer": label})

        for i in range(copies):
            aug_img = augment(img.copy(), rng)
            aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            dst_aug = images_dir / aug_name
            aug_img.save(dst_aug)
            aug_entries.append({"image": str(dst_aug.resolve()), "prompt": PROMPT, "answer": label})

    out = dst_dir / "eval.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for entry in aug_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"  {len(aug_entries)} images -> {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare clean and augmented eval JSONL for SmolVLM2 from Roboflow val+test splits."
    )
    parser.add_argument("--src", required=True, help="jersey_numbers dir with val.jsonl and test.jsonl.")
    parser.add_argument("--dst-clean", required=True, help="Output dir for clean eval JSONL.")
    parser.add_argument("--dst-aug", required=True, help="Output dir for augmented eval JSONL.")
    parser.add_argument("--copies", type=int, default=9, help="Augmented copies per image (default 9).")
    args = parser.parse_args(argv)

    src = Path(args.src)
    dst_clean = Path(args.dst_clean)
    dst_aug = Path(args.dst_aug)

    print("Merging val+test into clean eval...")
    entries = merge_splits(src, dst_clean)

    print(f"\nAugmenting clean eval ({args.copies} copies per image)...")
    build_augmented(entries, dst_aug, copies=args.copies)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
