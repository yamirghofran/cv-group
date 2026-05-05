from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from scripts.augment_dataset import expand_split


def merge_val_test(src_dir: Path, dst_eval_dir: Path) -> int:
    dst_eval_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in ["val", "test"]:
        split_dir = src_dir / split
        if not split_dir.exists():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            dst_cls = dst_eval_dir / cls_dir.name
            dst_cls.mkdir(exist_ok=True)
            for img in cls_dir.iterdir():
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                dst = dst_cls / f"{split}_{img.name}"
                shutil.copy2(img, dst)
                total += 1
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare clean and augmented eval datasets by merging val+test."
    )
    parser.add_argument("--src", required=True, help="Stratified dataset root (jersey_numbers_split).")
    parser.add_argument("--dst-clean", required=True, help="Output root for clean eval dataset.")
    parser.add_argument("--dst-aug", required=True, help="Output root for augmented eval dataset.")
    parser.add_argument("--copies", type=int, default=9, help="Augmented copies per image (default 9).")
    args = parser.parse_args(argv)

    src = Path(args.src)
    dst_clean = Path(args.dst_clean) / "eval"
    dst_aug = Path(args.dst_aug) / "eval"

    # Merge val+test once (same images in both augmented and clean datasets)
    print("Merging val+test into clean eval...")
    n = merge_val_test(src, dst_clean)
    print(f"  {n} images -> {dst_clean}")

    # Augment the merged clean eval
    print(f"\nAugmenting clean eval ({args.copies} copies per image)...")
    n = expand_split(dst_clean, dst_aug, copies=args.copies)
    print(f"  {n} images -> {dst_aug}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
