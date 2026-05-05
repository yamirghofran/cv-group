from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def resplit(src_dir: Path, dst_dir: Path, val_frac: float, test_frac: float, seed: int) -> None:
    rng = random.Random(seed)

    # Collect all images grouped by class
    by_class: dict[str, list[Path]] = defaultdict(list)
    for split in ["train", "val", "test"]:
        split_dir = src_dir / split
        if not split_dir.exists():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            images = [p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            by_class[cls_dir.name].extend(images)

    print(f"Found {len(by_class)} classes, {sum(len(v) for v in by_class.values())} total images.")

    for split in ["train", "val", "test"]:
        (dst_dir / split).mkdir(parents=True, exist_ok=True)

    skipped = []
    for cls_name, images in sorted(by_class.items()):
        rng.shuffle(images)
        n = len(images)

        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))

        if n < 3:
            skipped.append((cls_name, n))
            continue

        test_imgs = images[:n_test]
        val_imgs = images[n_test:n_test + n_val]
        train_imgs = images[n_test + n_val:]

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            cls_dst = dst_dir / split_name / cls_name
            cls_dst.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, cls_dst / img.name)

        print(f"  {cls_name:>4}: {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test")

    if skipped:
        print(f"\nSkipped {len(skipped)} classes with <3 images: {[c for c,_ in skipped]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-split dataset with stratified sampling per class.")
    parser.add_argument("--src", required=True, help="Source dataset root (with train/val/test subdirs).")
    parser.add_argument("--dst", required=True, help="Output dataset root.")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction for val (default 0.1).")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction for test (default 0.1).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    resplit(
        src_dir=Path(args.src),
        dst_dir=Path(args.dst),
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print(f"\nDone. New dataset at: {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
