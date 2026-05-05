from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    import PIL.ImageEnhance as E

    # Rotation
    img = img.rotate(rng.uniform(-25, 25), expand=False, fillcolor=(0, 0, 0))

    # Perspective warp
    if rng.random() < 0.6:
        w, h = img.size
        skew = rng.uniform(0.05, 0.2)
        dx, dy = int(w * skew), int(h * skew)
        coeffs = _perspective_coeffs(
            [(0, 0), (w, 0), (w, h), (0, h)],
            [
                (rng.randint(0, dx), rng.randint(0, dy)),
                (w - rng.randint(0, dx), rng.randint(0, dy)),
                (w - rng.randint(0, dx), h - rng.randint(0, dy)),
                (rng.randint(0, dx), h - rng.randint(0, dy)),
            ],
        )
        img = img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    # Brightness / contrast / saturation / hue
    img = E.Brightness(img).enhance(rng.uniform(0.5, 1.8))
    img = E.Contrast(img).enhance(rng.uniform(0.5, 1.8))
    img = E.Color(img).enhance(rng.uniform(0.4, 1.6))
    img = E.Sharpness(img).enhance(rng.uniform(0.3, 2.0))

    # Blur
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.0)))

    # Grayscale
    if rng.random() < 0.15:
        img = img.convert("L").convert("RGB")

    # Random erasing (simulate occlusion)
    if rng.random() < 0.4:
        arr = np.array(img)
        h, w = arr.shape[:2]
        eh = rng.randint(int(h * 0.05), int(h * 0.3))
        ew = rng.randint(int(w * 0.05), int(w * 0.3))
        y0 = rng.randint(0, h - eh)
        x0 = rng.randint(0, w - ew)
        arr[y0:y0+eh, x0:x0+ew] = rng.randint(0, 255)
        img = Image.fromarray(arr)

    # Noise
    if rng.random() < 0.3:
        arr = np.array(img, dtype=np.float32)
        arr += np.random.normal(0, rng.uniform(5, 20), arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def _perspective_coeffs(src, dst):
    matrix = []
    for (x, y), (X, Y) in zip(dst, src):
        matrix += [[x, y, 1, 0, 0, 0, -X*x, -X*y],
                   [0, 0, 0, x, y, 1, -Y*x, -Y*y]]
    A = np.array(matrix, dtype=np.float64)
    b = np.array([X for (X, _) in src] + [Y for (_, Y) in src], dtype=np.float64).reshape(-1)
    # least squares solve
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return result.tolist()


def expand_split(src_dir: Path, dst_dir: Path, copies: int, seed: int = 42) -> int:
    rng = random.Random(seed)
    total = 0
    class_dirs = sorted(p for p in src_dir.iterdir() if p.is_dir())
    for cls_dir in class_dirs:
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        if not images:
            continue
        out_cls = dst_dir / cls_dir.name
        out_cls.mkdir(parents=True, exist_ok=True)
        for img_path in images:
            # Always copy original
            img = Image.open(img_path).convert("RGB")
            img.save(out_cls / img_path.name)
            total += 1
            # Generate augmented copies
            for i in range(copies):
                aug = augment(img.copy(), rng)
                stem = img_path.stem
                aug.save(out_cls / f"{stem}_aug{i}{img_path.suffix}")
                total += 1
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline augmentation: expand training set.")
    parser.add_argument("--src", required=True, help="Source dataset root (with train/val/test dirs).")
    parser.add_argument("--dst", required=True, help="Output dataset root.")
    parser.add_argument("--copies", type=int, default=9, help="Augmented copies per image (default 9 → 10x dataset).")
    args = parser.parse_args(argv)

    src = Path(args.src)
    dst = Path(args.dst)

    for split in ["train", "val", "test"]:
        split_src = src / split
        split_dst = dst / split
        if not split_src.exists():
            continue
        # Only augment train; copy val/test as-is
        copies = args.copies if split == "train" else 0
        print(f"Processing {split} (copies={copies})...")
        n = expand_split(split_src, split_dst, copies=copies)
        print(f"  → {n} images in {split_dst}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
