from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from jersey_numbers.ocr.resnet18 import ResNet18OCR
from jersey_numbers.ocr.resnet34 import ResNetOCR
from jersey_numbers.ocr.smolvlm2 import RoboflowOCR


def _load_jsonl(path: Path) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _load_imagefolder(split_dir: Path) -> list[dict]:
    entries = []
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                entries.append({"image": str(img), "answer": cls_dir.name})
    return entries


def _load_split(data_dir: Path, split: str) -> list[dict] | None:
    jsonl_path = data_dir / f"{split}.jsonl"
    if jsonl_path.exists():
        return _load_jsonl(jsonl_path)
    split_dir = data_dir / split
    if split_dir.exists() and split_dir.is_dir():
        return _load_imagefolder(split_dir)
    return None


def _load_bgr(img_path: str) -> np.ndarray:
    bgr = cv2.imread(img_path)
    if bgr is None:
        bgr = np.array(Image.open(img_path).convert("RGB"))[..., ::-1].copy()
    return bgr


def _alignment_index(model_class_names: list[str], canonical: list[str]) -> list[int]:
    """Returns indices so that model_probs[alignment[i]] = canonical_probs[i]."""
    name_to_idx = {name: i for i, name in enumerate(model_class_names)}
    return [name_to_idx.get(cls, -1) for cls in canonical]


def build(
    data_dir: Path,
    output_dir: Path,
    resnet34_ckpt: Path,
    resnet18_ckpt: Path,
    roboflow_model: str,
    roboflow_api_key: str,
    splits: list[str],
    device_str: str = "auto",
) -> None:
    print("Loading models...")
    resnet34 = ResNetOCR(resnet34_ckpt, device_str=device_str)
    resnet18 = ResNet18OCR(resnet18_ckpt, device_str=device_str)
    smolvlm2 = RoboflowOCR(roboflow_model, api_key=roboflow_api_key)

    resnet34_classes = [resnet34.idx_to_class[i] for i in range(len(resnet34.idx_to_class))]
    resnet18_classes = [resnet18.idx_to_class[i] for i in range(len(resnet18.idx_to_class))]
    canonical = sorted(set(resnet34_classes) | set(resnet18_classes))

    align34 = _alignment_index(resnet34_classes, canonical)
    align18 = _alignment_index(resnet18_classes, canonical)

    print(f"Canonical class list: {len(canonical)} classes")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        entries = _load_split(data_dir, split)
        if entries is None:
            print(f"Skipping {split} — no JSONL or ImageFolder found in {data_dir}")
            continue

        print(f"\nProcessing {split} ({len(entries)} images)...")

        image_paths = []
        true_labels = []
        resnet34_probs_list = []
        resnet18_probs_list = []
        smolvlm2_probs_list = []

        for entry in tqdm(entries, desc=split):
            img_path = entry["image"]
            label = entry["answer"]
            bgr = _load_bgr(img_path)

            raw34, _ = resnet34.predict_proba(bgr)
            raw18, _ = resnet18.predict_proba(bgr)
            probs_vlm = smolvlm2.predict_proba(bgr, canonical)

            aligned34 = np.array([raw34[i] if i >= 0 else 0.0 for i in align34], dtype=np.float32)
            aligned18 = np.array([raw18[i] if i >= 0 else 0.0 for i in align18], dtype=np.float32)

            image_paths.append(img_path)
            true_labels.append(label)
            resnet34_probs_list.append(aligned34)
            resnet18_probs_list.append(aligned18)
            smolvlm2_probs_list.append(probs_vlm)

        out = output_dir / f"{split}.npz"
        np.savez(
            out,
            image_paths=np.array(image_paths),
            true_labels=np.array(true_labels),
            class_names=np.array(canonical),
            resnet34_probs=np.array(resnet34_probs_list),
            resnet18_probs=np.array(resnet18_probs_list),
            smolvlm2_probs=np.array(smolvlm2_probs_list),
        )
        print(f"Saved -> {out}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with all 3 models and save probability matrices.")
    parser.add_argument("--data", required=True, help="Dataset dir with val.jsonl and test.jsonl.")
    parser.add_argument("--output", required=True, help="Output dir for .npz files.")
    parser.add_argument("--resnet34", required=True, help="Path to resnet34.pth checkpoint.")
    parser.add_argument("--resnet18", required=True, help="Path to resnet18.pth checkpoint.")
    parser.add_argument("--roboflow-model", required=True, help="Roboflow model ID.")
    parser.add_argument("--roboflow-api-key", help="Roboflow API key (or set ROBOFLOW_API_KEY env var).")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], help="Splits to process (default: val test).")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    api_key = args.roboflow_api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Roboflow API key required. Pass --roboflow-api-key or set ROBOFLOW_API_KEY.")
    build(
        data_dir=Path(args.data),
        output_dir=Path(args.output),
        resnet34_ckpt=Path(args.resnet34),
        resnet18_ckpt=Path(args.resnet18),
        roboflow_model=args.roboflow_model,
        roboflow_api_key=api_key,
        splits=args.splits,
        device_str=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
