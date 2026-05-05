from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from jersey_numbers.ocr.smolvlm2 import RoboflowOCR, SmolVLM2OCR


def _load_jsonl(jsonl_path: Path) -> tuple[list[str], list[str]]:
    images, labels = [], []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            images.append(entry["image"])
            labels.append(entry["answer"])
    return images, labels


def _load_imagefolder(split_dir: Path) -> tuple[list[str], list[str]]:
    images, labels = [], []
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                images.append(str(img))
                labels.append(cls_dir.name)
    return images, labels


def _load_data(data_dir: Path, split: str) -> tuple[list[str], list[str]]:
    jsonl_path = data_dir / f"{split}.jsonl"
    if jsonl_path.exists():
        return _load_jsonl(jsonl_path)
    split_dir = data_dir / split
    if split_dir.exists():
        return _load_imagefolder(split_dir)
    raise FileNotFoundError(f"No {split}.jsonl or {split}/ directory found in {data_dir}")


def evaluate(
    data_dir: Path,
    split: str = "test",
    output_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    device_str: str = "auto",
    roboflow_model: str | None = None,
    roboflow_api_key: str | None = None,
) -> dict:
    if roboflow_model:
        if not roboflow_api_key:
            raise ValueError("--roboflow-api-key is required when using --roboflow-model")
        model = RoboflowOCR(roboflow_model, api_key=roboflow_api_key)
    elif checkpoint_path:
        model = SmolVLM2OCR(checkpoint_path, device_str=device_str)
    else:
        raise ValueError("Provide either --checkpoint or --roboflow-model")

    image_paths, true_labels = _load_data(data_dir, split)
    class_names = sorted(set(true_labels))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    num_classes = len(class_names)

    all_preds: list[int] = []
    all_labels: list[int] = []

    for img_path, true_label in tqdm(zip(image_paths, true_labels), total=len(image_paths), desc=f"Evaluating {split}"):
        bgr = cv2.imread(img_path)
        if bgr is None:
            bgr = np.array(Image.open(img_path).convert("RGB"))[..., ::-1]

        predicted_str, _ = model.predict(bgr)

        all_labels.append(class_to_idx[true_label])
        all_preds.append(class_to_idx.get(predicted_str, -1))

    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)

    overall_acc = float((all_preds_arr == all_labels_arr).mean())
    print(f"\nOverall accuracy on {split}: {overall_acc:.4f} ({overall_acc * 100:.2f}%)")

    counts = np.bincount(all_labels_arr, minlength=num_classes)
    per_class_correct = np.zeros(num_classes, dtype=int)
    for pred, label in zip(all_preds_arr, all_labels_arr):
        if pred == label:
            per_class_correct[label] += 1

    per_class_acc = np.where(counts > 0, per_class_correct / np.maximum(counts, 1), float("nan"))

    macro_acc = float(np.nanmean(per_class_acc))
    inv_freq = np.where(counts > 0, 1.0 / counts, 0.0)
    inv_freq = inv_freq / inv_freq.sum()
    weighted_acc = float(np.nansum(np.where(counts > 0, per_class_acc * inv_freq, 0.0)))

    print(f"Macro  accuracy (mean per-class):       {macro_acc:.4f} ({macro_acc * 100:.2f}%)")
    print(f"Weighted accuracy (inv-freq per-class): {weighted_acc:.4f} ({weighted_acc * 100:.2f}%)")

    print(f"\n{'Class':>6} {'Count':>6} {'Accuracy':>10}")
    print("-" * 26)
    for i in range(num_classes):
        acc_str = f"{per_class_acc[i]:.4f}" if counts[i] > 0 else "   N/A"
        print(f"{class_names[i]:>6} {counts[i]:>6} {acc_str:>10}")

    total = len(all_labels)
    rare = [class_names[i] for i in range(num_classes) if counts[i] > 0 and counts[i] / total < 0.01]
    if rare:
        print(f"\nRare classes (<1% of test set): {rare}")

    hallucinated = int((all_preds_arr == -1).sum())
    if hallucinated:
        print(f"Hallucinated (output not in known classes): {hallucinated} / {total}")

    _plot_confusion_matrix(all_labels_arr, all_preds_arr, class_names, output_dir)

    results = {
        "overall_accuracy": overall_acc,
        "macro_accuracy": macro_acc,
        "weighted_accuracy": weighted_acc,
        "hallucinated_count": hallucinated,
        "per_class": {
            class_names[i]: {
                "count": int(counts[i]),
                "accuracy": float(per_class_acc[i]) if counts[i] > 0 else None,
            }
            for i in range(num_classes)
        },
    }

    if output_dir:
        out = output_dir / "eval_smolvlm2_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results -> {out}")

    return results


def _plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    output_dir: Path | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed — skipping confusion matrix plot.")
        return

    mask = preds >= 0
    labels_clean = labels[mask]
    preds_clean = preds[mask]

    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(labels_clean, preds_clean):
        cm[true, pred] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    _, ax = plt.subplots(figsize=(max(10, n // 3), max(8, n // 3)))
    sns.heatmap(
        cm_norm,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.3,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("SmolVLM2 Jersey Number Confusion Matrix (row-normalised)")
    plt.tight_layout()

    if output_dir:
        out = output_dir / "confusion_matrix.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150)
        print(f"Saved confusion matrix -> {out}")
    else:
        plt.show()
    plt.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SmolVLM2 jersey number classifier.")
    parser.add_argument("--checkpoint", help="Path to fine-tuned LoRA checkpoint directory.")
    parser.add_argument("--roboflow-model", help="Roboflow model ID (e.g. basketball-jersey-numbers-ocr/3).")
    parser.add_argument("--roboflow-api-key", help="Roboflow API key (or set ROBOFLOW_API_KEY env var).")
    parser.add_argument("--data", required=True, help="Dataset root (JSONL dir or ImageFolder root).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "eval"])
    parser.add_argument("--output-dir", help="Directory for confusion matrix PNG and JSON results.")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    api_key = args.roboflow_api_key or os.getenv("ROBOFLOW_API_KEY")
    evaluate(
        data_dir=Path(args.data),
        split=args.split,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        device_str=args.device,
        roboflow_model=args.roboflow_model,
        roboflow_api_key=api_key,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
