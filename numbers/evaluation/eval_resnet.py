from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from jersey_numbers.ocr.resnet34 import load_checkpoint, _transforms


def evaluate(
    checkpoint_path: Path,
    data_dir: Path,
    split: str = "test",
    output_dir: Path | None = None,
    device_str: str = "auto",
) -> dict:
    model, idx_to_class, img_size = load_checkpoint(checkpoint_path, device_str)
    num_classes = len(idx_to_class)
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    dataset = datasets.ImageFolder(split_dir, transform=_transforms(img_size, augment=False))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    from jersey_numbers.ocr.resnet34 import resolve_device
    device = resolve_device(device_str)
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    overall_acc = float((all_preds == all_labels).mean())
    print(f"\nOverall accuracy on {split}: {overall_acc:.4f} ({overall_acc * 100:.2f}%)")

    # Per-class accuracy + imbalance analysis
    class_names = [idx_to_class[i] for i in range(num_classes)]
    counts = np.bincount(all_labels, minlength=num_classes)
    per_class_correct = np.zeros(num_classes, dtype=int)
    for pred, label in zip(all_preds, all_labels):
        if pred == label:
            per_class_correct[label] += 1

    per_class_acc = np.where(counts > 0, per_class_correct / np.maximum(counts, 1), float("nan"))

    # Macro accuracy: mean per-class (equal weight per class regardless of size)
    macro_acc = float(np.nanmean(per_class_acc))
    # Weighted accuracy: inverse-frequency weighted (rare classes count more)
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

    # Flag imbalanced classes (< 1% of dataset)
    total = len(all_labels)
    rare = [class_names[i] for i in range(num_classes) if counts[i] > 0 and counts[i] / total < 0.01]
    if rare:
        print(f"\nRare classes (<1% of eval set): {rare}")

    # Confusion matrix
    _plot_confusion_matrix(all_labels, all_preds, class_names, output_dir)

    results = {
        "overall_accuracy": overall_acc,
        "macro_accuracy": macro_acc,
        "weighted_accuracy": weighted_acc,
        "per_class": {
            class_names[i]: {
                "count": int(counts[i]),
                "accuracy": float(per_class_acc[i]) if counts[i] > 0 else None,
            }
            for i in range(num_classes)
        },
    }

    if output_dir:
        out = output_dir / "eval_resnet_results.json"
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
        print("skipping confusion matrix plot.")
        return

    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true, pred] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(max(10, n // 3), max(8, n // 3)))
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
    ax.set_title("ResNet-32 Jersey Number Confusion Matrix (row-normalised)")
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
    parser = argparse.ArgumentParser(description="Evaluate ResNet-32 jersey number classifier.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pth checkpoint.")
    parser.add_argument("--data", required=True, help="Dataset root with train/val/test subdirs.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "eval"])
    parser.add_argument("--output-dir", help="Directory for confusion matrix PNG and JSON results.")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    evaluate(
        checkpoint_path=Path(args.checkpoint),
        data_dir=Path(args.data),
        split=args.split,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        device_str=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
