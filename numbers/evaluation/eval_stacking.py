from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jersey_numbers.ensemble.stacking import StackingEnsemble


def evaluate(
    ensemble_path: Path,
    stacking_dir: Path,
    output_dir: Path | None = None,
) -> dict:
    ensemble = StackingEnsemble.load(ensemble_path)
    data = np.load(stacking_dir / "test.npz", allow_pickle=True)

    class_names = list(data["class_names"])
    true_labels = list(data["true_labels"])
    resnet34_probs = data["resnet34_probs"]
    resnet18_probs = data["resnet18_probs"]
    smolvlm2_probs = data["smolvlm2_probs"]

    valid = [(i, l) for i, l in enumerate(true_labels) if l in class_names]
    indices = [i for i, _ in valid]
    labels = [l for _, l in valid]

    preds = []
    for i in indices:
        pred, _ = ensemble.predict(resnet34_probs[i], resnet18_probs[i], smolvlm2_probs[i])
        preds.append(pred)

    overall_acc = float(sum(p == l for p, l in zip(preds, labels)) / len(labels))
    print(f"\nOverall accuracy: {overall_acc:.4f} ({overall_acc * 100:.2f}%)")

    label_array = np.array([class_names.index(l) for l in labels])
    pred_array = np.array([class_names.index(p) for p in preds])

    counts = np.bincount(label_array, minlength=len(class_names))
    per_class_correct = np.zeros(len(class_names), dtype=int)
    for true, pred in zip(label_array, pred_array):
        if true == pred:
            per_class_correct[true] += 1
    per_class_acc = np.where(counts > 0, per_class_correct / np.maximum(counts, 1), float("nan"))
    macro_acc = float(np.nanmean(per_class_acc))

    print(f"Macro  accuracy (mean per-class): {macro_acc:.4f} ({macro_acc * 100:.2f}%)")
    print(f"\n{'Class':>6} {'Count':>6} {'Accuracy':>10}")
    print("-" * 26)
    for i, cls in enumerate(class_names):
        acc_str = f"{per_class_acc[i]:.4f}" if counts[i] > 0 else "   N/A"
        print(f"{cls:>6} {counts[i]:>6} {acc_str:>10}")

    _plot_confusion_matrix(label_array, pred_array, class_names, output_dir)

    results = {
        "overall_accuracy": overall_acc,
        "macro_accuracy": macro_acc,
        "n_samples": len(labels),
        "per_class": {
            class_names[i]: {
                "count": int(counts[i]),
                "accuracy": float(per_class_acc[i]) if counts[i] > 0 else None,
            }
            for i in range(len(class_names))
        },
    }
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "eval_stacking_results.json"
        with out.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results -> {out}")

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
    ax.set_title("Stacking Ensemble Confusion Matrix (row-normalised)")
    plt.tight_layout()

    if output_dir:
        out = output_dir / "confusion_matrix.png"
        plt.savefig(out, dpi=150)
        print(f"Saved confusion matrix -> {out}")
    else:
        plt.show()
    plt.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate stacking ensemble on test.npz.")
    parser.add_argument("--ensemble", required=True, help="Path to stacking_ensemble.pkl.")
    parser.add_argument("--stacking-dir", required=True, help="Dir with test.npz.")
    parser.add_argument("--output-dir", help="Directory for confusion matrix PNG and JSON results.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    evaluate(
        ensemble_path=Path(args.ensemble),
        stacking_dir=Path(args.stacking_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
