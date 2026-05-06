from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS_DIR = Path("numbers/models")
OUTPUT_DIR = Path("numbers/models/plots")


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def plot_model_comparison(output_dir: Path) -> None:
    data = load_json(MODELS_DIR / "stacking/stacking_results.json")

    models = ["ResNet34", "ResNet18", "SmolVLM2\n(Roboflow)", "Hard Voting", "Stacking (LR)"]
    accuracies = [
        data["resnet34_accuracy"],
        data["resnet18_accuracy"],
        data["smolvlm2_accuracy"],
        data["hard_voting_accuracy"],
        data["stacking_accuracy"],
    ]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, [a * 100 for a in accuracies], color=colors, width=0.5, edgecolor="white")

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc * 100:.2f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Comparison — Non-Augmented Test Set", fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "model_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")


def plot_resnet_aug_vs_clean(output_dir: Path) -> None:
    r34_clean = load_json(MODELS_DIR / "eval/clean/eval_resnet_results.json")["overall_accuracy"]
    r34_aug = load_json(MODELS_DIR / "eval/aug/eval_resnet_results.json")["overall_accuracy"]
    r18_clean = load_json(MODELS_DIR / "eval18/clean/eval_resnet_results.json")["overall_accuracy"]
    r18_aug = load_json(MODELS_DIR / "eval18/aug/eval_resnet_results.json")["overall_accuracy"]

    x = np.arange(2)
    width = 0.35
    clean_vals = [r34_clean * 100, r18_clean * 100]
    aug_vals = [r34_aug * 100, r18_aug * 100]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_clean = ax.bar(x - width / 2, clean_vals, width, label="Clean", color="#2196F3", edgecolor="white")
    bars_aug = ax.bar(x + width / 2, aug_vals, width, label="Augmented", color="#FF9800", edgecolor="white")

    for bar, val in zip(list(bars_clean) + list(bars_aug), clean_vals + aug_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["ResNet34", "ResNet18"], fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("ResNet — Clean vs Augmented Evaluation", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "resnet_aug_vs_clean.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(OUTPUT_DIR)
    plot_resnet_aug_vs_clean(OUTPUT_DIR)


if __name__ == "__main__":
    main()
