from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from jersey_numbers.ensemble.stacking import StackingEnsemble


def _load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _individual_accuracy(probs: np.ndarray, true_idx: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == true_idx).mean())


def _hard_vote_accuracy(probs_list: list[np.ndarray], true_idx: np.ndarray) -> float:
    votes = np.stack([p.argmax(axis=1) for p in probs_list], axis=1)
    from scipy.stats import mode
    majority = mode(votes, axis=1, keepdims=False).mode
    return float((majority == true_idx).mean())


def train(
    stacking_dir: Path,
    output_dir: Path,
    C: float = 1.0,
    max_iter: int = 1000,
) -> None:
    meta_train_path = stacking_dir / "val.npz"
    meta_test_path = stacking_dir / "test.npz"

    if not meta_train_path.exists():
        raise FileNotFoundError(f"Meta-train file not found: {meta_train_path}")
    if not meta_test_path.exists():
        raise FileNotFoundError(f"Meta-test file not found: {meta_test_path}")

    train_data = _load_npz(meta_train_path)
    test_data = _load_npz(meta_test_path)

    class_names = list(train_data["class_names"])
    le = LabelEncoder()
    le.fit(class_names)

    y_train = le.transform(train_data["true_labels"])
    y_test = le.transform(test_data["true_labels"])

    X_train = np.concatenate([
        train_data["resnet34_probs"],
        train_data["resnet18_probs"],
        train_data["smolvlm2_probs"],
    ], axis=1)

    X_test = np.concatenate([
        test_data["resnet34_probs"],
        test_data["resnet18_probs"],
        test_data["smolvlm2_probs"],
    ], axis=1)

    print(f"Meta-train: {X_train.shape[0]} samples | Meta-test: {X_test.shape[0]} samples")
    print(f"Feature dim: {X_train.shape[1]} ({len(class_names)} classes × 3 models)")

    print("\nTraining Logistic Regression meta-model...")
    clf = LogisticRegression(C=C, max_iter=max_iter, multi_class="multinomial", solver="lbfgs")
    clf.fit(X_train, y_train)

    ensemble = StackingEnsemble(
        clf=clf,
        class_names=class_names,
        resnet34_classes=list(train_data["class_names"]),
        resnet18_classes=list(train_data["class_names"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(output_dir / "stacking_ensemble.pkl")

    stacking_preds = clf.predict(X_test)
    stacking_acc = accuracy_score(y_test, stacking_preds)

    resnet34_acc = _individual_accuracy(test_data["resnet34_probs"], le.transform(test_data["true_labels"]))
    resnet18_acc = _individual_accuracy(test_data["resnet18_probs"], le.transform(test_data["true_labels"]))
    smolvlm2_acc = _individual_accuracy(test_data["smolvlm2_probs"], le.transform(test_data["true_labels"]))
    vote_acc = _hard_vote_accuracy(
        [test_data["resnet34_probs"], test_data["resnet18_probs"], test_data["smolvlm2_probs"]],
        le.transform(test_data["true_labels"]),
    )

    print("\n" + "=" * 45)
    print(f"{'Model':<25} {'Accuracy':>10}")
    print("-" * 45)
    print(f"{'ResNet34':<25} {resnet34_acc:>10.4f} ({resnet34_acc*100:.2f}%)")
    print(f"{'ResNet18':<25} {resnet18_acc:>10.4f} ({resnet18_acc*100:.2f}%)")
    print(f"{'SmolVLM2 (Roboflow)':<25} {smolvlm2_acc:>10.4f} ({smolvlm2_acc*100:.2f}%)")
    print(f"{'Hard voting':<25} {vote_acc:>10.4f} ({vote_acc*100:.2f}%)")
    print(f"{'Stacking (LR)':<25} {stacking_acc:>10.4f} ({stacking_acc*100:.2f}%)")
    print("=" * 45)

    results = {
        "resnet34_accuracy": resnet34_acc,
        "resnet18_accuracy": resnet18_acc,
        "smolvlm2_accuracy": smolvlm2_acc,
        "hard_voting_accuracy": vote_acc,
        "stacking_accuracy": stacking_acc,
        "meta_train_samples": int(X_train.shape[0]),
        "meta_test_samples": int(X_test.shape[0]),
        "feature_dim": int(X_train.shape[1]),
        "lr_C": C,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "stacking_results.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results -> {out}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LR meta-model on stacked model probabilities.")
    parser.add_argument("--stacking-dir", required=True, help="Dir with val.npz and test.npz from build_stacking_dataset.py.")
    parser.add_argument("--output-dir", required=True, help="Output dir for results JSON.")
    parser.add_argument("--C", type=float, default=1.0, help="LR regularization strength (default 1.0).")
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    train(
        stacking_dir=Path(args.stacking_dir),
        output_dir=Path(args.output_dir),
        C=args.C,
        max_iter=args.max_iter,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
