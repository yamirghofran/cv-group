from __future__ import annotations

from pathlib import Path

import numpy as np


class StackingEnsemble:
    """Meta-model that combines ResNet18, ResNet34, and SmolVLM2 probability vectors."""

    def __init__(
        self,
        clf,
        class_names: list[str],
        resnet34_classes: list[str],
        resnet18_classes: list[str],
    ) -> None:
        self._clf = clf
        self.class_names = class_names
        self._align34 = _alignment_index(resnet34_classes, class_names)
        self._align18 = _alignment_index(resnet18_classes, class_names)

    def predict(
        self,
        resnet34_probs: np.ndarray,
        resnet18_probs: np.ndarray,
        smolvlm2_probs: np.ndarray,
    ) -> tuple[str, float]:
        aligned34 = np.array([resnet34_probs[i] if i >= 0 else 0.0 for i in self._align34], dtype=np.float32)
        aligned18 = np.array([resnet18_probs[i] if i >= 0 else 0.0 for i in self._align18], dtype=np.float32)
        x = np.concatenate([aligned34, aligned18, smolvlm2_probs]).reshape(1, -1)
        pred_idx = int(self._clf.predict(x)[0])
        proba = self._clf.predict_proba(x)[0]
        return self.class_names[pred_idx], float(proba[pred_idx])

    def predict_proba(
        self,
        resnet34_probs: np.ndarray,
        resnet18_probs: np.ndarray,
        smolvlm2_probs: np.ndarray,
    ) -> np.ndarray:
        aligned34 = np.array([resnet34_probs[i] if i >= 0 else 0.0 for i in self._align34], dtype=np.float32)
        aligned18 = np.array([resnet18_probs[i] if i >= 0 else 0.0 for i in self._align18], dtype=np.float32)
        x = np.concatenate([aligned34, aligned18, smolvlm2_probs]).reshape(1, -1)
        return self._clf.predict_proba(x)[0]

    def save(self, path: Path) -> None:
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "clf": self._clf,
                "class_names": self.class_names,
                "resnet34_classes": [self.class_names[i] if i >= 0 else "" for i in self._align34],
                "resnet18_classes": [self.class_names[i] if i >= 0 else "" for i in self._align18],
            }, f)

    @classmethod
    def load(cls, path: Path) -> StackingEnsemble:
        import pickle
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        return cls(
            clf=data["clf"],
            class_names=data["class_names"],
            resnet34_classes=data["resnet34_classes"],
            resnet18_classes=data["resnet18_classes"],
        )


def _alignment_index(model_classes: list[str], canonical: list[str]) -> list[int]:
    name_to_idx = {name: i for i, name in enumerate(model_classes)}
    return [name_to_idx.get(cls, -1) for cls in canonical]
