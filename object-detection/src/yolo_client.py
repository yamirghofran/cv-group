from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class YoloLocalSettings:
    weights_path: str = "object-detection/finetuning/runs/basketball_yolo11s/weights/best.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    device: str | None = None


class YoloLocalClient:
    """Local Ultralytics YOLO detector that mirrors RoboflowHostedClient.infer_frame."""

    def __init__(self, settings: YoloLocalSettings) -> None:
        self.settings = settings
        weights = Path(settings.weights_path)
        if not weights.exists():
            raise RuntimeError(f"YOLO weights not found: {weights}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: uv sync"
            ) from exc
        self._model = YOLO(str(weights))
        names = self._model.names
        if isinstance(names, dict):
            self._class_names = {int(k): str(v) for k, v in names.items()}
        else:
            self._class_names = {int(i): str(n) for i, n in enumerate(names)}

    @property
    def model_identifier(self) -> str:
        return f"yolo:{Path(self.settings.weights_path).stem}"

    def infer_frame(self, frame: np.ndarray, frame_name: str) -> dict[str, Any]:
        results = self._model.predict(
            source=frame,
            imgsz=self.settings.imgsz,
            conf=self.settings.conf,
            iou=self.settings.iou,
            device=self.settings.device,
            verbose=False,
        )
        predictions: list[dict[str, Any]] = []
        if not results:
            return {"predictions": predictions}
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return {"predictions": predictions}

        xyxy = _to_numpy(boxes.xyxy)
        confs = _to_numpy(boxes.conf)
        clses = _to_numpy(boxes.cls)
        for box, conf, cls_id in zip(xyxy, confs, clses):
            x1, y1, x2, y2 = (float(v) for v in box)
            class_id = int(cls_id)
            predictions.append(
                {
                    "class": self._class_names.get(class_id, str(class_id)),
                    "class_id": class_id,
                    "confidence": float(conf),
                    "x": (x1 + x2) / 2.0,
                    "y": (y1 + y2) / 2.0,
                    "width": x2 - x1,
                    "height": y2 - y1,
                }
            )
        return {"predictions": predictions}


def _to_numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)
