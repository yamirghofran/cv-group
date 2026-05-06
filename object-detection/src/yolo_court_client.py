from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .device import resolve_device


@dataclass(frozen=True)
class YoloCourtKeypointSettings:
    weights: str
    confidence: float = 0.25
    device: str = "auto"


class YoloCourtKeypointClient:
    """Local Ultralytics YOLO pose client that returns Roboflow-like keypoints."""

    def __init__(self, settings: YoloCourtKeypointSettings) -> None:
        weights_path = Path(settings.weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                "YOLO court keypoint weights not found: "
                f"{weights_path}. Download the shared checkpoint with: "
                "uvx hf download AnzeZ/basketball-court-yolo11m-pose best.pt "
                "--revision 4499465f27dfa29ffa6fe621d71124394375c1d8 "
                "--local-dir object-detection/models/court_yolo11m_pose_v19"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Run: uv sync --extra train") from exc

        self.settings = settings
        self.device = resolve_device(settings.device)
        self.model = YOLO(str(weights_path))

    def infer_frame(self, frame, frame_name: str) -> dict[str, Any]:
        results = self.model.predict(frame, conf=self.settings.confidence, device=self.device, verbose=False)
        if not results:
            return {"predictions": []}

        result = results[0]
        if result.keypoints is None or result.keypoints.xy is None:
            return {"predictions": []}

        keypoints_xy = result.keypoints.xy.cpu().numpy()
        keypoint_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
        box_conf = result.boxes.conf.cpu().numpy() if result.boxes is not None and result.boxes.conf is not None else None
        class_ids = result.boxes.cls.cpu().numpy() if result.boxes is not None and result.boxes.cls is not None else None

        if len(keypoints_xy) == 0:
            return {"predictions": []}
        instance_indexes = [int(box_conf.argmax())] if box_conf is not None and len(box_conf) else [0]

        predictions: list[dict[str, Any]] = []
        for instance_index in instance_indexes:
            instance_keypoints = keypoints_xy[instance_index]
            instance_confidence = float(box_conf[instance_index]) if box_conf is not None and len(box_conf) > instance_index else 1.0
            class_id = int(class_ids[instance_index]) if class_ids is not None and len(class_ids) > instance_index else 0
            class_name = str(self.model.names.get(class_id, "court")) if isinstance(self.model.names, dict) else "court"
            keypoints: list[dict[str, Any]] = []
            for keypoint_id, (x, y) in enumerate(instance_keypoints):
                confidence = (
                    float(keypoint_conf[instance_index][keypoint_id])
                    if keypoint_conf is not None
                    and len(keypoint_conf) > instance_index
                    and len(keypoint_conf[instance_index]) > keypoint_id
                    else instance_confidence
                )
                keypoints.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "class": str(keypoint_id),
                        "class_name": str(keypoint_id),
                        "class_id": int(keypoint_id),
                        "confidence": confidence,
                    }
                )
            predictions.append(
                {
                    "class": class_name,
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": instance_confidence,
                    "keypoints": keypoints,
                    "source_frame": frame_name,
                }
            )
        return {"predictions": predictions}
