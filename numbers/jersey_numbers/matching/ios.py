from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class OCRModel(Protocol):
    """Any of the three OCR backends (ResNet34, ResNet18, SmolVLM2) satisfies this."""

    def predict(self, bgr_image: np.ndarray) -> tuple[str, float]: ...
    def predict_batch(self, bgr_images: list[np.ndarray]) -> list[tuple[str, float]]: ...


def compute_ios(player_mask: np.ndarray, number_bbox: list[float]) -> float:
    """Intersection over Smaller Area: fraction of the number bbox covered by the player mask."""
    h, w = player_mask.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(number_bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    number_area = (x2 - x1) * (y2 - y1)
    player_area = int(np.count_nonzero(player_mask))
    smaller_area = min(number_area, player_area)
    if smaller_area == 0:
        return 0.0

    intersection = int(np.count_nonzero(player_mask[y1:y2, x1:x2]))
    return intersection / smaller_area


def compute_iou(player_mask: np.ndarray, number_bbox: list[float]) -> float:
    """Standard Intersection over Union for comparison with IoS."""
    h, w = player_mask.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(number_bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    number_area = (x2 - x1) * (y2 - y1)
    player_area = int(np.count_nonzero(player_mask))
    intersection = int(np.count_nonzero(player_mask[y1:y2, x1:x2]))
    union = player_area + number_area - intersection
    return intersection / union if union > 0 else 0.0


def _clamp_bbox(bbox: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x1 = int(round(max(0.0, min(float(bbox[0]), w))))
    y1 = int(round(max(0.0, min(float(bbox[1]), h))))
    x2 = int(round(max(0.0, min(float(bbox[2]), w))))
    y2 = int(round(max(0.0, min(float(bbox[3]), h))))
    return x1, y1, x2, y2


def load_mask(mask_path: str | Path) -> np.ndarray | None:
    return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)


def crop_number(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    """Crop the number region from a live frame using the detector bbox."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def match_frame(
    frame_id: int,
    number_detections: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    ios_threshold: float = 0.9,
    frame_image: np.ndarray | None = None,
    ocr_model: OCRModel | None = None,
) -> list[dict[str, Any]]:
    """Match number detections to player tracks for one live frame."""
    results: list[dict[str, Any]] = []

    loaded_masks = {track["track_id"]: load_mask(track["mask_path"]) for track in tracks}

    for det in number_detections:
        bbox = det["bbox_xyxy"]
        best_track_id: int | None = None
        best_ios = 0.0

        for track in tracks:
            mask = loaded_masks.get(track["track_id"])
            if mask is None:
                continue
            score = compute_ios(mask, bbox)
            if score > best_ios:
                best_ios = score
                best_track_id = int(track["track_id"])

        if best_track_id is None or best_ios < ios_threshold:
            continue

        match: dict[str, Any] = {
            "frame_id": frame_id,
            "track_id": best_track_id,
            "number_bbox_xyxy": bbox,
            "ios_score": best_ios,
            "detection_confidence": float(det.get("confidence", 0.0)),
            "predicted_number": None,
            "ocr_confidence": None,
        }

        if frame_image is not None and ocr_model is not None:
            number_crop = crop_number(frame_image, bbox)
            if number_crop is not None:
                predicted, conf = ocr_model.predict(number_crop)
                match["predicted_number"] = predicted
                match["ocr_confidence"] = conf

        results.append(match)

    return results
