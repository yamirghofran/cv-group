from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

import cv2
import requests

from .schemas import CourtKeypointRecord, DetectionRecord
from .utils import clamp_bbox_xyxy


@dataclass(frozen=True)
class RoboflowHostedSettings:
    api_url: str = "https://detect.roboflow.com"
    model_id: str = "basketball-player-detection-3-ycjdo"
    model_version: int = 13
    confidence: int = 40
    overlap: int = 30
    timeout_seconds: int = 60
    api_key_env: str = "ROBOFLOW_API_KEY"


class RoboflowHostedClient:
    """Small REST client for Roboflow Hosted API object detection."""

    def __init__(self, settings: RoboflowHostedSettings, api_key: str | None = None) -> None:
        self.settings = settings
        self.api_key = api_key or os.getenv(settings.api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"Missing Roboflow API key. Set {settings.api_key_env} or pass --mock-response for smoke tests."
            )

    @property
    def endpoint(self) -> str:
        return f"{self.settings.api_url.rstrip('/')}/{self.settings.model_id}/{self.settings.model_version}"

    def infer_frame(self, frame, frame_name: str) -> dict[str, Any]:
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError(f"Could not JPEG-encode frame for Roboflow: {frame_name}")
        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        params = {
            "api_key": self.api_key,
            "confidence": self.settings.confidence,
            "overlap": self.settings.overlap,
            "format": "json",
            "name": frame_name,
        }
        response = requests.post(
            self.endpoint,
            params=params,
            data=image_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Roboflow response was not a JSON object")
        return payload


def normalize_roboflow_predictions(
    payload: dict[str, Any],
    frame_width: int,
    frame_height: int,
    source: str = "roboflow",
) -> list[DetectionRecord]:
    predictions = payload.get("predictions", [])
    if not isinstance(predictions, list):
        raise ValueError("Roboflow payload field 'predictions' must be a list")

    records: list[DetectionRecord] = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        class_name = prediction.get("class") or prediction.get("class_name")
        if class_name is None:
            continue
        x = float(prediction.get("x", 0.0))
        y = float(prediction.get("y", 0.0))
        width = float(prediction.get("width", 0.0))
        height = float(prediction.get("height", 0.0))
        bbox = clamp_bbox_xyxy(
            [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0],
            frame_width,
            frame_height,
        )
        class_id = prediction.get("class_id")
        if class_id is not None:
            try:
                class_id = int(class_id)
            except (TypeError, ValueError):
                class_id = None
        confidence = float(prediction.get("confidence", 0.0))
        records.append(
            DetectionRecord(
                class_name=str(class_name),
                class_id=class_id,
                confidence=confidence,
                bbox_xyxy=bbox,
                source=source,
            )
        )
    return records


def normalize_roboflow_keypoints(payload: dict[str, Any], source: str = "roboflow") -> list[CourtKeypointRecord]:
    predictions = payload.get("predictions", [])
    if not isinstance(predictions, list):
        raise ValueError("Roboflow payload field 'predictions' must be a list")

    keypoints: list[CourtKeypointRecord] = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        for keypoint in prediction.get("keypoints", []) or []:
            if not isinstance(keypoint, dict):
                continue
            class_id = keypoint.get("class_id")
            if class_id is None:
                continue
            keypoints.append(
                CourtKeypointRecord(
                    class_name=str(keypoint.get("class_name") or keypoint.get("class") or class_id),
                    class_id=int(class_id),
                    confidence=float(keypoint.get("confidence", 0.0)),
                    image_xy=[float(keypoint.get("x", 0.0)), float(keypoint.get("y", 0.0))],
                    source=source,
                )
            )
    return keypoints
