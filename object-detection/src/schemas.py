from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PipelineModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectionRecord(PipelineModel):
    class_name: str
    class_id: int | None = None
    confidence: float
    bbox_xyxy: list[float]
    source: str = "roboflow"

    @field_validator("bbox_xyxy")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("bbox_xyxy must contain exactly four values")
        x1, y1, x2, y2 = value
        if x2 < x1 or y2 < y1:
            raise ValueError("bbox_xyxy must be ordered as x1, y1, x2, y2")
        return [float(v) for v in value]


class DetectionFrame(PipelineModel):
    frame_id: int
    timestamp_sec: float
    detections: list[DetectionRecord]


class DetectionOutput(PipelineModel):
    video: str
    fps: float
    width: int
    height: int
    model_id: str
    model_version: int
    frame_stride: int
    frames: list[DetectionFrame]
    classes_seen: list[str] = Field(default_factory=list)


class CorrectionBox(PipelineModel):
    track_hint: str | int | None = None
    bbox_xyxy: list[float]
    class_name: str = "player"
    note: str | None = None

    @field_validator("bbox_xyxy")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("bbox_xyxy must contain exactly four values")
        return [float(v) for v in value]


class Frame0Corrections(PipelineModel):
    video: str | None = None
    frame_id: int = 0
    boxes: list[CorrectionBox]

    @field_validator("frame_id")
    @classmethod
    def validate_frame_id(cls, value: int) -> int:
        if value != 0:
            raise ValueError("Frame-0 correction files must use frame_id 0")
        return value


class TrackRecord(PipelineModel):
    track_id: int
    bbox_xyxy: list[float]
    mask_path: str
    area: int
    score: float | None = None
    source_prompt_bbox_xyxy: list[float]


class TrackFrame(PipelineModel):
    frame_id: int
    timestamp_sec: float
    tracks: list[TrackRecord]


class TrackOutput(PipelineModel):
    video: str
    fps: float
    width: int
    height: int
    sam2_checkpoint: str | None = None
    mask_cleanup: dict[str, Any]
    frames: list[TrackFrame]


class CourtKeypointRecord(PipelineModel):
    class_name: str
    class_id: int
    confidence: float
    image_xy: list[float]
    source: str = "roboflow"

    @field_validator("image_xy")
    @classmethod
    def validate_xy(cls, value: list[float]) -> list[float]:
        if len(value) != 2:
            raise ValueError("image_xy must contain exactly two values")
        return [float(v) for v in value]


class CourtKeypointFrame(PipelineModel):
    frame_id: int
    timestamp_sec: float
    keypoints: list[CourtKeypointRecord]
    valid: bool
    reason: str | None = None


class CourtKeypointOutput(PipelineModel):
    video: str
    fps: float
    width: int
    height: int
    model_id: str
    model_version: int
    frame_stride: int
    keypoint_confidence: float
    min_keypoints: int
    frames: list[CourtKeypointFrame]


class BallFrame(PipelineModel):
    frame_id: int
    timestamp_sec: float
    raw_detection: DetectionRecord | None = None
    image_xy: list[float] | None = None
    confidence: float | None = None
    source: str = "missing"


class BallOutput(PipelineModel):
    video: str
    fps: float
    width: int
    height: int
    model_id: str
    model_version: int
    frame_stride: int
    frames: list[BallFrame]


class CourtPlayerProjection(PipelineModel):
    track_id: int
    image_xy: list[float]
    court_xy: list[float] | None = None
    valid: bool


class CourtBallProjection(PipelineModel):
    image_xy: list[float] | None = None
    court_xy: list[float] | None = None
    source: str = "missing"
    valid: bool = False


class CourtProjectionFrame(PipelineModel):
    frame_id: int
    timestamp_sec: float
    homography_source_frame: int | None = None
    players: list[CourtPlayerProjection]
    ball: CourtBallProjection | None = None


class CourtProjectionOutput(PipelineModel):
    video: str
    fps: float
    width: int
    height: int
    court_length_ft: float
    court_width_ft: float
    frames: list[CourtProjectionFrame]
