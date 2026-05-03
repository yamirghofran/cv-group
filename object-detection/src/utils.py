from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


PLAYER_LIKE_CLASS_NAMES = {
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
    "player-layup",
    "player-dribble",
    "player-screen",
    "player-fall",
}


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def is_player_like_class(class_name: str, configured_classes: Iterable[str] | None = None) -> bool:
    normalized = class_name.strip().lower()
    configured = {name.strip().lower() for name in configured_classes or PLAYER_LIKE_CLASS_NAMES}
    if normalized in configured:
        return True
    return normalized.startswith("player-")


def clamp_bbox_xyxy(bbox: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def bbox_area(bbox: Iterable[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center_x(bbox: Iterable[float]) -> float:
    x1, _, x2, _ = [float(v) for v in bbox]
    return (x1 + x2) / 2.0


def bbox_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def video_metadata(video_path: str | Path) -> dict[str, float | int]:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if fps <= 0:
        fps = 30.0
    return {"fps": fps, "width": width, "height": height, "frame_count": frame_count}


def iter_video_frames(video_path: str | Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_id = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame_id, frame
            frame_id += 1
    finally:
        capture.release()


def should_process_frame(frame_id: int, frame_stride: int) -> bool:
    if frame_id == 0:
        return True
    return frame_stride > 0 and frame_id % frame_stride == 0


def color_for_id(identifier: int | str) -> tuple[int, int, int]:
    if isinstance(identifier, str):
        seed = sum(ord(ch) for ch in identifier)
    else:
        seed = int(identifier)
    r = (37 * seed + 59) % 255
    g = (17 * seed + 131) % 255
    b = (97 * seed + 29) % 255
    return int(b), int(g), int(r)


def draw_label(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (label_width, label_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    top = max(0, y - label_height - baseline - 4)
    right = min(frame.shape[1] - 1, x + label_width + 6)
    cv2.rectangle(frame, (x, top), (right, top + label_height + baseline + 4), color, -1)
    cv2.putText(frame, text, (x + 3, top + label_height + 1), font, font_scale, (255, 255, 255), thickness)


def draw_bbox(
    frame: np.ndarray,
    bbox_xyxy: Iterable[float],
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    draw_label(frame, label, (x1, max(0, y1 - 2)), color)


def open_video_writer(path: str | Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    ensure_parent_dir(path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def frame_timestamp(frame_id: int, fps: float) -> float:
    if fps <= 0:
        return 0.0
    return float(frame_id) / float(fps)


def sample_interval_for_fps(video_fps: float, target_fps: float) -> int:
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    return max(1, int(round(video_fps / target_fps)))
