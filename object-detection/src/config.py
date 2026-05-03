from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "roboflow": {
        "api_url": "https://detect.roboflow.com",
        "model_id": "basketball-player-detection-3-ycjdo",
        "model_version": 13,
        "confidence": 40,
        "overlap": 30,
        "timeout_seconds": 60,
    },
    "detection": {
        "frame_stride": 5,
        "player_like_classes": [
            "player",
            "player-in-possession",
            "player-jump-shot",
            "player-layup-dunk",
            "player-shot-block",
            "player-layup",
            "player-dribble",
            "player-screen",
            "player-fall",
        ],
        "relevant_classes": [
            "ball",
            "ball-in-basket",
            "number",
            "player",
            "player-in-possession",
            "player-jump-shot",
            "player-layup-dunk",
            "player-shot-block",
            "referee",
            "rim",
        ],
    },
    "tracking": {
        "backend": "sam2",
        "sam2_checkpoint": "checkpoints/sam2.1_hiera_large.pt",
        "sam2_model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "device": "auto",
    },
    "mask_cleanup": {
        "distance_threshold": 80,
        "min_component_area": 50,
    },
    "handoff": {
        "crop_fps": 1.0,
    },
    "court_keypoints": {
        "backend": "roboflow",
        "api_url": "https://serverless.roboflow.com",
        "model_id": "basketball-court-detection-2",
        "model_version": 19,
        "confidence": 30,
        "overlap": 30,
        "yolo_weights": "object-detection/court_finetuning/runs/court_yolo11m_pose_v19/weights/best.pt",
        "yolo_confidence": 0.25,
        "yolo_device": "auto",
        "keypoint_confidence": 0.5,
        "frame_stride": 15,
        "min_keypoints": 4,
        "class_id_offset": 0,
    },
    "ball_tracking": {
        "frame_stride": 1,
        "min_confidence": 0.25,
        "max_missing_gap": 6,
        "max_jump_px": 450,
        "smoothing_window": 3,
    },
    "court_projection": {
        "court_length_ft": 94.0,
        "court_width_ft": 50.0,
        "out_of_bounds_margin_ft": 5.0,
    },
    "court_visualization": {
        "panel_height": 720,
        "trail_length": 45,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        default_path = Path("object-detection/config/default.yaml")
        if not default_path.exists():
            default_path = Path("config/default.yaml")
        path = default_path if default_path.exists() else None
    if path is None:
        return config

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise RuntimeError("Install pyyaml to load YAML config files.") from exc

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return deep_merge(config, loaded)


def nested_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
