from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "detector": {
        "backend": "yolo",
    },
    "roboflow": {
        "api_url": "https://detect.roboflow.com",
        "model_id": "basketball-player-detection-3-ycjdo",
        "model_version": 13,
        "confidence": 40,
        "overlap": 30,
        "timeout_seconds": 60,
    },
    "yolo": {
        "weights_path": "object-detection/finetuning/runs/basketball_yolo11s/weights/best.pt",
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.5,
        "device": None,
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
            "ball_vis_full",
            "ball_vis_partial",
            "ball_vis_visible",
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
