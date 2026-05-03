from __future__ import annotations

from pathlib import Path
from typing import Any

from ..device import resolve_device


DEFAULT_DATA_ROOT = Path("object-detection/data/court-yolo")
DEFAULT_RUNS_DIR = Path("object-detection/court_finetuning/runs")
DEFAULT_CHECKPOINTS_DIR = Path("object-detection/court_finetuning/checkpoints")
DEFAULT_PROJECT = "basketball-court-detection-2"
DEFAULT_WORKSPACE = "roboflow-jvuqo"
DEFAULT_VERSION = 19
DEFAULT_FORMAT = "yolov11"
DEFAULT_MODEL = "yolo11m-pose.pt"
DEFAULT_RUN_NAME = "court_yolo11m_pose_v19"


def resolve_pretrained(model: str, checkpoints_dir: Path) -> str:
    candidate = Path(model)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return model
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoints_dir / model)


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise RuntimeError("Install pyyaml to load YAML files.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def validate_pose_data_yaml(path: str | Path, expected_keypoints: int = 33) -> dict[str, Any]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")
    payload = load_yaml(data_path)
    missing = [key for key in ("train", "val", "names") if key not in payload]
    if missing:
        raise ValueError(f"{data_path} is missing required YOLO data.yaml keys: {', '.join(missing)}")

    names = payload["names"]
    if isinstance(names, dict):
        class_names = {int(key): str(value) for key, value in names.items()}
    elif isinstance(names, list):
        class_names = {index: str(value) for index, value in enumerate(names)}
    else:
        raise ValueError("data.yaml names must be a list or mapping")
    if class_names.get(0) != "court":
        print(f"Warning: expected class 0 to be 'court', found {class_names.get(0)!r}")

    kpt_shape = payload.get("kpt_shape")
    if not isinstance(kpt_shape, list) or len(kpt_shape) != 2:
        raise ValueError("Pose data.yaml must define kpt_shape, usually [33, 2] or [33, 3]")
    if int(kpt_shape[0]) != expected_keypoints:
        raise ValueError(f"Expected {expected_keypoints} court keypoints, found kpt_shape={kpt_shape}")
    if int(kpt_shape[1]) not in (2, 3):
        raise ValueError(f"Expected keypoint dimensions 2 or 3, found kpt_shape={kpt_shape}")
    if "flip_idx" not in payload:
        print("Warning: data.yaml has no flip_idx. Training defaults disable flips to avoid corrupting keypoint IDs.")
    return payload


__all__ = [
    "DEFAULT_CHECKPOINTS_DIR",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_FORMAT",
    "DEFAULT_MODEL",
    "DEFAULT_PROJECT",
    "DEFAULT_RUN_NAME",
    "DEFAULT_RUNS_DIR",
    "DEFAULT_VERSION",
    "DEFAULT_WORKSPACE",
    "resolve_device",
    "resolve_pretrained",
    "validate_pose_data_yaml",
]
