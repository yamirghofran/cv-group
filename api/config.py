from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value is not None else default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


@dataclass(frozen=True)
class ApiSettings:
    yolo_weights: Path
    yolo_imgsz: int
    yolo_conf: float
    yolo_iou: float
    yolo_device: str | None
    sam2_enabled: bool
    sam2_checkpoint: Path | None
    sam2_model_cfg: str
    sam2_device: str
    work_dir: Path
    frame_stride: int
    cleanup_distance_threshold: float
    cleanup_min_component_area: int
    max_upload_mb: int
    clustering_device: str
    clustering_batch_size: int
    clustering_n_teams: int
    # OCR Settings
    ocr_enabled: bool
    ocr_model_name: str  # "resnet", "x2", or "smol"
    ocr_ios_threshold: float
    ocr_device: str

    @classmethod
    def from_env(cls) -> "ApiSettings":
        sam2_enabled = _env_bool("BASKETBALL_API_SAM2_ENABLED", False)
        sam2_ckpt = _env(
            "BASKETBALL_API_SAM2_CHECKPOINT",
            "object-detection/checkpoints/sam2.1_hiera_large.pt",
        )
        return cls(
            yolo_weights=Path(
                _env(
                    "BASKETBALL_API_YOLO_WEIGHTS",
                    "object-detection/finetuning/runs/basketball_yolo11s/weights/best.pt",
                )  # type: ignore[arg-type]
            ),
            yolo_imgsz=_env_int("BASKETBALL_API_YOLO_IMGSZ", 640),
            yolo_conf=_env_float("BASKETBALL_API_YOLO_CONF", 0.25),
            yolo_iou=_env_float("BASKETBALL_API_YOLO_IOU", 0.5),
            yolo_device=_env("BASKETBALL_API_YOLO_DEVICE"),
            sam2_enabled=sam2_enabled,
            sam2_checkpoint=Path(sam2_ckpt) if sam2_enabled and sam2_ckpt else None,
            sam2_model_cfg=_env(
                "BASKETBALL_API_SAM2_MODEL_CFG",
                "configs/sam2.1/sam2.1_hiera_l.yaml",
            ) or "configs/sam2.1/sam2.1_hiera_l.yaml",
            sam2_device=_env("BASKETBALL_API_SAM2_DEVICE", "auto") or "auto",
            work_dir=Path(_env("BASKETBALL_API_WORK_DIR", "object-detection/outputs/api") or "object-detection/outputs/api"),
            frame_stride=_env_int("BASKETBALL_API_FRAME_STRIDE", 5),
            cleanup_distance_threshold=_env_float("BASKETBALL_API_CLEANUP_DIST", 80.0),
            cleanup_min_component_area=_env_int("BASKETBALL_API_CLEANUP_MIN_AREA", 50),
            max_upload_mb=_env_int("BASKETBALL_API_MAX_UPLOAD_MB", 500),
            clustering_device=_env("BASKETBALL_API_CLUSTERING_DEVICE", "cpu") or "cpu",
            clustering_batch_size=_env_int("BASKETBALL_API_CLUSTERING_BATCH_SIZE", 32),
            clustering_n_teams=_env_int("BASKETBALL_API_CLUSTERING_N_TEAMS", 2),
            # OCR Settings
            ocr_enabled=_env_bool("BASKETBALL_API_OCR_ENABLED", False),
            ocr_model_name=_env("BASKETBALL_API_OCR_MODEL", "resnet") or "resnet",
            ocr_ios_threshold=_env_float("BASKETBALL_API_OCR_IOS_THRESHOLD", 0.9),
            ocr_device=_env("BASKETBALL_API_OCR_DEVICE", "auto") or "auto",
        )
