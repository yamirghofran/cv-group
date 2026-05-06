"""Smoke-test a fine-tuned YOLO checkpoint on a video, image, or directory.

Annotated outputs land under ``object-detection/finetuning/runs/predict/<name>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


FINETUNING_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICT_DIR = FINETUNING_DIR / "runs" / "predict"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a YOLO checkpoint on a video/image source.")
    parser.add_argument("--weights", required=True, help="Path to a fine-tuned .pt file.")
    parser.add_argument("--source", required=True, help="Video file, image, or directory.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default=None, help="auto | cpu | mps | cuda index. Default: ultralytics auto.")
    parser.add_argument("--save", action="store_true", help="Save annotated output.")
    parser.add_argument(
        "--project",
        default=str(DEFAULT_PREDICT_DIR),
        help="Output root for predictions.",
    )
    parser.add_argument("--name", default="exp")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run: uv sync --extra train"
        ) from exc

    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise SystemExit(f"Weights file not found: {weights}")

    project_dir = Path(args.project).resolve()
    model = YOLO(str(weights))
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
