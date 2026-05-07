from __future__ import annotations

import argparse
from pathlib import Path

from .common import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test a fine-tuned YOLO court keypoint checkpoint.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Image, directory, or video path.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="object-detection/court_finetuning/runs/predict")
    parser.add_argument("--name", default="court_keypoints")
    parser.add_argument("--save", action="store_true", help="Save annotated predictions.")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO-format prediction text files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights not found: {weights_path}")
    source_path = Path(args.source)
    if not source_path.exists():
        raise SystemExit(f"Source not found: {source_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run: uv sync --extra train") from exc

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        imgsz=args.imgsz,
        device=resolve_device(args.device),
        project=args.project,
        name=args.name,
        save=args.save,
        save_txt=args.save_txt,
    )
    print(f"Predicted {len(results)} result(s). Output project: {args.project}/{args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
