from __future__ import annotations

import argparse
from pathlib import Path

from .common import (
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_MODEL,
    DEFAULT_RUN_NAME,
    DEFAULT_RUNS_DIR,
    resolve_device,
    resolve_pretrained,
    validate_pose_data_yaml,
)


def bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO pose on basketball court keypoints.")
    parser.add_argument("--data", required=True, help="Path to Roboflow/Ultralytics pose data.yaml.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base pose checkpoint. Default: yolo11m-pose.pt.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda | 0.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", type=bool_arg, default=True)
    parser.add_argument("--amp", type=bool_arg, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--checkpoints-dir", default=str(DEFAULT_CHECKPOINTS_DIR))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=0.20)
    parser.add_argument("--perspective", type=float, default=0.0005)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = Path(args.data).resolve()
    validate_pose_data_yaml(data_path)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run: uv sync --extra train") from exc

    project_dir = Path(args.project).resolve()
    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    pretrained = resolve_pretrained(args.model, checkpoints_dir)
    device = resolve_device(args.device)

    model = YOLO(pretrained)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        workers=args.workers,
        cache=args.cache,
        amp=args.amp,
        seed=args.seed,
        project=str(project_dir),
        name=args.name,
        resume=args.resume,
        exist_ok=args.resume,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        perspective=args.perspective,
    )

    save_dir = Path(getattr(results, "save_dir", project_dir / args.name))
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"
    print(f"Training complete. Best weights: {best_weights}")
    print(f"Last weights: {last_weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
