"""Fine-tune Ultralytics YOLO on the basketball dataset.

Reads a Roboflow YOLOv8-format export (a directory containing ``data.yaml``
plus ``train/``, ``valid/``, ``test/`` splits) and produces a fine-tuned
checkpoint under ``object-detection/finetuning/runs/<name>/weights/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


FINETUNING_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = FINETUNING_DIR / "runs"
DEFAULT_CHECKPOINTS_DIR = FINETUNING_DIR / "checkpoints"


def resolve_device(requested: str) -> str | int:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_pretrained(model: str, checkpoints_dir: Path) -> str:
    """If ``model`` is a bare filename, route the auto-download into ``checkpoints_dir``.

    Anything that already looks like a path (absolute, or contains a directory
    component) is passed through untouched.
    """
    candidate = Path(model)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return model
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoints_dir / model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on the basketball dataset.")
    parser.add_argument("--data", required=True, help="Path to the Roboflow data.yaml.")
    parser.add_argument("--model", default="yolo11s.pt", help="Base checkpoint (default: yolo11s.pt).")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience.")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda index (e.g. 0).")
    parser.add_argument(
        "--project",
        default=str(DEFAULT_RUNS_DIR),
        help="Output root for runs. Default: object-detection/finetuning/runs.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=str(DEFAULT_CHECKPOINTS_DIR),
        help="Where to store auto-downloaded pretrained weights.",
    )
    parser.add_argument("--name", default="basketball_yolo11s")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint in --project/--name.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run: uv sync --extra train"
        ) from exc

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise SystemExit(f"data.yaml not found: {data_path}")

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
        project=str(project_dir),
        name=args.name,
        resume=args.resume,
        seed=args.seed,
        exist_ok=args.resume,
    )

    save_dir = Path(getattr(results, "save_dir", project_dir / args.name))
    best_weights = save_dir / "weights" / "best.pt"
    print(f"Training complete. Best weights: {best_weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
