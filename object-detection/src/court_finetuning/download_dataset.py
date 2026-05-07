from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .common import DEFAULT_DATA_ROOT, DEFAULT_FORMAT, DEFAULT_PROJECT, DEFAULT_VERSION, DEFAULT_WORKSPACE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the Roboflow basketball court keypoint dataset.")
    parser.add_argument("--workspace", default=os.getenv("ROBOFLOW_COURT_WORKSPACE", DEFAULT_WORKSPACE))
    parser.add_argument("--project", default=os.getenv("ROBOFLOW_COURT_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--version", type=int, default=int(os.getenv("ROBOFLOW_COURT_VERSION", DEFAULT_VERSION)))
    parser.add_argument(
        "--format",
        default=os.getenv("ROBOFLOW_COURT_FORMAT", DEFAULT_FORMAT),
        help="Roboflow export format. Prefer yolov11 for YOLO11 pose; try yolov8 if unavailable.",
    )
    parser.add_argument(
        "--location",
        help="Dataset destination. Defaults to object-detection/data/court-yolo/<project>-<version>.",
    )
    parser.add_argument("--api-key-env", default="ROBOFLOW_API_KEY")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing Roboflow API key. Set {args.api_key_env} in .env")

    location = Path(args.location) if args.location else DEFAULT_DATA_ROOT / f"{args.project}-{args.version}"
    location.parent.mkdir(parents=True, exist_ok=True)

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise SystemExit("The roboflow package is not installed. Run: uv sync --extra train") from exc

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download(args.format, location=str(location))
    print(f"Downloaded {args.project} v{args.version} ({args.format}) to {dataset.location}")
    print(f"Data YAML: {Path(dataset.location) / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
