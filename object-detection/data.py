"""Download the labeled basketball dataset from Roboflow.

Reads credentials and project info from ``.env``. Defaults to the YOLOv8
export format so the result is consumable by Ultralytics training out of
the box; pass ``--format coco`` to pull the same labels for RF-DETR.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


SUPPORTED_FORMATS = ("yolov8", "yolov11", "coco", "yolov5pytorch", "voc")
DEFAULT_DATA_DIR = Path("object-detection/data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a Roboflow dataset for local training.")
    parser.add_argument(
        "--format",
        default=os.getenv("ROBOFLOW_FORMAT", "yolov8"),
        help="Roboflow export format. Default: yolov8 (Ultralytics). Use 'coco' for RF-DETR.",
    )
    parser.add_argument("--workspace", help="Override ROBOFLOW_WORKSPACE.")
    parser.add_argument("--project", help="Override ROBOFLOW_PROJECT.")
    parser.add_argument("--version", type=int, help="Override ROBOFLOW_VERSION.")
    parser.add_argument(
        "--location",
        help="Where to write the dataset. Defaults to object-detection/data/<format>/<project>-<version>.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY missing from .env")

    workspace = args.workspace or os.getenv("ROBOFLOW_WORKSPACE")
    project_name = args.project or os.getenv("ROBOFLOW_PROJECT")
    version_env = os.getenv("ROBOFLOW_VERSION")
    if args.version is not None:
        version = args.version
    elif version_env is not None:
        version = int(version_env)
    else:
        version = 1

    if not workspace or not project_name:
        raise SystemExit(
            "ROBOFLOW_WORKSPACE and ROBOFLOW_PROJECT must be set in .env "
            "(or pass --workspace and --project)."
        )

    if args.format not in SUPPORTED_FORMATS:
        print(f"Warning: '{args.format}' is not in the known list {SUPPORTED_FORMATS}; passing through anyway.")

    location = (
        Path(args.location)
        if args.location
        else DEFAULT_DATA_DIR / args.format / f"{project_name}-{version}"
    )
    location.parent.mkdir(parents=True, exist_ok=True)

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise SystemExit(
            "The 'roboflow' package is not installed. Run: uv sync --extra train"
        ) from exc

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download(args.format, location=str(location))
    print(f"Downloaded {project_name} v{version} ({args.format}) to {dataset.location}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())