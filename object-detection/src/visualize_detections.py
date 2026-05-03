from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .detection import draw_detection_frame
from .utils import iter_video_frames, open_video_writer, read_json, video_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render detection JSON as an annotated video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--output", required=True)
    return parser


def detections_by_frame(detection_data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    return {
        int(frame["frame_id"]): list(frame.get("detections", []))
        for frame in detection_data.get("frames", [])
    }


def render_detection_video(video_path: str | Path, detections_path: str | Path, output_path: str | Path) -> None:
    metadata = video_metadata(video_path)
    writer = open_video_writer(output_path, float(metadata["fps"]), int(metadata["width"]), int(metadata["height"]))
    frame_map = detections_by_frame(read_json(detections_path))
    try:
        for frame_id, frame in iter_video_frames(video_path):
            annotated = frame.copy()
            draw_detection_frame(annotated, frame_map.get(frame_id, []))
            writer.write(annotated)
    finally:
        writer.release()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render_detection_video(args.video, args.detections, args.output)
    print(f"Wrote detection visualization: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
