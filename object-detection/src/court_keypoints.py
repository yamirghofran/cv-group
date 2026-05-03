from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from tqdm import tqdm

from .config import load_config, nested_get
from .roboflow_client import RoboflowHostedClient, RoboflowHostedSettings, normalize_roboflow_keypoints
from .schemas import CourtKeypointFrame, CourtKeypointOutput
from .utils import (
    color_for_id,
    draw_label,
    ensure_dir,
    frame_timestamp,
    iter_video_frames,
    read_json,
    should_process_frame,
    video_metadata,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect basketball court keypoints with a Roboflow keypoint model.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", help="Output court keypoint JSON path.")
    parser.add_argument("--debug-frame-dir", help="Directory for annotated sampled frames.")
    parser.add_argument("--debug-frame-limit", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, help="Run keypoint detection every N frames.")
    parser.add_argument("--config", default="object-detection/config/default.yaml")
    parser.add_argument("--api-url", help="Roboflow keypoint API base URL.")
    parser.add_argument("--model-id", help="Roboflow court keypoint model ID.")
    parser.add_argument("--model-version", type=int, help="Roboflow court keypoint model version.")
    parser.add_argument("--confidence", type=int, help="Roboflow model confidence threshold, 0-100.")
    parser.add_argument("--overlap", type=int, help="Roboflow overlap threshold, 0-100.")
    parser.add_argument("--keypoint-confidence", type=float, help="Minimum keypoint confidence used for validity.")
    parser.add_argument("--min-keypoints", type=int, help="Minimum accepted keypoints per frame.")
    parser.add_argument("--api-key-env", default="ROBOFLOW_API_KEY")
    parser.add_argument("--mock-response", help="Roboflow-style keypoint response reused for every sampled frame.")
    parser.add_argument("--max-frames", type=int, help="Optional frame limit for smoke tests.")
    return parser


def default_output(video_path: str | Path) -> Path:
    return Path("object-detection/outputs/court_keypoints") / f"{Path(video_path).stem}_keypoints.json"


def default_debug_dir(video_path: str | Path) -> Path:
    return Path("object-detection/outputs/debug_court_keypoints") / Path(video_path).stem


def draw_keypoints(frame, keypoints: list[dict[str, Any]]) -> None:
    for keypoint in keypoints:
        x, y = [int(round(v)) for v in keypoint["image_xy"]]
        color = color_for_id(keypoint["class_id"])
        cv2.circle(frame, (x, y), 7, color, -1)
        draw_label(frame, f"{keypoint['class_id']} {keypoint['confidence']:.2f}", (x + 4, y - 4), color)


def run_court_keypoints(args: argparse.Namespace) -> CourtKeypointOutput:
    load_dotenv()
    config = load_config(args.config)
    video_path = Path(args.video)
    metadata = video_metadata(video_path)
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    frame_count = int(metadata["frame_count"])

    frame_stride = (
        args.frame_stride if args.frame_stride is not None else int(nested_get(config, "court_keypoints.frame_stride", 15))
    )
    if frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    keypoint_confidence = (
        args.keypoint_confidence
        if args.keypoint_confidence is not None
        else float(nested_get(config, "court_keypoints.keypoint_confidence", 0.5))
    )
    min_keypoints = (
        args.min_keypoints if args.min_keypoints is not None else int(nested_get(config, "court_keypoints.min_keypoints", 4))
    )

    settings = RoboflowHostedSettings(
        api_url=args.api_url or str(nested_get(config, "court_keypoints.api_url")),
        model_id=args.model_id or str(nested_get(config, "court_keypoints.model_id")),
        model_version=args.model_version
        if args.model_version is not None
        else int(nested_get(config, "court_keypoints.model_version")),
        confidence=args.confidence if args.confidence is not None else int(nested_get(config, "court_keypoints.confidence")),
        overlap=args.overlap if args.overlap is not None else int(nested_get(config, "court_keypoints.overlap")),
        timeout_seconds=int(nested_get(config, "roboflow.timeout_seconds", 60)),
        api_key_env=args.api_key_env,
    )
    output_path = Path(args.output) if args.output else default_output(video_path)
    debug_dir = Path(args.debug_frame_dir) if args.debug_frame_dir else default_debug_dir(video_path)
    ensure_dir(debug_dir)

    mock_payload = read_json(args.mock_response) if args.mock_response else None
    client = None if mock_payload is not None else RoboflowHostedClient(settings)
    frames: list[CourtKeypointFrame] = []
    debug_frames_saved = 0

    progress = tqdm(total=frame_count or None, desc="Court keypoints", unit="frame")
    for frame_id, frame in iter_video_frames(video_path):
        if args.max_frames is not None and frame_id >= args.max_frames:
            break
        if should_process_frame(frame_id, frame_stride):
            frame_name = f"{video_path.stem}_court_{frame_id:06d}.jpg"
            payload = mock_payload if mock_payload is not None else client.infer_frame(frame, frame_name)  # type: ignore[union-attr]
            keypoints = normalize_roboflow_keypoints(payload)
            accepted_count = sum(1 for keypoint in keypoints if keypoint.confidence >= keypoint_confidence)
            valid = accepted_count >= min_keypoints
            frames.append(
                CourtKeypointFrame(
                    frame_id=frame_id,
                    timestamp_sec=frame_timestamp(frame_id, fps),
                    keypoints=keypoints,
                    valid=valid,
                    reason=None if valid else f"only {accepted_count} keypoints above confidence",
                )
            )

            if debug_frames_saved < args.debug_frame_limit:
                annotated = frame.copy()
                draw_keypoints(annotated, [keypoint.model_dump(mode="json") for keypoint in keypoints])
                cv2.imwrite(str(debug_dir / f"frame_{frame_id:06d}.jpg"), annotated)
                debug_frames_saved += 1
        progress.update(1)
    progress.close()

    output = CourtKeypointOutput(
        video=str(video_path),
        fps=fps,
        width=width,
        height=height,
        model_id=settings.model_id,
        model_version=settings.model_version,
        frame_stride=frame_stride,
        keypoint_confidence=keypoint_confidence,
        min_keypoints=min_keypoints,
        frames=frames,
    )
    write_json(output_path, output.model_dump(mode="json"))
    print(f"Wrote court keypoints: {output_path}")
    print(f"Wrote court keypoint debug frames: {debug_dir}")
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_court_keypoints(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
