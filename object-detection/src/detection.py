from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from tqdm import tqdm

from .config import load_config, nested_get
from .roboflow_client import RoboflowHostedClient, RoboflowHostedSettings, normalize_roboflow_predictions
from .schemas import DetectionFrame, DetectionOutput
from .yolo_client import YoloLocalClient, YoloLocalSettings
from .utils import (
    color_for_id,
    draw_bbox,
    ensure_dir,
    frame_timestamp,
    iter_video_frames,
    open_video_writer,
    read_json,
    should_process_frame,
    video_metadata,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Roboflow basketball detections on sampled video frames.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", help="Output detection JSON path.")
    parser.add_argument("--debug-video", help="Optional annotated debug video path.")
    parser.add_argument("--debug-frame-dir", help="Directory for annotated sampled debug frames.")
    parser.add_argument("--debug-frame-limit", type=int, default=25, help="Maximum sampled debug frames to save.")
    parser.add_argument("--frame-stride", type=int, help="Run detection on frame 0 and every N frames.")
    parser.add_argument("--config", default="object-detection/config/default.yaml", help="Pipeline YAML config path.")
    parser.add_argument("--detector-backend", choices=["roboflow", "yolo"], help="Detector backend.")
    parser.add_argument("--api-url", help="Roboflow API base URL.")
    parser.add_argument("--model-id", help="Roboflow project/model ID.")
    parser.add_argument("--model-version", type=int, help="Roboflow model version.")
    parser.add_argument("--confidence", type=int, help="Roboflow confidence threshold, 0-100.")
    parser.add_argument("--overlap", type=int, help="Roboflow overlap threshold, 0-100.")
    parser.add_argument("--timeout-seconds", type=int, help="Roboflow request timeout.")
    parser.add_argument("--api-key-env", default="ROBOFLOW_API_KEY", help="Environment variable with API key.")
    parser.add_argument("--yolo-weights", help="Path to a fine-tuned YOLO .pt file.")
    parser.add_argument("--yolo-imgsz", type=int, help="YOLO inference image size.")
    parser.add_argument("--yolo-conf", type=float, help="YOLO confidence threshold (0-1).")
    parser.add_argument("--yolo-iou", type=float, help="YOLO NMS IoU threshold (0-1).")
    parser.add_argument("--yolo-device", help="YOLO device override: cpu | mps | cuda index.")
    parser.add_argument(
        "--mock-response",
        help="Optional Roboflow-style JSON response reused for every detected frame. Avoids network calls.",
    )
    parser.add_argument("--max-frames", type=int, help="Optional processing limit for smoke tests.")
    return parser


def default_detection_output(video_path: str | Path) -> Path:
    stem = Path(video_path).stem
    return Path("object-detection/outputs/detections") / f"{stem}_detections.json"


def default_debug_frame_dir(video_path: str | Path) -> Path:
    return Path("object-detection/outputs/debug_detections") / Path(video_path).stem


def draw_detection_frame(frame, detections: list[dict[str, Any]]) -> None:
    for detection in detections:
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        color = color_for_id(detection["class_name"])
        draw_bbox(frame, detection["bbox_xyxy"], label, color)


def build_detector(
    backend: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    skip_client: bool,
) -> tuple[Any, str, int]:
    """Construct the active detector client and return (client, model_id, model_version).

    ``skip_client`` lets callers skip instantiation (e.g. when --mock-response is used)
    while still resolving identifiers for the output JSON.
    """
    if backend == "roboflow":
        settings = RoboflowHostedSettings(
            api_url=args.api_url or str(nested_get(config, "roboflow.api_url")),
            model_id=args.model_id or str(nested_get(config, "roboflow.model_id")),
            model_version=args.model_version
            if args.model_version is not None
            else int(nested_get(config, "roboflow.model_version")),
            confidence=args.confidence
            if args.confidence is not None
            else int(nested_get(config, "roboflow.confidence")),
            overlap=args.overlap if args.overlap is not None else int(nested_get(config, "roboflow.overlap")),
            timeout_seconds=args.timeout_seconds
            if args.timeout_seconds is not None
            else int(nested_get(config, "roboflow.timeout_seconds")),
            api_key_env=args.api_key_env,
        )
        client = None if skip_client else RoboflowHostedClient(settings)
        return client, settings.model_id, settings.model_version

    if backend == "yolo":
        device_cfg = nested_get(config, "yolo.device")
        settings = YoloLocalSettings(
            weights_path=args.yolo_weights or str(nested_get(config, "yolo.weights_path")),
            imgsz=args.yolo_imgsz if args.yolo_imgsz is not None else int(nested_get(config, "yolo.imgsz", 640)),
            conf=args.yolo_conf if args.yolo_conf is not None else float(nested_get(config, "yolo.conf", 0.25)),
            iou=args.yolo_iou if args.yolo_iou is not None else float(nested_get(config, "yolo.iou", 0.5)),
            device=args.yolo_device if args.yolo_device is not None else (str(device_cfg) if device_cfg else None),
        )
        client = None if skip_client else YoloLocalClient(settings)
        model_id = f"yolo:{Path(settings.weights_path).stem}"
        return client, model_id, 0

    raise ValueError(f"Unknown detector backend: {backend!r}. Expected 'roboflow' or 'yolo'.")


def run_detection(args: argparse.Namespace) -> DetectionOutput:
    load_dotenv()
    config = load_config(args.config)
    video_path = Path(args.video)
    metadata = video_metadata(video_path)
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    frame_count = int(metadata["frame_count"])

    frame_stride = args.frame_stride if args.frame_stride is not None else int(nested_get(config, "detection.frame_stride", 5))
    if frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    output_path = Path(args.output) if args.output else default_detection_output(video_path)
    debug_frame_dir = Path(args.debug_frame_dir) if args.debug_frame_dir else default_debug_frame_dir(video_path)
    ensure_dir(debug_frame_dir)

    backend = args.detector_backend or str(nested_get(config, "detector.backend", "yolo"))
    mock_payload = read_json(args.mock_response) if args.mock_response else None
    client, model_id, model_version = build_detector(backend, args, config, skip_client=mock_payload is not None)
    writer = None
    if args.debug_video:
        writer = open_video_writer(args.debug_video, fps, width, height)

    frames: list[DetectionFrame] = []
    classes_seen: set[str] = set()
    debug_frames_saved = 0
    debug_frame_ids_saved: set[int] = set()

    try:
        progress = tqdm(total=frame_count or None, desc="Detecting", unit="frame")
        for frame_id, frame in iter_video_frames(video_path):
            if args.max_frames is not None and frame_id >= args.max_frames:
                break

            detections_for_frame: list[dict[str, Any]] = []
            if should_process_frame(frame_id, frame_stride):
                frame_name = f"{video_path.stem}_frame_{frame_id:06d}.jpg"
                payload = mock_payload if mock_payload is not None else client.infer_frame(frame, frame_name)  # type: ignore[union-attr]
                normalized = normalize_roboflow_predictions(payload, width, height)
                detections_for_frame = [record.model_dump(mode="json") for record in normalized]
                classes_seen.update(record.class_name for record in normalized)
                frames.append(
                    DetectionFrame(
                        frame_id=frame_id,
                        timestamp_sec=frame_timestamp(frame_id, fps),
                        detections=normalized,
                    )
                )

            should_save_debug_frame = (
                frame_id == 0
                or (
                    bool(detections_for_frame)
                    and debug_frames_saved < args.debug_frame_limit
                    and frame_id not in debug_frame_ids_saved
                )
            )
            if writer is not None or should_save_debug_frame:
                annotated = frame.copy()
                draw_detection_frame(annotated, detections_for_frame)
                if writer is not None:
                    writer.write(annotated)
                if should_save_debug_frame:
                    cv2.imwrite(str(debug_frame_dir / f"frame_{frame_id:06d}.jpg"), annotated)
                    debug_frame_ids_saved.add(frame_id)
                    if frame_id != 0:
                        debug_frames_saved += 1

            progress.update(1)
        progress.close()
    finally:
        if writer is not None:
            writer.release()

    output = DetectionOutput(
        video=str(video_path),
        fps=fps,
        width=width,
        height=height,
        model_id=model_id,
        model_version=model_version,
        frame_stride=frame_stride,
        frames=frames,
        classes_seen=sorted(classes_seen),
    )
    write_json(output_path, output.model_dump(mode="json"))
    print(
        "Detection complete: "
        f"backend={backend}, model={model_id}/{model_version}, "
        f"sampled_frames={len(frames)}, classes_seen={sorted(classes_seen)}"
    )
    print(f"Wrote detections: {output_path}")
    if args.debug_video:
        print(f"Wrote debug video: {args.debug_video}")
    print(f"Wrote debug frames: {debug_frame_dir}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_detection(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
