from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from .config import load_config, nested_get
from .roboflow_client import RoboflowHostedClient, RoboflowHostedSettings, normalize_roboflow_predictions
from .schemas import BallFrame, BallOutput, DetectionRecord
from .utils import bbox_area, frame_timestamp, iter_video_frames, read_json, should_process_frame, video_metadata, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect and smooth basketball ball movement across a video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", help="Output ball tracking JSON path.")
    parser.add_argument("--frame-stride", type=int, help="Run ball detection every N frames.")
    parser.add_argument("--config", default="object-detection/config/default.yaml")
    parser.add_argument("--api-url", help="Roboflow API base URL.")
    parser.add_argument("--model-id", help="Roboflow player/ball detection model ID.")
    parser.add_argument("--model-version", type=int, help="Roboflow model version.")
    parser.add_argument("--confidence", type=int, help="Roboflow confidence threshold, 0-100.")
    parser.add_argument("--overlap", type=int, help="Roboflow overlap threshold, 0-100.")
    parser.add_argument("--min-confidence", type=float, help="Minimum accepted ball confidence, 0-1.")
    parser.add_argument("--max-missing-gap", type=int, help="Maximum gap to interpolate.")
    parser.add_argument("--max-jump-px", type=float, help="Reject detected ball jumps above this distance.")
    parser.add_argument("--prediction-gate-px", type=float, help="Maximum distance from constant-velocity prediction.")
    parser.add_argument("--smoothing-window", type=int, help="Odd moving-average window for detected/interpolated points.")
    parser.add_argument("--api-key-env", default="ROBOFLOW_API_KEY")
    parser.add_argument("--mock-response", help="Roboflow-style response reused for every sampled frame.")
    parser.add_argument("--max-frames", type=int, help="Optional frame limit for smoke tests.")
    return parser


def default_output(video_path: str | Path) -> Path:
    return Path("object-detection/outputs/ball") / f"{Path(video_path).stem}_ball.json"


def detection_center(detection: DetectionRecord) -> list[float]:
    x1, y1, x2, y2 = detection.bbox_xyxy
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def ball_detection_candidates(detections: list[DetectionRecord], min_confidence: float) -> list[DetectionRecord]:
    return [
        detection
        for detection in detections
        if detection.class_name.lower() == "ball" and detection.confidence >= min_confidence and bbox_area(detection.bbox_xyxy) > 0
    ]


def pick_ball_detection(
    detections: list[DetectionRecord],
    min_confidence: float,
    predicted_xy: list[float] | None = None,
    prediction_gate_px: float | None = None,
) -> DetectionRecord | None:
    candidates = ball_detection_candidates(detections, min_confidence)
    if not candidates:
        return None
    if predicted_xy is None or prediction_gate_px is None or prediction_gate_px <= 0:
        return max(candidates, key=lambda detection: detection.confidence)

    prediction = np.array(predicted_xy, dtype=float)
    scored: list[tuple[float, DetectionRecord]] = []
    for detection in candidates:
        distance = float(np.linalg.norm(np.array(detection_center(detection), dtype=float) - prediction))
        if distance > prediction_gate_px:
            continue
        distance_penalty = min(1.0, distance / prediction_gate_px)
        score = detection.confidence - 0.4 * distance_penalty
        scored.append((score, detection))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def predict_ball_xy(last_xy: np.ndarray | None, velocity_xy: np.ndarray | None, frames_since_last: int) -> list[float] | None:
    if last_xy is None:
        return None
    velocity = velocity_xy if velocity_xy is not None else np.zeros(2, dtype=float)
    predicted = last_xy + velocity * max(1, frames_since_last)
    return [float(predicted[0]), float(predicted[1])]


def interpolate_and_smooth_ball_frames(
    frames: list[BallFrame],
    max_missing_gap: int,
    max_jump_px: float,
    smoothing_window: int,
) -> list[BallFrame]:
    cleaned = [frame.model_copy(deep=True) for frame in frames]
    last_detected_xy: np.ndarray | None = None
    for frame in cleaned:
        if frame.source != "detected" or frame.image_xy is None:
            continue
        current = np.array(frame.image_xy, dtype=float)
        if last_detected_xy is not None and float(np.linalg.norm(current - last_detected_xy)) > max_jump_px:
            frame.image_xy = None
            frame.confidence = None
            frame.source = "missing"
            continue
        last_detected_xy = current

    detected_indexes = [index for index, frame in enumerate(cleaned) if frame.source == "detected" and frame.image_xy is not None]
    for left, right in zip(detected_indexes, detected_indexes[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > max_missing_gap:
            continue
        start = np.array(cleaned[left].image_xy, dtype=float)
        end = np.array(cleaned[right].image_xy, dtype=float)
        for step, index in enumerate(range(left + 1, right), start=1):
            alpha = step / (gap + 1)
            point = (1 - alpha) * start + alpha * end
            cleaned[index].image_xy = [float(point[0]), float(point[1])]
            cleaned[index].confidence = None
            cleaned[index].source = "interpolated"

    if smoothing_window > 1:
        if smoothing_window % 2 == 0:
            smoothing_window += 1
        radius = smoothing_window // 2
        smoothed = [frame.model_copy(deep=True) for frame in cleaned]
        for index, frame in enumerate(cleaned):
            if frame.image_xy is None:
                continue
            points = [
                np.array(cleaned[j].image_xy, dtype=float)
                for j in range(max(0, index - radius), min(len(cleaned), index + radius + 1))
                if cleaned[j].image_xy is not None
            ]
            if len(points) >= 2:
                mean = np.mean(points, axis=0)
                smoothed[index].image_xy = [float(mean[0]), float(mean[1])]
        cleaned = smoothed
    return cleaned


def run_ball_tracking(args: argparse.Namespace) -> BallOutput:
    load_dotenv()
    config = load_config(args.config)
    video_path = Path(args.video)
    metadata = video_metadata(video_path)
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    frame_count = int(metadata["frame_count"])

    frame_stride = args.frame_stride if args.frame_stride is not None else int(nested_get(config, "ball_tracking.frame_stride", 1))
    min_confidence = (
        args.min_confidence if args.min_confidence is not None else float(nested_get(config, "ball_tracking.min_confidence", 0.25))
    )
    prediction_gate_px = (
        args.prediction_gate_px
        if args.prediction_gate_px is not None
        else float(nested_get(config, "ball_tracking.prediction_gate_px", 300))
    )
    settings = RoboflowHostedSettings(
        api_url=args.api_url or str(nested_get(config, "roboflow.api_url")),
        model_id=args.model_id or str(nested_get(config, "roboflow.model_id")),
        model_version=args.model_version if args.model_version is not None else int(nested_get(config, "roboflow.model_version")),
        confidence=args.confidence if args.confidence is not None else int(nested_get(config, "roboflow.confidence")),
        overlap=args.overlap if args.overlap is not None else int(nested_get(config, "roboflow.overlap")),
        timeout_seconds=int(nested_get(config, "roboflow.timeout_seconds", 60)),
        api_key_env=args.api_key_env,
    )
    output_path = Path(args.output) if args.output else default_output(video_path)
    mock_payload = read_json(args.mock_response) if args.mock_response else None
    client = None if mock_payload is not None else RoboflowHostedClient(settings)

    frames: list[BallFrame] = []
    last_accepted_xy: np.ndarray | None = None
    last_velocity_xy: np.ndarray | None = None
    last_accepted_frame_id: int | None = None
    progress = tqdm(total=frame_count or None, desc="Ball detection", unit="frame")
    for frame_id, frame in iter_video_frames(video_path):
        if args.max_frames is not None and frame_id >= args.max_frames:
            break
        ball_detection = None
        if should_process_frame(frame_id, frame_stride):
            frame_name = f"{video_path.stem}_ball_{frame_id:06d}.jpg"
            payload = mock_payload if mock_payload is not None else client.infer_frame(frame, frame_name)  # type: ignore[union-attr]
            frames_since_last = frame_id - last_accepted_frame_id if last_accepted_frame_id is not None else 1
            predicted_xy = predict_ball_xy(last_accepted_xy, last_velocity_xy, frames_since_last)
            ball_detection = pick_ball_detection(
                normalize_roboflow_predictions(payload, width, height),
                min_confidence,
                predicted_xy=predicted_xy,
                prediction_gate_px=prediction_gate_px,
            )

        if ball_detection is None:
            frames.append(BallFrame(frame_id=frame_id, timestamp_sec=frame_timestamp(frame_id, fps)))
        else:
            current_xy = np.array(detection_center(ball_detection), dtype=float)
            if last_accepted_xy is not None and last_accepted_frame_id is not None:
                frame_delta = max(1, frame_id - last_accepted_frame_id)
                last_velocity_xy = (current_xy - last_accepted_xy) / frame_delta
            last_accepted_xy = current_xy
            last_accepted_frame_id = frame_id
            frames.append(
                BallFrame(
                    frame_id=frame_id,
                    timestamp_sec=frame_timestamp(frame_id, fps),
                    raw_detection=ball_detection,
                    image_xy=detection_center(ball_detection),
                    confidence=ball_detection.confidence,
                    source="detected",
                )
            )
        progress.update(1)
    progress.close()

    frames = interpolate_and_smooth_ball_frames(
        frames,
        max_missing_gap=args.max_missing_gap
        if args.max_missing_gap is not None
        else int(nested_get(config, "ball_tracking.max_missing_gap", 6)),
        max_jump_px=args.max_jump_px if args.max_jump_px is not None else float(nested_get(config, "ball_tracking.max_jump_px", 450)),
        smoothing_window=args.smoothing_window
        if args.smoothing_window is not None
        else int(nested_get(config, "ball_tracking.smoothing_window", 3)),
    )

    output = BallOutput(
        video=str(video_path),
        fps=fps,
        width=width,
        height=height,
        model_id=settings.model_id,
        model_version=settings.model_version,
        frame_stride=frame_stride,
        frames=frames,
    )
    write_json(output_path, output.model_dump(mode="json"))
    print(f"Wrote ball tracking: {output_path}")
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_ball_tracking(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
