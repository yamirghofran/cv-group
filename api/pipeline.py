from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.detection import draw_detection_frame
from src.roboflow_client import normalize_roboflow_predictions
from src.schemas import DetectionFrame, DetectionOutput
from src.tracking import build_track_output, load_frame0_boxes
from src.utils import (
    ensure_dir,
    frame_timestamp,
    iter_video_frames,
    open_video_writer,
    should_process_frame,
    video_metadata,
    write_json,
)
from src.visualize_tracks import render_track_video
from src.yolo_client import YoloLocalClient

from .config import ApiSettings


@dataclass
class ProcessResult:
    job_id: str
    work_dir: Path
    detections_path: Path
    classes_seen: list[str]
    frames_sampled: int
    annotated_video_path: Path
    sam2_used: bool
    extras: dict[str, Any] = field(default_factory=dict)


def process_video(
    *,
    video_path: Path,
    yolo: YoloLocalClient,
    sam2_predictor: Any | None,
    settings: ApiSettings,
) -> ProcessResult:
    """Run YOLO detection (always) and SAM2 tracking + render (if predictor is loaded).

    Extension point for colleagues: append additional steps after detection or tracking
    using the artifacts in ``work_dir`` (detections.json, tracks.json, masks/, crops/).
    """
    job_id = uuid.uuid4().hex[:12]
    work_dir = settings.work_dir / job_id
    ensure_dir(work_dir)

    detection_output = _run_yolo_detection(
        video_path=video_path,
        yolo=yolo,
        frame_stride=settings.frame_stride,
    )
    detections_path = work_dir / "detections.json"
    write_json(detections_path, detection_output.model_dump(mode="json"))

    annotated_video_path = work_dir / "annotated.mp4"
    if sam2_predictor is not None:
        annotated_video_path = _run_tracking_and_render(
            video_path=video_path,
            detections_path=detections_path,
            work_dir=work_dir,
            sam2_predictor=sam2_predictor,
            settings=settings,
        )
    else:
        _render_yolo_overlay_video(
            video_path=video_path,
            detection_output=detection_output,
            output_path=annotated_video_path,
        )

    return ProcessResult(
        job_id=job_id,
        work_dir=work_dir,
        detections_path=detections_path,
        classes_seen=list(detection_output.classes_seen),
        frames_sampled=len(detection_output.frames),
        annotated_video_path=annotated_video_path,
        sam2_used=sam2_predictor is not None,
    )


def _run_yolo_detection(
    *, video_path: Path, yolo: YoloLocalClient, frame_stride: int
) -> DetectionOutput:
    metadata = video_metadata(video_path)
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])

    frames: list[DetectionFrame] = []
    classes_seen: set[str] = set()

    for frame_id, frame in iter_video_frames(video_path):
        if not should_process_frame(frame_id, frame_stride):
            continue
        frame_name = f"{video_path.stem}_frame_{frame_id:06d}.jpg"
        payload = yolo.infer_frame(frame, frame_name)
        records = normalize_roboflow_predictions(payload, width, height, source="yolo")
        classes_seen.update(record.class_name for record in records)
        frames.append(
            DetectionFrame(
                frame_id=frame_id,
                timestamp_sec=frame_timestamp(frame_id, fps),
                detections=records,
            )
        )

    return DetectionOutput(
        video=str(video_path),
        fps=fps,
        width=width,
        height=height,
        model_id=yolo.model_identifier,
        model_version=0,
        frame_stride=frame_stride,
        frames=frames,
        classes_seen=sorted(classes_seen),
    )


def _render_yolo_overlay_video(
    *,
    video_path: Path,
    detection_output: DetectionOutput,
    output_path: Path,
) -> Path:
    detections_by_frame: dict[int, list[dict[str, Any]]] = {
        frame.frame_id: [det.model_dump(mode="json") for det in frame.detections]
        for frame in detection_output.frames
    }

    writer = open_video_writer(
        output_path, detection_output.fps, detection_output.width, detection_output.height
    )
    # Carry the most recent sampled detections forward so the overlay reads as
    # continuous instead of flashing once every frame_stride frames.
    last_detections: list[dict[str, Any]] = []
    try:
        for frame_id, frame in iter_video_frames(video_path):
            if frame_id in detections_by_frame:
                last_detections = detections_by_frame[frame_id]
            draw_detection_frame(frame, last_detections)
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def _run_tracking_and_render(
    *,
    video_path: Path,
    detections_path: Path,
    work_dir: Path,
    sam2_predictor: Any,
    settings: ApiSettings,
) -> Path:
    metadata = video_metadata(video_path)
    width = int(metadata["width"])
    height = int(metadata["height"])

    player_like = [
        "player",
        "player-in-possession",
        "player-jump-shot",
        "player-layup-dunk",
        "player-shot-block",
    ]
    prompt_boxes, prompt_source = load_frame0_boxes(
        detections_path, None, width, height, player_like
    )
    if not prompt_boxes:
        raise RuntimeError("No frame-0 player boxes were detected — cannot prompt SAM2.")

    mask_dir = work_dir / "masks"
    ensure_dir(mask_dir)
    track_output, qa = build_track_output(
        video_path=video_path,
        prompt_boxes=prompt_boxes,
        prompt_source=prompt_source,
        backend="sam2",
        mask_dir=mask_dir,
        checkpoint=str(settings.sam2_checkpoint) if settings.sam2_checkpoint else None,
        model_cfg=settings.sam2_model_cfg,
        device=settings.sam2_device,
        cleanup_distance_threshold=settings.cleanup_distance_threshold,
        cleanup_min_component_area=settings.cleanup_min_component_area,
        max_frames=None,
        predictor=sam2_predictor,
    )

    tracks_path = work_dir / "tracks.json"
    write_json(tracks_path, track_output.model_dump(mode="json"))
    write_json(work_dir / "tracking_qa.json", qa)

    annotated_path = work_dir / "annotated.mp4"
    render_track_video(video_path, tracks_path, annotated_path, mask_alpha=0.35)
    return annotated_path
