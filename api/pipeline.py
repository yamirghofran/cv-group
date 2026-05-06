from __future__ import annotations

import argparse
import sys
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
    read_json,
    should_process_frame,
    video_metadata,
    write_json,
)
from src.visualize_tracks import render_track_video
from src.yolo_client import YoloLocalClient

from .config import ApiSettings

# Add numbers directory to path for OCR imports
_NUMBERS_DIR = Path(__file__).parent.parent / "numbers"
if str(_NUMBERS_DIR) not in sys.path:
    sys.path.insert(0, str(_NUMBERS_DIR))

from jersey_numbers.matching.ios import match_frame, crop_number, load_mask


@dataclass
class ProcessResult:
    job_id: str
    work_dir: Path
    detections_path: Path
    classes_seen: list[str]
    frames_sampled: int
    annotated_video_path: Path
    sam2_used: bool
    ocr_enabled: bool
    teams_path: Path | None = None
    jersey_numbers: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def process_video(
    *,
    video_path: Path,
    yolo: YoloLocalClient,
    sam2_predictor: Any | None,
    ocr_model: Any | None,
    settings: ApiSettings,
) -> ProcessResult:
    """Run YOLO detection (always) and SAM2 tracking + render (if predictor is loaded).

    When SAM2 tracking produces player crops, the pipeline also runs the
    team-clustering stage and annotates the output video with team-coloured
    overlays and labels.

    When OCR is enabled and SAM2 tracking is active, jersey number recognition
    is run as an additional step after tracking.

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
    teams_path: Path | None = None
    tracks_path: Path | None = None
    mask_dir: Path | None = None

    if sam2_predictor is not None:
        crop_dir = work_dir / "crops"
        annotated_video_path, tracks_path, mask_dir, teams_path = (
            _run_tracking_and_render(
                video_path=video_path,
                detections_path=detections_path,
                work_dir=work_dir,
                crop_dir=crop_dir,
                sam2_predictor=sam2_predictor,
                settings=settings,
            )
        )
    else:
        _render_yolo_overlay_video(
            video_path=video_path,
            detection_output=detection_output,
            output_path=annotated_video_path,
        )

    # Run OCR if enabled and we have tracks with masks
    jersey_numbers: dict[int, list[dict[str, Any]]] = {}
    if ocr_model is not None and tracks_path is not None and mask_dir is not None:
        jersey_numbers = _run_ocr_matching(
            video_path=video_path,
            tracks_path=tracks_path,
            detections_path=detections_path,
            mask_dir=mask_dir,
            work_dir=work_dir,
            ocr_model=ocr_model,
            settings=settings,
        )

    return ProcessResult(
        job_id=job_id,
        work_dir=work_dir,
        detections_path=detections_path,
        classes_seen=list(detection_output.classes_seen),
        frames_sampled=len(detection_output.frames),
        annotated_video_path=annotated_video_path,
        sam2_used=sam2_predictor is not None,
        ocr_enabled=ocr_model is not None,
        teams_path=teams_path,
        jersey_numbers=jersey_numbers,
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
    crop_dir: Path,
    sam2_predictor: Any,
    settings: ApiSettings,
) -> tuple[Path, Path, Path, Path | None]:
    """Run SAM2 tracking, optional team clustering, and render the annotated video.

    Returns ``(annotated_video_path, tracks_path, mask_dir, teams_path_or_none)``.
    """
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

    # Export player crops for downstream team clustering
    from src.export_crops import export_crops

    crop_summary = export_crops(
        video_path, track_output.model_dump(mode="json"), crop_dir, target_fps=1.0
    )
    qa["crop_summary"] = crop_summary
    write_json(work_dir / "tracking_qa.json", qa)

    # --- Team clustering ---
    teams_path: Path | None = None
    team_lookup: dict[int, str] | None = None

    if crop_summary.get("saved_crops", 0) > 0:
        teams_path, team_lookup = _run_team_clustering(
            crop_dir=crop_dir,
            work_dir=work_dir,
            settings=settings,
        )
        if team_lookup:
            _enrich_tracks_with_teams(tracks_path, team_lookup)

    annotated_path = work_dir / "annotated.mp4"
    render_track_video(
        video_path, tracks_path, annotated_path,
        mask_alpha=0.35,
        team_lookup=team_lookup,
    )
    return annotated_path, tracks_path, mask_dir, teams_path


def _run_team_clustering(
    *,
    crop_dir: Path,
    work_dir: Path,
    settings: ApiSettings,
) -> tuple[Path, dict[int, str]]:
    """Run the team-clustering stage on exported crops.

    Returns ``(teams_json_path, track_id_to_team_name_lookup)``.
    """
    from clustering.assign import build_track_team_lookup
    from clustering.cli import run as run_clustering

    teams_path = work_dir / "teams.json"
    args = argparse.Namespace(
        crops_dir=crop_dir,
        output=teams_path,
        device=settings.clustering_device,
        batch_size=settings.clustering_batch_size,
        n_teams=settings.clustering_n_teams,
        method="siglip+umap+kmeans",
    )
    team_output = run_clustering(args)

    team_lookup = build_track_team_lookup(team_output)
    return teams_path, team_lookup


def _enrich_tracks_with_teams(
    tracks_path: Path,
    team_lookup: dict[int, str],
) -> None:
    """Read *tracks_path*, annotate each TrackRecord with ``team_name``, and rewrite."""
    import json

    tracks_data = json.loads(tracks_path.read_text())

    for frame in tracks_data.get("frames", []):
        for track in frame.get("tracks", []):
            track_id = track.get("track_id")
            if track_id is not None and track_id in team_lookup:
                track["team_name"] = team_lookup[track_id]

    write_json(tracks_path, tracks_data)


def _run_ocr_matching(
    *,
    video_path: Path,
    tracks_path: Path,
    detections_path: Path,
    mask_dir: Path,
    work_dir: Path,
    ocr_model: Any,
    settings: ApiSettings,
) -> dict[int, list[dict[str, Any]]]:
    """Run OCR matching to associate jersey numbers with player tracks.

    Returns dict mapping frame_id to list of matches with jersey numbers.
    """
    import cv2

    # Load detections and tracks
    detections_data = read_json(detections_path)
    tracks_data = read_json(tracks_path)

    # Get number detections from YOLO (class_name contains "number")
    jersey_numbers: dict[int, list[dict[str, Any]]] = {}

    # Open video for frame extraction
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return jersey_numbers

    try:
        for frame_data in tracks_data.get("frames", []):
            frame_id = frame_data["frame_id"]

            # Get number detections for this frame
            number_detections = []
            for det_frame in detections_data.get("frames", []):
                if det_frame["frame_id"] == frame_id:
                    for det in det_frame.get("detections", []):
                        if "number" in det.get("class_name", "").lower():
                            number_detections.append(det)

            if not number_detections:
                continue

            # Get tracks for this frame
            tracks = frame_data.get("tracks", [])
            if not tracks:
                continue

            # Read frame from video
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue

            # Run matching with OCR
            matches = match_frame(
                frame_id=frame_id,
                number_detections=number_detections,
                tracks=tracks,
                ios_threshold=settings.ocr_ios_threshold,
                frame_image=frame,
                ocr_model=ocr_model,
            )

            if matches:
                jersey_numbers[frame_id] = matches

    finally:
        cap.release()

    # Save jersey numbers to JSON
    if jersey_numbers:
        write_json(work_dir / "jersey_numbers.json", jersey_numbers)

    return jersey_numbers
