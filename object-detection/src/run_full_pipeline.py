"""End-to-end pipeline runner.

Chains every stage of the basketball CV pipeline against a single video and
writes all intermediate artifacts plus the final side-by-side visualization
to one output directory:

    detections.json          YOLO per-frame object detections
    ball.json                YOLO-only ball detections + smoothing
    tracks.json + masks/     SAM2 player tracks
    crops/                   1 FPS player crops for clustering
    court_keypoints.json     fine-tuned YOLO11m-pose court calibration
    court_tracks.json        players + ball projected to NBA court coords
    court_mapping_qa.json    homography / projection diagnostics
    teams.json               SigLIP -> UMAP -> KMeans team assignments
    jersey_numbers.json      OCR + IoS jersey-number matches
    annotated_tracks.mp4     broadcast view with team colours + jersey numbers
    annotated_court.mp4      side-by-side broadcast + 2D court view

Stages can be turned off with ``--no-<stage>`` flags. The defaults assume the
fine-tuned YOLO weights and the local fine-tuned court keypoint weights are
already present on disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv

from .ball_tracking import run_ball_tracking
from .config import load_config, nested_get
from .court_keypoints import run_court_keypoints
from .detection import run_detection
from .export_crops import export_crops
from .project_to_court import run_projection
from .tracking import build_track_output, load_frame0_boxes
from .utils import ensure_dir, read_json, video_metadata, write_json
from .visualize_court import render_side_by_side
from .visualize_tracks import build_track_jersey_lookup, render_track_video


_NUMBERS_DIR = Path(__file__).resolve().parents[2] / "numbers"
if _NUMBERS_DIR.exists() and str(_NUMBERS_DIR) not in sys.path:
    sys.path.insert(0, str(_NUMBERS_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full basketball CV pipeline against one video.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output-dir", required=True, help="Directory to write all pipeline artifacts to.")
    parser.add_argument("--config", default="object-detection/config/default.yaml")

    parser.add_argument("--frame-stride", type=int, help="Detection sampling stride (default from config).")
    parser.add_argument("--ball-frame-stride", type=int, help="Ball detection stride (default from config, usually 1).")
    parser.add_argument("--court-frame-stride", type=int, help="Court keypoint stride (default from config, usually 15).")

    parser.add_argument("--yolo-weights", help="Override YOLO detection weights path.")
    parser.add_argument("--court-weights", help="Override court keypoint YOLO weights path.")
    parser.add_argument("--sam2-checkpoint", help="Override SAM2 checkpoint path.")
    parser.add_argument("--sam2-model-cfg", help="Override SAM2 model cfg.")
    parser.add_argument("--device", help="Device override for YOLO + SAM2 (e.g. cpu, mps, cuda, 0).")

    parser.add_argument("--max-frames", type=int, help="Cap total processed frames (smoke tests).")

    parser.add_argument("--no-tracking", action="store_true", help="Skip SAM2 tracking and everything downstream of it.")
    parser.add_argument("--no-clustering", action="store_true", help="Skip team clustering.")
    parser.add_argument("--no-ocr", action="store_true", help="Skip jersey-number OCR + matching.")
    parser.add_argument("--no-court-mapping", action="store_true", help="Skip ball detection, court keypoints, projection, and the side-by-side render.")

    parser.add_argument("--ocr-model", default="resnet", help="OCR model name: resnet | resnet18 | x2 | smol.")
    parser.add_argument("--ocr-ios-threshold", type=float, default=0.9)
    parser.add_argument("--ocr-device", default="auto")
    parser.add_argument("--clustering-device", default="cpu")
    parser.add_argument("--clustering-batch-size", type=int, default=32)
    parser.add_argument("--clustering-n-teams", type=int, default=2)

    parser.add_argument("--mask-alpha", type=float, default=0.35)
    parser.add_argument("--court-mask-alpha", type=float, default=0.25)
    parser.add_argument("--panel-height", type=int)
    parser.add_argument("--trail-length", type=int)
    return parser


def _argns(**kwargs: Any) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _detection_args(args: argparse.Namespace, video: Path, detections_path: Path) -> argparse.Namespace:
    return _argns(
        video=str(video),
        output=str(detections_path),
        debug_video=None,
        debug_frame_dir=str(detections_path.parent / "debug_detections"),
        debug_frame_limit=10,
        frame_stride=args.frame_stride,
        config=args.config,
        detector_backend="yolo",
        api_url=None,
        model_id=None,
        model_version=None,
        confidence=None,
        overlap=None,
        timeout_seconds=None,
        api_key_env="ROBOFLOW_API_KEY",
        yolo_weights=args.yolo_weights,
        yolo_imgsz=None,
        yolo_conf=None,
        yolo_iou=None,
        yolo_device=args.device,
        mock_response=None,
        max_frames=args.max_frames,
    )


def _ball_args(args: argparse.Namespace, video: Path, ball_path: Path) -> argparse.Namespace:
    return _argns(
        video=str(video),
        output=str(ball_path),
        frame_stride=args.ball_frame_stride,
        config=args.config,
        api_url=None,
        model_id=None,
        model_version=None,
        confidence=None,
        overlap=None,
        min_confidence=None,
        max_missing_gap=None,
        max_jump_px=None,
        prediction_gate_px=None,
        smoothing_window=None,
        api_key_env="ROBOFLOW_API_KEY",
        mock_response=None,
        max_frames=args.max_frames,
        detector_backend="yolo",
        yolo_weights=args.yolo_weights,
        yolo_imgsz=None,
        yolo_conf=None,
        yolo_iou=None,
        yolo_device=args.device,
    )


def _court_keypoints_args(args: argparse.Namespace, video: Path, court_kp_path: Path) -> argparse.Namespace:
    # Use local YOLO weights when provided; otherwise let court_keypoints.py
    # read the backend from config (defaults to "roboflow").
    court_weights = getattr(args, "court_weights", None)
    backend = "yolo" if court_weights else None
    return _argns(
        video=str(video),
        output=str(court_kp_path),
        debug_frame_dir=str(court_kp_path.parent / "debug_court_keypoints"),
        debug_frame_limit=10,
        frame_stride=args.court_frame_stride,
        config=args.config,
        backend=backend,
        api_url=None,
        model_id=None,
        model_version=None,
        confidence=None,
        overlap=None,
        weights=court_weights,
        yolo_confidence=None,
        device=args.device,
        keypoint_confidence=None,
        min_keypoints=None,
        api_key_env="ROBOFLOW_API_KEY",
        mock_response=None,
        max_frames=args.max_frames,
    )


def _projection_args(
    args: argparse.Namespace,
    video: Path,
    tracks_path: Path,
    court_kp_path: Path,
    ball_path: Path,
    court_tracks_path: Path,
    qa_path: Path,
) -> argparse.Namespace:
    return _argns(
        video=str(video),
        tracks=str(tracks_path),
        court_keypoints=str(court_kp_path),
        ball=str(ball_path),
        output=str(court_tracks_path),
        qa_report=str(qa_path),
        config=args.config,
        keypoint_confidence=None,
        min_keypoints=None,
        class_id_offset=None,
        out_of_bounds_margin_ft=None,
        smooth_window=None,
        ball_smooth_window=None,
        max_player_jump_ft=None,
        max_ball_jump_ft=None,
        max_reused_player_frames=None,
        disable_projection_smoothing=False,
        ball_player_snap_distance_px=None,
        ball_max_player_distance_ft=None,
        ball_max_rim_distance_ft=None,
        disable_ball_player_snap=False,
        disable_ball_context_filter=False,
    )


def _court_visualization_args(
    args: argparse.Namespace,
    video: Path,
    tracks_path: Path,
    court_tracks_path: Path,
    output_path: Path,
) -> argparse.Namespace:
    return _argns(
        video=str(video),
        tracks=str(tracks_path),
        court_tracks=str(court_tracks_path),
        output=str(output_path),
        config=args.config,
        panel_height=args.panel_height,
        trail_length=args.trail_length,
        mask_alpha=args.court_mask_alpha,
        teams=None,
        jersey_numbers=None,
    )


def _run_tracking(
    *,
    video: Path,
    detections_path: Path,
    tracks_path: Path,
    mask_dir: Path,
    qa_path: Path,
    config: dict[str, Any],
    sam2_checkpoint: str,
    sam2_model_cfg: str,
    device: str,
) -> dict[str, Any]:
    metadata = video_metadata(video)
    width, height = int(metadata["width"]), int(metadata["height"])
    player_like = list(nested_get(config, "detection.player_like_classes", []) or []) or [
        "player",
        "player-in-possession",
        "player-jump-shot",
        "player-layup-dunk",
        "player-shot-block",
    ]
    prompt_boxes, prompt_source = load_frame0_boxes(detections_path, None, width, height, player_like)
    if not prompt_boxes:
        raise RuntimeError("No frame-0 player boxes detected — cannot prompt SAM2.")
    cleanup = nested_get(config, "mask_cleanup", {}) or {}
    track_output, qa = build_track_output(
        video_path=video,
        prompt_boxes=prompt_boxes,
        prompt_source=prompt_source,
        backend="sam2",
        mask_dir=mask_dir,
        checkpoint=sam2_checkpoint,
        model_cfg=sam2_model_cfg,
        device=device,
        cleanup_distance_threshold=float(cleanup.get("distance_threshold", 80.0)),
        cleanup_min_component_area=int(cleanup.get("min_component_area", 50)),
        max_frames=None,
    )
    write_json(tracks_path, track_output.model_dump(mode="json"))
    write_json(qa_path, qa)
    return track_output.model_dump(mode="json")


def _run_clustering(
    *,
    crops_dir: Path,
    teams_path: Path,
    device: str,
    batch_size: int,
    n_teams: int,
) -> dict[int, str]:
    from clustering.assign import build_track_team_lookup
    from clustering.cli import run as run_clustering

    cluster_args = _argns(
        crops_dir=crops_dir,
        output=teams_path,
        device=device,
        batch_size=batch_size,
        n_teams=n_teams,
        method="siglip+umap+kmeans",
    )
    team_output = run_clustering(cluster_args)
    return build_track_team_lookup(team_output)


def _enrich_tracks_with_teams(tracks_path: Path, team_lookup: dict[int, str]) -> None:
    tracks_data = read_json(tracks_path)
    for frame in tracks_data.get("frames", []):
        for track in frame.get("tracks", []):
            track_id = track.get("track_id")
            if track_id is not None and int(track_id) in team_lookup:
                track["team_name"] = team_lookup[int(track_id)]
    write_json(tracks_path, tracks_data)


def _run_ocr_matching(
    *,
    video: Path,
    detections_path: Path,
    tracks_path: Path,
    mask_dir: Path,
    output_path: Path,
    ocr_model_name: str,
    ios_threshold: float,
    device: str,
) -> dict[str, list[dict[str, Any]]]:
    from jersey_numbers.matching.ios import match_frame
    from jersey_numbers.ocr.model_factory import create_ocr_model

    detections_data = read_json(detections_path)
    tracks_data = read_json(tracks_path)
    ocr_model = create_ocr_model(ocr_model_name, device=device)

    detections_by_frame: dict[int, list[dict[str, Any]]] = {
        int(frame["frame_id"]): [
            det for det in frame.get("detections", []) if "number" in det.get("class_name", "").lower()
        ]
        for frame in detections_data.get("frames", [])
    }

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for OCR: {video}")

    jersey_numbers: dict[str, list[dict[str, Any]]] = {}
    try:
        for frame_data in tracks_data.get("frames", []):
            frame_id = int(frame_data["frame_id"])
            number_detections = detections_by_frame.get(frame_id, [])
            if not number_detections:
                continue
            tracks = frame_data.get("tracks", [])
            if not tracks:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                continue
            matches = match_frame(
                frame_id=frame_id,
                number_detections=number_detections,
                tracks=tracks,
                ios_threshold=ios_threshold,
                frame_image=frame,
                ocr_model=ocr_model,
            )
            if matches:
                jersey_numbers[str(frame_id)] = matches
    finally:
        cap.release()

    write_json(output_path, jersey_numbers)
    return jersey_numbers


def run_pipeline(args: argparse.Namespace) -> dict[str, Path]:
    load_dotenv()
    config = load_config(args.config)
    video = Path(args.video).resolve()
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    detections_path = output_dir / "detections.json"
    ball_path = output_dir / "ball.json"
    tracks_path = output_dir / "tracks.json"
    mask_dir = output_dir / "masks"
    crops_dir = output_dir / "crops"
    tracking_qa_path = output_dir / "tracking_qa.json"
    teams_path = output_dir / "teams.json"
    court_kp_path = output_dir / "court_keypoints.json"
    court_tracks_path = output_dir / "court_tracks.json"
    court_qa_path = output_dir / "court_mapping_qa.json"
    jersey_path = output_dir / "jersey_numbers.json"
    annotated_tracks_path = output_dir / "annotated_tracks.mp4"
    annotated_court_path = output_dir / "annotated_court.mp4"

    artifacts: dict[str, Path] = {}

    print("[1/9] YOLO detection")
    run_detection(_detection_args(args, video, detections_path))
    artifacts["detections"] = detections_path

    if args.no_court_mapping:
        ball_path = None  # type: ignore[assignment]
    else:
        print("[2/9] Ball detection (YOLO, every frame)")
        run_ball_tracking(_ball_args(args, video, ball_path))
        artifacts["ball"] = ball_path

    if args.no_tracking:
        if not tracks_path.exists():
            print("Skipping tracking and downstream stages (no existing tracks found).")
            return artifacts
        print(f"[3/9] Skipping SAM2 — reusing {tracks_path}")
    else:
        print("[3/9] SAM2 tracking")
        sam2_checkpoint = args.sam2_checkpoint or str(nested_get(config, "tracking.sam2_checkpoint"))
        sam2_model_cfg = args.sam2_model_cfg or str(nested_get(config, "tracking.sam2_model_cfg"))
        sam2_device = args.device or str(nested_get(config, "tracking.device", "auto"))
        _run_tracking(
            video=video,
            detections_path=detections_path,
            tracks_path=tracks_path,
            mask_dir=mask_dir,
            qa_path=tracking_qa_path,
            config=config,
            sam2_checkpoint=sam2_checkpoint,
            sam2_model_cfg=sam2_model_cfg,
            device=sam2_device,
        )
    artifacts["tracks"] = tracks_path
    artifacts["masks"] = mask_dir

    print("[4/9] Crop export (1 FPS)")
    crop_summary = export_crops(video, read_json(tracks_path), crops_dir, target_fps=1.0)
    qa = read_json(tracking_qa_path) if tracking_qa_path.exists() else {}
    qa["crop_summary"] = crop_summary
    write_json(tracking_qa_path, qa)
    artifacts["crops"] = crops_dir

    team_lookup: dict[int, str] | None = None
    if args.no_clustering and teams_path.exists():
        print(f"[5/9] Skipping clustering — reusing {teams_path}")
        from clustering.assign import build_track_team_lookup
        from clustering.schemas import TeamOutput
        team_lookup = build_track_team_lookup(TeamOutput.model_validate_json(teams_path.read_text()))
        artifacts["teams"] = teams_path
    elif not args.no_clustering and crop_summary.get("saved_crops", 0) > 0:
        print("[5/9] Team clustering")
        team_lookup = _run_clustering(
            crops_dir=crops_dir,
            teams_path=teams_path,
            device=args.clustering_device,
            batch_size=args.clustering_batch_size,
            n_teams=args.clustering_n_teams,
        )
        if team_lookup:
            _enrich_tracks_with_teams(tracks_path, team_lookup)
        artifacts["teams"] = teams_path
    else:
        print("[5/9] Skipping team clustering")

    jersey_lookup: dict[int, str] | None = None
    if not args.no_ocr:
        print("[6/9] OCR + jersey-number matching")
        jersey_numbers = _run_ocr_matching(
            video=video,
            detections_path=detections_path,
            tracks_path=tracks_path,
            mask_dir=mask_dir,
            output_path=jersey_path,
            ocr_model_name=args.ocr_model,
            ios_threshold=args.ocr_ios_threshold,
            device=args.ocr_device,
        )
        jersey_lookup = build_track_jersey_lookup(jersey_numbers)
        artifacts["jersey_numbers"] = jersey_path
    else:
        print("[6/9] Skipping OCR")

    if not args.no_court_mapping:
        print("[7/9] Court keypoints (local YOLO11m-pose)")
        run_court_keypoints(_court_keypoints_args(args, video, court_kp_path))
        artifacts["court_keypoints"] = court_kp_path

        print("[8/9] Court projection")
        run_projection(
            _projection_args(args, video, tracks_path, court_kp_path, ball_path, court_tracks_path, court_qa_path)
        )
        artifacts["court_tracks"] = court_tracks_path
    else:
        print("[7/9, 8/9] Skipping court mapping")

    print("[9a/9] Rendering broadcast view (annotated_tracks.mp4)")
    render_track_video(
        video,
        tracks_path,
        annotated_tracks_path,
        mask_alpha=args.mask_alpha,
        team_lookup=team_lookup,
        jersey_lookup=jersey_lookup,
    )
    artifacts["annotated_tracks"] = annotated_tracks_path

    if not args.no_court_mapping:
        print("[9b/9] Rendering side-by-side (annotated_court.mp4)")
        render_side_by_side(
            _court_visualization_args(args, video, tracks_path, court_tracks_path, annotated_court_path),
            team_lookup=team_lookup,
            jersey_lookup=jersey_lookup,
        )
        artifacts["annotated_court"] = annotated_court_path

    print()
    print("Pipeline complete. Artifacts:")
    for key, path in artifacts.items():
        print(f"  {key:>20s}: {path}")
    return artifacts


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
