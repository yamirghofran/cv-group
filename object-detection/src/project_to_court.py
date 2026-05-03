from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, nested_get
from .court_geometry import (
    HomographyResult,
    NBACourt,
    bottom_center_from_bbox,
    compute_homography_from_keypoints,
    is_inside_court,
    project_points,
)
from .schemas import (
    BallFrame,
    CourtBallProjection,
    CourtKeypointFrame,
    CourtPlayerProjection,
    CourtProjectionFrame,
    CourtProjectionOutput,
    TrackFrame,
)
from .utils import read_json, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project tracked players and ball movement to a 2D basketball court.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--court-keypoints", required=True)
    parser.add_argument("--ball", help="Optional ball tracking JSON.")
    parser.add_argument("--output", help="Output court-coordinate JSON path.")
    parser.add_argument("--qa-report", help="Output court mapping QA JSON path.")
    parser.add_argument("--config", default="object-detection/config/default.yaml")
    parser.add_argument("--keypoint-confidence", type=float)
    parser.add_argument("--min-keypoints", type=int)
    parser.add_argument("--class-id-offset", type=int)
    parser.add_argument("--out-of-bounds-margin-ft", type=float)
    return parser


def default_output(video_path: str | Path) -> Path:
    return Path("object-detection/outputs/court_tracks") / f"{Path(video_path).stem}_court_tracks.json"


def default_qa_report(video_path: str | Path) -> Path:
    return Path("object-detection/outputs/reports") / f"{Path(video_path).stem}_court_mapping_qa.json"


def frames_by_id(frames: list[dict[str, Any]], validator):
    return {int(frame["frame_id"]): validator.model_validate(frame) for frame in frames}


def build_homographies(
    court_keypoint_data: dict[str, Any],
    court: NBACourt,
    min_confidence: float,
    min_keypoints: int,
    class_id_offset: int,
) -> list[HomographyResult]:
    homographies: list[HomographyResult] = []
    for raw_frame in court_keypoint_data.get("frames", []):
        frame = CourtKeypointFrame.model_validate(raw_frame)
        matrix, meta = compute_homography_from_keypoints(
            frame.keypoints,
            court=court,
            min_confidence=min_confidence,
            min_keypoints=min_keypoints,
            class_id_offset=class_id_offset,
        )
        if matrix is None:
            continue
        homographies.append(
            HomographyResult(
                matrix=matrix,
                source_frame_id=frame.frame_id,
                inliers=int(meta["inliers"]),
                used_keypoint_ids=list(meta["used_keypoint_ids"]),  # type: ignore[arg-type]
            )
        )
    return sorted(homographies, key=lambda item: item.source_frame_id)


def nearest_homography(homographies: list[HomographyResult], frame_id: int) -> HomographyResult | None:
    if not homographies:
        return None
    return min(homographies, key=lambda item: abs(item.source_frame_id - frame_id))


def project_track_frame(
    frame: TrackFrame,
    homography: HomographyResult | None,
    court: NBACourt,
    margin: float,
) -> list[CourtPlayerProjection]:
    players: list[CourtPlayerProjection] = []
    image_points = [bottom_center_from_bbox(track.bbox_xyxy) for track in frame.tracks]
    projected = project_points(homography.matrix, image_points) if homography is not None and image_points else np.empty((0, 2))
    for index, track in enumerate(frame.tracks):
        court_xy = projected[index].tolist() if homography is not None and len(projected) > index else None
        valid = court_xy is not None and is_inside_court(court_xy, court, margin)
        players.append(
            CourtPlayerProjection(
                track_id=track.track_id,
                image_xy=image_points[index],
                court_xy=[float(court_xy[0]), float(court_xy[1])] if valid and court_xy is not None else None,
                valid=valid,
            )
        )
    return players


def project_ball_frame(
    frame_id: int,
    ball_frames: dict[int, BallFrame],
    homography: HomographyResult | None,
    court: NBACourt,
    margin: float,
) -> CourtBallProjection | None:
    ball_frame = ball_frames.get(frame_id)
    if ball_frame is None:
        return None
    if ball_frame.image_xy is None or homography is None:
        return CourtBallProjection(image_xy=ball_frame.image_xy, source=ball_frame.source, valid=False)
    projected = project_points(homography.matrix, [ball_frame.image_xy])[0].tolist()
    valid = is_inside_court(projected, court, margin)
    return CourtBallProjection(
        image_xy=ball_frame.image_xy,
        court_xy=[float(projected[0]), float(projected[1])] if valid else None,
        source=ball_frame.source,
        valid=valid,
    )


def run_projection(args: argparse.Namespace) -> CourtProjectionOutput:
    config = load_config(args.config)
    video_path = Path(args.video)
    tracks_data = read_json(args.tracks)
    court_keypoint_data = read_json(args.court_keypoints)
    ball_data = read_json(args.ball) if args.ball else None
    court = NBACourt(
        length=float(nested_get(config, "court_projection.court_length_ft", 94.0)),
        width=float(nested_get(config, "court_projection.court_width_ft", 50.0)),
    )
    min_confidence = (
        args.keypoint_confidence
        if args.keypoint_confidence is not None
        else float(nested_get(config, "court_keypoints.keypoint_confidence", 0.5))
    )
    min_keypoints = (
        args.min_keypoints if args.min_keypoints is not None else int(nested_get(config, "court_keypoints.min_keypoints", 4))
    )
    class_id_offset = (
        args.class_id_offset
        if args.class_id_offset is not None
        else int(nested_get(config, "court_keypoints.class_id_offset", 0))
    )
    margin = (
        args.out_of_bounds_margin_ft
        if args.out_of_bounds_margin_ft is not None
        else float(nested_get(config, "court_projection.out_of_bounds_margin_ft", 5.0))
    )

    homographies = build_homographies(court_keypoint_data, court, min_confidence, min_keypoints, class_id_offset)
    if not homographies:
        raise RuntimeError("No valid court homographies found. Check court keypoint debug frames or lower thresholds.")

    track_frames = [TrackFrame.model_validate(frame) for frame in tracks_data.get("frames", [])]
    ball_frames = frames_by_id(ball_data.get("frames", []), BallFrame) if ball_data else {}
    output_frames: list[CourtProjectionFrame] = []
    missing_homographies = 0
    invalid_player_points = 0
    invalid_ball_points = 0

    for frame in track_frames:
        homography = nearest_homography(homographies, frame.frame_id)
        if homography is None:
            missing_homographies += 1
        players = project_track_frame(frame, homography, court, margin)
        invalid_player_points += sum(1 for player in players if not player.valid)
        ball = project_ball_frame(frame.frame_id, ball_frames, homography, court, margin) if ball_data else None
        if ball is not None and not ball.valid and ball.source != "missing":
            invalid_ball_points += 1
        output_frames.append(
            CourtProjectionFrame(
                frame_id=frame.frame_id,
                timestamp_sec=frame.timestamp_sec,
                homography_source_frame=homography.source_frame_id if homography is not None else None,
                players=players,
                ball=ball,
            )
        )

    output = CourtProjectionOutput(
        video=str(video_path),
        fps=float(tracks_data["fps"]),
        width=int(tracks_data["width"]),
        height=int(tracks_data["height"]),
        court_length_ft=court.length,
        court_width_ft=court.width,
        frames=output_frames,
    )
    output_path = Path(args.output) if args.output else default_output(video_path)
    qa_path = Path(args.qa_report) if args.qa_report else default_qa_report(video_path)
    write_json(output_path, output.model_dump(mode="json"))
    write_json(
        qa_path,
        {
            "video": str(video_path),
            "homographies": len(homographies),
            "homography_source_frames": [item.source_frame_id for item in homographies],
            "frames_projected": len(output_frames),
            "missing_homography_frames": missing_homographies,
            "invalid_player_points": invalid_player_points,
            "invalid_ball_points": invalid_ball_points,
        },
    )
    print(f"Wrote court projection: {output_path}")
    print(f"Wrote court mapping QA: {qa_path}")
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_projection(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
