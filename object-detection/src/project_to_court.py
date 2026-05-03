from __future__ import annotations

import argparse
from collections import defaultdict, deque
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
    parser.add_argument("--smooth-window", type=int, help="Moving average window for projected player court coordinates.")
    parser.add_argument("--ball-smooth-window", type=int, help="Moving average window for projected ball coordinates.")
    parser.add_argument("--max-player-jump-ft", type=float, help="Reject per-frame projected player jumps larger than this.")
    parser.add_argument("--max-ball-jump-ft", type=float, help="Reject per-frame projected ball jumps larger than this.")
    parser.add_argument(
        "--max-reused-player-frames",
        type=int,
        help="Maximum consecutive frames to reuse a player's last good court point after bad projection.",
    )
    parser.add_argument("--disable-projection-smoothing", action="store_true")
    parser.add_argument(
        "--ball-player-snap-distance-px",
        type=float,
        help="Snap ball to nearest player when image-space ball is this close to a player bbox.",
    )
    parser.add_argument(
        "--ball-max-player-distance-ft",
        type=float,
        help="Keep direct ball projections only when close enough to a projected player.",
    )
    parser.add_argument(
        "--ball-max-rim-distance-ft",
        type=float,
        help="Keep direct ball projections near either rim even if not close to a player.",
    )
    parser.add_argument("--disable-ball-player-snap", action="store_true")
    parser.add_argument("--disable-ball-context-filter", action="store_true")
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


def bbox_distance_to_point(point_xy: list[float], bbox_xyxy: list[float], padding_px: float = 0.0) -> float:
    x, y = [float(value) for value in point_xy]
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    x1 -= padding_px
    y1 -= padding_px
    x2 += padding_px
    y2 += padding_px
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy))


def nearest_player_to_ball(
    ball_image_xy: list[float],
    track_frame: TrackFrame,
    players: list[CourtPlayerProjection],
    snap_distance_px: float,
) -> tuple[CourtPlayerProjection | None, float]:
    players_by_track_id = {player.track_id: player for player in players if player.valid and player.court_xy is not None}
    nearest_player: CourtPlayerProjection | None = None
    nearest_distance = float("inf")
    for track in track_frame.tracks:
        player = players_by_track_id.get(track.track_id)
        if player is None:
            continue
        x1, y1, x2, y2 = [float(value) for value in track.bbox_xyxy]
        dynamic_padding = min(30.0, 0.10 * max(x2 - x1, y2 - y1))
        distance = bbox_distance_to_point(ball_image_xy, track.bbox_xyxy, padding_px=dynamic_padding)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_player = player
    return nearest_player, nearest_distance


def nearest_player_court_distance(ball_court_xy: list[float], players: list[CourtPlayerProjection]) -> float:
    distances = [
        point_distance(ball_court_xy, [float(player.court_xy[0]), float(player.court_xy[1])])
        for player in players
        if player.valid and player.court_xy is not None
    ]
    return min(distances, default=float("inf"))


def nearest_rim_distance(ball_court_xy: list[float], court: NBACourt) -> float:
    rim_points = [list(court.vertices[6]), list(court.vertices[26])]
    return min(point_distance(ball_court_xy, rim_point) for rim_point in rim_points)


def resolve_ball_projection(
    frame_id: int,
    ball_frames: dict[int, BallFrame],
    homography: HomographyResult | None,
    court: NBACourt,
    margin: float,
    track_frame: TrackFrame,
    players: list[CourtPlayerProjection],
    snap_distance_px: float,
    max_player_distance_ft: float,
    max_rim_distance_ft: float,
    player_snap_enabled: bool,
    context_filter_enabled: bool,
) -> tuple[CourtBallProjection | None, dict[str, int]]:
    stats = {
        "ball_direct_accepted": 0,
        "ball_player_snaps": 0,
        "ball_airborne_or_uncertain": 0,
        "ball_direct_out_of_bounds": 0,
    }
    ball_frame = ball_frames.get(frame_id)
    if ball_frame is None:
        return None, stats
    if ball_frame.image_xy is None or homography is None:
        return CourtBallProjection(image_xy=ball_frame.image_xy, source=ball_frame.source, valid=False), stats

    source = ball_frame.source
    ball_image_xy = [float(ball_frame.image_xy[0]), float(ball_frame.image_xy[1])]
    direct_xy = project_points(homography.matrix, [ball_image_xy])[0].tolist()
    direct_court_xy = [float(direct_xy[0]), float(direct_xy[1])]
    direct_inside_court = is_inside_court(direct_court_xy, court, margin)
    nearest_image_player, image_distance = nearest_player_to_ball(ball_image_xy, track_frame, players, snap_distance_px)

    if player_snap_enabled and nearest_image_player is not None and image_distance <= snap_distance_px:
        stats["ball_player_snaps"] += 1
        return (
            CourtBallProjection(
                image_xy=ball_image_xy,
                court_xy=[float(nearest_image_player.court_xy[0]), float(nearest_image_player.court_xy[1])],  # type: ignore[index]
                source=f"player_snap_{source}",
                valid=True,
            ),
            stats,
        )

    if not direct_inside_court:
        stats["ball_direct_out_of_bounds"] += 1
        stats["ball_airborne_or_uncertain"] += 1
        return CourtBallProjection(image_xy=ball_image_xy, source=f"airborne_or_uncertain_{source}", valid=False), stats

    if context_filter_enabled:
        player_distance = nearest_player_court_distance(direct_court_xy, players)
        rim_distance = nearest_rim_distance(direct_court_xy, court)
        if player_distance > max_player_distance_ft and rim_distance > max_rim_distance_ft:
            stats["ball_airborne_or_uncertain"] += 1
            return CourtBallProjection(image_xy=ball_image_xy, source=f"airborne_or_uncertain_{source}", valid=False), stats

    stats["ball_direct_accepted"] += 1
    return CourtBallProjection(image_xy=ball_image_xy, court_xy=direct_court_xy, source=f"direct_{source}", valid=True), stats


def point_distance(point_a: list[float], point_b: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(point_a, dtype=np.float32) - np.asarray(point_b, dtype=np.float32)))


def mean_point(points: deque[list[float]]) -> list[float]:
    array = np.asarray(points, dtype=np.float32)
    mean = array.mean(axis=0)
    return [float(mean[0]), float(mean[1])]


def smooth_projection_frames(
    frames: list[CourtProjectionFrame],
    player_window: int,
    ball_window: int,
    max_player_jump_ft: float,
    max_ball_jump_ft: float,
    max_reused_player_frames: int,
) -> tuple[list[CourtProjectionFrame], dict[str, int]]:
    player_window = max(1, player_window)
    ball_window = max(1, ball_window)
    max_reused_player_frames = max(0, max_reused_player_frames)

    player_histories: dict[int, deque[list[float]]] = defaultdict(lambda: deque(maxlen=player_window))
    player_last_good: dict[int, list[float]] = {}
    player_reuse_streak: dict[int, int] = defaultdict(int)
    ball_history: deque[list[float]] = deque(maxlen=ball_window)
    ball_last_good: list[float] | None = None
    stats = {
        "player_points_smoothed": 0,
        "player_jump_rejections": 0,
        "player_reused_previous_points": 0,
        "ball_points_smoothed": 0,
        "ball_jump_rejections": 0,
    }
    smoothed_frames: list[CourtProjectionFrame] = []

    for frame in frames:
        smoothed_players: list[CourtPlayerProjection] = []
        for player in frame.players:
            track_id = player.track_id
            raw_xy = [float(player.court_xy[0]), float(player.court_xy[1])] if player.valid and player.court_xy else None
            candidate_xy = raw_xy

            if raw_xy is not None:
                previous = player_last_good.get(track_id)
                if previous is not None and max_player_jump_ft > 0 and point_distance(raw_xy, previous) > max_player_jump_ft:
                    candidate_xy = previous
                    player_reuse_streak[track_id] += 1
                    stats["player_jump_rejections"] += 1
                else:
                    player_reuse_streak[track_id] = 0
            elif track_id in player_last_good and player_reuse_streak[track_id] < max_reused_player_frames:
                candidate_xy = player_last_good[track_id]
                player_reuse_streak[track_id] += 1
                stats["player_reused_previous_points"] += 1

            if candidate_xy is None:
                smoothed_players.append(player)
                continue

            history = player_histories[track_id]
            history.append(candidate_xy)
            smoothed_xy = mean_point(history) if player_window > 1 else candidate_xy
            player_last_good[track_id] = smoothed_xy
            smoothed_players.append(
                CourtPlayerProjection(
                    track_id=track_id,
                    image_xy=player.image_xy,
                    court_xy=smoothed_xy,
                    valid=True,
                )
            )
            stats["player_points_smoothed"] += 1

        ball = frame.ball
        smoothed_ball = ball
        if ball is not None and ball.valid and ball.court_xy is not None:
            raw_ball_xy = [float(ball.court_xy[0]), float(ball.court_xy[1])]
            candidate_ball_xy = raw_ball_xy
            if ball_last_good is not None and max_ball_jump_ft > 0 and point_distance(raw_ball_xy, ball_last_good) > max_ball_jump_ft:
                candidate_ball_xy = ball_last_good
                stats["ball_jump_rejections"] += 1
            ball_history.append(candidate_ball_xy)
            smoothed_ball_xy = mean_point(ball_history) if ball_window > 1 else candidate_ball_xy
            ball_last_good = smoothed_ball_xy
            smoothed_ball = CourtBallProjection(
                image_xy=ball.image_xy,
                court_xy=smoothed_ball_xy,
                source=ball.source,
                valid=True,
            )
            stats["ball_points_smoothed"] += 1

        smoothed_frames.append(
            CourtProjectionFrame(
                frame_id=frame.frame_id,
                timestamp_sec=frame.timestamp_sec,
                homography_source_frame=frame.homography_source_frame,
                players=smoothed_players,
                ball=smoothed_ball,
            )
        )

    return smoothed_frames, stats


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
    smoothing_enabled = not bool(getattr(args, "disable_projection_smoothing", False))
    player_smooth_window = (
        args.smooth_window
        if getattr(args, "smooth_window", None) is not None
        else int(nested_get(config, "court_projection.smoothing_window", 5))
    )
    ball_smooth_window = (
        args.ball_smooth_window
        if getattr(args, "ball_smooth_window", None) is not None
        else int(nested_get(config, "court_projection.ball_smoothing_window", 3))
    )
    max_player_jump = (
        args.max_player_jump_ft
        if getattr(args, "max_player_jump_ft", None) is not None
        else float(nested_get(config, "court_projection.max_player_jump_ft", 12.0))
    )
    max_ball_jump = (
        args.max_ball_jump_ft
        if getattr(args, "max_ball_jump_ft", None) is not None
        else float(nested_get(config, "court_projection.max_ball_jump_ft", 35.0))
    )
    max_reused_player_frames = (
        args.max_reused_player_frames
        if getattr(args, "max_reused_player_frames", None) is not None
        else int(nested_get(config, "court_projection.max_reused_player_frames", 10))
    )
    ball_player_snap_distance_px = (
        args.ball_player_snap_distance_px
        if getattr(args, "ball_player_snap_distance_px", None) is not None
        else float(nested_get(config, "court_projection.ball_player_snap_distance_px", 90.0))
    )
    ball_max_player_distance_ft = (
        args.ball_max_player_distance_ft
        if getattr(args, "ball_max_player_distance_ft", None) is not None
        else float(nested_get(config, "court_projection.ball_max_player_distance_ft", 20.0))
    )
    ball_max_rim_distance_ft = (
        args.ball_max_rim_distance_ft
        if getattr(args, "ball_max_rim_distance_ft", None) is not None
        else float(nested_get(config, "court_projection.ball_max_rim_distance_ft", 15.0))
    )
    ball_player_snap_enabled = not bool(getattr(args, "disable_ball_player_snap", False))
    ball_context_filter_enabled = not bool(getattr(args, "disable_ball_context_filter", False))

    homographies = build_homographies(court_keypoint_data, court, min_confidence, min_keypoints, class_id_offset)
    if not homographies:
        raise RuntimeError("No valid court homographies found. Check court keypoint debug frames or lower thresholds.")

    track_frames = [TrackFrame.model_validate(frame) for frame in tracks_data.get("frames", [])]
    ball_frames = frames_by_id(ball_data.get("frames", []), BallFrame) if ball_data else {}
    output_frames: list[CourtProjectionFrame] = []
    missing_homographies = 0
    raw_invalid_player_points = 0
    raw_invalid_ball_points = 0
    ball_projection_stats = {
        "ball_direct_accepted": 0,
        "ball_player_snaps": 0,
        "ball_airborne_or_uncertain": 0,
        "ball_direct_out_of_bounds": 0,
    }

    for frame in track_frames:
        homography = nearest_homography(homographies, frame.frame_id)
        if homography is None:
            missing_homographies += 1
        players = project_track_frame(frame, homography, court, margin)
        raw_invalid_player_points += sum(1 for player in players if not player.valid)
        if ball_data:
            ball, ball_stats = resolve_ball_projection(
                frame.frame_id,
                ball_frames,
                homography,
                court,
                margin,
                frame,
                players,
                snap_distance_px=ball_player_snap_distance_px,
                max_player_distance_ft=ball_max_player_distance_ft,
                max_rim_distance_ft=ball_max_rim_distance_ft,
                player_snap_enabled=ball_player_snap_enabled,
                context_filter_enabled=ball_context_filter_enabled,
            )
            for key, value in ball_stats.items():
                ball_projection_stats[key] += value
        else:
            ball = None
        if ball is not None and not ball.valid and ball.source != "missing":
            raw_invalid_ball_points += 1
        output_frames.append(
            CourtProjectionFrame(
                frame_id=frame.frame_id,
                timestamp_sec=frame.timestamp_sec,
                homography_source_frame=homography.source_frame_id if homography is not None else None,
                players=players,
                ball=ball,
            )
        )

    smoothing_stats = {
        "player_points_smoothed": 0,
        "player_jump_rejections": 0,
        "player_reused_previous_points": 0,
        "ball_points_smoothed": 0,
        "ball_jump_rejections": 0,
    }
    if smoothing_enabled:
        output_frames, smoothing_stats = smooth_projection_frames(
            output_frames,
            player_window=player_smooth_window,
            ball_window=ball_smooth_window,
            max_player_jump_ft=max_player_jump,
            max_ball_jump_ft=max_ball_jump,
            max_reused_player_frames=max_reused_player_frames,
        )

    final_invalid_player_points = sum(1 for frame in output_frames for player in frame.players if not player.valid)
    final_invalid_ball_points = sum(
        1
        for frame in output_frames
        if frame.ball is not None and not frame.ball.valid and frame.ball.source != "missing"
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
            "raw_invalid_player_points": raw_invalid_player_points,
            "raw_invalid_ball_points": raw_invalid_ball_points,
            "final_invalid_player_points": final_invalid_player_points,
            "final_invalid_ball_points": final_invalid_ball_points,
            "projection_smoothing_enabled": smoothing_enabled,
            "projection_smoothing": {
                "player_window": player_smooth_window,
                "ball_window": ball_smooth_window,
                "max_player_jump_ft": max_player_jump,
                "max_ball_jump_ft": max_ball_jump,
                "max_reused_player_frames": max_reused_player_frames,
                **smoothing_stats,
            },
            "ball_context": {
                "player_snap_enabled": ball_player_snap_enabled,
                "context_filter_enabled": ball_context_filter_enabled,
                "player_snap_distance_px": ball_player_snap_distance_px,
                "max_player_distance_ft": ball_max_player_distance_ft,
                "max_rim_distance_ft": ball_max_rim_distance_ft,
                **ball_projection_stats,
            },
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
