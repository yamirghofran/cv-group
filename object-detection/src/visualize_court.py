from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import load_config, nested_get
from .court_geometry import NBACourt, court_to_pixel, draw_court
from .utils import color_for_id, draw_bbox, iter_video_frames, open_video_writer, read_json, video_metadata
from .visualize_tracks import overlay_mask, resolve_mask_path, tracks_by_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render source video beside top-down court movement.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--court-tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="object-detection/config/default.yaml")
    parser.add_argument("--panel-height", type=int)
    parser.add_argument("--trail-length", type=int)
    parser.add_argument("--mask-alpha", type=float, default=0.25)
    return parser


def court_frames_by_id(court_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(frame["frame_id"]): frame for frame in court_data.get("frames", [])}


def resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = target_height / height
    return cv2.resize(frame, (int(round(width * scale)), target_height))


def draw_source_tracks(frame: np.ndarray, tracks: list[dict[str, Any]], tracks_path: str | Path, mask_alpha: float) -> None:
    for track in tracks:
        track_id = int(track["track_id"])
        color = color_for_id(track_id)
        overlay_mask(frame, resolve_mask_path(track["mask_path"], tracks_path), color, mask_alpha)
        draw_bbox(frame, track["bbox_xyxy"], f"ID {track_id}", color)


def render_source_panel(
    frame: np.ndarray,
    tracks: list[dict[str, Any]],
    tracks_path: str | Path,
    mask_alpha: float,
    target_height: int,
) -> np.ndarray:
    annotated = frame.copy()
    draw_source_tracks(annotated, tracks, tracks_path, mask_alpha)
    return resize_to_height(annotated, target_height)


def draw_court_motion(
    court_image: np.ndarray,
    frame_id: int,
    court_map: dict[int, dict[str, Any]],
    court: NBACourt,
    scale: float,
    padding: int,
    trail_length: int,
) -> None:
    start_frame = max(0, frame_id - trail_length)
    by_track: dict[int, list[tuple[float, float]]] = {}
    ball_points: list[tuple[float, float]] = []
    for past_frame_id in range(start_frame, frame_id + 1):
        frame = court_map.get(past_frame_id)
        if frame is None:
            continue
        for player in frame.get("players", []):
            if not player.get("valid") or player.get("court_xy") is None:
                continue
            by_track.setdefault(int(player["track_id"]), []).append(tuple(player["court_xy"]))
        ball = frame.get("ball")
        if ball and ball.get("valid") and ball.get("court_xy") is not None:
            ball_points.append(tuple(ball["court_xy"]))

    for track_id, points in by_track.items():
        color = color_for_id(track_id)
        pixel_points = [court_to_pixel(point, court, scale, padding) for point in points]
        for pt1, pt2 in zip(pixel_points, pixel_points[1:]):
            cv2.line(court_image, pt1, pt2, color, 2)
        if pixel_points:
            cv2.circle(court_image, pixel_points[-1], 7, color, -1)
            cv2.putText(court_image, str(track_id), (pixel_points[-1][0] + 7, pixel_points[-1][1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    ball_pixels = [court_to_pixel(point, court, scale, padding) for point in ball_points]
    for pt1, pt2 in zip(ball_pixels, ball_pixels[1:]):
        cv2.line(court_image, pt1, pt2, (0, 165, 255), 2)
    if ball_pixels:
        cv2.circle(court_image, ball_pixels[-1], 6, (0, 120, 255), -1)


def render_side_by_side(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    video_path = Path(args.video)
    metadata = video_metadata(video_path)
    fps = float(metadata["fps"])
    panel_height = args.panel_height or int(nested_get(config, "court_visualization.panel_height", 720))
    trail_length = args.trail_length or int(nested_get(config, "court_visualization.trail_length", 45))
    court = NBACourt(
        length=float(nested_get(config, "court_projection.court_length_ft", 94.0)),
        width=float(nested_get(config, "court_projection.court_width_ft", 50.0)),
    )
    tracks_data = read_json(args.tracks)
    court_data = read_json(args.court_tracks)
    track_map = tracks_by_frame(tracks_data)
    court_map = court_frames_by_id(court_data)
    max_frame = max(court_map) if court_map else -1
    if max_frame < 0:
        raise RuntimeError("Court track JSON has no frames to visualize.")

    court_scale = (panel_height - 70) / court.width
    court_base = draw_court(court, scale=court_scale, padding=35)
    court_panel = resize_to_height(court_base, panel_height)
    source_width = int(round(int(metadata["width"]) * (panel_height / int(metadata["height"]))))
    output_width = source_width + court_panel.shape[1]
    writer = open_video_writer(args.output, fps, output_width, panel_height)

    try:
        for frame_id, frame in iter_video_frames(video_path):
            if frame_id > max_frame:
                break
            source_panel = render_source_panel(frame, track_map.get(frame_id, []), args.tracks, args.mask_alpha, panel_height)
            court_frame = court_base.copy()
            draw_court_motion(court_frame, frame_id, court_map, court, court_scale, 35, trail_length)
            court_frame = resize_to_height(court_frame, panel_height)
            combined = np.hstack([source_panel, court_frame])
            writer.write(combined)
    finally:
        writer.release()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render_side_by_side(args)
    print(f"Wrote court movement visualization: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
