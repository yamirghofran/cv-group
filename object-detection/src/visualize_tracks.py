from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .utils import color_for_id, draw_bbox, iter_video_frames, open_video_writer, read_json, video_metadata


# Default team colours in BGR.  Keys are team names as produced by
# ``clustering.assign.build_track_team_lookup`` (e.g. "Team A", "Team B").
# Falls back to a simple cycle if more teams are added.
TEAM_PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 255),     # Red   – "Team A"
    (255, 165, 0),   # Blue  – "Team B"
    (0, 255, 0),     # Green – "Team C" (for n_teams > 2)
    (0, 255, 255),   # Yellow – "Team D"
]


def _team_color(team_name: str) -> tuple[int, int, int]:
    """Return a BGR colour for a team name, cycling through TEAM_PALETTE."""
    idx = ord(team_name[-1]) - ord("A") if len(team_name) == 6 and team_name.startswith("Team ") else 0
    return TEAM_PALETTE[idx % len(TEAM_PALETTE)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render track JSON and masks as an annotated video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-alpha", type=float, default=0.35)
    parser.add_argument("--teams", help="Optional path to a TeamOutput JSON file for team-coloured overlays.")
    return parser


def tracks_by_frame(track_data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    return {int(frame["frame_id"]): list(frame.get("tracks", [])) for frame in track_data.get("frames", [])}


def resolve_mask_path(mask_path: str, tracks_path: str | Path) -> Path:
    path = Path(mask_path)
    if path.exists() or path.is_absolute():
        return path
    candidate = Path(tracks_path).parent / path
    if candidate.exists():
        return candidate
    return path


def overlay_mask(frame: np.ndarray, mask_path: Path, color: tuple[int, int, int], alpha: float) -> None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape[:2] != frame.shape[:2]:
        return
    color_layer = np.zeros_like(frame)
    color_layer[mask > 0] = color
    blended = cv2.addWeighted(frame, 1.0, color_layer, alpha, 0.0)
    frame[mask > 0] = blended[mask > 0]


def render_track_video(video_path: str | Path, tracks_path: str | Path, output_path: str | Path, mask_alpha: float, team_lookup: dict[int, str] | None = None) -> None:
    metadata = video_metadata(video_path)
    writer = open_video_writer(output_path, float(metadata["fps"]), int(metadata["width"]), int(metadata["height"]))
    track_map = tracks_by_frame(read_json(tracks_path))
    max_track_frame = max(track_map) if track_map else -1
    try:
        for frame_id, frame in iter_video_frames(video_path):
            if frame_id > max_track_frame:
                break
            annotated = frame.copy()
            for track in track_map.get(frame_id, []):
                track_id = int(track["track_id"])
                team_name = team_lookup.get(track_id) if team_lookup else None
                if team_name:
                    color = _team_color(team_name)
                    label = f"ID {track_id} ({team_name})"
                else:
                    color = color_for_id(track_id)
                    label = f"ID {track_id}"
                mask_path = resolve_mask_path(track["mask_path"], tracks_path)
                overlay_mask(annotated, mask_path, color, mask_alpha)
                draw_bbox(annotated, track["bbox_xyxy"], label, color)
            writer.write(annotated)
    finally:
        writer.release()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    team_lookup: dict[int, str] | None = None
    if args.teams:
        from clustering.assign import build_track_team_lookup
        from clustering.schemas import TeamOutput

        team_data = TeamOutput.model_validate_json(Path(args.teams).read_text())
        team_lookup = build_track_team_lookup(team_data)
    render_track_video(args.video, args.tracks, args.output, args.mask_alpha, team_lookup=team_lookup)
    print(f"Wrote track visualization: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
