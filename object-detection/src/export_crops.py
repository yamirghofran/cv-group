from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2

from .utils import clamp_bbox_xyxy, ensure_dir, iter_video_frames, read_json, sample_interval_for_fps, video_metadata


def frames_by_id(tracks_data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    return {int(frame["frame_id"]): list(frame.get("tracks", [])) for frame in tracks_data.get("frames", [])}


def export_crops(
    video_path: str | Path,
    tracks_data: dict[str, Any],
    crop_dir: str | Path,
    target_fps: float = 1.0,
) -> dict[str, Any]:
    metadata = video_metadata(video_path)
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    interval = sample_interval_for_fps(fps, target_fps)
    track_map = frames_by_id(tracks_data)
    output_dir = Path(crop_dir)
    ensure_dir(output_dir)

    saved = 0
    by_track: dict[int, int] = {}
    for frame_id, frame in iter_video_frames(video_path):
        if frame_id % interval != 0:
            continue
        for track in track_map.get(frame_id, []):
            bbox = clamp_bbox_xyxy(track["bbox_xyxy"], width, height)
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            track_id = int(track["track_id"])
            track_dir = output_dir / f"track_{track_id:03d}"
            ensure_dir(track_dir)
            crop_path = track_dir / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(crop_path), crop)
            saved += 1
            by_track[track_id] = by_track.get(track_id, 0) + 1

    return {"crop_dir": str(output_dir), "saved_crops": saved, "crops_by_track": by_track, "target_fps": target_fps}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export player crops from track JSON.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--crop-dir", required=True)
    parser.add_argument("--target-fps", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_crops(args.video, read_json(args.tracks), args.crop_dir, args.target_fps)
    print(f"Exported {result['saved_crops']} crops to {result['crop_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
