from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from jersey_numbers.ocr.resnet34 import ResNetOCR


def compute_ios(player_mask: np.ndarray, number_bbox: list[float]) -> float:
    """Intersection over Smaller Area: fraction of the number bbox covered by the player mask"""
    h, w = player_mask.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(number_bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    number_area = (x2 - x1) * (y2 - y1)
    player_area = int(np.count_nonzero(player_mask))
    smaller_area = min(number_area, player_area)
    if smaller_area == 0:
        return 0.0

    intersection = int(np.count_nonzero(player_mask[y1:y2, x1:x2]))
    return intersection / smaller_area


def compute_iou(player_mask: np.ndarray, number_bbox: list[float]) -> float:
    """Standard Intersection over Union for comparison with IoS."""
    h, w = player_mask.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(number_bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    number_area = (x2 - x1) * (y2 - y1)
    player_area = int(np.count_nonzero(player_mask))
    intersection = int(np.count_nonzero(player_mask[y1:y2, x1:x2]))
    union = player_area + number_area - intersection
    return intersection / union if union > 0 else 0.0


def _clamp_bbox(bbox: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x1 = int(round(max(0.0, min(float(bbox[0]), w))))
    y1 = int(round(max(0.0, min(float(bbox[1]), h))))
    x2 = int(round(max(0.0, min(float(bbox[2]), w))))
    y2 = int(round(max(0.0, min(float(bbox[3]), h))))
    return x1, y1, x2, y2


def load_mask(mask_path: str | Path) -> np.ndarray | None:
    return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)


def crop_number(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    """Crop the number region from a video frame using the RF-DETR bbox."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _seek_frame(cap: cv2.VideoCapture, frame_id: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    return frame if ok else None


def match_frame(
    frame_id: int,
    number_detections: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    ios_threshold: float = 0.9,
    frame_image: np.ndarray | None = None,
    ocr_model: ResNetOCR | None = None,
) -> list[dict[str, Any]]:
    """Match number detections to player tracks for one frame."""
    results: list[dict[str, Any]] = []

    for det in number_detections:
        if str(det.get("class_name", "")) != "number":
            continue
        bbox = det["bbox_xyxy"]
        best_track_id: int | None = None
        best_ios = 0.0

        for track in tracks:
            mask = load_mask(track["mask_path"])
            if mask is None:
                continue
            score = compute_ios(mask, bbox)
            if score > best_ios:
                best_ios = score
                best_track_id = int(track["track_id"])

        if best_track_id is None or best_ios < ios_threshold:
            continue

        match: dict[str, Any] = {
            "frame_id": frame_id,
            "track_id": best_track_id,
            "number_bbox_xyxy": bbox,
            "ios_score": best_ios,
            "detection_confidence": float(det.get("confidence", 0.0)),
            "predicted_number": None,
            "ocr_confidence": None,
        }

        if frame_image is not None and ocr_model is not None:
            number_crop = crop_number(frame_image, bbox)
            if number_crop is not None:
                predicted, conf = ocr_model.predict(number_crop)
                match["predicted_number"] = predicted
                match["ocr_confidence"] = conf

        results.append(match)

    return results


def run_matching(
    tracks_path: Path,
    detections_path: Path,
    video_path: Path,
    ios_threshold: float = 0.9,
    ocr_model: ResNetOCR | None = None,
) -> dict[str, Any]:
    """Match number detections to player tracks across all frames"""
    with tracks_path.open(encoding="utf-8") as f:
        tracks_data = json.load(f)
    with detections_path.open(encoding="utf-8") as f:
        detections_data = json.load(f)

    det_by_frame: dict[int, list[dict[str, Any]]] = {}
    for frame in detections_data.get("frames", []):
        dets = frame.get("detections", [])
        number_dets = [d for d in dets if d.get("class_name") == "number"]
        if number_dets:
            det_by_frame[int(frame["frame_id"])] = number_dets

    track_by_frame: dict[int, list[dict[str, Any]]] = {
        int(f["frame_id"]): f.get("tracks", [])
        for f in tracks_data.get("frames", [])
    }

    frames_to_process = sorted(set(det_by_frame) & set(track_by_frame))

    all_matches: list[dict[str, Any]] = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        for frame_id in frames_to_process:
            frame_image = _seek_frame(cap, frame_id)
            matches = match_frame(
                frame_id=frame_id,
                number_detections=det_by_frame[frame_id],
                tracks=track_by_frame[frame_id],
                ios_threshold=ios_threshold,
                frame_image=frame_image,
                ocr_model=ocr_model,
            )
            all_matches.extend(matches)
    finally:
        cap.release()

    return {
        "video": tracks_data.get("video", ""),
        "ios_threshold": ios_threshold,
        "ocr_model": ocr_model.__class__.__name__ if ocr_model else None,
        "total_matches": len(all_matches),
        "matches": all_matches,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Match jersey number detections to player tracks via IoS.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--tracks", required=True, help="Tracks JSON from object-detection module.")
    parser.add_argument("--detections", required=True, help="Detections JSON from object-detection module.")
    parser.add_argument("--output", required=True, help="Output JSON path for matches.")
    parser.add_argument("--ios-threshold", type=float, default=0.9, help="Minimum IoS to accept a match.")
    parser.add_argument("--checkpoint", help="Optional ResNet-32 checkpoint (.pth) to run OCR on matched crops.")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    ocr_model = None
    if args.checkpoint:
        from jersey_numbers.ocr.resnet34 import ResNetOCR
        ocr_model = ResNetOCR(Path(args.checkpoint), device_str=args.device)

    result = run_matching(
        tracks_path=Path(args.tracks),
        detections_path=Path(args.detections),
        video_path=Path(args.video),
        ios_threshold=args.ios_threshold,
        ocr_model=ocr_model,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    ocr_str = f" + OCR ({args.checkpoint})" if args.checkpoint else " (no OCR)"
    print(f"Matched {result['total_matches']} numbers{ocr_str}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
