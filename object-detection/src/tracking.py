from __future__ import annotations

import argparse
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from tqdm import tqdm

from .config import load_config, nested_get
from .export_crops import export_crops
from .mask_cleanup import clean_mask
from .schemas import Frame0Corrections, TrackFrame, TrackOutput, TrackRecord
from .utils import (
    bbox_area,
    bbox_center_x,
    bbox_from_mask,
    clamp_bbox_xyxy,
    ensure_dir,
    frame_timestamp,
    is_player_like_class,
    iter_video_frames,
    mask_area,
    read_json,
    video_metadata,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track basketball players from frame-0 boxes with SAM2.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--frame0-overrides", help="Optional manual frame-0 correction JSON.")
    parser.add_argument("--output", help="Output track JSON path.")
    parser.add_argument("--mask-dir", help="Output directory for binary masks.")
    parser.add_argument("--config", default="object-detection/config/default.yaml")
    parser.add_argument("--tracker-backend", choices=["sam2", "mock"], help="Tracking backend.")
    parser.add_argument("--sam2-checkpoint", help="SAM2 checkpoint path.")
    parser.add_argument("--sam2-model-cfg", help="SAM2 model config path.")
    parser.add_argument("--device", help="Tracking device: auto, cpu, cuda, or mps.")
    parser.add_argument("--cleanup-distance-threshold", type=float)
    parser.add_argument("--cleanup-min-component-area", type=int)
    parser.add_argument("--crop-dir", help="Output player crop directory.")
    parser.add_argument("--crop-fps", type=float, help="Crop sampling FPS.")
    parser.add_argument("--qa-report", help="Output tracking QA JSON path.")
    parser.add_argument("--max-frames", type=int, help="Optional limit for smoke tests.")
    return parser


def default_tracking_output(video_path: str | Path) -> Path:
    return Path("object-detection/outputs/tracks") / f"{Path(video_path).stem}_tracks.json"


def default_mask_dir(video_path: str | Path) -> Path:
    return Path("object-detection/outputs/masks") / Path(video_path).stem


def default_crop_dir(video_path: str | Path) -> Path:
    return Path("object-detection/outputs/crops") / Path(video_path).stem


def default_qa_report(video_path: str | Path) -> Path:
    return Path("object-detection/outputs/reports") / f"{Path(video_path).stem}_tracking_qa.json"


def load_frame0_boxes(
    detections_path: str | Path,
    overrides_path: str | Path | None,
    width: int,
    height: int,
    player_like_classes: list[str],
) -> tuple[list[list[float]], str]:
    if overrides_path and Path(overrides_path).exists():
        corrections = Frame0Corrections.model_validate(read_json(overrides_path))
        if corrections.boxes:
            boxes = [
                clamp_bbox_xyxy(box.bbox_xyxy, width, height)
                for box in corrections.boxes
                if bbox_area(clamp_bbox_xyxy(box.bbox_xyxy, width, height)) > 0
            ]
            if boxes:
                return sorted(boxes, key=bbox_center_x), "frame0_override"

    detections = read_json(detections_path)
    for frame in detections.get("frames", []):
        if int(frame.get("frame_id", -1)) != 0:
            continue
        boxes = []
        for detection in frame.get("detections", []):
            class_name = str(detection.get("class_name", ""))
            if not is_player_like_class(class_name, player_like_classes):
                continue
            bbox = clamp_bbox_xyxy(detection["bbox_xyxy"], width, height)
            if bbox_area(bbox) > 0:
                boxes.append(bbox)
        return sorted(boxes, key=bbox_center_x), "detections_frame0"
    return [], "none"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mock_track_masks(video_path: str | Path, prompt_boxes: list[list[float]], max_frames: int | None = None) -> Iterator[tuple[int, int, np.ndarray, float | None]]:
    for frame_id, frame in iter_video_frames(video_path):
        if max_frames is not None and frame_id >= max_frames:
            break
        height, width = frame.shape[:2]
        for track_id, box in enumerate(prompt_boxes, start=1):
            x1, y1, x2, y2 = [int(round(v)) for v in clamp_bbox_xyxy(box, width, height)]
            mask = np.zeros((height, width), dtype=np.uint8)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1
            yield frame_id, track_id, mask, 1.0


def write_video_frames_to_directory(video_path: str | Path, frames_dir: str | Path, max_frames: int | None = None) -> int:
    ensure_dir(frames_dir)
    count = 0
    for frame_id, frame in iter_video_frames(video_path):
        if max_frames is not None and frame_id >= max_frames:
            break
        cv2.imwrite(str(Path(frames_dir) / f"{frame_id:06d}.jpg"), frame)
        count += 1
    return count


def sam2_track_masks(
    video_path: str | Path,
    prompt_boxes: list[list[float]],
    checkpoint: str,
    model_cfg: str,
    device: str,
    max_frames: int | None = None,
) -> Iterator[tuple[int, int, np.ndarray, float | None]]:
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 is not installed. Install SAM2, or run with --tracker-backend mock for smoke tests."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="sam2_frames_") as temp_dir:
        frame_count = write_video_frames_to_directory(video_path, temp_dir, max_frames=max_frames)
        if frame_count == 0:
            raise RuntimeError(f"No frames extracted from video: {video_path}")

        predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
        inference_state = predictor.init_state(video_path=temp_dir)
        if hasattr(predictor, "reset_state"):
            predictor.reset_state(inference_state)

        with torch.inference_mode():
            for track_id, box in enumerate(prompt_boxes, start=1):
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=track_id,
                    box=np.array(box, dtype=np.float32),
                )

            for frame_id, object_ids, mask_logits in predictor.propagate_in_video(inference_state):
                logits = mask_logits.detach().cpu().numpy() if hasattr(mask_logits, "detach") else np.asarray(mask_logits)
                for index, object_id in enumerate(object_ids):
                    raw_mask = np.squeeze(logits[index])
                    mask = (raw_mask > 0).astype(np.uint8)
                    yield int(frame_id), int(object_id), mask, None


def build_track_output(
    video_path: str | Path,
    prompt_boxes: list[list[float]],
    prompt_source: str,
    backend: str,
    mask_dir: str | Path,
    checkpoint: str | None,
    model_cfg: str | None,
    device: str,
    cleanup_distance_threshold: float,
    cleanup_min_component_area: int,
    max_frames: int | None,
) -> tuple[TrackOutput, dict[str, Any]]:
    metadata = video_metadata(video_path)
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    frame_count = int(metadata["frame_count"])
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    output_mask_dir = Path(mask_dir)
    ensure_dir(output_mask_dir)

    frames_by_id: dict[int, list[TrackRecord]] = defaultdict(list)
    prompt_by_track_id = {track_id: box for track_id, box in enumerate(prompt_boxes, start=1)}

    if backend == "mock":
        mask_iter = mock_track_masks(video_path, prompt_boxes, max_frames=max_frames)
        sam2_checkpoint = "mock"
    else:
        if not checkpoint or not model_cfg:
            raise RuntimeError("SAM2 backend requires --sam2-checkpoint and --sam2-model-cfg.")
        mask_iter = sam2_track_masks(video_path, prompt_boxes, checkpoint, model_cfg, device, max_frames=max_frames)
        sam2_checkpoint = checkpoint

    total_entries = 0
    empty_masks = 0
    areas: list[int] = []
    progress = tqdm(mask_iter, desc=f"Tracking ({backend})", unit="mask")
    for frame_id, track_id, raw_mask, score in progress:
        cleaned_mask = clean_mask(
            raw_mask,
            distance_threshold=cleanup_distance_threshold,
            min_component_area=cleanup_min_component_area,
        )
        area = mask_area(cleaned_mask)
        if area == 0:
            empty_masks += 1
        areas.append(area)
        bbox = bbox_from_mask(cleaned_mask)
        mask_path = output_mask_dir / f"frame_{frame_id:06d}_track_{track_id:03d}.png"
        cv2.imwrite(str(mask_path), cleaned_mask.astype(np.uint8) * 255)
        frames_by_id[frame_id].append(
            TrackRecord(
                track_id=track_id,
                bbox_xyxy=bbox,
                mask_path=str(mask_path),
                area=area,
                score=score,
                source_prompt_bbox_xyxy=prompt_by_track_id[track_id],
            )
        )
        total_entries += 1

    frames = [
        TrackFrame(
            frame_id=frame_id,
            timestamp_sec=frame_timestamp(frame_id, fps),
            tracks=sorted(frames_by_id.get(frame_id, []), key=lambda track: track.track_id),
        )
        for frame_id in range(frame_count)
    ]
    output = TrackOutput(
        video=str(video_path),
        fps=fps,
        width=width,
        height=height,
        sam2_checkpoint=sam2_checkpoint,
        mask_cleanup={
            "distance_threshold": cleanup_distance_threshold,
            "min_component_area": cleanup_min_component_area,
        },
        frames=frames,
    )
    expected_entries = frame_count * len(prompt_boxes)
    qa = {
        "video": str(video_path),
        "tracker_backend": backend,
        "prompt_source": prompt_source,
        "frames_expected": frame_count,
        "tracks_initialized": len(prompt_boxes),
        "expected_track_entries": expected_entries,
        "total_track_entries": total_entries,
        "missing_track_entries": max(0, expected_entries - total_entries),
        "empty_masks": empty_masks,
        "average_track_area": float(sum(areas) / len(areas)) if areas else 0.0,
    }
    return output, qa


def run_tracking(args: argparse.Namespace) -> TrackOutput:
    config = load_config(args.config)
    video_path = Path(args.video)
    metadata = video_metadata(video_path)
    width = int(metadata["width"])
    height = int(metadata["height"])

    backend = args.tracker_backend or str(nested_get(config, "tracking.backend", "sam2"))
    checkpoint = args.sam2_checkpoint or str(nested_get(config, "tracking.sam2_checkpoint", ""))
    model_cfg = args.sam2_model_cfg or str(nested_get(config, "tracking.sam2_model_cfg", ""))
    device = resolve_device(args.device or str(nested_get(config, "tracking.device", "auto")))
    player_like_classes = list(nested_get(config, "detection.player_like_classes", []))
    cleanup_distance = (
        args.cleanup_distance_threshold
        if args.cleanup_distance_threshold is not None
        else float(nested_get(config, "mask_cleanup.distance_threshold", 80))
    )
    cleanup_min_area = (
        args.cleanup_min_component_area
        if args.cleanup_min_component_area is not None
        else int(nested_get(config, "mask_cleanup.min_component_area", 50))
    )
    crop_fps = args.crop_fps if args.crop_fps is not None else float(nested_get(config, "handoff.crop_fps", 1.0))

    prompt_boxes, prompt_source = load_frame0_boxes(
        args.detections,
        args.frame0_overrides,
        width,
        height,
        player_like_classes,
    )
    if not prompt_boxes:
        raise RuntimeError(
            "No valid frame-0 player boxes found. Inspect detection debug frame and add "
            "--frame0-overrides object-detection/outputs/corrections/<clip>_frame0_boxes.json."
        )

    output_path = Path(args.output) if args.output else default_tracking_output(video_path)
    mask_dir = Path(args.mask_dir) if args.mask_dir else default_mask_dir(video_path)
    crop_dir = Path(args.crop_dir) if args.crop_dir else default_crop_dir(video_path)
    qa_report = Path(args.qa_report) if args.qa_report else default_qa_report(video_path)

    output, qa = build_track_output(
        video_path=video_path,
        prompt_boxes=prompt_boxes,
        prompt_source=prompt_source,
        backend=backend,
        mask_dir=mask_dir,
        checkpoint=checkpoint,
        model_cfg=model_cfg,
        device=device,
        cleanup_distance_threshold=cleanup_distance,
        cleanup_min_component_area=cleanup_min_area,
        max_frames=args.max_frames,
    )

    output_payload = output.model_dump(mode="json")
    write_json(output_path, output_payload)
    crop_summary = export_crops(video_path, output_payload, crop_dir, target_fps=crop_fps)
    qa["crop_summary"] = crop_summary
    write_json(qa_report, qa)

    print(
        "Tracking complete: "
        f"backend={backend}, prompts={len(prompt_boxes)}, source={prompt_source}, device={device}"
    )
    print(f"Wrote tracks: {output_path}")
    print(f"Wrote masks: {mask_dir}")
    print(f"Wrote crops: {crop_dir}")
    print(f"Wrote QA report: {qa_report}")
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_tracking(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
