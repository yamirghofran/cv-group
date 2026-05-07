from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Basketball CV pipeline demo.")
    p.add_argument("--video", required=True)
    p.add_argument("--yolo-weights", default="object-detection/finetuning/runs/basketball_yolo11s/weights/best.pt")
    p.add_argument("--sam2-checkpoint", help="SAM2 checkpoint (.pt). Omit to run detection-only.")
    p.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--ocr-checkpoint", default="numbers/models/resnet34.pth", help="ResNet OCR checkpoint.")
    p.add_argument("--ios-threshold", type=float, default=0.9)
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--output-dir", default="demo_output")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-tracking", action="store_true", help="Reuse existing tracks.json and masks — skip SAM2.")
    p.add_argument("--skip-clustering", action="store_true", help="Reuse existing teams.json — skip SigLIP clustering.")
    return p


# ── Team colours (BGR) ────────────────────────────────────────────────────────
TEAM_COLORS = {
    0: (220, 60, 60),    # blue
    1: (60, 60, 220),    # red
    None: (160, 160, 160),  # unknown / referee
}
TEAM_NAMES = {0: "Team A", 1: "Team B", None: "?"}


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Step helpers ──────────────────────────────────────────────────────────────

def _run_detection(video_path, yolo_weights, frame_stride, device, out_dir):
    import cv2
    from src.yolo_client import YoloLocalClient, YoloLocalSettings
    from src.utils import video_metadata, iter_video_frames, should_process_frame, frame_timestamp, write_json
    from src.roboflow_client import normalize_roboflow_predictions
    from src.schemas import DetectionFrame, DetectionOutput

    yolo = YoloLocalClient(YoloLocalSettings(
        weights_path=yolo_weights,
        imgsz=640, conf=0.25, iou=0.5,
        device=device if device != "cpu" else None,
    ))
    meta = video_metadata(video_path)
    fps, w, h = float(meta["fps"]), int(meta["width"]), int(meta["height"])
    frames, classes_seen = [], set()
    for frame_id, frame in iter_video_frames(video_path):
        if not should_process_frame(frame_id, frame_stride):
            continue
        payload = yolo.infer_frame(frame, f"demo_frame_{frame_id:06d}.jpg")
        records = normalize_roboflow_predictions(payload, w, h, source="yolo")
        classes_seen.update(r.class_name for r in records)
        frames.append(DetectionFrame(frame_id=frame_id, timestamp_sec=frame_timestamp(frame_id, fps), detections=records))

    det_output = DetectionOutput(
        video=str(video_path), fps=fps, width=w, height=h,
        model_id=yolo.model_identifier, model_version=0,
        frame_stride=frame_stride, frames=frames, classes_seen=sorted(classes_seen),
    )
    path = out_dir / "detections.json"
    write_json(path, det_output.model_dump(mode="json"))
    print(f"    -> {len(frames)} frames, classes: {sorted(classes_seen)}")
    print(f"    -> saved {path}")
    return path


def _run_tracking(video_path, detections_path, sam2_checkpoint, sam2_model_cfg, device, out_dir):
    from src.tracking import build_track_output, load_frame0_boxes
    from src.utils import write_json

    w_h = _video_wh(video_path)
    player_classes = ["player", "player-in-possession", "player-jump-shot",
                      "player-layup-dunk", "player-shot-block"]
    prompt_boxes, prompt_source = load_frame0_boxes(
        detections_path, None, w_h[0], w_h[1], player_classes
    )
    if not prompt_boxes:
        print("    WARNING: no players detected in frame 0 — skipping tracking.")
        return None

    mask_dir = out_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    track_output, _ = build_track_output(
        video_path=video_path, prompt_boxes=prompt_boxes, prompt_source=prompt_source,
        backend="sam2", mask_dir=mask_dir, checkpoint=sam2_checkpoint,
        model_cfg=sam2_model_cfg, device=device,
        cleanup_distance_threshold=80.0, cleanup_min_component_area=50, max_frames=None,
    )
    path = out_dir / "tracks.json"
    write_json(path, track_output.model_dump(mode="json"))
    total = sum(len(fr.tracks) for fr in track_output.frames)
    print(f"    -> {len(track_output.frames)} frames, {total} track entries")
    print(f"    -> saved {path}")
    return path


def _extract_crops(video_path, tracks_path, out_dir) -> Path:
    """Save one crop image per (track, frame) from player bboxes."""
    import cv2
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    with tracks_path.open() as f:
        trk_data = json.load(f)

    cap = cv2.VideoCapture(str(video_path))
    try:
        for fr in trk_data.get("frames", []):
            frame_id = int(fr["frame_id"])
            if not fr.get("tracks"):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            for track in fr["tracks"]:
                tid = int(track["track_id"])
                x1, y1, x2, y2 = track["bbox_xyxy"]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                cv2.imwrite(str(crops_dir / f"track_{tid:03d}_frame_{frame_id:06d}.png"), crop)
    finally:
        cap.release()

    n = len(list(crops_dir.glob("*.png")))
    print(f"    -> {n} crops saved to {crops_dir}")
    return crops_dir


def _run_clustering(crops_dir, device, out_dir) -> Path:
    from clustering.classifier import TeamClassifier
    from clustering.load_crops import load_crops
    from clustering.schemas import TeamAssignment, TeamOutput

    pairs = load_crops(crops_dir)
    if not pairs:
        raise ValueError("No crops found for clustering.")

    paths, images = zip(*pairs, strict=True)
    clf = TeamClassifier(device=device if device != "mps" else "cpu")
    labels = clf.fit_predict(list(images))

    assignments = []
    for path, label in zip(paths, labels, strict=True):
        # filename: track_003_frame_000025.png -> track_id=3
        try:
            tid = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            tid = None
        assignments.append(TeamAssignment(crop_path=str(path), cluster_id=int(label), track_id=tid))

    output = TeamOutput(crops_dir=str(crops_dir), classifier_method="siglip+umap+kmeans",
                        n_teams=2, assignments=assignments)
    path = out_dir / "teams.json"
    path.write_text(output.model_dump_json(indent=2))
    print(f"    -> team assignments saved to {path}")
    return path


def _run_matching(video_path, detections_path, tracks_path, ios_threshold, ocr_checkpoint, device, out_dir) -> Path:
    import cv2
    from jersey_numbers.matching.ios import match_frame, OCRModel

    ocr: OCRModel | None = None
    if ocr_checkpoint and Path(ocr_checkpoint).exists():
        from jersey_numbers.ocr.resnet34 import ResNetOCR
        ocr = ResNetOCR(Path(ocr_checkpoint), device_str=device)
        print(f"    -> OCR: {ocr_checkpoint}")
    else:
        print("     OCR checkpoint not found, skipping number reading.")

    with detections_path.open() as f:
        det_data = json.load(f)
    with tracks_path.open() as f:
        trk_data = json.load(f)

    det_by_frame = {
        int(fr["frame_id"]): [d for d in fr["detections"] if d.get("class_name") == "number"]
        for fr in det_data.get("frames", [])
    }
    trk_by_frame = {int(fr["frame_id"]): fr.get("tracks", []) for fr in trk_data.get("frames", [])}

    all_matches: list[dict] = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        for frame_id in sorted(set(det_by_frame) & set(trk_by_frame)):
            if not det_by_frame[frame_id]:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            all_matches.extend(match_frame(
                frame_id=frame_id,
                number_detections=det_by_frame[frame_id],
                tracks=trk_by_frame[frame_id],
                ios_threshold=ios_threshold,
                frame_image=frame if ok else None,
                ocr_model=ocr,
            ))
    finally:
        cap.release()

    path = out_dir / "matches.json"
    with path.open("w") as f:
        json.dump(all_matches, f, indent=2)
    print(f"    -> {len(all_matches)} matches found")
    return path


def _render_video(video_path, tracks_path, matches_path, teams_path, out_dir) -> Path:
    import cv2
    import numpy as np

    # Build track -> jersey number (majority vote)
    with matches_path.open() as f:
        matches = json.load(f)
    number_votes: dict[int, list[str]] = {}
    for m in matches:
        if m.get("predicted_number"):
            number_votes.setdefault(m["track_id"], []).append(m["predicted_number"])
    track_number: dict[int, str] = {
        tid: Counter(nums).most_common(1)[0][0]
        for tid, nums in number_votes.items()
    }

    # Build track -> team (majority vote across crop labels)
    track_team: dict[int, int] = {}
    if teams_path and teams_path.exists():
        with teams_path.open() as f:
            teams_data = json.load(f)
        team_votes: dict[int, list[int]] = {}
        for a in teams_data.get("assignments", []):
            tid = a.get("track_id")
            if tid is not None:
                team_votes.setdefault(tid, []).append(a["cluster_id"])
        track_team = {
            tid: Counter(votes).most_common(1)[0][0]
            for tid, votes in team_votes.items()
        }

    with tracks_path.open() as f:
        trk_data = json.load(f)

    def _tracks_by_frame(data):
        return {int(fr["frame_id"]): fr.get("tracks", []) for fr in data.get("frames", [])}

    track_map = _tracks_by_frame(trk_data)
    max_frame = max(track_map) if track_map else 0

    def _meta(path):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return fps, w, h

    fps, w, h = _meta(video_path)
    out_path = out_dir / "annotated.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame_id > max_frame:
                break
            annotated = frame.copy()
            for track in track_map.get(frame_id, []):
                tid = int(track["track_id"])
                team = track_team.get(tid)
                color = TEAM_COLORS[team]

                # mask overlay
                mask_path = Path(track["mask_path"])
                if not mask_path.exists():
                    mask_path = tracks_path.parent / track["mask_path"]
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None and mask.shape[:2] == (h, w):
                    mask2d = mask.squeeze() > 0
                    layer = np.zeros_like(annotated)
                    layer[mask2d] = color
                    blended = cv2.addWeighted(annotated, 1.0, layer, 0.4, 0.0)
                    annotated[mask2d] = blended[mask2d]

                # bounding box
                x1, y1, x2, y2 = [int(v) for v in track["bbox_xyxy"]]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # label: jersey number + team
                number = track_number.get(tid, "?")
                team_name = TEAM_NAMES[team]
                label = f"#{number} {team_name}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(annotated)
            frame_id += 1
    finally:
        cap.release()
        writer.release()

    print(f"    -> saved {out_path}")
    return out_path


def _video_wh(video_path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


# Main 
def main(argv=None):
    args = _build_parser().parse_args(argv)
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    print(f"Device: {device}\n")

    # 1. Detection
    print("[1/5] YOLO detection…")
    detections_path = _run_detection(video_path, args.yolo_weights, args.frame_stride, device, out_dir)

    if not args.sam2_checkpoint:
        print("\nSkipping steps 2-5 (no --sam2-checkpoint). Detection saved.")
        return 0

    # 2. Tracking
    tracks_path = out_dir / "tracks.json"
    if args.skip_tracking and tracks_path.exists():
        print(f"\n[2/5] Skipping SAM2 — reusing {tracks_path}")
    else:
        print("\n[2/5] SAM2 tracking…")
        tracks_path = _run_tracking(video_path, detections_path, args.sam2_checkpoint,
                                    args.sam2_model_cfg, device, out_dir)
        if tracks_path is None:
            return 1

    # 3. Extract crops for team clustering
    print("\n[3/5] Extracting player crops…")
    crops_dir = _extract_crops(video_path, tracks_path, out_dir)

    # 4. Team clustering
    teams_path = out_dir / "teams.json"
    if args.skip_clustering and teams_path.exists():
        print(f"\n[4/5] Skipping clustering — reusing {teams_path}")
    else:
        print("\n[4/5] Team clustering (SigLIP + UMAP + KMeans)…")
        teams_path = _run_clustering(crops_dir, device, out_dir)

    # 5. OCR matching
    print("\n[5/5] IoS matching + OCR…")
    matches_path = _run_matching(video_path, detections_path, tracks_path,
                                 args.ios_threshold, args.ocr_checkpoint, device, out_dir)

    # 6. Render
    print("\nRendering annotated video…")
    out_video = _render_video(video_path, tracks_path, matches_path, teams_path, out_dir)

    # Summary
    with matches_path.open() as f:
        matches = json.load(f)
    number_votes: dict[int, list] = {}
    for m in matches:
        if m.get("predicted_number"):
            number_votes.setdefault(m["track_id"], []).append(m["predicted_number"])

    print("\n Results")
    with teams_path.open() as f:
        teams_data = json.load(f)
    team_votes2: dict[int, list] = {}
    for a in teams_data.get("assignments", []):
        tid = a.get("track_id")
        if tid is not None:
            team_votes2.setdefault(tid, []).append(a["cluster_id"])
    track_team2 = {tid: Counter(v).most_common(1)[0][0] for tid, v in team_votes2.items()}

    for tid in sorted(track_team2):
        team = TEAM_NAMES[track_team2.get(tid)]
        nums = number_votes.get(tid, [])
        num = Counter(nums).most_common(1)[0][0] if nums else "?"
        print(f"  Track {tid:2d} -> {team}, jersey #{num}")

    print(f"\nOutput video: {out_video.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
