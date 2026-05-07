"""demo.py — Basketball CV pipeline demo

Runs the full pipeline against a single video:
  [1] YOLO detection → [2] ball tracking → [3] SAM2 tracking → [4] crop export →
  [5] team clustering → [6] OCR jersey numbers → [7] court keypoints →
  [8] court projection → [9a] annotated broadcast video + [9b] 2D court minimap

Outputs are written to --output-dir (default: demo_output/).

Usage:
    uv run python demo.py --video clip_short.mp4
    uv run python demo.py --video clip_short.mp4 --skip-tracking --skip-clustering
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root, then fall back to object-detection/.env
_root = Path(__file__).parent
load_dotenv(_root / ".env")
load_dotenv(_root / "object-detection" / ".env")

# Add sub-package roots to sys.path so package imports work when running as a
# top-level script (i.e. without `uv run -m`).
for _p in ["object-detection", "numbers"]:
    _abs = str(Path(__file__).parent / _p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

_DEFAULT_SAM2 = "object-detection/checkpoints/sam2.1_hiera_large.pt"
_DEFAULT_COURT_WEIGHTS = "object-detection/models/court_yolo11m_pose_v19/best.pt"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Basketball CV pipeline demo — produces an annotated video with team colours "
            "and jersey numbers, plus a 2D court minimap showing player movement."
        )
    )
    p.add_argument("--video", required=True, help="Input video path (.mp4).")
    p.add_argument(
        "--sam2-checkpoint",
        default=_DEFAULT_SAM2,
        help=f"SAM2 weights (.pt). Default: {_DEFAULT_SAM2}. "
             "Download from: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    )
    p.add_argument("--sam2-model-cfg", default=None,
                   help="SAM2 config yaml (default: configs/sam2.1/sam2.1_hiera_l.yaml).")
    p.add_argument("--yolo-weights", default=None,
                   help="YOLO detection weights (default: from object-detection/config/default.yaml).")
    p.add_argument("--court-weights", default=None,
                   help="Court-keypoint YOLO weights (default: from config). "
                        "If absent, the court minimap step is skipped automatically.")
    p.add_argument("--ocr-model", default="resnet34",
                   help="OCR model: resnet34 | resnet18 | smolvlm2 | stacking (default: resnet34).")
    p.add_argument("--ios-threshold", type=float, default=0.9,
                   help="Minimum IoS score to accept a jersey-number match (default: 0.9).")
    p.add_argument("--frame-stride", type=int, default=None,
                   help="Run detection every N frames (default: 5, from config).")
    p.add_argument("--output-dir", default="demo_output",
                   help="Directory to write all output files (default: demo_output).")
    p.add_argument("--device", default=None,
                   help="Device: cpu | cuda | mps (default: auto-detected).")
    p.add_argument("--skip-tracking", action="store_true",
                   help="Reuse existing tracks.json + masks/ — skip SAM2 (fast re-run).")
    p.add_argument("--skip-clustering", action="store_true",
                   help="Reuse existing teams.json — skip SigLIP clustering (fast re-run).")
    p.add_argument("--skip-court", action="store_true",
                   help="Skip ball tracking, court keypoints, and court minimap video.")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    video = Path(args.video)
    if not video.exists():
        print(f"ERROR: video not found: {video}", file=sys.stderr)
        return 1

    # ── SAM2 checkpoint check ────────────────────────────────────────────────
    sam2_ckpt = Path(args.sam2_checkpoint)
    no_tracking = args.skip_tracking
    if not no_tracking and not sam2_ckpt.exists():
        print(
            f"\nWARNING: SAM2 checkpoint not found at '{sam2_ckpt}'.\n"
            "  Running detection only (steps 1-2). To enable the full pipeline,\n"
            "  download the checkpoint (~900 MB):\n\n"
            "  # Windows (PowerShell):\n"
            f"  Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt' "
            f"-OutFile '{sam2_ckpt}'\n\n"
            "  # Linux / macOS:\n"
            f"  curl -L -o {sam2_ckpt} "
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt\n"
        )
        no_tracking = True

    # ── Court weights check ──────────────────────────────────────────────────
    # If local YOLO weights don't exist the pipeline falls back to Roboflow API
    # automatically (as long as ROBOFLOW_API_KEY is set).
    skip_court = args.skip_court

    # ── Resolve device (SAM2 doesn't accept "auto") ──────────────────────────
    device = args.device
    if not device or device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Device: {device}\n")

    # ── Delegate to the full pipeline ────────────────────────────────────────
    from src.run_full_pipeline import run_pipeline
    import argparse as _ap

    fp_args = _ap.Namespace(
        video=str(video),
        output_dir=args.output_dir,
        config="object-detection/config/default.yaml",
        frame_stride=args.frame_stride,
        ball_frame_stride=None,
        court_frame_stride=None,
        yolo_weights=args.yolo_weights,
        court_weights=args.court_weights,
        sam2_checkpoint=str(sam2_ckpt) if not no_tracking else None,
        sam2_model_cfg=args.sam2_model_cfg,
        device=device,
        max_frames=None,
        no_tracking=no_tracking,
        no_clustering=args.skip_clustering,
        no_ocr=False,
        no_court_mapping=skip_court,
        ocr_model=args.ocr_model,
        ocr_ios_threshold=args.ios_threshold,
        ocr_device=device,
        clustering_device="cpu",
        clustering_batch_size=32,
        clustering_n_teams=2,
        mask_alpha=0.35,
        court_mask_alpha=0.25,
        panel_height=None,
        trail_length=None,
    )

    run_pipeline(fp_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
