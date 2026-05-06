from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from jersey_numbers.matching.ios import compute_ios, compute_iou, load_mask


THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]


def _score_all_pairs(
    tracks_path: Path,
    detections_path: Path,
) -> list[dict[str, float]]:
    """Compute both IoS and IoU for every (number_detection, track) pair."""
    with tracks_path.open(encoding="utf-8") as f:
        tracks_data = json.load(f)
    with detections_path.open(encoding="utf-8") as f:
        detections_data = json.load(f)

    det_by_frame: dict[int, list[dict[str, Any]]] = {}
    for frame in detections_data.get("frames", []):
        det_by_frame[int(frame["frame_id"])] = frame.get("detections", [])

    pairs: list[dict[str, float]] = []
    for frame in tracks_data.get("frames", []):
        frame_id = int(frame["frame_id"])
        number_dets = [d for d in det_by_frame.get(frame_id, []) if d.get("class_name") == "number"]
        if not number_dets:
            continue
        for det in number_dets:
            bbox = det["bbox_xyxy"]
            for track in frame.get("tracks", []):
                mask = load_mask(track["mask_path"])
                if mask is None:
                    continue
                ios = compute_ios(mask, bbox)
                iou = compute_iou(mask, bbox)
                pairs.append({"ios": ios, "iou": iou})

    return pairs


def compare_metrics(
    tracks_path: Path,
    detections_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    print("Scoring all (detection, track) pairs — this reads every mask file…")
    pairs = _score_all_pairs(tracks_path, detections_path)
    if not pairs:
        print("No number detections found. Check that detections JSON contains class 'number'.")
        return {}

    print(f"Scored {len(pairs)} pairs across all frames.\n")

    ios_scores = np.array([p["ios"] for p in pairs])
    iou_scores = np.array([p["iou"] for p in pairs])

    results: dict[str, Any] = {"thresholds": {}}

    print(f"{'Threshold':>10} {'IoS matches':>12} {'IoU matches':>12} {'IoS gain':>10}")
    print("-" * 48)

    for t in THRESHOLDS:
        ios_matches = int((ios_scores >= t).sum())
        iou_matches = int((iou_scores >= t).sum())
        gain = ios_matches - iou_matches
        print(f"{t:>10.2f} {ios_matches:>12} {iou_matches:>12} {gain:>+10}")
        results["thresholds"][str(t)] = {
            "ios_matches": ios_matches,
            "iou_matches": iou_matches,
            "ios_gain_over_iou": gain,
        }

    results["total_pairs"] = len(pairs)
    results["ios_mean"] = float(ios_scores.mean())
    results["iou_mean"] = float(iou_scores.mean())

    print(f"\nMean IoS={ios_scores.mean():.4f}  Mean IoU={iou_scores.mean():.4f}")
    _plot_comparison(results, output_dir)

    if output_dir:
        out = output_dir / "eval_matching_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results: {out}")

    return results


def _plot_comparison(results: dict[str, Any], output_dir: Path | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    thresholds = [float(t) for t in results["thresholds"]]
    ios_vals = [results["thresholds"][str(t)]["ios_matches"] for t in thresholds]
    iou_vals = [results["thresholds"][str(t)]["iou_matches"] for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, ios_vals, marker="o", label="IoS", linewidth=2)
    ax.plot(thresholds, iou_vals, marker="s", label="IoU", linewidth=2, linestyle="--")
    ax.axvline(x=0.9, color="gray", linestyle=":", label="threshold=0.9")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Number of matched detections")
    ax.set_title("IoS vs IoU: matches retained at each threshold")
    ax.legend()
    plt.tight_layout()

    if output_dir:
        out = output_dir / "ios_vs_iou.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150)
        print(f"Saved plot: {out}")
    else:
        plt.show()
    plt.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare IoS vs IoU matching across thresholds.")
    parser.add_argument("--tracks", required=True, help="Tracks JSON from object-detection module.")
    parser.add_argument("--detections", required=True, help="Detections JSON from object-detection module.")
    parser.add_argument("--output-dir", help="Directory for results JSON and plot PNG.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    compare_metrics(
        tracks_path=Path(args.tracks),
        detections_path=Path(args.detections),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
