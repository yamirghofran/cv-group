from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .utils import bbox_from_mask, ensure_parent_dir


def clean_mask(mask: np.ndarray, distance_threshold: float = 80, min_component_area: int = 50) -> np.ndarray:
    """Remove distant disconnected fragments from a binary player mask."""
    binary = (mask > 0).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return binary

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if component_count <= 1:
        return binary

    component_ids = list(range(1, component_count))
    largest_id = max(component_ids, key=lambda idx: stats[idx, cv2.CC_STAT_AREA])
    largest_center = centroids[largest_id]
    cleaned = labels == largest_id

    for component_id in component_ids:
        if component_id == largest_id:
            continue
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        center = centroids[component_id]
        distance = float(np.linalg.norm(center - largest_center))
        if distance <= distance_threshold:
            cleaned |= labels == component_id

    return cleaned.astype(np.uint8)


def save_cleaned_mask(input_path: str | Path, output_path: str | Path, distance_threshold: float, min_component_area: int) -> None:
    mask = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {input_path}")
    cleaned = clean_mask(mask, distance_threshold=distance_threshold, min_component_area=min_component_area)
    ensure_parent_dir(output_path)
    cv2.imwrite(str(output_path), cleaned * 255)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean a binary player mask with connected components.")
    parser.add_argument("--input", required=True, help="Input mask PNG.")
    parser.add_argument("--output", required=True, help="Output cleaned mask PNG.")
    parser.add_argument("--distance-threshold", type=float, default=80)
    parser.add_argument("--min-component-area", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    save_cleaned_mask(args.input, args.output, args.distance_threshold, args.min_component_area)
    bbox = bbox_from_mask(cv2.imread(str(args.output), cv2.IMREAD_GRAYSCALE))
    print(f"Wrote cleaned mask: {args.output}; bbox_xyxy={bbox}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
