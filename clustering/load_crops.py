"""Load player crops from a directory into the format the TeamClassifier expects.

Reads PNG/JPEG files in deterministic (filename-sorted) order and returns BGR
numpy arrays as produced by cv2.imread, matching how the rest of the pipeline
moves images around.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CROP_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_crop_paths(crops_dir: str | Path) -> list[Path]:
    """Return crop file paths in a directory, sorted by filename."""
    directory = Path(crops_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in CROP_EXTENSIONS
    )


def load_crops(crops_dir: str | Path) -> list[tuple[Path, np.ndarray]]:
    """Load all crops from a directory.

    Returns a list of (path, image) pairs. Images are BGR uint8 numpy arrays as
    returned by cv2.imread. Order matches `list_crop_paths`.

    Raises NotADirectoryError if `crops_dir` is not a directory, and ValueError
    if any file fails to decode.
    """
    pairs: list[tuple[Path, np.ndarray]] = []
    for path in list_crop_paths(crops_dir):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to decode crop: {path}")
        pairs.append((path, image))
    return pairs
