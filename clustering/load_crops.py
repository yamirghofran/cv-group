"""Load player crops from a directory into the format the TeamClassifier expects.

Reads PNG/JPEG files in deterministic (filename-sorted) order and returns BGR
numpy arrays as produced by cv2.imread, matching how the rest of the pipeline
moves images around.

Supports two directory layouts:
  * **Flat** – all crops in a single directory (legacy behaviour).
  * **Hierarchical** – per-track sub-directories named ``track_###/`` each
    containing frame crops, as produced by ``export_crops.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

CROP_EXTENSIONS = (".png", ".jpg", ".jpeg")

_TRACK_DIR_RE = re.compile(r"^track_(\d+)$", re.IGNORECASE)


def parse_track_id_from_path(crop_path: Path) -> int | None:
    """Extract a track ID from a crop path that lives under a ``track_###/`` directory.

    Scans path parts left-to-right and returns the integer from the first
    component matching ``track_<digits>``.  Returns ``None`` when no such
    component is found (e.g. crops in a flat directory layout).

    Examples::

        >>> parse_track_id_from_path(Path("crops/clip/track_003/frame_000150.jpg"))
        3
        >>> parse_track_id_from_path(Path("crops/clip/player.png"))
        None
    """
    for part in crop_path.parts:
        match = _TRACK_DIR_RE.match(part)
        if match:
            return int(match.group(1))
    return None


def _has_track_subdirs(directory: Path) -> bool:
    """Return True if *directory* contains at least one ``track_*/`` subdirectory."""
    return any(
        child.is_dir() and _TRACK_DIR_RE.match(child.name)
        for child in directory.iterdir()
    )


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


def _list_track_crop_paths(crops_dir: Path) -> list[Path]:
    """Recursively collect crop paths from ``track_*/`` sub-directories."""
    paths: list[Path] = []
    track_dirs = sorted(
        child
        for child in crops_dir.iterdir()
        if child.is_dir() and _TRACK_DIR_RE.match(child.name)
    )
    for track_dir in track_dirs:
        paths.extend(
            sorted(
                p
                for p in track_dir.iterdir()
                if p.is_file() and p.suffix.lower() in CROP_EXTENSIONS
            )
        )
    return paths


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


def load_track_crops(crops_dir: str | Path) -> list[tuple[Path, np.ndarray, int | None]]:
    """Load crops from a directory, returning ``(path, image, track_id)`` triples.

    Handles both directory layouts transparently:

    * **Flat** – images directly in *crops_dir*. ``track_id`` will be ``None``
      for every crop.
    * **Hierarchical** – sub-directories named ``track_###/`` (as produced by
      ``export_crops.py``). ``track_id`` is parsed from the directory name.

    Images are BGR uint8 numpy arrays as returned by ``cv2.imread``.
    Crops are sorted deterministically: by track directory first, then by
    filename within each track.

    Raises:
        NotADirectoryError: if *crops_dir* does not exist or is not a directory.
        ValueError: if any file fails to decode.
    """
    directory = Path(crops_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if _has_track_subdirs(directory):
        paths = _list_track_crop_paths(directory)
    else:
        paths = list_crop_paths(directory)

    if not paths:
        return []

    triples: list[tuple[Path, np.ndarray, int | None]] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to decode crop: {path}")
        track_id = parse_track_id_from_path(path)
        triples.append((path, image, track_id))
    return triples
