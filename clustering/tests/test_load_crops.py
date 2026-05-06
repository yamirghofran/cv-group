from pathlib import Path

import cv2
import numpy as np
import pytest

from clustering.load_crops import (
    list_crop_paths,
    load_crops,
    load_track_crops,
    parse_track_id_from_path,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sample_crops"
TRACK_FIXTURES = Path(__file__).parent / "fixtures" / "track_crops"


# ---------------------------------------------------------------------------
# parse_track_id_from_path
# ---------------------------------------------------------------------------


def test_parse_track_id_from_track_subdir() -> None:
    assert parse_track_id_from_path(Path("crops/clip/track_003/frame_000150.jpg")) == 3


def test_parse_track_id_from_multi_digit_track() -> None:
    assert parse_track_id_from_path(Path("output/track_012/frame_000000.jpg")) == 12


def test_parse_track_id_returns_none_for_flat_path() -> None:
    assert parse_track_id_from_path(Path("crops/player_001.png")) is None


def test_parse_track_id_case_insensitive() -> None:
    assert parse_track_id_from_path(Path("crops/Track_005/frame.png")) == 5


def test_parse_track_id_ignores_non_track_dirs() -> None:
    assert parse_track_id_from_path(Path("crops/track_something/frame.jpg")) is None


# ---------------------------------------------------------------------------
# list_crop_paths (existing tests)
# ---------------------------------------------------------------------------


def test_list_crop_paths_finds_fixtures() -> None:
    paths = list_crop_paths(FIXTURES)
    assert len(paths) == 6
    assert all(p.suffix == ".png" for p in paths)


def test_list_crop_paths_returns_sorted() -> None:
    paths = list_crop_paths(FIXTURES)
    assert paths == sorted(paths)


def test_list_crop_paths_filters_extensions(tmp_path: Path) -> None:
    (tmp_path / "good.png").write_bytes(cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint8))[1].tobytes())
    (tmp_path / "ignored.txt").write_text("not an image")
    (tmp_path / "subdir").mkdir()

    paths = list_crop_paths(tmp_path)

    assert [p.name for p in paths] == ["good.png"]


def test_list_crop_paths_raises_for_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        list_crop_paths(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# load_crops (existing tests)
# ---------------------------------------------------------------------------


def test_load_crops_returns_path_image_pairs() -> None:
    pairs = load_crops(FIXTURES)
    assert len(pairs) == 6
    for path, image in pairs:
        assert isinstance(path, Path)
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert image.shape == (32, 32, 3)


def test_load_crops_preserves_path_order() -> None:
    pairs = load_crops(FIXTURES)
    paths = [path for path, _ in pairs]
    assert paths == list_crop_paths(FIXTURES)


def test_load_crops_raises_on_undecodable_file(tmp_path: Path) -> None:
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not a real PNG")

    with pytest.raises(ValueError, match="Failed to decode crop"):
        load_crops(tmp_path)


# ---------------------------------------------------------------------------
# load_track_crops — flat directory
# ---------------------------------------------------------------------------


def test_load_track_crops_flat_returns_none_track_ids() -> None:
    triples = load_track_crops(FIXTURES)
    assert len(triples) == 6
    for path, image, track_id in triples:
        assert isinstance(path, Path)
        assert isinstance(image, np.ndarray)
        assert track_id is None


def test_load_track_crops_flat_preserves_path_order() -> None:
    triples = load_track_crops(FIXTURES)
    paths = [p for p, _, _ in triples]
    assert paths == list_crop_paths(FIXTURES)


# ---------------------------------------------------------------------------
# load_track_crops — hierarchical track_*/ directory
# ---------------------------------------------------------------------------


def test_load_track_crops_hierarchical_finds_all_crops() -> None:
    triples = load_track_crops(TRACK_FIXTURES)
    assert len(triples) == 6


def test_load_track_crops_hierarchical_parses_track_ids() -> None:
    triples = load_track_crops(TRACK_FIXTURES)
    track_ids = {tid for _, _, tid in triples}
    assert track_ids == {1, 2}


def test_load_track_crops_hierarchical_groups_by_track() -> None:
    triples = load_track_crops(TRACK_FIXTURES)
    by_track: dict[int, list[Path]] = {}
    for path, _, tid in triples:
        assert tid is not None
        by_track.setdefault(tid, []).append(path)

    assert len(by_track[1]) == 3
    assert len(by_track[2]) == 3


def test_load_track_crops_hierarchical_sorts_by_track_then_frame() -> None:
    triples = load_track_crops(TRACK_FIXTURES)
    # track_001 crops come first (sorted by track dir name), then track_002
    track_ids = [tid for _, _, tid in triples]
    assert track_ids[:3] == [1, 1, 1]
    assert track_ids[3:] == [2, 2, 2]


# ---------------------------------------------------------------------------
# load_track_crops — edge cases
# ---------------------------------------------------------------------------


def test_load_track_crops_empty_directory(tmp_path: Path) -> None:
    triples = load_track_crops(tmp_path)
    assert triples == []


def test_load_track_crops_raises_on_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        load_track_crops(tmp_path / "nope")


def test_load_track_crops_raises_on_undecodable_file(tmp_path: Path) -> None:
    track_dir = tmp_path / "track_001"
    track_dir.mkdir()
    (track_dir / "bad.jpg").write_bytes(b"not an image")

    with pytest.raises(ValueError, match="Failed to decode crop"):
        load_track_crops(tmp_path)
