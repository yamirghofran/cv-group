from pathlib import Path

import cv2
import numpy as np
import pytest

from clustering.load_crops import list_crop_paths, load_crops

FIXTURES = Path(__file__).parent / "fixtures" / "sample_crops"


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
