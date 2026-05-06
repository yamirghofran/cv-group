import numpy as np

from src.court_geometry import (
    NBACourt,
    bottom_center_from_bbox,
    compute_homography_from_keypoints,
    is_inside_court,
    project_points,
)
from src.schemas import CourtKeypointRecord


def test_bottom_center_from_bbox():
    assert bottom_center_from_bbox([10, 20, 30, 60]) == [20.0, 60.0]


def test_compute_homography_from_synthetic_keypoints():
    court = NBACourt()
    image_points = {
        0: [100, 100],
        5: [100, 600],
        27: [1000, 100],
        32: [1000, 600],
        16: [550, 350],
    }
    keypoints = [
        CourtKeypointRecord(class_name=str(class_id), class_id=class_id, confidence=0.95, image_xy=xy)
        for class_id, xy in image_points.items()
    ]

    matrix, meta = compute_homography_from_keypoints(keypoints, court, min_confidence=0.5, min_keypoints=4)
    projected = project_points(matrix, [[550, 350]])[0]  # type: ignore[arg-type]

    assert meta["inliers"] >= 4
    assert np.allclose(projected, np.array(court.vertices[16]), atol=0.5)


def test_compute_homography_deduplicates_keypoint_ids_by_confidence():
    court = NBACourt()
    keypoints = [
        CourtKeypointRecord(class_name="0", class_id=0, confidence=0.1, image_xy=[999, 999]),
        CourtKeypointRecord(class_name="0", class_id=0, confidence=0.95, image_xy=[100, 100]),
        CourtKeypointRecord(class_name="5", class_id=5, confidence=0.95, image_xy=[100, 600]),
        CourtKeypointRecord(class_name="27", class_id=27, confidence=0.95, image_xy=[1000, 100]),
        CourtKeypointRecord(class_name="32", class_id=32, confidence=0.95, image_xy=[1000, 600]),
    ]

    matrix, meta = compute_homography_from_keypoints(keypoints, court, min_confidence=0.0, min_keypoints=4)

    assert matrix is not None
    assert meta["used_keypoint_ids"] == [0, 5, 27, 32]


def test_out_of_court_margin_check():
    court = NBACourt()

    assert is_inside_court([0, 0], court, margin=0)
    assert is_inside_court([-2, 20], court, margin=5)
    assert not is_inside_court([-6, 20], court, margin=5)
