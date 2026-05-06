from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from .schemas import CourtKeypointRecord


@dataclass(frozen=True)
class NBACourt:
    """NBA court geometry in feet with keypoint IDs aligned to Roboflow court models."""

    length: float = 94.0
    width: float = 50.0
    paint_width: float = 16.0
    paint_length: float = 19.0
    free_throw_line_distance: float = 15.0
    center_circle_radius: float = 6.0
    restricted_area_radius: float = 4.0
    rim_radius: float = 0.75
    baseline_to_rim_center: float = 5.25
    three_point_arc_radius: float = 23.75
    straight_three_point_length: float = 13.92
    sideline_to_corner_three: float = 3.0
    baseline_to_throw_line: float = 27.4
    throw_line_length: float = 1.65

    @property
    def paint_y1(self) -> float:
        return (self.width - self.paint_width) / 2.0

    @property
    def paint_y2(self) -> float:
        return self.paint_y1 + self.paint_width

    @property
    def center_y(self) -> float:
        return self.width / 2.0

    @property
    def vertices(self) -> list[tuple[float, float]]:
        py1, py2, cy = self.paint_y1, self.paint_y2, self.center_y
        return [
            (0.0, 0.0),
            (0.0, self.sideline_to_corner_three),
            (0.0, py1),
            (0.0, py2),
            (0.0, self.width - self.sideline_to_corner_three),
            (0.0, self.width),
            (self.baseline_to_rim_center, cy),
            (self.straight_three_point_length, self.sideline_to_corner_three),
            (self.straight_three_point_length, self.width - self.sideline_to_corner_three),
            (self.paint_length, py1),
            (self.paint_length, cy),
            (self.paint_length, py2),
            (self.baseline_to_throw_line, 0.0),
            (self.baseline_to_rim_center + self.three_point_arc_radius, cy),
            (self.baseline_to_throw_line, self.width),
            (self.length / 2.0, 0.0),
            (self.length / 2.0, cy),
            (self.length / 2.0, self.width),
            (self.length - self.baseline_to_throw_line, 0.0),
            (self.length - self.baseline_to_rim_center - self.three_point_arc_radius, cy),
            (self.length - self.baseline_to_throw_line, self.width),
            (self.length - self.paint_length, py1),
            (self.length - self.paint_length, cy),
            (self.length - self.paint_length, py2),
            (self.length - self.straight_three_point_length, self.sideline_to_corner_three),
            (self.length - self.straight_three_point_length, self.width - self.sideline_to_corner_three),
            (self.length - self.baseline_to_rim_center, cy),
            (self.length, 0.0),
            (self.length, self.sideline_to_corner_three),
            (self.length, py1),
            (self.length, py2),
            (self.length, self.width - self.sideline_to_corner_three),
            (self.length, self.width),
        ]

    @property
    def edges(self) -> list[tuple[int, int]]:
        return [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (2, 9), (9, 10), (10, 11), (11, 3),
            (1, 7), (4, 8), (1, 4),
            (0, 12), (12, 15), (15, 18), (18, 27),
            (15, 16), (16, 17),
            (5, 14), (14, 17), (17, 20), (20, 32),
            (27, 28), (28, 29), (29, 30), (30, 31), (31, 32),
            (28, 24), (31, 25), (28, 31),
            (29, 21), (21, 22), (22, 23), (23, 30),
        ]


@dataclass
class HomographyResult:
    matrix: np.ndarray
    source_frame_id: int
    inliers: int
    used_keypoint_ids: list[int] = field(default_factory=list)


def bottom_center_from_bbox(bbox_xyxy: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return [(x1 + x2) / 2.0, y2]


def compute_homography_from_keypoints(
    keypoints: list[CourtKeypointRecord],
    court: NBACourt,
    min_confidence: float,
    min_keypoints: int,
    class_id_offset: int = 0,
    ransac_reprojection_threshold: float = 8.0,
) -> tuple[np.ndarray | None, dict[str, int | list[int]]]:
    by_vertex: dict[int, CourtKeypointRecord] = {}
    vertices = court.vertices

    for keypoint in keypoints:
        if keypoint.confidence < min_confidence:
            continue
        vertex_index = keypoint.class_id - class_id_offset
        if vertex_index < 0 or vertex_index >= len(vertices):
            continue
        previous = by_vertex.get(vertex_index)
        if previous is None or keypoint.confidence > previous.confidence:
            by_vertex[vertex_index] = keypoint

    used_indexes = sorted(by_vertex)
    source_points = [by_vertex[index].image_xy for index in used_indexes]
    target_points = [vertices[index] for index in used_indexes]
    used_ids = [by_vertex[index].class_id for index in used_indexes]

    if len(source_points) < min_keypoints:
        return None, {"inliers": 0, "used_keypoint_ids": used_ids}

    matrix, inlier_mask = cv2.findHomography(
        np.array(source_points, dtype=np.float32),
        np.array(target_points, dtype=np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reprojection_threshold,
    )
    inliers = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else 0
    if matrix is None or inliers < min_keypoints:
        return None, {"inliers": inliers, "used_keypoint_ids": used_ids}
    return matrix, {"inliers": inliers, "used_keypoint_ids": used_ids}


def project_points(matrix: np.ndarray, points: list[list[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return array.reshape(0, 2)
    reshaped = array.reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(reshaped, matrix)
    return projected.reshape(-1, 2)


def is_inside_court(point_xy: Iterable[float], court: NBACourt, margin: float) -> bool:
    x, y = [float(v) for v in point_xy]
    return -margin <= x <= court.length + margin and -margin <= y <= court.width + margin


def court_to_pixel(point_xy: Iterable[float], court: NBACourt, scale: float, padding: int) -> tuple[int, int]:
    x, y = [float(v) for v in point_xy]
    return int(round(x * scale + padding)), int(round(y * scale + padding))


def draw_court(
    court: NBACourt,
    scale: float = 10.0,
    padding: int = 35,
    background_color: tuple[int, int, int] = (58, 124, 68),
    line_color: tuple[int, int, int] = (245, 245, 245),
) -> np.ndarray:
    width_px = int(round(court.length * scale + 2 * padding))
    height_px = int(round(court.width * scale + 2 * padding))
    image = np.full((height_px, width_px, 3), background_color, dtype=np.uint8)

    for start, end in court.edges:
        cv2.line(
            image,
            court_to_pixel(court.vertices[start], court, scale, padding),
            court_to_pixel(court.vertices[end], court, scale, padding),
            line_color,
            2,
        )

    left_rim = court_to_pixel(court.vertices[6], court, scale, padding)
    right_rim = court_to_pixel(court.vertices[26], court, scale, padding)
    center = court_to_pixel(court.vertices[16], court, scale, padding)
    cv2.circle(image, center, int(round(court.center_circle_radius * scale)), line_color, 2)
    cv2.circle(image, left_rim, max(2, int(round(court.rim_radius * scale))), line_color, 2)
    cv2.circle(image, right_rim, max(2, int(round(court.rim_radius * scale))), line_color, 2)

    for rim, side in [(left_rim, "left"), (right_rim, "right")]:
        start_angle, end_angle = (180, 360) if side == "left" else (0, 180)
        cv2.ellipse(
            image,
            rim,
            (int(round(court.restricted_area_radius * scale)), int(round(court.restricted_area_radius * scale))),
            90,
            start_angle,
            end_angle,
            line_color,
            2,
        )

    return image
