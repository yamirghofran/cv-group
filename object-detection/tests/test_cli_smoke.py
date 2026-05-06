import json
from pathlib import Path

import cv2
import numpy as np

from src.detection import main as detection_main
from src.ball_tracking import main as ball_tracking_main
from src.court_keypoints import main as court_keypoints_main
from src.project_to_court import main as project_to_court_main
from src.tracking import main as tracking_main
from src.visualize_court import main as visualize_court_main


def write_synthetic_video(path: Path, frame_count: int = 4) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 4.0, (64, 48))
    assert writer.isOpened()
    try:
        for frame_id in range(frame_count):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            cv2.rectangle(frame, (10 + frame_id, 8), (25 + frame_id, 35), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_detection_and_mock_tracking_cli_smoke(tmp_path):
    video_path = tmp_path / "clip.mp4"
    write_synthetic_video(video_path)
    mock_response = tmp_path / "mock_roboflow.json"
    mock_response.write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "x": 18,
                        "y": 21,
                        "width": 16,
                        "height": 28,
                        "class": "player",
                        "confidence": 0.95,
                    },
                    {
                        "x": 18,
                        "y": 14,
                        "width": 5,
                        "height": 6,
                        "class": "number",
                        "confidence": 0.75,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    detections_path = tmp_path / "detections.json"
    tracks_path = tmp_path / "tracks.json"
    masks_dir = tmp_path / "masks"
    crops_dir = tmp_path / "crops"
    qa_path = tmp_path / "qa.json"

    assert detection_main(
        [
            "--video",
            str(video_path),
            "--output",
            str(detections_path),
            "--mock-response",
            str(mock_response),
            "--frame-stride",
            "2",
        ]
    ) == 0
    assert detections_path.exists()

    assert tracking_main(
        [
            "--video",
            str(video_path),
            "--detections",
            str(detections_path),
            "--tracker-backend",
            "mock",
            "--output",
            str(tracks_path),
            "--mask-dir",
            str(masks_dir),
            "--crop-dir",
            str(crops_dir),
            "--qa-report",
            str(qa_path),
        ]
    ) == 0

    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    assert len(tracks["frames"]) == 4
    assert tracks["frames"][0]["tracks"][0]["track_id"] == 1
    assert list(masks_dir.glob("*.png"))
    assert qa_path.exists()


def test_court_mapping_cli_smoke(tmp_path):
    video_path = tmp_path / "clip.mp4"
    write_synthetic_video(video_path)
    detection_response = tmp_path / "mock_detection.json"
    detection_response.write_text(
        json.dumps(
            {
                "predictions": [
                    {"x": 18, "y": 21, "width": 16, "height": 28, "class": "player", "confidence": 0.95},
                    {"x": 30, "y": 20, "width": 4, "height": 4, "class": "ball", "confidence": 0.9},
                ]
            }
        ),
        encoding="utf-8",
    )
    court_response = tmp_path / "mock_court.json"
    court_response.write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "class": "court",
                        "keypoints": [
                            {"x": 0, "y": 0, "class_name": "0", "class_id": 0, "confidence": 0.95},
                            {"x": 0, "y": 48, "class_name": "5", "class_id": 5, "confidence": 0.95},
                            {"x": 64, "y": 0, "class_name": "27", "class_id": 27, "confidence": 0.95},
                            {"x": 64, "y": 48, "class_name": "32", "class_id": 32, "confidence": 0.95},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    detections_path = tmp_path / "detections.json"
    tracks_path = tmp_path / "tracks.json"
    masks_dir = tmp_path / "masks"
    crops_dir = tmp_path / "crops"
    tracking_qa_path = tmp_path / "tracking_qa.json"
    court_keypoints_path = tmp_path / "court_keypoints.json"
    ball_path = tmp_path / "ball.json"
    court_tracks_path = tmp_path / "court_tracks.json"
    court_qa_path = tmp_path / "court_qa.json"
    visualization_path = tmp_path / "court_visualization.mp4"

    assert detection_main(
        [
            "--video",
            str(video_path),
            "--output",
            str(detections_path),
            "--mock-response",
            str(detection_response),
        ]
    ) == 0
    assert tracking_main(
        [
            "--video",
            str(video_path),
            "--detections",
            str(detections_path),
            "--tracker-backend",
            "mock",
            "--output",
            str(tracks_path),
            "--mask-dir",
            str(masks_dir),
            "--crop-dir",
            str(crops_dir),
            "--qa-report",
            str(tracking_qa_path),
        ]
    ) == 0
    assert court_keypoints_main(
        [
            "--video",
            str(video_path),
            "--output",
            str(court_keypoints_path),
            "--mock-response",
            str(court_response),
            "--frame-stride",
            "2",
            "--debug-frame-dir",
            str(tmp_path / "debug_court"),
        ]
    ) == 0
    assert ball_tracking_main(
        [
            "--video",
            str(video_path),
            "--output",
            str(ball_path),
            "--mock-response",
            str(detection_response),
        ]
    ) == 0
    assert project_to_court_main(
        [
            "--video",
            str(video_path),
            "--tracks",
            str(tracks_path),
            "--court-keypoints",
            str(court_keypoints_path),
            "--ball",
            str(ball_path),
            "--output",
            str(court_tracks_path),
            "--qa-report",
            str(court_qa_path),
        ]
    ) == 0
    assert visualize_court_main(
        [
            "--video",
            str(video_path),
            "--tracks",
            str(tracks_path),
            "--court-tracks",
            str(court_tracks_path),
            "--output",
            str(visualization_path),
            "--panel-height",
            "180",
        ]
    ) == 0

    court_tracks = json.loads(court_tracks_path.read_text(encoding="utf-8"))
    assert court_tracks["frames"][0]["players"][0]["valid"]
    assert court_tracks["frames"][0]["ball"]["valid"]
    assert court_qa_path.exists()
    assert visualization_path.exists()
