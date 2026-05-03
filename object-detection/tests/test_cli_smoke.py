import json
from pathlib import Path

import cv2
import numpy as np

from src.detection import main as detection_main
from src.tracking import main as tracking_main


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
