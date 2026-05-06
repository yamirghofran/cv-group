import json
from pathlib import Path

from src.project_to_court import run_projection


class Args:
    video: str
    tracks: str
    court_keypoints: str
    ball: str | None
    output: str
    qa_report: str
    config: str = "object-detection/config/default.yaml"
    keypoint_confidence = None
    min_keypoints = None
    class_id_offset = None
    out_of_bounds_margin_ft = None


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_project_tracks_to_court_with_synthetic_homography(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"placeholder")
    tracks = tmp_path / "tracks.json"
    keypoints = tmp_path / "keypoints.json"
    ball = tmp_path / "ball.json"
    output = tmp_path / "court_tracks.json"
    qa = tmp_path / "qa.json"

    write_json(
        tracks,
        {
            "video": str(video),
            "fps": 30,
            "width": 1000,
            "height": 600,
            "mask_cleanup": {},
            "frames": [
                {
                    "frame_id": 0,
                    "timestamp_sec": 0.0,
                    "tracks": [
                        {
                            "track_id": 1,
                            "bbox_xyxy": [450, 250, 550, 350],
                            "mask_path": "missing.png",
                            "area": 100,
                            "score": None,
                            "source_prompt_bbox_xyxy": [450, 250, 550, 350],
                        }
                    ],
                }
            ],
        },
    )
    write_json(
        keypoints,
        {
            "video": str(video),
            "fps": 30,
            "width": 1000,
            "height": 600,
            "model_id": "court",
            "model_version": 1,
            "frame_stride": 1,
            "keypoint_confidence": 0.5,
            "min_keypoints": 4,
            "frames": [
                {
                    "frame_id": 0,
                    "timestamp_sec": 0.0,
                    "valid": True,
                    "reason": None,
                    "keypoints": [
                        {"class_name": "0", "class_id": 0, "confidence": 0.95, "image_xy": [100, 100], "source": "test"},
                        {"class_name": "5", "class_id": 5, "confidence": 0.95, "image_xy": [100, 600], "source": "test"},
                        {"class_name": "27", "class_id": 27, "confidence": 0.95, "image_xy": [1000, 100], "source": "test"},
                        {"class_name": "32", "class_id": 32, "confidence": 0.95, "image_xy": [1000, 600], "source": "test"},
                    ],
                }
            ],
        },
    )
    write_json(
        ball,
        {
            "video": str(video),
            "fps": 30,
            "width": 1000,
            "height": 600,
            "model_id": "player",
            "model_version": 1,
            "frame_stride": 1,
            "frames": [{"frame_id": 0, "timestamp_sec": 0.0, "image_xy": [500, 300], "confidence": 0.8, "source": "detected"}],
        },
    )
    args = Args()
    args.video = str(video)
    args.tracks = str(tracks)
    args.court_keypoints = str(keypoints)
    args.ball = str(ball)
    args.output = str(output)
    args.qa_report = str(qa)

    run_projection(args)
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["frames"][0]["players"][0]["valid"]
    assert data["frames"][0]["ball"]["valid"]
