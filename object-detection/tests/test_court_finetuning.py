import pytest

from src.court_finetuning.common import validate_pose_data_yaml
from src.court_keypoints import build_client


def test_validate_pose_data_yaml_accepts_33_keypoints(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        """
train: train/images
val: valid/images
names:
  0: court
kpt_shape: [33, 3]
""",
        encoding="utf-8",
    )

    payload = validate_pose_data_yaml(data_yaml)

    assert payload["kpt_shape"] == [33, 3]


def test_validate_pose_data_yaml_rejects_wrong_keypoint_count(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        """
train: train/images
val: valid/images
names:
  0: court
kpt_shape: [17, 3]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Expected 33"):
        validate_pose_data_yaml(data_yaml)


def test_yolo_backend_requires_existing_weights(tmp_path):
    class Args:
        backend = "yolo"
        weights = str(tmp_path / "missing.pt")
        yolo_confidence = None
        device = None
        api_url = None
        model_id = None
        model_version = None
        confidence = None
        overlap = None
        api_key_env = "ROBOFLOW_API_KEY"

    config = {
        "court_keypoints": {
            "yolo_weights": str(tmp_path / "missing.pt"),
            "yolo_confidence": 0.25,
            "yolo_device": "cpu",
        }
    }

    with pytest.raises(FileNotFoundError, match="YOLO court keypoint weights not found"):
        build_client(Args(), config)
