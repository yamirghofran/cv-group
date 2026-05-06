from src.roboflow_client import normalize_roboflow_keypoints


def test_normalize_roboflow_keypoints():
    payload = {
        "predictions": [
            {
                "class": "court",
                "keypoints": [
                    {"x": 10, "y": 20, "class_name": "corner", "class_id": 0, "confidence": 0.91},
                    {"x": 30, "y": 40, "class_name": "paint", "class_id": 2, "confidence": 0.72},
                ],
            }
        ]
    }

    keypoints = normalize_roboflow_keypoints(payload)

    assert len(keypoints) == 2
    assert keypoints[0].class_id == 0
    assert keypoints[0].image_xy == [10.0, 20.0]
    assert keypoints[1].class_name == "paint"
