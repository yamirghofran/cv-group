from src.roboflow_client import normalize_roboflow_predictions


def test_normalize_roboflow_predictions_converts_center_bbox_to_xyxy():
    payload = {
        "predictions": [
            {
                "x": 50,
                "y": 40,
                "width": 20,
                "height": 10,
                "class": "player",
                "class_id": 3,
                "confidence": 0.91,
            }
        ]
    }

    records = normalize_roboflow_predictions(payload, frame_width=100, frame_height=80)

    assert len(records) == 1
    assert records[0].bbox_xyxy == [40.0, 35.0, 60.0, 45.0]
    assert records[0].class_name == "player"
    assert records[0].class_id == 3


def test_normalize_roboflow_predictions_clamps_to_frame():
    payload = {
        "predictions": [
            {
                "x": 5,
                "y": 5,
                "width": 20,
                "height": 20,
                "class": "number",
                "confidence": 0.5,
            }
        ]
    }

    records = normalize_roboflow_predictions(payload, frame_width=100, frame_height=80)

    assert records[0].bbox_xyxy == [0.0, 0.0, 15.0, 15.0]
