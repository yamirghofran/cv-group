import numpy as np
import cv2

from src.visualize_court import render_source_panel


def test_source_panel_draws_tracks_before_resizing(tmp_path):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), np.zeros((48, 64), dtype=np.uint8))
    track = {
        "track_id": 1,
        "bbox_xyxy": [10, 8, 25, 35],
        "mask_path": str(mask_path),
    }

    panel = render_source_panel(frame, [track], tmp_path / "tracks.json", mask_alpha=0.25, target_height=96)
    nonzero_y, nonzero_x = np.where(np.any(panel > 0, axis=2))

    assert panel.shape[:2] == (96, 128)
    assert int(nonzero_x.max()) > 70
    assert int(nonzero_y.max()) > 60
