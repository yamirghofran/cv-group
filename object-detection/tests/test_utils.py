import numpy as np

from src.utils import bbox_from_mask, clamp_bbox_xyxy, is_player_like_class


def test_player_like_filter_catches_action_classes():
    assert is_player_like_class("player")
    assert is_player_like_class("player-jump-shot")
    assert is_player_like_class("player-layup-dunk")
    assert not is_player_like_class("referee")
    assert not is_player_like_class("number")


def test_clamp_bbox_xyxy_orders_and_clamps():
    assert clamp_bbox_xyxy([120, -10, 10, 90], width=100, height=80) == [10.0, 0.0, 100.0, 80.0]


def test_bbox_from_mask_returns_exclusive_max_corner():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:10, 7:12] = 1

    assert bbox_from_mask(mask) == [7.0, 5.0, 12.0, 10.0]
