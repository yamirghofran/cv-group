import numpy as np

from src.mask_cleanup import clean_mask
from src.utils import bbox_from_mask


def test_clean_mask_removes_distant_fragment():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:50] = 1
    mask[80:85, 80:85] = 1

    cleaned = clean_mask(mask, distance_threshold=20, min_component_area=1)

    assert cleaned[30, 30] == 1
    assert cleaned[82, 82] == 0


def test_clean_mask_keeps_nearby_component():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:50] = 1
    mask[55:65, 48:58] = 1

    cleaned = clean_mask(mask, distance_threshold=40, min_component_area=1)

    assert cleaned[30, 30] == 1
    assert cleaned[60, 52] == 1


def test_empty_mask_bbox_is_zero():
    mask = np.zeros((20, 30), dtype=np.uint8)
    cleaned = clean_mask(mask)

    assert bbox_from_mask(cleaned) == [0.0, 0.0, 0.0, 0.0]
