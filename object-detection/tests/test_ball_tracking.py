from src.ball_tracking import interpolate_and_smooth_ball_frames
from src.schemas import BallFrame


def test_ball_interpolation_fills_short_gap():
    frames = [
        BallFrame(frame_id=0, timestamp_sec=0.0, image_xy=[0, 0], confidence=0.9, source="detected"),
        BallFrame(frame_id=1, timestamp_sec=1.0),
        BallFrame(frame_id=2, timestamp_sec=2.0, image_xy=[20, 0], confidence=0.9, source="detected"),
    ]

    output = interpolate_and_smooth_ball_frames(frames, max_missing_gap=2, max_jump_px=100, smoothing_window=1)

    assert output[1].source == "interpolated"
    assert output[1].image_xy == [10.0, 0.0]


def test_ball_interpolation_leaves_long_gap_missing():
    frames = [
        BallFrame(frame_id=0, timestamp_sec=0.0, image_xy=[0, 0], confidence=0.9, source="detected"),
        BallFrame(frame_id=1, timestamp_sec=1.0),
        BallFrame(frame_id=2, timestamp_sec=2.0),
        BallFrame(frame_id=3, timestamp_sec=3.0, image_xy=[30, 0], confidence=0.9, source="detected"),
    ]

    output = interpolate_and_smooth_ball_frames(frames, max_missing_gap=1, max_jump_px=100, smoothing_window=1)

    assert output[1].source == "missing"
    assert output[2].source == "missing"


def test_ball_tracking_rejects_large_jump():
    frames = [
        BallFrame(frame_id=0, timestamp_sec=0.0, image_xy=[0, 0], confidence=0.9, source="detected"),
        BallFrame(frame_id=1, timestamp_sec=1.0, image_xy=[1000, 0], confidence=0.9, source="detected"),
    ]

    output = interpolate_and_smooth_ball_frames(frames, max_missing_gap=1, max_jump_px=100, smoothing_window=1)

    assert output[1].source == "missing"
    assert output[1].image_xy is None
