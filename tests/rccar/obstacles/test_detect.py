import cv2
import numpy as np

from rccar.obstacles.detect import Obstacle, define_corridor, detect_obstacles

FRAME_SHAPE = (240, 320, 3)


def _blank_frame():
    return np.zeros(FRAME_SHAPE, dtype=np.uint8)


def _all_road_mask():
    return np.full((240, 320), 255, dtype=np.uint8)


def test_obstacle_blob_inside_corridor_is_detected():
    # Curb on the left at x=100 -> corridor is x >= 100 (right side of curb).
    corridor_mask = define_corridor(FRAME_SHAPE, curb_side="left", curb_x=100)

    road_mask = _all_road_mask()
    # Punch a non-road rectangle well inside the corridor (x in [150,190),
    # y in [100,140)) -> actual center at (169.5, 119.5).
    x0, y0, x1, y1 = 150, 100, 190, 140
    road_mask[y0:y1, x0:x1] = 0

    frame = _blank_frame()
    obstacles = detect_obstacles(frame, road_mask, corridor_mask, min_blob_area=50)

    assert len(obstacles) == 1
    obs = obstacles[0]
    assert isinstance(obs, Obstacle)
    expected_cx = (x0 + x1 - 1) / 2.0
    expected_cy = (y0 + y1 - 1) / 2.0
    assert abs(obs.centroid[0] - expected_cx) <= 2.0
    assert abs(obs.centroid[1] - expected_cy) <= 2.0
    assert obs.area == (x1 - x0) * (y1 - y0)
    assert obs.bbox == (x0, y0, x1 - x0, y1 - y0)


def test_obstacle_blob_outside_corridor_is_ignored():
    # Curb on the left at x=200 -> corridor is x >= 200.
    corridor_mask = define_corridor(FRAME_SHAPE, curb_side="left", curb_x=200)

    road_mask = _all_road_mask()
    # Punch a non-road rectangle entirely to the left of the corridor
    # (x in [10,50)), i.e. off-road but outside the drivable corridor.
    road_mask[100:140, 10:50] = 0

    frame = _blank_frame()
    obstacles = detect_obstacles(frame, road_mask, corridor_mask, min_blob_area=50)

    assert obstacles == []


def test_tiny_noise_speckle_below_min_area_is_ignored():
    # Fallback mode (no curb) -> centered 60% band corridor, which easily
    # covers the frame center where we place the speckle.
    corridor_mask = define_corridor(FRAME_SHAPE, curb_side=None)

    road_mask = _all_road_mask()
    # 3x3 speckle (area 9) well within the fallback band, but below the
    # default min_blob_area of 50.
    road_mask[120:123, 160:163] = 0

    frame = _blank_frame()
    obstacles = detect_obstacles(frame, road_mask, corridor_mask, min_blob_area=50)

    assert obstacles == []


def test_define_corridor_fallback_band_is_centered():
    mask = define_corridor(FRAME_SHAPE, curb_side="none", fallback_band_frac=0.6)
    width = FRAME_SHAPE[1]
    # Middle 60% of 320 = 192px, centered -> columns [64, 256).
    assert mask[0, 0] == 0
    assert mask[0, width // 2] == 255
    assert mask[0, width - 1] == 0
