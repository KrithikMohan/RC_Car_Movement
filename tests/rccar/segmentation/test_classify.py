"""Tests for the adaptive road/non-road pixel classifier (T8)."""

import numpy as np

from rccar.segmentation.classify import classify_frame, should_rebuild_model
from rccar.segmentation.model import RoadColorModel, build_road_color_model

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

ROAD_BGR = (60, 120, 80)
OBSTACLE_BGR = (200, 30, 220)  # a hue/saturation far from the road color


def _iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union > 0 else 1.0


def test_classify_frame_separates_road_from_obstacle_with_high_iou():
    """Build a road-color model from a solid road-color frame, then classify
    a two-tone frame (road rectangle + differently-colored obstacle
    rectangle). The predicted road mask should closely match the true road
    region (IoU > 0.9)."""
    # Model built from a solid road-color frame (equivalent to sampling an
    # ROI that is 100% road-colored).
    road_only_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    road_only_frame[:, :] = ROAD_BGR
    model = build_road_color_model(road_only_frame)
    assert model.histogram.sum() > 0

    # Two-tone frame: road color everywhere except an obstacle rectangle.
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:, :] = ROAD_BGR
    gt_road_mask = np.full((FRAME_HEIGHT, FRAME_WIDTH), 255, dtype=np.uint8)

    obstacle_slice = (slice(50, 180), slice(100, 260))
    frame[obstacle_slice] = OBSTACLE_BGR
    gt_road_mask[obstacle_slice] = 0

    pred_mask = classify_frame(frame, model)

    assert pred_mask.shape == (FRAME_HEIGHT, FRAME_WIDTH)
    assert pred_mask.dtype == np.uint8
    assert set(np.unique(pred_mask)).issubset({0, 255})

    # Road region should be (mostly) classified as road...
    assert np.mean(pred_mask[gt_road_mask == 255] == 255) > 0.9
    # ...and the obstacle region should be (mostly) classified as non-road.
    assert np.mean(pred_mask[obstacle_slice] == 0) > 0.9

    iou = _iou(pred_mask, gt_road_mask)
    assert iou > 0.9, f"expected IoU > 0.9, got {iou:.4f}"


def test_classify_frame_all_road_does_not_crash_or_divide_by_zero():
    """A frame that is 100% road color (no obstacle) should classify
    cleanly -- no exceptions, no NaN/inf from a degenerate normalization,
    and the resulting mask should be a valid, mostly-road mask."""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:, :] = ROAD_BGR

    model = build_road_color_model(frame)
    pred_mask = classify_frame(frame, model)

    assert pred_mask.shape == (FRAME_HEIGHT, FRAME_WIDTH)
    assert pred_mask.dtype == np.uint8
    assert np.all(np.isfinite(pred_mask))
    # Nearly everything should be classified as road.
    assert np.mean(pred_mask == 255) > 0.99


def test_classify_frame_empty_model_histogram_returns_all_zero_mask():
    """If the model's histogram is all-zero (e.g. built from an ROI with no
    overlap with the frame, or a tiny frame far outside the configured
    trapezoid), classify_frame must not crash/div-by-zero, and should
    return a valid all-zero (all non-road) mask."""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:, :] = ROAD_BGR

    # Directly construct a model with an all-zero histogram, the case
    # build_road_color_model documents for a degenerate/empty ROI (see
    # test_roi_model.py's out-of-bounds tests) -- classify_frame must
    # handle this shape deterministically regardless of how it arose.
    model = RoadColorModel(
        histogram=np.zeros((30, 32), dtype=np.float32),
        h_bins=30,
        s_bins=32,
        h_range=(0, 180),
        s_range=(0, 256),
    )

    pred_mask = classify_frame(frame, model)

    assert pred_mask.shape == (FRAME_HEIGHT, FRAME_WIDTH)
    assert pred_mask.dtype == np.uint8
    assert np.all(pred_mask == 0)


def test_should_rebuild_model_every_k_frames():
    assert should_rebuild_model(0, 30) is True
    assert should_rebuild_model(1, 30) is False
    assert should_rebuild_model(29, 30) is False
    assert should_rebuild_model(30, 30) is True
    assert should_rebuild_model(60, 30) is True

    # K=1 always rebuilds.
    assert all(should_rebuild_model(i, 1) for i in range(5))
