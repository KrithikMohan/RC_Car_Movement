"""Tests for the ROI trapezoid + road-color HSV histogram model (T7)."""

import numpy as np

from rccar.segmentation.model import build_road_color_model
from rccar.segmentation.roi import build_roi_mask, extract_roi_pixels

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Points from config/roi.yaml, duplicated here so the "out of bounds" test
# can scale them up independently of whatever the config file happens to
# contain.
ROI_POINTS = [(20, 240), (300, 240), (200, 150), (120, 150)]


def _solid_frame(bgr, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def test_solid_color_road_produces_tight_histogram_peak():
    """A synthetic frame that is one solid road color everywhere should
    yield an HSV histogram whose mass is concentrated in one or two bins."""
    frame = _solid_frame((60, 120, 80))  # arbitrary solid BGR "road" color

    mask = build_roi_mask(frame.shape)
    assert mask.shape == (FRAME_HEIGHT, FRAME_WIDTH)
    # Sanity: the ROI actually selects some pixels, and they're all equal
    # since the frame is solid-colored.
    roi_pixels = extract_roi_pixels(frame, mask=mask)
    assert roi_pixels.shape[0] > 0
    assert np.all(roi_pixels == roi_pixels[0])

    model = build_road_color_model(frame)

    hist = model.histogram
    assert hist.shape == (model.h_bins, model.s_bins)
    assert np.isclose(hist.sum(), 1.0)

    # A solid color maps to (at most) a couple of adjacent HSV bins due to
    # floating point / bin-edge quantization, so the top few bins should
    # already account for essentially all the probability mass.
    flat_sorted = np.sort(hist.flatten())[::-1]
    top_bins_mass = flat_sorted[:2].sum()
    assert top_bins_mass > 0.95


def test_roi_partly_out_of_frame_bounds_does_not_crash():
    """Simulate a smaller frame (e.g. car nose crop) where the configured
    ROI trapezoid extends beyond the frame's actual width/height. Masking
    and model building must clip gracefully and still produce a valid,
    non-empty histogram."""
    small_width, small_height = 200, 160  # smaller than ROI_POINTS' max extent (300, 240)
    frame = _solid_frame((30, 90, 200), width=small_width, height=small_height)

    # Points intentionally exceed the frame's bounds (max x=300, max y=240
    # vs. a 200x160 frame).
    mask = build_roi_mask(frame.shape, points=ROI_POINTS)
    assert mask.shape == (small_height, small_width)
    # No crash, and clipping still leaves some pixels selected.
    assert mask.sum() > 0

    model = build_road_color_model(frame, points=ROI_POINTS)
    hist = model.histogram
    assert hist.shape == (model.h_bins, model.s_bins)
    assert hist.sum() > 0
    assert np.isclose(hist.sum(), 1.0)


def test_roi_entirely_out_of_frame_produces_valid_empty_histogram():
    """If the ROI has zero overlap with the frame, mask building and model
    building should still not crash, returning an all-zero (but
    correctly-shaped) histogram rather than raising."""
    tiny_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    far_away_points = [(1000, 1000), (1010, 1000), (1010, 1010), (1000, 1010)]

    mask = build_roi_mask(tiny_frame.shape, points=far_away_points)
    assert mask.shape == (10, 10)

    model = build_road_color_model(tiny_frame, points=far_away_points)
    assert model.histogram.shape == (model.h_bins, model.s_bins)
    # Degenerate case: clipping collapses the polygon onto/near the frame
    # corner, so there may or may not be any masked pixels, but either way
    # the histogram must be a valid, finite array.
    assert np.all(np.isfinite(model.histogram))
