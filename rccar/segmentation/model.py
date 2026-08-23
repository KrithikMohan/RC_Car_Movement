"""HSV histogram model of "road color", built from the near-field ROI.

The model is a simple, well-understood 2D color histogram over the Hue and
Saturation channels of the HSV colorspace (Value/brightness is deliberately
excluded so the model is more robust to lighting/shadow changes across the
frame). It is:

1. Convert the frame to HSV.
2. Mask to the trapezoid ROI (see ``rccar.segmentation.roi``).
3. Compute a 2D histogram over (H, S) with ``cv2.calcHist``.
4. Normalize it to a probability distribution (values sum to 1).

The returned :class:`RoadColorModel` is intentionally simple/serializable —
a normalized histogram array plus the bin counts and channel ranges used to
build it — so a later classification stage (T8) can look up how "road-like"
a given HSV pixel is by indexing into the histogram.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from rccar.segmentation.roi import Point, build_roi_mask

# OpenCV's H channel is [0, 180), S channel is [0, 256).
H_RANGE: Tuple[int, int] = (0, 180)
S_RANGE: Tuple[int, int] = (0, 256)

DEFAULT_H_BINS = 30
DEFAULT_S_BINS = 32


@dataclass
class RoadColorModel:
    """A normalized 2D (Hue, Saturation) histogram describing "road color".

    Attributes
    ----------
    histogram:
        ``np.ndarray`` of shape ``(h_bins, s_bins)``, dtype float32/float64,
        normalized so ``histogram.sum() == 1.0`` (or the array is all zeros
        if no ROI pixels were available to sample).
    h_bins, s_bins:
        Number of bins used for the Hue and Saturation channels.
    h_range, s_range:
        ``(low, high)`` value ranges used for each channel when building the
        histogram (OpenCV convention: H in [0, 180), S in [0, 256)).
    """

    histogram: np.ndarray
    h_bins: int
    s_bins: int
    h_range: Tuple[int, int]
    s_range: Tuple[int, int]

    def bin_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the (h_edges, s_edges) bin boundaries for the histogram."""
        h_edges = np.linspace(self.h_range[0], self.h_range[1], self.h_bins + 1)
        s_edges = np.linspace(self.s_range[0], self.s_range[1], self.s_bins + 1)
        return h_edges, s_edges


def build_road_color_model(
    frame: np.ndarray,
    points: Optional[Sequence[Point]] = None,
    config_path: Optional[Path] = None,
    mask: Optional[np.ndarray] = None,
    h_bins: int = DEFAULT_H_BINS,
    s_bins: int = DEFAULT_S_BINS,
) -> RoadColorModel:
    """Build a :class:`RoadColorModel` from the ROI pixels of ``frame``.

    Parameters
    ----------
    frame:
        BGR image array of shape ``(H, W, 3)`` (as produced by OpenCV/
        ``rccar.capture``).
    points, config_path:
        Optional overrides for the ROI trapezoid; see
        :func:`rccar.segmentation.roi.build_roi_mask`.
    mask:
        Precomputed ROI mask to reuse instead of rebuilding it.
    h_bins, s_bins:
        Histogram resolution for the Hue and Saturation channels.

    Returns
    -------
    A :class:`RoadColorModel`. If the ROI does not overlap the frame (e.g.
    a degenerate/out-of-bounds configuration), the histogram is all zeros
    rather than raising, so downstream code can check
    ``model.histogram.sum() == 0`` and treat it as "no data".
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected a BGR frame with shape (H, W, 3), got {frame.shape}")

    if mask is None:
        mask = build_roi_mask(frame.shape, points=points, config_path=config_path)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1],
        mask=mask,
        histSize=[h_bins, s_bins],
        ranges=[H_RANGE[0], H_RANGE[1], S_RANGE[0], S_RANGE[1]],
    )

    total = hist.sum()
    if total > 0:
        hist = hist / total
    # else: no ROI pixels sampled (e.g. ROI entirely clipped away) -- leave
    # as an all-zero, still-valid-shape histogram.

    return RoadColorModel(
        histogram=hist,
        h_bins=h_bins,
        s_bins=s_bins,
        h_range=H_RANGE,
        s_range=S_RANGE,
    )
