"""Adaptive road/non-road pixel classification (T8).

Given a :class:`~rccar.segmentation.model.RoadColorModel` built from the
near-field ROI (T7), classify every pixel of a *full* frame as road or
non-road by looking up how likely its (Hue, Saturation) value is under the
model's histogram -- a "back-projection" of the histogram onto the frame,
conceptually the same idea as ``cv2.calcBackProject``.

Implementation notes
---------------------
Rather than calling ``cv2.calcBackProject`` directly (which has some fiddly
dtype/scale conventions -- it wants the histogram pre-scaled to ``[0, 255]``
and returns a ``CV_8U`` image), we do the equivalent lookup ourselves with
plain vectorized numpy indexing:

1. Convert the frame to HSV.
2. Map each pixel's (H, S) value to its histogram bin index (vectorized,
   no Python-level pixel loop).
3. Index directly into ``model.histogram`` to get each pixel's raw
   probability mass as a "road-likelihood" score.
4. Normalize scores by the histogram's peak bin (so the road's *most common*
   color scores 1.0, regardless of how many bins/how peaked the histogram
   is), then threshold.

This is just as fast as ``calcBackProject`` (it's vectorized numpy over the
whole frame) and keeps the normalization explicit and easy to reason about.

Threshold choice
-----------------
``model.histogram`` is normalized so all bins sum to 1, so absolute bin
values shrink as bin count grows and are not a stable basis for a fixed
threshold. Instead we normalize by the histogram's own peak value (the most
common road color always scores 1.0) and threshold that ratio. The default,
``threshold=0.05``, means "a pixel counts as road if its color is at least
5% as common in the ROI sample as the single most common road color" --
lenient enough to tolerate lighting variation/quantization noise across the
bin grid, strict enough to reject colors that are rare-to-absent in the
road sample (e.g. a differently-colored obstacle).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from rccar.segmentation.model import RoadColorModel

DEFAULT_THRESHOLD = 0.05


def classify_frame(
    frame: np.ndarray,
    model: RoadColorModel,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Classify every pixel of ``frame`` as road (255) or non-road (0).

    Parameters
    ----------
    frame:
        BGR image array of shape ``(H, W, 3)``.
    model:
        A :class:`RoadColorModel` (see :mod:`rccar.segmentation.model`),
        typically built from the near-field ROI of a recent frame.
    threshold:
        Minimum road-likelihood score, normalized to the histogram's peak
        bin (i.e. in ``[0, 1]``), for a pixel to be classified as road. See
        module docstring for the reasoning behind the default.

    Returns
    -------
    ``np.ndarray`` of shape ``(H, W)``, dtype ``uint8``, with 255 for pixels
    classified as road and 0 otherwise. If ``model.histogram`` is all-zero
    (e.g. built from an empty/out-of-frame ROI) this returns an all-zero
    mask rather than raising or dividing by zero.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected a BGR frame with shape (H, W, 3), got {frame.shape}")

    height, width = frame.shape[:2]

    hist = model.histogram
    peak = float(hist.max()) if hist.size else 0.0
    if peak <= 0.0:
        # No data in the model (e.g. empty ROI) -- nothing can be
        # classified as road. Valid, correctly-shaped, all-zero mask.
        return np.zeros((height, width), dtype=np.uint8)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float64)
    s = hsv[:, :, 1].astype(np.float64)

    h_lo, h_hi = model.h_range
    s_lo, s_hi = model.s_range

    h_bin = np.floor((h - h_lo) / (h_hi - h_lo) * model.h_bins).astype(np.int32)
    s_bin = np.floor((s - s_lo) / (s_hi - s_lo) * model.s_bins).astype(np.int32)
    np.clip(h_bin, 0, model.h_bins - 1, out=h_bin)
    np.clip(s_bin, 0, model.s_bins - 1, out=s_bin)

    likelihood = hist[h_bin, s_bin]
    normalized = likelihood / peak

    mask = np.where(normalized > threshold, np.uint8(255), np.uint8(0))
    return mask


def should_rebuild_model(frame_index: int, rebuild_interval_k: int) -> bool:
    """Should the road-color model be rebuilt at ``frame_index``?

    Expresses the "rebuild every K frames" policy: the model is expensive
    enough (histogram over the ROI) that we don't want to redo it every
    frame, so it's refreshed periodically to adapt to slowly-changing
    lighting/road conditions instead.

    Parameters
    ----------
    frame_index:
        0-based (or any monotonically increasing) frame counter.
    rebuild_interval_k:
        Rebuild every K frames. Must be a positive integer; ``K=1`` rebuilds
        every frame, ``K=30`` rebuilds every 30th frame, etc.

    Returns
    -------
    ``True`` on frames where the model should be rebuilt (including frame
    0, so a model is always built on the first frame seen).
    """
    if rebuild_interval_k <= 0:
        raise ValueError(f"rebuild_interval_k must be positive, got {rebuild_interval_k}")
    return frame_index % rebuild_interval_k == 0


@dataclass
class AdaptiveClassifier:
    """Tiny stateful wrapper tying "rebuild every K frames" to classification.

    Not required for :func:`classify_frame` to work (that function is pure
    and stateless), but convenient for a video loop: call
    :meth:`process_frame` once per frame and it takes care of deciding when
    to rebuild ``self.model`` from the current frame's ROI vs. reusing the
    last-built one.
    """

    rebuild_interval_k: int = 30
    threshold: float = DEFAULT_THRESHOLD
    model: "RoadColorModel | None" = None
    _frame_index: int = 0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Classify ``frame``, rebuilding the model every K frames."""
        from rccar.segmentation.model import build_road_color_model

        if self.model is None or should_rebuild_model(self._frame_index, self.rebuild_interval_k):
            self.model = build_road_color_model(frame)

        mask = classify_frame(frame, self.model, threshold=self.threshold)
        self._frame_index += 1
        return mask
