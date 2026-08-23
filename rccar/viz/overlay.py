"""Debug overlay rendering (mask, curb line, decision) for video/window output.

T26 -- a debug/visualization tool that draws, on top of each raw camera
frame, the three pieces of state that are otherwise invisible when the
pipeline runs headless: the road/non-road segmentation mask, the curb line
used to steer, and the resulting speed/steer decision.

Sourcing mask/curb_line data
-----------------------------
``rccar.main.process_frame`` already returns everything needed to *drive*
the car (speed, steer, curb_state, curb_side, ...), but deliberately does
not return the road mask or the raw curb line segment -- those are
perception-internal artifacts, not decision outputs, and the T25 pipeline
has no reason to carry them around.

Rather than modify ``process_frame``'s return contract (risking every other
caller/test of it), this module calls the same lower-level pieces
``process_frame`` itself calls, alongside it, for exactly the two extra
artifacts it doesn't expose:

- The road mask: produced by ``state.classifier.process_frame(frame)``,
  which ``rccar.main.process_frame`` already calls as its first internal
  step. Since ``AdaptiveClassifier`` is stateful (it only rebuilds its
  road-color model every K frames, see
  ``rccar.segmentation.classify.should_rebuild_model``), calling it a
  *second* time per frame would double-advance its internal frame counter
  and desync the rebuild schedule from the main pipeline. Instead of doing
  that, ``run_viz`` wraps ``state.classifier.process_frame`` once (for the
  duration of the run) to capture the mask it returns on its single,
  pipeline-driven call -- see ``run_viz`` for the exact mechanism.
- The curb line segment: re-derived via ``rccar.curb.hough_stub.detect_lines``
  + the same near-horizontal filtering / side selection / longest-line-wins
  heuristic that ``rccar.curb.detect.detect_curb_side`` and
  ``rccar.main.estimate_curb_offset_cm`` already use internally, picking the
  single best line on the *tracked* curb side (from the result dict's
  ``curb_side``) to draw. This mirrors ``estimate_curb_offset_cm``'s own
  line-selection logic so the drawn line is the same one driving the steer
  decision.
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import cv2
import numpy as np

from rccar.capture.source import FrameSource
from rccar.curb.detect import (
    HORIZONTAL_ANGLE_THRESHOLD_DEG,
    _is_near_horizontal,
    _line_length,
    _line_midpoint_x,
)
from rccar.curb.hough_stub import Line, detect_lines
from rccar.serial_client.protocol import SpeedTier

# BGR colors.
_MASK_TINT_COLOR = (0, 200, 0)  # green tint over road pixels
_MASK_ALPHA = 0.35
_CURB_LINE_COLOR = (0, 255, 255)  # yellow
_CURB_LINE_THICKNESS = 2

_SPEED_COLORS = {
    SpeedTier.FULL: (0, 200, 0),  # green
    SpeedTier.SLOW: (0, 200, 255),  # yellow/orange
    SpeedTier.STOP: (0, 0, 255),  # red
}

_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_SCALE = 0.6
_TEXT_THICKNESS = 2
_TEXT_ORIGIN = (10, 24)
_TEXT_LINE_SPACING = 26


def draw_overlay(
    frame: np.ndarray,
    mask: Optional[np.ndarray],
    curb_line: Optional[Tuple[int, int, int, int]],
    speed: SpeedTier,
    steer: int,
    curb_state: str,
) -> np.ndarray:
    """Return a NEW frame with the debug overlay drawn on top of ``frame``.

    Parameters
    ----------
    frame:
        BGR frame, shape ``(H, W, 3)``. Not mutated.
    mask:
        Optional road/non-road mask (uint8, 255=road/0=non-road), shape
        ``(H, W)``. If given, blended as a semi-transparent green tint over
        road pixels via ``cv2.addWeighted``. Skipped cleanly if ``None``.
    curb_line:
        Optional ``(x1, y1, x2, y2)`` pixel line segment. Drawn as a yellow
        line via ``cv2.line`` if given. Skipped cleanly if ``None``.
    speed:
        Current speed tier decision; rendered as color-coded text
        (green=FULL, yellow=SLOW, red=STOP).
    steer:
        Current steer decision (percent deflection, negative=left).
    curb_state:
        ``"tracking"`` or ``"fallback"``, rendered as text.

    Returns
    -------
    A new ``np.ndarray`` (the input ``frame`` is never modified in place).
    """
    out = frame.copy()

    if mask is not None:
        tint = np.zeros_like(out)
        tint[:, :] = _MASK_TINT_COLOR
        road_bool = mask.astype(bool)
        blended = cv2.addWeighted(out, 1.0 - _MASK_ALPHA, tint, _MASK_ALPHA, 0.0)
        out[road_bool] = blended[road_bool]

    if curb_line is not None:
        x1, y1, x2, y2 = curb_line
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), _CURB_LINE_COLOR, _CURB_LINE_THICKNESS)

    speed_color = _SPEED_COLORS.get(speed, (255, 255, 255))
    speed_name = speed.name if isinstance(speed, SpeedTier) else str(speed)

    lines = [
        (f"speed: {speed_name}", speed_color),
        (f"steer: {steer}", (255, 255, 255)),
        (f"curb: {curb_state}", (255, 255, 255)),
    ]
    for i, (text, color) in enumerate(lines):
        origin = (_TEXT_ORIGIN[0], _TEXT_ORIGIN[1] + i * _TEXT_LINE_SPACING)
        cv2.putText(out, text, origin, _TEXT_FONT, _TEXT_SCALE, color, _TEXT_THICKNESS, cv2.LINE_AA)

    return out


def _best_curb_line(frame: np.ndarray, curb_side: Optional[str]) -> Optional[Line]:
    """Re-derive the single curb line segment driving the current decision.

    Mirrors ``rccar.main.estimate_curb_offset_cm``'s line-selection logic
    (same filtering + "longest line on the tracked side wins" heuristic) so
    the line drawn here is the same one the steer decision is based on.
    Returns ``None`` if ``curb_side`` isn't ``"left"``/``"right"`` or no
    line survives filtering on that side.
    """
    if curb_side not in ("left", "right"):
        return None

    lines = detect_lines(frame)
    candidates = [
        line for line in lines if not _is_near_horizontal(line, HORIZONTAL_ANGLE_THRESHOLD_DEG)
    ]
    if not candidates:
        return None

    width = frame.shape[1]
    center_x = width / 2.0
    if curb_side == "left":
        side_candidates = [c for c in candidates if _line_midpoint_x(c) < center_x]
    else:
        side_candidates = [c for c in candidates if _line_midpoint_x(c) >= center_x]

    if not side_candidates:
        return None

    return max(side_candidates, key=_line_length)


def run_viz(
    source: FrameSource,
    output_path: Optional[str] = None,
    show_window: bool = True,
    homography: Optional[np.ndarray] = None,
    max_frames: Optional[int] = None,
) -> None:
    """Run the perception/decision pipeline over ``source``, rendering an
    overlay (mask + curb line + decision text) for each frame.

    For each frame: builds the road mask and runs the full decision
    pipeline via ``rccar.main.PipelineState``/``process_frame`` (reused
    as-is, unmodified), re-derives the curb line segment via
    :func:`_best_curb_line` (see module docstring for why this is a
    separate, stateless re-derivation rather than a second call into
    ``process_frame``'s internals), draws the overlay via
    :func:`draw_overlay`, then writes it to ``output_path`` (if given) via
    ``cv2.VideoWriter`` and/or shows it via ``cv2.imshow`` (if
    ``show_window`` is True). If ``show_window`` is False, ``cv2.imshow``
    is never called -- this function runs cleanly headless (no GUI/display
    required).

    ``homography``: if not given, ``config/homography.yaml`` (relative to
    the repo root) is loaded via ``rccar.calibration.homography_api.load_homography``.

    Road mask sourcing: the mask displayed is produced by an
    ``AdaptiveClassifier`` owned by this function's ``PipelineState`` (the
    same classifier the pipeline's ``process_frame`` calls internally to
    produce the mask it uses for the decision) -- ``process_frame`` calls
    ``state.classifier.process_frame(frame)`` exactly once per frame as its
    first step, so the mask used for both the decision and the overlay
    is one and the same; no double-classification occurs.
    """
    from rccar.calibration.homography_api import load_homography
    from rccar.curb.confidence import CurbConfidenceTracker
    from rccar.decision.smoothing import MajorityVoteSmoother
    from rccar.main import PipelineState, process_frame
    from rccar.segmentation.classify import AdaptiveClassifier

    if homography is None:
        homography = load_homography("config/homography.yaml")

    state = PipelineState(
        classifier=AdaptiveClassifier(),
        curb_tracker=CurbConfidenceTracker(),
        homography=homography,
        speed_smoother=MajorityVoteSmoother(),
        steer_smoother=MajorityVoteSmoother(),
    )

    writer: Optional[cv2.VideoWriter] = None
    frame_count = 0

    # Wrap the classifier's process_frame once (not per-frame) so the mask
    # it produces -- process_frame's first internal step -- is captured for
    # the overlay without ever calling the classifier a second time (which
    # would double-advance its "rebuild every K frames" counter and desync
    # it from the main pipeline; see module docstring).
    mask_holder: dict = {}
    _orig_classify = state.classifier.process_frame

    def _capturing_classify(f, _orig=_orig_classify, _holder=mask_holder):
        m = _orig(f)
        _holder["mask"] = m
        return m

    state.classifier.process_frame = _capturing_classify  # type: ignore[method-assign]

    try:
        while max_frames is None or frame_count < max_frames:
            frame = source.read()
            if frame is None:
                break

            result = process_frame(frame, state)
            mask = mask_holder.get("mask")
            curb_line = _best_curb_line(frame, result.get("curb_side"))

            overlaid = draw_overlay(
                frame,
                mask,
                curb_line,
                result["speed"],
                result["steer"],
                result["curb_state"],
            )

            if output_path is not None:
                if writer is None:
                    height, width = overlaid.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"run_viz: failed to open VideoWriter for output path {output_path!r}"
                        )
                writer.write(overlaid)

            if show_window:
                cv2.imshow("rccar viz", overlaid)
                cv2.waitKey(1)

            frame_count += 1
    finally:
        state.classifier.process_frame = _orig_classify  # type: ignore[method-assign]
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()


class VizRunner:
    """Thin object wrapper around :func:`run_viz` for callers that prefer a
    class-based interface (e.g. to hold config across multiple runs)."""

    def __init__(
        self,
        output_path: Optional[str] = None,
        show_window: bool = True,
        homography: Optional[np.ndarray] = None,
        max_frames: Optional[int] = None,
    ):
        self.output_path = output_path
        self.show_window = show_window
        self.homography = homography
        self.max_frames = max_frames

    def run(self, source: FrameSource) -> None:
        run_viz(
            source,
            output_path=self.output_path,
            show_window=self.show_window,
            homography=self.homography,
            max_frames=self.max_frames,
        )


def build_frame_source(args: argparse.Namespace) -> FrameSource:
    """Construct the FrameSource selected by the parsed CLI args.

    Mirrors ``rccar.main.build_frame_source``.
    """
    from rccar.capture.file import VideoFileSource
    from rccar.capture.live import LiveCameraSource

    if args.source_file is not None:
        return VideoFileSource(args.source_file)
    return LiveCameraSource(args.device_index)
