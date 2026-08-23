"""Curb confidence tracker: smooths ``detect_curb_side`` output over time.

``detect_curb_side`` (see :mod:`rccar.curb.detect`) is a per-frame
detector -- it can flicker frame to frame (a momentary occlusion, a shadow,
a bad Hough vote) even when a real curb is present. :class:`CurbConfidenceTracker`
turns that noisy per-frame signal into a stable, hysteresis-based state the
rest of the driving stack can act on: either "tracking" a curb side, or
"fallback" to lane-center driving when the curb has genuinely been lost.

State-machine rules (implemented exactly, see module tests)
-------------------------------------------------------------
* A frame counts as **found** iff ``side != "none"`` and
  ``confidence >= min_confidence``.
* The tracker keeps a running count of **consecutive** not-found frames.
* Starting in (or already in) ``"tracking"``: the tracker only flips to
  ``"fallback"`` once the consecutive not-found count reaches
  ``window_n + 1``. Fewer consecutive misses than that -- e.g. a single
  dropped frame -- do *not* cause a fallback; ``curb_available`` stays
  ``True`` and ``current_side`` keeps its last-known value throughout.
* Reappearance rule (chosen for simplicity, see task docstring): once in
  ``"fallback"``, the tracker flips back to ``"tracking"`` on the very
  next **found** frame -- there is no separate "confirm it's really back"
  delay. This is deliberately the simplest of the reasonable options; a
  system that wants more hysteresis on the way back in can wrap this
  tracker or require several found updates before acting on
  ``curb_available``.
* Side changes while already ``"tracking"`` get the same ``window_n + 1``
  grace window as a disappearance does: a **found** frame reporting a
  *different* side than the one currently tracked does not immediately
  switch ``current_side`` -- it must be seen ``window_n + 1`` consecutive
  times first. Without this, a single stray misdetection (e.g. a Hough
  line briefly picked from something other than the real curb) instantly
  flips ``current_side``, which flips the steering direction the very
  same frame -- steering "away" from a wrongly-reported side drives
  straight into the real curb on the other side of the frame instead of
  away from it. A momentary flicker keeps the previously tracked side
  until the new one is confirmed; a genuine side change (car actually
  crossing to the other curb) still gets picked up, just not on frame one.
* Before any frames have been seen, the tracker starts in ``"fallback"``
  with no side (nothing has been found yet, so there is nothing to track).
"""

from collections import deque
from pathlib import Path
from typing import Optional

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "curb.yaml"

_DEFAULT_WINDOW_N = 5
_DEFAULT_MIN_CONFIDENCE = 0.2


def _load_defaults() -> tuple:
    """Best-effort load of (window_n, min_confidence) from config/curb.yaml.

    Falls back to the hardcoded defaults above if the file is missing or
    malformed, so the tracker remains usable even outside a full checkout.
    """
    try:
        with open(_CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return (
            int(cfg.get("confidence_window_n", _DEFAULT_WINDOW_N)),
            float(cfg.get("min_confidence_threshold", _DEFAULT_MIN_CONFIDENCE)),
        )
    except (OSError, ValueError, yaml.YAMLError):
        return _DEFAULT_WINDOW_N, _DEFAULT_MIN_CONFIDENCE


class CurbConfidenceTracker:
    """Stateful tracker turning per-frame curb detections into a stable state.

    Parameters
    ----------
    window_n:
        Number of frames of "grace" before a run of not-found frames is
        treated as a real loss of the curb. The tracker flips to
        ``"fallback"`` only after ``window_n + 1`` CONSECUTIVE not-found
        frames. Defaults to the value in ``config/curb.yaml``
        (``confidence_window_n``) if not given.
    min_confidence:
        Minimum per-frame confidence (from ``detect_curb_side``) for a
        frame to count as "found". Defaults to the value in
        ``config/curb.yaml`` (``min_confidence_threshold``) if not given.
    """

    def __init__(self, window_n: Optional[int] = None, min_confidence: Optional[float] = None):
        default_window_n, default_min_confidence = _load_defaults()
        self.window_n: int = window_n if window_n is not None else default_window_n
        self.min_confidence: float = (
            min_confidence if min_confidence is not None else default_min_confidence
        )

        # Ring buffer of the last `window_n` (found: bool, side: Optional[str])
        # observations, most-recent last. Retained for observability/future
        # use; the fallback/tracking decision itself is driven by the
        # consecutive-miss counter below.
        self._history: deque = deque(maxlen=self.window_n)

        self._consecutive_missing: int = 0
        self._state: str = "fallback"
        self._current_side: Optional[str] = None
        # Consecutive-count of a candidate side differing from
        # `_current_side` while already tracking -- see the module
        # docstring's "Side changes while already tracking" rule.
        self._pending_side: Optional[str] = None
        self._pending_count: int = 0

    def update(self, side: str, confidence: float) -> None:
        """Feed one frame's ``detect_curb_side`` output into the tracker."""
        found = side != "none" and confidence >= self.min_confidence
        self._history.append((found, side if found else None))

        if found:
            self._consecutive_missing = 0
            if self._state != "tracking" or self._current_side is None or side == self._current_side:
                self._current_side = side
                self._pending_side = None
                self._pending_count = 0
            else:
                # Candidate side disagrees with the one currently tracked:
                # require window_n + 1 consecutive frames on the new side
                # before actually switching (see module docstring).
                if side == self._pending_side:
                    self._pending_count += 1
                else:
                    self._pending_side = side
                    self._pending_count = 1
                if self._pending_count >= self.window_n + 1:
                    self._current_side = side
                    self._pending_side = None
                    self._pending_count = 0
            # Reappearance rule: the very next found frame flips fallback
            # back to tracking (see module docstring).
            self._state = "tracking"
        else:
            self._consecutive_missing += 1
            self._pending_side = None
            self._pending_count = 0
            if self._state == "tracking" and self._consecutive_missing >= self.window_n + 1:
                self._state = "fallback"
                self._current_side = None

    @property
    def curb_available(self) -> bool:
        """True if the curb is currently considered available (tracking)."""
        return self._state == "tracking"

    @property
    def current_side(self) -> Optional[str]:
        """``"left"``/``"right"`` while tracking, else ``None``."""
        return self._current_side if self.curb_available else None

    @property
    def state(self) -> str:
        """``"tracking"`` or ``"fallback"`` (fallback = lane-center mode)."""
        return self._state
