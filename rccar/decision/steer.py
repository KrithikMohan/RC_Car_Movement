"""Steering-offset computation from curb distance (T19).

Given the side of the curb the car is currently tracking (``"left"`` or
``"right"``) and the car's current lateral offset from that curb, compute a
``steer`` value in the serial protocol's ``-100..100`` range
(see ``docs/serial_protocol.md`` section 3.2: negative = steer left,
positive = steer right) that nudges the car toward holding a configured
``target_offset_cm`` distance from the curb.

Sign convention (verified against the protocol and encoded in
``compute_steer`` below):

* ``curb_side == "left"``: the curb is to the car's left.
    - Too close to it (``current_offset_cm < target_offset_cm``) -> steer
      AWAY from the curb -> steer RIGHT -> positive.
    - Too far from it (``current_offset_cm > target_offset_cm``) -> steer
      TOWARD the curb -> steer LEFT -> negative.
    - Formula: ``error = current_offset_cm - target_offset_cm``;
      ``steer = clamp(round(-error * gain), steer_min, steer_max)``, i.e.
      steer = clamp(round((target_offset_cm - current_offset_cm) * gain)).
      Note the sign is negated relative to a naive ``error * gain`` --
      that naive formula would steer the wrong way for the left-curb case,
      so it is flipped here to match the spec above.

* ``curb_side == "right"``: the curb is to the car's right (mirror image).
    - Too close to it (``current_offset_cm < target_offset_cm``) -> steer
      AWAY from the curb -> steer LEFT -> negative.
    - Too far from it (``current_offset_cm > target_offset_cm``) -> steer
      TOWARD the curb -> steer RIGHT -> positive.
    - Formula: ``steer = clamp(round(error * gain), steer_min, steer_max)``
      where ``error = current_offset_cm - target_offset_cm`` (no sign flip
      needed here -- it already matches the spec above).

* ``curb_side`` is ``None``/``"none"`` (fallback / no curb tracked): steer
  straight (``0``), regardless of any (necessarily stale) offset value --
  this is the "don't chase stale curb data" requirement.
"""

from typing import Optional

_DEFAULT_TARGET_OFFSET_CM = 30.0
_DEFAULT_GAIN = 2.0
_DEFAULT_STEER_MIN = -100
_DEFAULT_STEER_MAX = 100


def _clamp(value: int, steer_min: int, steer_max: int) -> int:
    return max(steer_min, min(steer_max, value))


def compute_steer(
    curb_side: Optional[str],
    current_offset_cm: Optional[float],
    target_offset_cm: float = _DEFAULT_TARGET_OFFSET_CM,
    gain: float = _DEFAULT_GAIN,
    steer_min: int = _DEFAULT_STEER_MIN,
    steer_max: int = _DEFAULT_STEER_MAX,
) -> int:
    """Compute a steer value (protocol range ``[steer_min, steer_max]``).

    Parameters
    ----------
    curb_side:
        ``"left"`` or ``"right"`` -- which side the tracked curb is on.
        ``None`` (or ``"none"``) means fallback / no curb tracked: the
        function returns ``0`` (straight) unconditionally in that case,
        ignoring ``current_offset_cm`` even if it happens to be a
        left-over/stale value from before the curb was lost.
    current_offset_cm:
        Current measured lateral offset (cm) from the curb line. Ignored
        when ``curb_side`` indicates fallback.
    target_offset_cm:
        Desired lateral offset (cm) to hold from the curb.
    gain:
        Proportional gain (steer units per cm of error), applied before
        clamping.
    steer_min, steer_max:
        Clamp bounds, matching the serial protocol's ``steer`` field range.

    Returns
    -------
    int
        Steer value clamped to ``[steer_min, steer_max]``.
    """
    if curb_side is None or curb_side == "none":
        return 0

    if current_offset_cm is None:
        # No usable measurement even though a side is reported -- nothing
        # sensible to correct toward, so hold straight.
        return 0

    error = current_offset_cm - target_offset_cm

    if curb_side == "left":
        # Too close (error < 0) -> steer right (positive) -> negate.
        raw = -error * gain
    elif curb_side == "right":
        # Too close (error < 0) -> steer left (negative) -> no negation.
        raw = error * gain
    else:
        raise ValueError(f"curb_side must be 'left', 'right', 'none', or None, got {curb_side!r}")

    return _clamp(round(raw), steer_min, steer_max)


class SteerController:
    """Thin stateful wrapper around :func:`compute_steer`.

    Reads a ``CurbConfidenceTracker``-like object's ``.state`` and
    ``.current_side`` and calls :func:`compute_steer` with the right
    ``curb_side``. Critically, it forces ``curb_side=None`` (straight,
    steer=0) whenever ``tracker.state == "fallback"``, even if
    ``tracker.current_side`` happens to be a stale non-None value -- this
    is what prevents the controller from chasing stale curb data. In
    practice ``CurbConfidenceTracker.current_side`` already returns
    ``None`` while in fallback (see ``rccar.curb.confidence``), so this
    check is a defensive belt-and-suspenders guard for any tracker-like
    object that doesn't make that same guarantee.
    """

    def __init__(
        self,
        target_offset_cm: float = _DEFAULT_TARGET_OFFSET_CM,
        gain: float = _DEFAULT_GAIN,
        steer_min: int = _DEFAULT_STEER_MIN,
        steer_max: int = _DEFAULT_STEER_MAX,
    ):
        self.target_offset_cm = target_offset_cm
        self.gain = gain
        self.steer_min = steer_min
        self.steer_max = steer_max

    def compute(self, tracker, current_offset_cm: Optional[float]) -> int:
        """Compute a steer value from a tracker's current state.

        ``tracker`` need only expose ``.state`` (``"tracking"``/``"fallback"``)
        and ``.current_side`` (``Optional[str]``), matching
        ``CurbConfidenceTracker``.
        """
        curb_side = tracker.current_side if tracker.state == "tracking" else None
        return compute_steer(
            curb_side,
            current_offset_cm,
            target_offset_cm=self.target_offset_cm,
            gain=self.gain,
            steer_min=self.steer_min,
            steer_max=self.steer_max,
        )
