"""Temporal smoothing of per-frame decision outputs via majority vote (T20).

Per-frame decision functions (:func:`rccar.decision.speed.decide_speed_tier`,
:func:`rccar.decision.steer.compute_steer`) can flicker frame to frame even
when nothing about the world has meaningfully changed -- a single noisy
detection can flip ``SpeedTier.FULL`` to ``SpeedTier.STOP`` and back within
a couple of frames. :class:`MajorityVoteSmoother` absorbs that jitter by
keeping a small ring buffer of the last ``window_size`` decision outputs and
returning the majority value over that buffer on every update.

This is explicitly **not** object tracking -- it has no notion of what is
being tracked, only a sliding window of already-decided output values (e.g.
``SpeedTier`` members, or steer ints). It is intentionally generic over any
hashable value type so the same class can smooth speed tiers, steer values,
or anything else with a well-defined "majority" notion.
"""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import Any, Hashable, Optional

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "smoothing.yaml"

_DEFAULT_WINDOW_SIZE = 3


def load_window_size(config_path: Optional[str] = None) -> int:
    """Best-effort load of ``window_size`` from ``config/smoothing.yaml``.

    Reads from ``config_path`` if given, otherwise from the default
    ``config/smoothing.yaml`` (resolved relative to the repo root). Falls
    back to :data:`_DEFAULT_WINDOW_SIZE` if the file is missing or
    malformed, so callers remain usable even outside a full checkout --
    mirrors the pattern used by ``rccar.curb.confidence._load_defaults``
    and ``rccar.decision.speed.load_thresholds``.
    """
    path = Path(config_path) if config_path is not None else _CONFIG_PATH
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return int(cfg.get("window_size", _DEFAULT_WINDOW_SIZE))
    except (OSError, ValueError, yaml.YAMLError):
        return _DEFAULT_WINDOW_SIZE


class MajorityVoteSmoother:
    """Smooths a stream of decision-output values via a sliding majority vote.

    Not object tracking: this class holds only a small ``collections.deque``
    ring buffer of the last ``window_size`` values handed to :meth:`update`
    -- it has no model of the underlying scene. Works for any hashable
    value type (``SpeedTier`` members, plain ints for steer, strings, ...);
    the value type is never hardcoded here so the same smoother instance
    type can be reused for speed, steer, or anything else.

    Parameters
    ----------
    window_size:
        Number of most-recent values to vote over. Defaults to the value
        in ``config/smoothing.yaml`` (``window_size``) if not given --
        3 by default (see that file for why an odd window is chosen).
    """

    def __init__(self, window_size: Optional[int] = None):
        self.window_size: int = (
            window_size if window_size is not None else load_window_size()
        )
        self._buffer: deque = deque(maxlen=self.window_size)

    def update(self, value: Hashable) -> Any:
        """Push ``value`` into the ring buffer and return the majority vote.

        The majority is computed over whatever is CURRENTLY in the buffer
        *after* ``value`` has been added -- i.e. this frame's value is
        always included in the vote it participates in.

        Startup behavior (buffer not yet full)
        ---------------------------------------
        Before ``window_size`` frames have been seen, the vote is simply
        taken over however many values are in the buffer so far (1 or 2
        for the default ``window_size=3``) -- this method never waits for
        a full window and never raises. With only one value seen, that
        value trivially "wins" the vote.

        Tie-break rule
        --------------
        When two or more distinct values in the buffer share the same
        (maximum) vote count -- which can only happen while the buffer is
        still filling up for an odd ``window_size`` with at most two
        distinct live values, but is handled generally regardless of
        window size or value cardinality -- the tie is broken in favor of
        the MOST RECENTLY ADDED value among the tied candidates. This is
        implemented by scanning the buffer from newest to oldest and
        returning the first value whose count equals the max count: for a
        genuine (non-tied) majority this is just that majority value; for
        a tie it is the most recent of the tied values. This keeps the
        smoother responsive to the newest evidence during tie situations,
        rather than arbitrarily favoring whichever value happened to
        appear first.
        """
        self._buffer.append(value)
        counts = Counter(self._buffer)
        max_count = max(counts.values())
        for candidate in reversed(self._buffer):
            if counts[candidate] == max_count:
                return candidate
        # Unreachable: self._buffer is never empty here since we just
        # appended a value.
        raise AssertionError("unreachable: buffer is non-empty after append")

    def reset(self) -> None:
        """Clear the ring buffer, returning the smoother to startup state."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
