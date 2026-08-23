"""Speed-tier decision from nearest-obstacle distance (T18).

Maps a nearest-obstacle real-world distance (cm), as produced by
``rccar.obstacles.distance.nearest_obstacle_real_distance``, to a
:class:`~rccar.serial_client.protocol.SpeedTier` the car should drive at,
using two tunable thresholds loaded from ``config/thresholds.yaml`` rather
than hardcoded magic numbers.

Boundary behavior (both comparisons are strict ``<``, see
``decide_speed_tier`` below):

* At exactly ``distance_cm == stop_distance_cm``: NOT STOP. ``stop_distance_cm``
  is not ``< stop_distance_cm``, so the value falls through to the SLOW
  check -> ``SpeedTier.SLOW`` wins at this boundary.
* At exactly ``distance_cm == slow_distance_cm``: NOT SLOW. ``slow_distance_cm``
  is not ``< slow_distance_cm``, so the value falls through to the default
  -> ``SpeedTier.FULL`` wins at this boundary.

In other words, each threshold belongs to the *safer/faster* side that is
adjacent to it going outward: the stop boundary itself is already outside
"emergency stop" territory (SLOW), and the slow boundary itself is already
outside "be careful" territory (FULL).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from rccar.serial_client.protocol import SpeedTier

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "thresholds.yaml"

_DEFAULT_STOP_DISTANCE_CM = 30.0
_DEFAULT_SLOW_DISTANCE_CM = 100.0


def load_thresholds(config_path: Optional[str] = None) -> tuple:
    """Best-effort load of ``(stop_distance_cm, slow_distance_cm)``.

    Reads from ``config_path`` if given, otherwise from the default
    ``config/thresholds.yaml`` (resolved relative to the repo root). Falls
    back to the hardcoded defaults above -- matching the plan's defaults --
    if the file is missing or malformed, so callers remain usable even
    outside a full checkout (mirrors the pattern used by
    ``rccar.curb.confidence._load_defaults``).
    """
    path = Path(config_path) if config_path is not None else _CONFIG_PATH
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return (
            float(cfg.get("stop_distance_cm", _DEFAULT_STOP_DISTANCE_CM)),
            float(cfg.get("slow_distance_cm", _DEFAULT_SLOW_DISTANCE_CM)),
        )
    except (OSError, ValueError, yaml.YAMLError):
        return _DEFAULT_STOP_DISTANCE_CM, _DEFAULT_SLOW_DISTANCE_CM


def decide_speed_tier(
    distance_cm: Optional[float],
    stop_distance_cm: float = _DEFAULT_STOP_DISTANCE_CM,
    slow_distance_cm: float = _DEFAULT_SLOW_DISTANCE_CM,
) -> SpeedTier:
    """Map a nearest-obstacle distance (cm) to a :class:`SpeedTier`.

    Rules:
    * ``distance_cm is None`` (no obstacle detected in range) ->
      ``SpeedTier.FULL``.
    * ``distance_cm < stop_distance_cm`` -> ``SpeedTier.STOP``.
    * ``distance_cm < slow_distance_cm`` -> ``SpeedTier.SLOW``.
    * otherwise -> ``SpeedTier.FULL``.

    Boundary sides (both comparisons are strict ``<``):

    * Exactly at ``stop_distance_cm`` -> **not** STOP (30 is not < 30) ->
      falls through to the SLOW check -> ``SpeedTier.SLOW``.
    * Exactly at ``slow_distance_cm`` -> **not** SLOW (100 is not < 100) ->
      falls through to the default -> ``SpeedTier.FULL``.

    ``stop_distance_cm`` and ``slow_distance_cm`` default to the same
    hardcoded values as :data:`_DEFAULT_STOP_DISTANCE_CM` /
    :data:`_DEFAULT_SLOW_DISTANCE_CM` (which also match
    ``config/thresholds.yaml``), but callers that want the config-driven
    values should obtain them via :func:`load_thresholds` and pass them in
    explicitly -- this function itself is a pure function with no I/O.
    """
    if distance_cm is None:
        return SpeedTier.FULL
    if distance_cm < stop_distance_cm:
        return SpeedTier.STOP
    if distance_cm < slow_distance_cm:
        return SpeedTier.SLOW
    return SpeedTier.FULL
