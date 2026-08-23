"""Tests for T18: speed-tier decision from nearest-obstacle distance."""

import pytest

from rccar.decision.speed import decide_speed_tier
from rccar.serial_client.protocol import SpeedTier


@pytest.mark.parametrize(
    "distance_cm, expected",
    [
        # Clear interior points.
        (200.0, SpeedTier.FULL),  # far away -> full speed
        (50.0, SpeedTier.SLOW),  # mid-range -> slow down
        (10.0, SpeedTier.STOP),  # very close -> stop
        # No obstacle detected at all.
        (None, SpeedTier.FULL),
        # Boundary at stop_distance_cm=30.0: exactly at the boundary is
        # NOT < 30, so it falls through to SLOW (documented in
        # decide_speed_tier's docstring).
        (30.0, SpeedTier.SLOW),
        # Boundary at slow_distance_cm=100.0: exactly at the boundary is
        # NOT < 100, so it falls through to FULL.
        (100.0, SpeedTier.FULL),
        # Interior points just inside each boundary.
        (29.9, SpeedTier.STOP),
        (99.9, SpeedTier.SLOW),
    ],
)
def test_decide_speed_tier(distance_cm, expected):
    assert decide_speed_tier(distance_cm) == expected


def test_decide_speed_tier_custom_thresholds():
    # With custom thresholds, the same boundary rules apply relative to
    # the passed-in values rather than the defaults.
    assert decide_speed_tier(20.0, stop_distance_cm=25.0, slow_distance_cm=50.0) == SpeedTier.STOP
    assert decide_speed_tier(25.0, stop_distance_cm=25.0, slow_distance_cm=50.0) == SpeedTier.SLOW
    assert decide_speed_tier(50.0, stop_distance_cm=25.0, slow_distance_cm=50.0) == SpeedTier.FULL
