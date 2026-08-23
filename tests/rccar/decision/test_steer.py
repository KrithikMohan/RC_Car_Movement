"""Tests for T19: steering offset from curb distance."""

import pytest

from rccar.curb.confidence import CurbConfidenceTracker
from rccar.decision.steer import SteerController, compute_steer


class TestComputeSteerTooClose:
    """Car too close to curb -> steer away from it."""

    def test_left_curb_too_close_steers_right(self):
        # target=30cm, current=10cm -> too close to a LEFT curb -> steer
        # away -> RIGHT -> positive.
        steer = compute_steer("left", current_offset_cm=10.0, target_offset_cm=30.0, gain=2.0)
        assert steer > 0

    def test_right_curb_too_close_steers_left(self):
        # target=30cm, current=10cm -> too close to a RIGHT curb -> steer
        # away -> LEFT -> negative.
        steer = compute_steer("right", current_offset_cm=10.0, target_offset_cm=30.0, gain=2.0)
        assert steer < 0


class TestComputeSteerTooFar:
    """Car too far from curb -> steer toward it."""

    def test_left_curb_too_far_steers_left(self):
        # target=30cm, current=60cm -> too far from a LEFT curb -> steer
        # toward it -> LEFT -> negative.
        steer = compute_steer("left", current_offset_cm=60.0, target_offset_cm=30.0, gain=2.0)
        assert steer < 0

    def test_right_curb_too_far_steers_right(self):
        # target=30cm, current=60cm -> too far from a RIGHT curb -> steer
        # toward it -> RIGHT -> positive.
        steer = compute_steer("right", current_offset_cm=60.0, target_offset_cm=30.0, gain=2.0)
        assert steer > 0


class TestComputeSteerFallback:
    """No curb / fallback mode -> steer straight, ignoring stale offsets."""

    @pytest.mark.parametrize("current_offset_cm", [None, 0.0, 10.0, 60.0, -1000.0])
    def test_none_side_returns_zero(self, current_offset_cm):
        assert compute_steer(None, current_offset_cm) == 0

    @pytest.mark.parametrize("current_offset_cm", [None, 0.0, 10.0, 60.0, -1000.0])
    def test_literal_none_string_side_returns_zero(self, current_offset_cm):
        assert compute_steer("none", current_offset_cm) == 0


class TestComputeSteerClamping:
    """Extreme offset errors must clamp to [steer_min, steer_max]."""

    def test_extreme_too_close_left_clamps_to_max(self):
        steer = compute_steer(
            "left", current_offset_cm=-1000.0, target_offset_cm=30.0, gain=2.0,
            steer_min=-100, steer_max=100,
        )
        assert steer == 100

    def test_extreme_too_far_left_clamps_to_min(self):
        steer = compute_steer(
            "left", current_offset_cm=1000.0, target_offset_cm=30.0, gain=2.0,
            steer_min=-100, steer_max=100,
        )
        assert steer == -100

    def test_extreme_too_close_right_clamps_to_min(self):
        steer = compute_steer(
            "right", current_offset_cm=-1000.0, target_offset_cm=30.0, gain=2.0,
            steer_min=-100, steer_max=100,
        )
        assert steer == -100

    def test_extreme_too_far_right_clamps_to_max(self):
        steer = compute_steer(
            "right", current_offset_cm=1000.0, target_offset_cm=30.0, gain=2.0,
            steer_min=-100, steer_max=100,
        )
        assert steer == 100

    def test_clamped_value_never_exceeds_bounds(self):
        for offset in (-5000.0, -100.0, 0.0, 30.0, 100.0, 5000.0):
            for side in ("left", "right"):
                steer = compute_steer(side, offset, target_offset_cm=30.0, gain=2.0)
                assert -100 <= steer <= 100


class TestSteerControllerDoesNotChaseStaleData:
    """SteerController must ignore current_side while tracker is in fallback."""

    def test_fallback_state_forces_straight_even_with_stale_side(self):
        tracker = CurbConfidenceTracker(window_n=2, min_confidence=0.2)
        # Get it tracking a left curb first.
        tracker.update("left", 0.9)
        assert tracker.state == "tracking"
        assert tracker.current_side == "left"

        # Now drop enough consecutive frames to force fallback.
        for _ in range(tracker.window_n + 1):
            tracker.update("none", 0.0)
        assert tracker.state == "fallback"
        # Tracker itself already clears current_side, but SteerController
        # must not chase it even if that guarantee didn't hold.
        assert tracker.current_side is None

        controller = SteerController(target_offset_cm=30.0, gain=2.0)
        steer = controller.compute(tracker, current_offset_cm=10.0)
        assert steer == 0

    def test_tracking_state_computes_normal_steer(self):
        tracker = CurbConfidenceTracker(window_n=2, min_confidence=0.2)
        tracker.update("left", 0.9)
        assert tracker.state == "tracking"

        controller = SteerController(target_offset_cm=30.0, gain=2.0)
        steer = controller.compute(tracker, current_offset_cm=10.0)
        assert steer > 0
