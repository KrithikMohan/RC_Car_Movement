from rccar.curb.confidence import CurbConfidenceTracker


def test_curb_visible_every_frame_stays_tracking():
    tracker = CurbConfidenceTracker(window_n=5, min_confidence=0.2)

    for _ in range(10):
        tracker.update("left", 0.8)
        assert tracker.curb_available is True
        assert tracker.state == "tracking"
        assert tracker.current_side == "left"


def test_curb_disappearing_flips_to_fallback_exactly_at_n_plus_1():
    window_n = 3
    tracker = CurbConfidenceTracker(window_n=window_n, min_confidence=0.2)

    # Establish tracking first.
    tracker.update("left", 0.8)
    tracker.update("left", 0.8)
    assert tracker.state == "tracking"
    assert tracker.curb_available is True

    # Curb disappears for window_n + 1 = 4 consecutive frames. Should stay
    # in "tracking" for the first `window_n` (3) missing frames, and only
    # flip to "fallback" on the 4th consecutive missing frame.
    for i in range(1, window_n + 1):
        tracker.update("none", 0.0)
        assert tracker.state == "tracking", f"flipped to fallback too early at miss #{i}"
        assert tracker.curb_available is True

    # The (window_n + 1)-th consecutive missing frame flips it.
    tracker.update("none", 0.0)
    assert tracker.state == "fallback"
    assert tracker.curb_available is False
    assert tracker.current_side is None


def test_curb_reappearing_after_fallback_flips_back_to_tracking():
    window_n = 3
    tracker = CurbConfidenceTracker(window_n=window_n, min_confidence=0.2)

    # Drive it into fallback.
    tracker.update("right", 0.9)
    for _ in range(window_n + 1):
        tracker.update("none", 0.0)
    assert tracker.state == "fallback"
    assert tracker.curb_available is False
    assert tracker.current_side is None

    # Reappearance rule: flips back to "tracking" on the very next found
    # frame.
    tracker.update("right", 0.7)
    assert tracker.state == "tracking"
    assert tracker.curb_available is True
    assert tracker.current_side == "right"

    # Stays tracking on subsequent found frames.
    tracker.update("right", 0.6)
    assert tracker.state == "tracking"
    assert tracker.curb_available is True
    assert tracker.current_side == "right"


def test_initial_state_before_any_update_is_fallback():
    tracker = CurbConfidenceTracker(window_n=5, min_confidence=0.2)

    assert tracker.state == "fallback"
    assert tracker.curb_available is False
    assert tracker.current_side is None


def test_defaults_loaded_when_not_specified():
    tracker = CurbConfidenceTracker()

    assert tracker.window_n == 5
    assert tracker.min_confidence == 0.2
