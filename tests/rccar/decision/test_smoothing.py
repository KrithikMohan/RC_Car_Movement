"""Tests for T20: temporal smoothing (decision-level majority vote)."""

from rccar.decision.smoothing import MajorityVoteSmoother
from rccar.serial_client.protocol import SpeedTier


def test_flicker_sequence_smooths_to_majority_at_each_step():
    """Single-frame flicker (STOP, FULL, FULL) fed one update() at a time.

    Step 1: buffer=[STOP]                 -> only item      -> STOP
    Step 2: buffer=[STOP, FULL]            -> tie, buffer not
            yet full -> documented tie-break rule (most recently added
            among tied candidates) -> FULL
    Step 3: buffer=[STOP, FULL, FULL]      -> window full, FULL has 2/3
            votes -> genuine majority -> FULL (key assertion: the plan's
            stated expectation that this smooths to FULL by majority).
    """
    smoother = MajorityVoteSmoother(window_size=3)

    step1 = smoother.update(SpeedTier.STOP)
    assert step1 == SpeedTier.STOP

    step2 = smoother.update(SpeedTier.FULL)
    assert step2 == SpeedTier.FULL  # documented tie-break: most recent wins

    step3 = smoother.update(SpeedTier.FULL)
    assert step3 == SpeedTier.FULL  # key assertion: 2/3 majority -> FULL


def test_startup_partial_buffer_returns_only_item_without_crashing():
    """Edge case: window not yet full at startup (a single update())."""
    smoother = MajorityVoteSmoother(window_size=3)

    result = smoother.update(SpeedTier.SLOW)

    assert result == SpeedTier.SLOW
    assert len(smoother) == 1


def test_default_window_size_loaded_from_config():
    """With no explicit window_size, the default (3, from config/smoothing.yaml)
    is used."""
    smoother = MajorityVoteSmoother()
    assert smoother.window_size == 3


def test_generic_over_plain_ints_not_just_speedtier():
    """The smoother is not hardcoded to SpeedTier -- plain ints work too."""
    smoother = MajorityVoteSmoother(window_size=3)
    assert smoother.update(10) == 10
    assert smoother.update(20) == 20  # tie -> most recent wins
    assert smoother.update(20) == 20  # 2/3 majority


def test_reset_clears_buffer_back_to_startup_state():
    smoother = MajorityVoteSmoother(window_size=3)
    smoother.update(SpeedTier.STOP)
    smoother.update(SpeedTier.STOP)
    assert len(smoother) == 2

    smoother.reset()

    assert len(smoother) == 0
    # After reset, behaves like a fresh startup buffer again.
    assert smoother.update(SpeedTier.FULL) == SpeedTier.FULL
    assert len(smoother) == 1
