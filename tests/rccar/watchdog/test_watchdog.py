"""Tests for the watchdog (T24): stale frame / stall / serial failure -> forced STOP.

Uses a fully mocked, injectable clock (a simple counter under test control,
never real time.sleep) and a MagicMock serial client, per the plan's T24
test cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rccar.serial_client.client import SerialClientError
from rccar.serial_client.protocol import SpeedTier, decode_command, encode_command
from rccar.watchdog.watchdog import Watchdog


class FakeClock:
    """A simple controllable clock: starts at 0.0, advance() moves it forward."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


def make_watchdog(frame_timeout_ms: float = 500.0):
    serial_client = MagicMock()
    clock = FakeClock()
    wd = Watchdog(serial_client, frame_timeout_ms=frame_timeout_ms, clock=clock)
    return wd, serial_client, clock


def test_happy_normal_frame_cadence_no_trip():
    """Frames arrive steadily with gaps well under frame_timeout_ms -> no trip."""
    wd, serial_client, clock = make_watchdog(frame_timeout_ms=500.0)

    wd.on_frame_received()
    for _ in range(10):
        clock.advance(0.05)  # 50ms, well under 500ms timeout
        assert wd.check_frame_staleness() is False
        wd.on_frame_received()

    serial_client.write.assert_not_called()
    assert wd.last_stop_send_failed is False


def test_edge_frame_gap_600ms_trips_and_sends_stop():
    """A 600ms gap since the last frame trips staleness and sends STOP."""
    wd, serial_client, clock = make_watchdog(frame_timeout_ms=500.0)

    wd.on_frame_received()
    clock.advance(0.6)  # 600ms > 500ms timeout

    tripped = wd.check_frame_staleness()

    assert tripped is True
    serial_client.write.assert_called_once()
    sent_bytes = serial_client.write.call_args[0][0]
    speed, steer = decode_command(sent_bytes)
    assert speed == SpeedTier.STOP
    assert sent_bytes == encode_command(SpeedTier.STOP, 0)


def test_before_any_frame_received_not_stale():
    """Before on_frame_received() has ever been called, staleness must not
    false-trigger."""
    wd, serial_client, _clock = make_watchdog()
    assert wd.check_frame_staleness() is False
    serial_client.write.assert_not_called()


def test_check_processing_stall_trips_and_sends_stop():
    wd, serial_client, clock = make_watchdog()
    loop_start = clock()
    clock.advance(1.0)  # 1000ms

    tripped = wd.check_processing_stall(loop_start, max_stall_ms=300.0)

    assert tripped is True
    serial_client.write.assert_called_once()
    speed, _steer = decode_command(serial_client.write.call_args[0][0])
    assert speed == SpeedTier.STOP


def test_check_processing_stall_does_not_trip_when_within_budget():
    wd, serial_client, clock = make_watchdog()
    loop_start = clock()
    clock.advance(0.1)  # 100ms

    tripped = wd.check_processing_stall(loop_start, max_stall_ms=300.0)

    assert tripped is False
    serial_client.write.assert_not_called()


def test_error_serial_write_raises_inside_write_command_does_not_crash():
    """write_command catches SerialClientError, attempts a last-ditch STOP,
    does not propagate, and returns False."""
    wd, serial_client, _clock = make_watchdog()
    serial_client.write.side_effect = SerialClientError("port gone")

    result = wd.write_command(SpeedTier.SLOW, 10)

    assert result is False
    # send_stop's own attempt also went through serial_client.write (and
    # also raised, since the mock always raises) -- watchdog must not
    # propagate that either.
    assert wd.last_stop_send_failed is True
    assert isinstance(wd.last_stop_send_exception, SerialClientError)
    assert serial_client.write.call_count == 2  # original attempt + send_stop attempt


def test_error_send_stop_itself_fails_does_not_crash():
    """Even if the last-ditch send_stop() call also fails, send_stop must
    not raise -- it should report failure via its return value/state."""
    serial_client = MagicMock()
    serial_client.write.side_effect = SerialClientError("still gone")
    clock = FakeClock()
    wd = Watchdog(serial_client, frame_timeout_ms=500.0, clock=clock)

    result = wd.send_stop()

    assert result is False
    assert wd.last_stop_send_failed is True
    assert isinstance(wd.last_stop_send_exception, SerialClientError)


def test_send_stop_success_resets_failure_state():
    wd, serial_client, _clock = make_watchdog()

    result = wd.send_stop()

    assert result is True
    serial_client.write.assert_called_once_with(encode_command(SpeedTier.STOP, 0))
    assert wd.last_stop_send_failed is False
    assert wd.last_stop_send_exception is None
