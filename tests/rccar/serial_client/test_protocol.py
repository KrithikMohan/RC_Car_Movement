import pytest

from rccar.serial_client.protocol import (
    ProtocolError,
    SpeedTier,
    decode_command,
    encode_command,
    encode_version,
)


class TestEncodeVersion:
    def test_encode_version(self):
        assert encode_version() == b"V,1\n"


class TestRoundTrip:
    @pytest.mark.parametrize(
        "speed,steer",
        [
            (SpeedTier.STOP, -100),
            (SpeedTier.SLOW, 0),
            (SpeedTier.FULL, 100),
            (SpeedTier.FULL, 5),
            (SpeedTier.STOP, 0),
        ],
    )
    def test_round_trip(self, speed, steer):
        wire = encode_command(speed, steer)
        decoded_speed, decoded_steer = decode_command(wire)
        assert decoded_speed == int(speed)
        assert decoded_steer == steer

    def test_encode_command_wire_format(self):
        assert encode_command(SpeedTier.FULL, -15) == b"S,2,-15\n"
        assert encode_command(SpeedTier.SLOW, 10) == b"S,1,10\n"
        assert encode_command(SpeedTier.STOP, 0) == b"S,0,0\n"


class TestSteerRangeEdges:
    def test_steer_min(self):
        wire = encode_command(SpeedTier.FULL, -100)
        assert wire == b"S,2,-100\n"
        assert decode_command(wire) == (2, -100)

    def test_steer_max(self):
        wire = encode_command(SpeedTier.FULL, 100)
        assert wire == b"S,2,100\n"
        assert decode_command(wire) == (2, 100)


class TestEncodeValidation:
    def test_invalid_speed_raises_value_error(self):
        with pytest.raises(ValueError):
            encode_command(3, 0)

    def test_steer_out_of_range_raises_value_error(self):
        with pytest.raises(ValueError):
            encode_command(SpeedTier.FULL, 101)
        with pytest.raises(ValueError):
            encode_command(SpeedTier.FULL, -101)


class TestDecodeMalformed:
    @pytest.mark.parametrize(
        "line",
        [
            b"S,2\n",  # wrong field count (missing steer)
            b"S,2,-15,0\n",  # wrong field count (extra field)
            b"S,abc,5\n",  # non-numeric speed
            b"S,2,left\n",  # non-numeric steer
            b"S,2,5",  # missing trailing newline
            b"S,2,150\n",  # steer out of range
            b"S,3,0\n",  # invalid speed value
            b"S,5,0\n",  # invalid speed value
            b"S,02,5\n",  # leading zero (speed)
            b"S,2,05\n",  # leading zero (steer)
            b"S,+2,05\n",  # leading plus
            b"S,2,+5\n",  # leading plus on steer
            b"S,2,5.0\n",  # decimal
            b"S, 2,5\n",  # embedded whitespace
            b"X,1,2\n",  # unrecognized tag
            b"S,2,-101\n",  # out of range negative steer
        ],
    )
    def test_raises_protocol_error(self, line):
        with pytest.raises(ProtocolError):
            decode_command(line)

    def test_raises_protocol_error_not_bare_exception_type(self):
        # ProtocolError specifically, not a generic ValueError/Exception subclass
        # that isn't ProtocolError.
        try:
            decode_command(b"S,abc,5\n")
        except ProtocolError:
            pass
        except Exception:
            pytest.fail("decode_command raised a non-ProtocolError exception")
        else:
            pytest.fail("decode_command did not raise")

    def test_decode_accepts_str_too(self):
        assert decode_command("S,2,-15\n") == (2, -15)

    def test_valid_stop_message(self):
        assert decode_command(b"S,0,0\n") == (0, 0)
