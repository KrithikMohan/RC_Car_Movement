"""Serial wire protocol encode/decode for the RC-car Pi <-> MCU link.

Implements the protocol defined in `docs/serial_protocol.md` (v1):

    V,<version>\\n                  version header, sent once on connect
    S,<speed>,<steer>\\n            drive command, sent every decision tick

    speed  ::= "0" | "1" | "2"          0=STOP  1=SLOW  2=FULL
    steer  ::= integer in [-100, 100]   % of max deflection, negative=left

Encoding is strict (no leading zeros/plus signs/decimals/whitespace) and
decoding is strict: any malformed input raises `ProtocolError` rather than
silently returning a wrong/partial value.
"""

from __future__ import annotations

from enum import IntEnum

PROTOCOL_VERSION = 1

STEER_MIN = -100
STEER_MAX = 100


class ProtocolError(Exception):
    """Raised when decoding a wire message that does not conform to spec."""

    pass


class SpeedTier(IntEnum):
    STOP = 0
    SLOW = 1
    FULL = 2


def encode_command(speed: "SpeedTier | int", steer: int) -> bytes:
    """Encode a drive command as wire bytes: b"S,<speed>,<steer>\\n".

    Validates `speed` is one of {0, 1, 2} and `steer` is in [-100, 100].
    Raises `ValueError` if either value is out of range.
    """
    speed_int = int(speed)
    if speed_int not in (SpeedTier.STOP, SpeedTier.SLOW, SpeedTier.FULL):
        raise ValueError(
            f"speed must be one of 0 (STOP), 1 (SLOW), 2 (FULL); got {speed_int!r}"
        )
    if not isinstance(steer, int) or isinstance(steer, bool):
        raise ValueError(f"steer must be an int; got {steer!r}")
    if not (STEER_MIN <= steer <= STEER_MAX):
        raise ValueError(
            f"steer must be in [{STEER_MIN}, {STEER_MAX}]; got {steer!r}"
        )
    return f"S,{speed_int},{steer}\n".encode("ascii")


def encode_version() -> bytes:
    """Encode the version header: b"V,<version>\\n"."""
    return f"V,{PROTOCOL_VERSION}\n".encode("ascii")


def _parse_strict_int(field: str, name: str) -> int:
    """Parse `field` as a strict integer per spec.

    Rejects leading '+', leading zeros (other than a bare "0"), decimals,
    embedded whitespace, and any other non-canonical encoding.
    """
    if field == "":
        raise ProtocolError(f"{name} field is empty")

    body = field
    negative = False
    if body.startswith("-"):
        negative = True
        body = body[1:]

    if body == "":
        raise ProtocolError(f"{name} field {field!r} has no digits")

    if not body.isdigit():
        raise ProtocolError(f"{name} field {field!r} is not a valid integer")

    # isdigit() can be True for non-ASCII digit characters (e.g. superscripts,
    # full-width digits); ensure strictly ASCII 0-9.
    if not all(c in "0123456789" for c in body):
        raise ProtocolError(f"{name} field {field!r} is not a valid integer")

    if len(body) > 1 and body[0] == "0":
        raise ProtocolError(f"{name} field {field!r} has a leading zero")

    if negative and body == "0":
        raise ProtocolError(f"{name} field {field!r} has invalid negative zero")

    value = int(body)
    return -value if negative else value


def decode_command(line: "bytes | str") -> tuple[int, int]:
    """Strictly decode an `S,<speed>,<steer>\\n` wire message.

    Returns (speed, steer) as plain ints. Raises `ProtocolError` on any
    malformed input: wrong field count, non-numeric fields, missing/wrong
    prefix, missing trailing newline, malformed numeric encoding
    (leading zeros/plus/decimals/whitespace), or out-of-range values.
    """
    if isinstance(line, bytes):
        try:
            text = line.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ProtocolError(f"line is not valid ASCII: {line!r}") from exc
    elif isinstance(line, str):
        text = line
    else:
        raise ProtocolError(f"line must be bytes or str; got {type(line)!r}")

    if not text.endswith("\n"):
        raise ProtocolError(f"missing trailing newline terminator: {text!r}")

    body = text[:-1]

    if "\r" in body:
        raise ProtocolError(f"unexpected CR byte in message: {text!r}")

    fields = body.split(",")

    if len(fields) == 0 or fields[0] != "S":
        raise ProtocolError(f"missing or unrecognized 'S' tag: {text!r}")

    if len(fields) != 3:
        raise ProtocolError(
            f"expected exactly 2 fields after 'S' tag (speed, steer), "
            f"got {len(fields) - 1}: {text!r}"
        )

    speed_field, steer_field = fields[1], fields[2]

    speed = _parse_strict_int(speed_field, "speed")
    steer = _parse_strict_int(steer_field, "steer")

    if speed not in (SpeedTier.STOP, SpeedTier.SLOW, SpeedTier.FULL):
        raise ProtocolError(f"speed value {speed} is not one of 0, 1, 2: {text!r}")

    if not (STEER_MIN <= steer <= STEER_MAX):
        raise ProtocolError(
            f"steer value {steer} is outside [{STEER_MIN}, {STEER_MAX}]: {text!r}"
        )

    return speed, steer
