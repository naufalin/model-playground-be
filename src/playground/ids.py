"""Encode/decode IDs — integer internally, opaque string externally."""

from sqids import Sqids

# min_length=8 pads short encodings for consistent-looking IDs
_sqids = Sqids(min_length=8)


def encode(int_id: int) -> str:
    """Encode an integer ID to an opaque string."""
    return _sqids.encode([int_id])


def decode(encoded: str) -> int:
    """Decode an opaque string back to an integer ID.

    Raises ValueError if the string is not a valid encoded ID.
    """
    result = _sqids.decode(encoded)
    if not result:
        raise ValueError(f"Invalid ID: {encoded}")
    return result[0]
