"""JWT token creation and verification."""

from datetime import UTC, datetime, timedelta

from jose import JWTError, jwt

from playground.config import get_settings
from playground.ids import decode as decode_id
from playground.ids import encode as encode_id


def create_access_token(user_id: int) -> str:
    """Create a JWT access token with sqids-encoded user ID as `sub`."""
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": encode_id(user_id),
        "iat": now,
        "exp": now + timedelta(hours=settings.access_token_expire_hours),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> int | None:
    """Decode a JWT and return the integer user ID, or None on failure."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        sub: str | None = payload.get("sub")
        if sub is None:
            return None
        return decode_id(sub)
    except (JWTError, ValueError):
        return None
