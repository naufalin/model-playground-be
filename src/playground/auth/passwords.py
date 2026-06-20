"""Password hashing helpers."""

from functools import lru_cache

from pwdlib import PasswordHash


@lru_cache
def _password_hash() -> PasswordHash:
    return PasswordHash.recommended()


def hash_password(password: str) -> str:
    return _password_hash().hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return _password_hash().verify(password, hashed_password)
