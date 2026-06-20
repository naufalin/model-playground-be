import pytest
from pydantic import ValidationError

from playground.app import app
from playground.auth.passwords import hash_password, verify_password
from playground.auth.schemas import LoginRequest, SignupRequest


def test_hash_password_uses_argon2_and_verifies() -> None:
    hashed = hash_password("correct horse battery staple")

    assert hashed.startswith("$argon2")
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong password", hashed)


def test_signup_rejects_short_password() -> None:
    with pytest.raises(ValidationError):
        SignupRequest(email="user@example.com", password="short")


def test_signup_rejects_password_longer_than_128_characters() -> None:
    with pytest.raises(ValidationError):
        SignupRequest(email="user@example.com", password="x" * 129)


def test_login_rejects_password_longer_than_128_characters() -> None:
    with pytest.raises(ValidationError):
        LoginRequest(email="user@example.com", password="x" * 129)


def test_openapi_uses_oauth_token_endpoint() -> None:
    scheme = app.openapi()["components"]["securitySchemes"]["OAuth2PasswordBearer"]

    assert scheme["flows"]["password"]["tokenUrl"] == "/auth/token"
