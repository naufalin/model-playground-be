"""Auth router — signup, login, me."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from playground.db.models import User
from playground.db.repos.user_repo import UserRepo
from playground.deps import get_user_repo
from playground.ids import encode as encode_id

from .deps import get_current_user
from .jwt import create_access_token
from .passwords import hash_password, verify_password
from .schemas import LoginRequest, SignupRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


def _user_to_out(user: User) -> UserOut:
    return UserOut(
        id=encode_id(user.id),
        email=user.email,
        display_name=user.display_name,
    )


@router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def signup(
    body: SignupRequest,
    user_repo: UserRepo = Depends(get_user_repo),
) -> UserOut:
    """Register a new user."""
    existing = await user_repo.get_by_email(body.email)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = await user_repo.create_user(
        email=body.email,
        hashed_password=hash_password(body.password),
        display_name=body.display_name,
    )
    return _user_to_out(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    user_repo: UserRepo = Depends(get_user_repo),
) -> TokenResponse:
    """Authenticate and return a JWT."""
    return await _login_with_credentials(body.email, body.password, user_repo)


@router.post("/token", response_model=TokenResponse)
async def token(
    form: OAuth2PasswordRequestForm = Depends(),
    user_repo: UserRepo = Depends(get_user_repo),
) -> TokenResponse:
    """OAuth2-compatible token endpoint for Swagger UI."""
    return await _login_with_credentials(form.username, form.password, user_repo)


async def _login_with_credentials(
    email: str,
    password: str,
    user_repo: UserRepo,
) -> TokenResponse:
    user = await user_repo.get_by_email(email)
    if user is None or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)) -> UserOut:
    """Return the authenticated user's profile."""
    return _user_to_out(current_user)
