"""User repository — CRUD for auth."""

from sqlalchemy import select

from playground.db.connection import Database
from playground.db.models import User


class UserRepo:
    def __init__(self, db: Database):
        self.db = db

    async def create_user(
        self,
        email: str,
        hashed_password: str,
        display_name: str | None = None,
    ) -> User:
        async with self.db.session() as s:
            user = User(email=email, hashed_password=hashed_password, display_name=display_name)
            s.add(user)
            await s.flush()
            return user

    async def get_by_email(self, email: str) -> User | None:
        async with self.db.session() as s:
            result = await s.execute(select(User).where(User.email == email))
            return result.scalar_one_or_none()

    async def get_by_id(self, user_id: int) -> User | None:
        async with self.db.session() as s:
            return await s.get(User, user_id)
