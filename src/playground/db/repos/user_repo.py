"""User repository — CRUD for auth."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from playground.db.models import User


class UserRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_user(
        self,
        email: str,
        hashed_password: str,
        display_name: str | None = None,
    ) -> User:
        user = User(email=email, hashed_password=hashed_password, display_name=display_name)
        self.session.add(user)
        await self.session.flush()
        return user

    async def get_by_email(self, email: str) -> User | None:
        result = await self.session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_by_id(self, user_id: int) -> User | None:
        return await self.session.get(User, user_id)
