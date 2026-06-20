"""Playground session repository — CRUD scoped to user."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from playground.db.connection import Database
from playground.db.models import PlaygroundSession


class SessionRepo:
    def __init__(self, db: Database | AsyncSession):
        self.db = db

    @asynccontextmanager
    async def _session(self) -> AsyncGenerator[AsyncSession]:
        if isinstance(self.db, AsyncSession):
            yield self.db
        else:
            async with self.db.session() as session:
                yield session

    async def create(self, user_id: int, title: str = "New Playground") -> PlaygroundSession:
        async with self._session() as s:
            sess = PlaygroundSession(user_id=user_id, title=title)
            s.add(sess)
            await s.flush()
            return sess

    async def list_by_user(
        self, user_id: int, limit: int = 20, offset: int = 0
    ) -> list[PlaygroundSession]:
        async with self._session() as s:
            result = await s.execute(
                select(PlaygroundSession)
                .where(PlaygroundSession.user_id == user_id)
                .order_by(PlaygroundSession.updated_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

    async def count_by_user(self, user_id: int) -> int:
        async with self._session() as s:
            result = await s.execute(
                select(func.count()).select_from(PlaygroundSession).where(
                    PlaygroundSession.user_id == user_id
                )
            )
            return result.scalar_one()

    async def get(self, session_id: int) -> PlaygroundSession | None:
        async with self._session() as s:
            return await s.get(PlaygroundSession, session_id)

    async def get_if_owner(self, session_id: int, user_id: int) -> PlaygroundSession | None:
        """Return session only if it belongs to the user."""
        async with self._session() as s:
            result = await s.execute(
                select(PlaygroundSession).where(
                    PlaygroundSession.id == session_id,
                    PlaygroundSession.user_id == user_id,
                )
            )
            return result.scalar_one_or_none()

    async def update_title(
        self,
        session_id: int,
        user_id: int,
        title: str,
    ) -> PlaygroundSession | None:
        async with self._session() as s:
            result = await s.execute(
                select(PlaygroundSession).where(
                    PlaygroundSession.id == session_id,
                    PlaygroundSession.user_id == user_id,
                )
            )
            sess = result.scalar_one_or_none()
            if sess is None:
                return None
            sess.title = title
            await s.flush()
            return sess

    async def delete(self, session_id: int) -> None:
        async with self._session() as s:
            sess = await s.get(PlaygroundSession, session_id)
            if sess:
                await s.delete(sess)
