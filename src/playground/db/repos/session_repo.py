"""Playground session repository — CRUD scoped to user."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from playground.db.models import PlaygroundSession


class SessionRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        user_id: int,
        title: str = "New Playground",
        mode: str = "compare",
        tools: list[str] | None = None,
        skills: list[str] | None = None,
        orchestration: dict | None = None,
        system_prompt_name: str | None = None,
        system_prompt_content: str | None = None,
    ) -> PlaygroundSession:
        sess = PlaygroundSession(
            user_id=user_id,
            title=title,
            mode=mode,
            tools_json=tools,
            skills_json=skills,
            orchestration_json=orchestration,
            system_prompt_name=system_prompt_name,
            system_prompt_content=system_prompt_content,
        )
        self.session.add(sess)
        await self.session.flush()
        return sess

    async def update_system_prompt(
        self,
        session_id: int,
        user_id: int,
        name: str,
        content: str,
    ) -> PlaygroundSession | None:
        sess = await self.get_if_owner(session_id, user_id)
        if sess is None:
            return None
        sess.system_prompt_name = name
        sess.system_prompt_content = content
        await self.session.flush()
        return sess

    async def list_by_user(
        self, user_id: int, limit: int = 20, offset: int = 0
    ) -> list[PlaygroundSession]:
        result = await self.session.execute(
            select(PlaygroundSession)
            .where(PlaygroundSession.user_id == user_id)
            .order_by(PlaygroundSession.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def count_by_user(self, user_id: int) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(PlaygroundSession)
            .where(PlaygroundSession.user_id == user_id)
        )
        return result.scalar_one()

    async def get(self, session_id: int) -> PlaygroundSession | None:
        return await self.session.get(PlaygroundSession, session_id)

    async def get_if_owner(self, session_id: int, user_id: int) -> PlaygroundSession | None:
        """Return session only if it belongs to the user."""
        result = await self.session.execute(
            select(PlaygroundSession).where(
                PlaygroundSession.id == session_id,
                PlaygroundSession.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_if_owner_for_update(
        self, session_id: int, user_id: int
    ) -> PlaygroundSession | None:
        result = await self.session.execute(
            select(PlaygroundSession)
            .where(
                PlaygroundSession.id == session_id,
                PlaygroundSession.user_id == user_id,
            )
            .with_for_update()
        )
        return result.scalar_one_or_none()

    async def update_title(
        self,
        session_id: int,
        user_id: int,
        title: str,
    ) -> PlaygroundSession | None:
        result = await self.session.execute(
            select(PlaygroundSession).where(
                PlaygroundSession.id == session_id,
                PlaygroundSession.user_id == user_id,
            )
        )
        sess = result.scalar_one_or_none()
        if sess is None:
            return None
        sess.title = title
        await self.session.flush()
        return sess

    async def update_tools(
        self,
        session_id: int,
        user_id: int,
        tools: list[str] | None,
    ) -> PlaygroundSession | None:
        result = await self.session.execute(
            select(PlaygroundSession).where(
                PlaygroundSession.id == session_id,
                PlaygroundSession.user_id == user_id,
            )
        )
        sess = result.scalar_one_or_none()
        if sess is None:
            return None
        sess.tools_json = tools
        await self.session.flush()
        return sess

    async def update_skills(
        self,
        session_id: int,
        user_id: int,
        skills: list[str] | None,
    ) -> PlaygroundSession | None:
        sess = await self.get_if_owner(session_id, user_id)
        if sess is None:
            return None
        sess.skills_json = skills
        await self.session.flush()
        return sess

    async def update_orchestration(self, session_id: int, user_id: int, orchestration: dict | None):
        sess = await self.get_if_owner(session_id, user_id)
        if sess is None:
            return None
        sess.orchestration_json = orchestration
        await self.session.flush()
        return sess

    async def delete(self, session_id: int) -> None:
        sess = await self.session.get(PlaygroundSession, session_id)
        if sess:
            await self.session.delete(sess)
