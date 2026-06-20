"""Model thread and message repository."""

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from playground.db.connection import Database
from playground.db.models import Message, ModelThread


class ThreadRepo:
    def __init__(self, db: Database):
        self.db = db

    async def create(
        self,
        playground_session_id: int,
        provider: str,
        model_name: str,
        runtime_session_id: str,
        model_id: int | None = None,
    ) -> ModelThread:
        async with self.db.session() as s:
            thread = ModelThread(
                playground_session_id=playground_session_id,
                model_id=model_id,
                provider=provider,
                model_name=model_name,
                runtime_session_id=runtime_session_id,
            )
            s.add(thread)
            await s.flush()
            return thread

    async def get_by_session(self, playground_session_id: int) -> list[ModelThread]:
        """Get all threads for a playground session, with messages."""
        async with self.db.session() as s:
            result = await s.execute(
                select(ModelThread)
                .where(ModelThread.playground_session_id == playground_session_id)
                .options(selectinload(ModelThread.messages))
                .order_by(ModelThread.created_at)
            )
            return list(result.scalars().unique().all())

    async def get(self, thread_id: int) -> ModelThread | None:
        async with self.db.session() as s:
            result = await s.execute(
                select(ModelThread)
                .where(ModelThread.id == thread_id)
                .options(selectinload(ModelThread.messages))
            )
            return result.scalar_one_or_none()

    async def get_by_session_and_model(
        self, playground_session_id: int, provider: str, model_name: str
    ) -> ModelThread | None:
        """Find existing thread for a specific model in a session."""
        async with self.db.session() as s:
            result = await s.execute(
                select(ModelThread).where(
                    ModelThread.playground_session_id == playground_session_id,
                    ModelThread.provider == provider,
                    ModelThread.model_name == model_name,
                )
            )
            return result.scalar_one_or_none()

    async def add_message(
        self,
        thread_id: int,
        role: str,
        content: str,
        latency_ms: int | None = None,
    ) -> Message:
        async with self.db.session() as s:
            msg = Message(
                thread_id=thread_id,
                role=role,
                content=content,
                latency_ms=latency_ms,
            )
            s.add(msg)
            await s.flush()
            return msg

    async def get_messages(self, thread_id: int) -> list[Message]:
        async with self.db.session() as s:
            result = await s.execute(
                select(Message)
                .where(Message.thread_id == thread_id)
                .order_by(Message.created_at)
            )
            return list(result.scalars().all())
