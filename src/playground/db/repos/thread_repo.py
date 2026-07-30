"""Model thread and message repository."""

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from playground.db.models import Message, ModelThread


class ThreadRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        playground_session_id: int,
        provider: str,
        model_name: str,
        runtime_session_id: str,
        model_id: int | None = None,
    ) -> ModelThread:
        thread = ModelThread(
            playground_session_id=playground_session_id,
            model_id=model_id,
            provider=provider,
            model_name=model_name,
            runtime_session_id=runtime_session_id,
        )
        self.session.add(thread)
        await self.session.flush()
        return thread

    async def get_by_session(self, playground_session_id: int) -> list[ModelThread]:
        """Get all threads for a playground session, with messages ordered by creation."""
        result = await self.session.execute(
            select(ModelThread)
            .where(ModelThread.playground_session_id == playground_session_id)
            .options(selectinload(ModelThread.messages))
            .order_by(ModelThread.created_at)
        )
        return list(result.scalars().unique().all())

    async def get(self, thread_id: int) -> ModelThread | None:
        result = await self.session.execute(
            select(ModelThread)
            .where(ModelThread.id == thread_id)
            .options(selectinload(ModelThread.messages))
        )
        return result.scalar_one_or_none()

    async def get_for_update(self, thread_id: int) -> ModelThread | None:
        """Load a thread under a row lock before replacing its conversation tail."""
        result = await self.session.execute(
            select(ModelThread)
            .where(ModelThread.id == thread_id)
            .options(selectinload(ModelThread.messages))
            .with_for_update()
        )
        return result.scalar_one_or_none()

    async def get_by_session_and_model(
        self, playground_session_id: int, provider: str, model_name: str
    ) -> ModelThread | None:
        """Find existing thread for a specific model in a session."""
        result = await self.session.execute(
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
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_input: dict | None = None,
        output_preview: str | None = None,
        viz_html: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        usage_json: dict | None = None,
        thinking_json: dict | None = None,
        request_options_json: dict | None = None,
        output_delta_count: int | None = None,
        selected_skill: str | None = None,
        turn_id: str | None = None,
        transcript_sequence: int | None = None,
    ) -> Message:
        msg = Message(
            thread_id=thread_id,
            role=role,
            content=content,
            latency_ms=latency_ms,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_input=tool_input,
            output_preview=output_preview,
            viz_html=viz_html,
            provider=provider,
            model=model,
            usage_json=usage_json,
            thinking_json=thinking_json,
            request_options_json=request_options_json,
            output_delta_count=output_delta_count,
            selected_skill=selected_skill,
            turn_id=turn_id,
            transcript_sequence=transcript_sequence,
        )
        self.session.add(msg)
        await self.session.flush()
        return msg

    async def replace_tail_with_user_message(
        self,
        thread: ModelThread,
        target_message_id: int,
        runtime_session_id: str,
        content: str,
        request_options_json: dict[str, str],
    ) -> ModelThread:
        """Point a thread at a fork and replace the target user turn onward."""
        await self.session.execute(
            delete(Message).where(
                Message.thread_id == thread.id,
                Message.id >= target_message_id,
            )
        )
        thread.runtime_session_id = runtime_session_id
        await self.add_message(
            thread.id,
            role="user",
            content=content,
            request_options_json=request_options_json,
        )
        return thread

    async def get_messages(self, thread_id: int) -> list[Message]:
        result = await self.session.execute(
            select(Message)
            .where(Message.thread_id == thread_id)
            .order_by(Message.created_at, Message.id)
        )
        return list(result.scalars().all())
