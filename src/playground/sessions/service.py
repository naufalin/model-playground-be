from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from playground.db.connection import Database
from playground.db.models import ModelThread
from playground.db.repos.model_repo import ModelRepo
from playground.db.repos.session_repo import SessionRepo
from playground.db.repos.thread_repo import ThreadRepo
from playground.ids import decode, encode
from playground.runtime.client import AgentRuntimeClient
from playground.sessions.chat_capture import (
    CapturedMessage,
    ChatThreadCapture,
    ChatThreadInfo,
    _request_options,
)
from playground.sessions.fanout import fanout_chat
from playground.sessions.schemas import (
    MessageOut,
    PlaygroundDetail,
    PlaygroundListOut,
    PlaygroundOut,
    ThreadOut,
)

ModelSelection = tuple[str, str, str | None]


class PlaygroundError(Exception):
    status_code = 400

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class PlaygroundNotFoundError(PlaygroundError):
    status_code = 404


class ModelNotFoundError(PlaygroundError):
    status_code = 400


def _decode_id(encoded_id: str, detail: str) -> int:
    try:
        return decode(encoded_id)
    except ValueError as exc:
        raise PlaygroundNotFoundError(detail) from exc


class PlaygroundService:
    def __init__(self, db: Database, runtime: AgentRuntimeClient) -> None:
        self.db = db
        self.runtime = runtime

    async def create_playground(
        self,
        user_id: int,
        title: str,
        tools: list[str] | None = None,
    ) -> PlaygroundOut:
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            playground = await session_repo.create(user_id=user_id, title=title, tools=tools)
            return PlaygroundOut(
                id=encode(playground.id),
                title=playground.title,
                tools=playground.tools_json,
                created_at=playground.created_at,
            )

    async def list_playgrounds(self, user_id: int, limit: int, offset: int) -> PlaygroundListOut:
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            sessions = await session_repo.list_by_user(user_id=user_id, limit=limit, offset=offset)
            total = await session_repo.count_by_user(user_id)
            items = [
                PlaygroundOut(
                    id=encode(s.id),
                    title=s.title,
                    tools=s.tools_json,
                    created_at=s.created_at,
                )
                for s in sessions
            ]
            return PlaygroundListOut(sessions=items, total=total)

    async def get_playground(self, encoded_id: str, user_id: int) -> PlaygroundDetail:
        session_id = _decode_id(encoded_id, "Playground not found")
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            thread_repo = ThreadRepo(session)
            model_repo = ModelRepo(session)

            playground = await session_repo.get_if_owner(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")

            threads = await thread_repo.get_by_session(session_id)
            thread_outs: list[ThreadOut] = []
            for thread in threads:
                model = await model_repo.get_by_provider_model(thread.provider, thread.model_name)
                display_name = model.display_name if model else thread.model_name
                # Sort messages by creation time — selectinload doesn't guarantee order
                sorted_messages = sorted(thread.messages, key=lambda m: m.created_at)
                messages = [
                    MessageOut(
                        id=message.id,
                        role=message.role,
                        content=message.content,
                        latency_ms=message.latency_ms,
                        provider=message.provider,
                        model=message.model,
                        usage=message.usage_json,
                        thinking=message.thinking_json,
                        tool_name=message.tool_name,
                        tool_call_id=message.tool_call_id,
                        tool_input=message.tool_input,
                        output_preview=message.output_preview,
                        viz_html=message.viz_html,
                        output_delta_count=message.output_delta_count,
                        request_options=message.request_options_json,
                        created_at=message.created_at,
                    )
                    for message in sorted_messages
                ]
                thread_outs.append(
                    ThreadOut(
                        id=encode(thread.id),
                        provider=thread.provider,
                        model_name=thread.model_name,
                        display_name=display_name,
                        messages=messages,
                    )
                )

            return PlaygroundDetail(
                id=encode(playground.id),
                title=playground.title,
                tools=playground.tools_json,
                created_at=playground.created_at,
                threads=thread_outs,
            )

    async def update_playground(
        self,
        encoded_id: str,
        user_id: int,
        title: str | None = None,
        tools: list[str] | None = None,
        update_tools: bool = False,
    ) -> PlaygroundOut:
        session_id = _decode_id(encoded_id, "Playground not found")
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            playground = await session_repo.get_if_owner(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")
            if title is not None:
                playground = await session_repo.update_title(session_id, user_id, title)
            if update_tools:
                playground = await session_repo.update_tools(session_id, user_id, tools)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")
            return PlaygroundOut(
                id=encode(playground.id),
                title=playground.title,
                tools=playground.tools_json,
                created_at=playground.created_at,
            )

    async def delete_playground(self, encoded_id: str, user_id: int) -> None:
        session_id = _decode_id(encoded_id, "Playground not found")
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            playground = await session_repo.get_if_owner(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")
            await session_repo.delete(session_id)

    async def stream_multi_chat(
        self,
        encoded_id: str,
        user_id: int,
        message: str,
        models: list[ModelSelection],
        tools: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        session_id = _decode_id(encoded_id, "Playground not found")
        threads = await self._prepare_multi_chat(session_id, user_id, message, models, tools)

        async def _stream() -> AsyncGenerator[str, None]:
            captures = {
                encode(thread.id): ChatThreadCapture(
                    ChatThreadInfo(
                        id=thread.id,
                        encoded_id=encode(thread.id),
                        provider=thread.provider,
                        model_name=thread.model_name,
                        request_options=_request_options(
                            thread.provider,
                            thread.model_name,
                            reasoning_effort,
                        ),
                    )
                )
                for thread, reasoning_effort in threads
            }

            async for event in fanout_chat(self.runtime, threads, message, tools):
                capture = captures.get(event.get("thread_id"))
                if capture is None or event.get("type") == "thread_start":
                    yield _sse(event)
                    continue

                observed = capture.observe(event)
                if event.get("type") == "done":
                    yield _sse(capture.thread_done_event())
                elif observed is not None:
                    yield _sse(observed)

            async with self.db.session() as session:
                thread_repo = ThreadRepo(session)
                for capture in captures.values():
                    await _persist_captured_messages(thread_repo, capture.thread.id, capture)

            yield _sse({"type": "all_done"})

        return _stream()

    async def stream_single_chat(
        self,
        encoded_id: str,
        thread_encoded_id: str,
        user_id: int,
        message: str,
        tools: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        session_id = _decode_id(encoded_id, "Playground not found")
        thread_id = _decode_id(thread_encoded_id, "Thread not found")
        thread = await self._prepare_single_chat(session_id, thread_id, user_id, message)

        async def _stream() -> AsyncGenerator[str, None]:
            thread_id_enc = encode(thread_id)
            capture = ChatThreadCapture(
                ChatThreadInfo(
                    id=thread_id,
                    encoded_id=thread_id_enc,
                    provider=thread.provider,
                    model_name=thread.model_name,
                    request_options=_request_options(thread.provider, thread.model_name, None),
                )
            )
            yield _sse(capture.start_event())
            try:
                async for event in self.runtime.chat_stream(
                    thread.runtime_session_id,
                    message,
                    provider=thread.provider,
                    model=thread.model_name,
                    tools=tools,
                ):
                    event["thread_id"] = thread_id_enc
                    observed = capture.observe(event)
                    if observed is not None:
                        yield _sse(observed)
            except Exception as exc:
                yield _sse(capture.error_event(exc))

            async with self.db.session() as session:
                thread_repo = ThreadRepo(session)
                await _persist_captured_messages(thread_repo, thread_id, capture)

            yield _sse(capture.thread_done_event())
            yield _sse({"type": "all_done"})

        return _stream()

    async def _prepare_multi_chat(
        self,
        session_id: int,
        user_id: int,
        message: str,
        models: list[ModelSelection],
        tools: list[str] | None = None,
    ) -> list[tuple[ModelThread, str | None]]:
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            thread_repo = ThreadRepo(session)
            model_repo = ModelRepo(session)

            playground = await session_repo.get_if_owner(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")

            threads = []
            for provider, model_name, reasoning_effort in models:
                model = await model_repo.get_by_provider_model(provider, model_name)
                if model is None:
                    raise ModelNotFoundError(f"Model not found: {provider}/{model_name}")

                thread = await thread_repo.get_by_session_and_model(
                    session_id,
                    provider,
                    model_name,
                )
                if thread is None:
                    runtime_session_id = await self.runtime.create_session(
                        title=f"{provider}/{model_name}",
                        tools=tools,
                    )
                    thread = await thread_repo.create(
                        playground_session_id=session_id,
                        provider=provider,
                        model_name=model_name,
                        runtime_session_id=runtime_session_id,
                        model_id=model.id,
                    )
                threads.append((thread, reasoning_effort))

            for thread, reasoning_effort in threads:
                await thread_repo.add_message(
                    thread.id,
                    role="user",
                    content=message,
                    request_options_json=_request_options(
                        thread.provider,
                        thread.model_name,
                        reasoning_effort,
                    ),
                )

            return threads

    async def _prepare_single_chat(
        self,
        session_id: int,
        thread_id: int,
        user_id: int,
        message: str,
    ) -> ModelThread:
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            thread_repo = ThreadRepo(session)

            playground = await session_repo.get_if_owner(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")

            thread = await thread_repo.get(thread_id)
            if thread is None or thread.playground_session_id != session_id:
                raise PlaygroundNotFoundError("Thread not found")

            await thread_repo.add_message(thread_id, role="user", content=message)
            return thread


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _persist_captured_messages(
    thread_repo: ThreadRepo,
    thread_id: int,
    capture: ChatThreadCapture,
) -> None:
    for message in capture.captured_messages():
        await _persist_captured_message(thread_repo, thread_id, message)


async def _persist_captured_message(
    thread_repo: ThreadRepo,
    thread_id: int,
    message: CapturedMessage,
) -> None:
    await thread_repo.add_message(
        thread_id,
        role=message.role,
        content=message.content,
        latency_ms=message.latency_ms,
        tool_name=message.tool_name,
        tool_call_id=message.tool_call_id,
        tool_input=message.tool_input,
        output_preview=message.output_preview,
        viz_html=message.viz_html,
        provider=message.provider,
        model=message.model,
        usage_json=message.usage_json,
        thinking_json=message.thinking_json,
        request_options_json=message.request_options_json,
        output_delta_count=message.output_delta_count,
    )
