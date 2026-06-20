from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator

from playground.db.connection import Database
from playground.db.models import ModelThread
from playground.db.repos.model_repo import ModelRepo
from playground.db.repos.session_repo import SessionRepo
from playground.db.repos.thread_repo import ThreadRepo
from playground.ids import decode, encode
from playground.runtime.client import AgentRuntimeClient
from playground.sessions.fanout import fanout_chat
from playground.sessions.schemas import (
    MessageOut,
    PlaygroundDetail,
    PlaygroundListOut,
    PlaygroundOut,
    ThreadOut,
)


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

    async def create_playground(self, user_id: int, title: str) -> PlaygroundOut:
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            playground = await session_repo.create(user_id=user_id, title=title)
            return PlaygroundOut(
                id=encode(playground.id),
                title=playground.title,
                created_at=playground.created_at,
            )

    async def list_playgrounds(self, user_id: int, limit: int, offset: int) -> PlaygroundListOut:
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            sessions = await session_repo.list_by_user(user_id=user_id, limit=limit, offset=offset)
            total = await session_repo.count_by_user(user_id)
            items = [
                PlaygroundOut(id=encode(s.id), title=s.title, created_at=s.created_at)
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
                messages = [
                    MessageOut(
                        id=message.id,
                        role=message.role,
                        content=message.content,
                        latency_ms=message.latency_ms,
                        created_at=message.created_at,
                    )
                    for message in thread.messages
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
                created_at=playground.created_at,
                threads=thread_outs,
            )

    async def update_playground(self, encoded_id: str, user_id: int, title: str) -> PlaygroundOut:
        session_id = _decode_id(encoded_id, "Playground not found")
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            playground = await session_repo.update_title(session_id, user_id, title)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")
            return PlaygroundOut(
                id=encode(playground.id),
                title=playground.title,
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
        models: list[tuple[str, str]],
    ) -> AsyncGenerator[str, None]:
        session_id = _decode_id(encoded_id, "Playground not found")
        threads = await self._prepare_multi_chat(session_id, user_id, message, models)

        async def _stream() -> AsyncGenerator[str, None]:
            thread_texts: dict[int, str] = {thread.id: "" for thread in threads}
            thread_latencies: dict[int, int] = {}

            async for chunk in fanout_chat(self.runtime, threads, message):
                try:
                    data = json.loads(chunk.removeprefix("data: ").strip())
                    tid_encoded = data.get("thread_id")
                    if tid_encoded and data.get("type") == "text_delta":
                        tid = decode(tid_encoded)
                        thread_texts[tid] += data.get("delta", "")
                    if tid_encoded and data.get("type") == "thread_done":
                        tid = decode(tid_encoded)
                        thread_latencies[tid] = data.get("latency_ms", 0)
                except (json.JSONDecodeError, ValueError):
                    pass
                yield chunk

            async with self.db.session() as session:
                thread_repo = ThreadRepo(session)
                for thread in threads:
                    content = thread_texts.get(thread.id, "")
                    if content:
                        await thread_repo.add_message(
                            thread.id,
                            role="assistant",
                            content=content,
                            latency_ms=thread_latencies.get(thread.id),
                        )

        return _stream()

    async def stream_single_chat(
        self,
        encoded_id: str,
        thread_encoded_id: str,
        user_id: int,
        message: str,
    ) -> AsyncGenerator[str, None]:
        session_id = _decode_id(encoded_id, "Playground not found")
        thread_id = _decode_id(thread_encoded_id, "Thread not found")
        thread = await self._prepare_single_chat(session_id, thread_id, user_id, message)

        async def _stream() -> AsyncGenerator[str, None]:
            thread_id_enc = encode(thread_id)
            full_text = ""
            latency_ms = 0
            start = time.monotonic()
            try:
                async for event in self.runtime.chat_stream(thread.runtime_session_id, message):
                    event["thread_id"] = thread_id_enc
                    if event.get("type") == "text_delta":
                        full_text += event.get("delta", "")
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:
                error_event = {
                    "type": "error",
                    "thread_id": thread_id_enc,
                    "error": str(exc),
                }
                yield f"data: {json.dumps(error_event)}\n\n"
            latency_ms = int((time.monotonic() - start) * 1000)

            if full_text:
                async with self.db.session() as session:
                    thread_repo = ThreadRepo(session)
                    await thread_repo.add_message(
                        thread_id,
                        role="assistant",
                        content=full_text,
                        latency_ms=latency_ms,
                    )

            done_event = {
                "type": "thread_done",
                "thread_id": thread_id_enc,
                "latency_ms": latency_ms,
                "content": full_text,
            }
            yield f"data: {json.dumps(done_event)}\n\n"
            yield f"data: {json.dumps({'type': 'all_done'})}\n\n"

        return _stream()

    async def _prepare_multi_chat(
        self,
        session_id: int,
        user_id: int,
        message: str,
        models: list[tuple[str, str]],
    ) -> list[ModelThread]:
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            thread_repo = ThreadRepo(session)
            model_repo = ModelRepo(session)

            playground = await session_repo.get_if_owner(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")

            threads = []
            for provider, model_name in models:
                model = await model_repo.get_by_provider_model(provider, model_name)
                if model is None:
                    raise ModelNotFoundError(f"Model not found: {provider}/{model_name}")

                thread = await thread_repo.get_by_session_and_model(
                    session_id,
                    provider,
                    model_name,
                )
                if thread is None:
                    runtime_session_id = await self.runtime.create_session(provider, model_name)
                    thread = await thread_repo.create(
                        playground_session_id=session_id,
                        provider=provider,
                        model_name=model_name,
                        runtime_session_id=runtime_session_id,
                        model_id=model.id,
                    )
                threads.append(thread)

            for thread in threads:
                await thread_repo.add_message(thread.id, role="user", content=message)

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
