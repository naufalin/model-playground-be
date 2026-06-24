from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

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
        models: list[ModelSelection],
    ) -> AsyncGenerator[str, None]:
        session_id = _decode_id(encoded_id, "Playground not found")
        threads = await self._prepare_multi_chat(session_id, user_id, message, models)

        async def _stream() -> AsyncGenerator[str, None]:
            thread_texts: dict[int, str] = {thread.id: "" for thread, _ in threads}
            thread_done: dict[int, dict[str, Any]] = {}
            timeline_events: dict[int, list[dict[str, Any]]] = {
                thread.id: [] for thread, _ in threads
            }
            request_options = {
                thread.id: _request_options(thread.provider, thread.model_name, reasoning_effort)
                for thread, reasoning_effort in threads
            }

            async for chunk in fanout_chat(self.runtime, threads, message):
                try:
                    data = json.loads(chunk.removeprefix("data: ").strip())
                    tid_encoded = data.get("thread_id")
                    if tid_encoded and data.get("type") == "text_delta":
                        tid = decode(tid_encoded)
                        thread_texts[tid] += data.get("delta", "")
                    if tid_encoded and data.get("type") == "thread_done":
                        tid = decode(tid_encoded)
                        thread_done[tid] = data
                    if tid_encoded and data.get("type") in {
                        "thinking_delta",
                        "tool_start",
                        "tool_end",
                    }:
                        tid = decode(tid_encoded)
                        _append_timeline_event(timeline_events.setdefault(tid, []), data)
                except (json.JSONDecodeError, ValueError):
                    pass
                yield chunk

            async with self.db.session() as session:
                thread_repo = ThreadRepo(session)
                for thread, _reasoning_effort in threads:
                    done = thread_done.get(thread.id, {})
                    timeline = timeline_events.get(thread.id, [])
                    has_thinking = any(event.get("type") == "thinking" for event in timeline)
                    if not has_thinking and done.get("thinking"):
                        thinking_text = _thinking_text(
                            done.get("thinking"),
                            done.get("provider") or thread.provider,
                        )
                        if thinking_text:
                            timeline.append(
                                {
                                    "type": "thinking",
                                    "kind": _thinking_kind(done.get("provider") or thread.provider),
                                    "content": thinking_text,
                                    "thinking": done.get("thinking"),
                                }
                            )

                    for event in timeline:
                        await _persist_timeline_event(thread_repo, thread.id, event)

                    content = thread_texts.get(thread.id, "")
                    content = done.get("content") or content
                    if content:
                        await thread_repo.add_message(
                            thread.id,
                            role="assistant",
                            content=content,
                            latency_ms=done.get("latency_ms"),
                            provider=done.get("provider"),
                            model=done.get("model"),
                            usage_json=done.get("usage"),
                            thinking_json=done.get("thinking"),
                            request_options_json=request_options.get(thread.id),
                            output_delta_count=done.get("output_delta_count"),
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
            start_event = {
                "type": "thread_start",
                "thread_id": thread_id_enc,
                "provider": thread.provider,
                "model": thread.model_name,
            }
            yield f"data: {json.dumps(start_event)}\n\n"
            start = time.monotonic()
            first_token_ms: int | None = None
            done_event: dict[str, Any] | None = None
            timeline_events: list[dict[str, Any]] = []
            request_options = _request_options(thread.provider, thread.model_name, None)
            try:
                async for event in self.runtime.chat_stream(
                    thread.runtime_session_id,
                    message,
                    provider=thread.provider,
                    model=thread.model_name,
                ):
                    if event.get("type") == "done":
                        done_event = event
                        continue
                    event["thread_id"] = thread_id_enc
                    if event.get("type") == "text_delta":
                        if first_token_ms is None and event.get("delta", "").strip():
                            first_token_ms = int((time.monotonic() - start) * 1000)
                        full_text += event.get("delta", "")
                    if event.get("type") in {"thinking_delta", "tool_start", "tool_end"}:
                        _append_timeline_event(timeline_events, event)
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:
                error_event = {
                    "type": "error",
                    "thread_id": thread_id_enc,
                    "error": str(exc),
                }
                yield f"data: {json.dumps(error_event)}\n\n"
            latency_ms = int((time.monotonic() - start) * 1000)
            done = done_event or {}
            content = done.get("content") or full_text
            usage = _usage_with_ttft(done.get("usage"), first_token_ms)

            has_thinking = any(event.get("type") == "thinking" for event in timeline_events)
            if not has_thinking and done.get("thinking"):
                thinking_text = _thinking_text(
                    done.get("thinking"),
                    done.get("provider") or thread.provider,
                )
                if thinking_text:
                    timeline_events.append(
                        {
                            "type": "thinking",
                            "kind": _thinking_kind(done.get("provider") or thread.provider),
                            "content": thinking_text,
                            "thinking": done.get("thinking"),
                        }
                    )

            if content or timeline_events:
                async with self.db.session() as session:
                    thread_repo = ThreadRepo(session)
                    for event in timeline_events:
                        await _persist_timeline_event(thread_repo, thread_id, event)
                    if content:
                        await thread_repo.add_message(
                            thread_id,
                            role="assistant",
                            content=content,
                            latency_ms=latency_ms,
                            provider=done.get("provider") or thread.provider,
                            model=done.get("model") or thread.model_name,
                            usage_json=usage,
                            thinking_json=done.get("thinking"),
                            request_options_json=request_options,
                            output_delta_count=done.get("output_delta_count"),
                        )

            done_event = {
                "type": "thread_done",
                "thread_id": thread_id_enc,
                "latency_ms": latency_ms,
                "content": content,
                "provider": done.get("provider") or thread.provider,
                "model": done.get("model") or thread.model_name,
                "usage": usage,
                "thinking": done.get("thinking"),
                "output_delta_count": done.get("output_delta_count"),
            }
            yield f"data: {json.dumps(done_event)}\n\n"
            yield f"data: {json.dumps({'type': 'all_done'})}\n\n"

        return _stream()

    async def _prepare_multi_chat(
        self,
        session_id: int,
        user_id: int,
        message: str,
        models: list[ModelSelection],
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
                        title=f"{provider}/{model_name}"
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


def _request_options(
    provider: str,
    model_name: str,
    reasoning_effort: str | None,
) -> dict[str, str]:
    options = {"provider": provider, "model": model_name}
    if reasoning_effort:
        options["reasoning_effort"] = reasoning_effort
    return options


TOOL_NAME_MAX_LENGTH = 100
OUTPUT_PREVIEW_MAX_LENGTH = 500


def _bounded_text(value: str, max_length: int) -> str:
    return value if len(value) <= max_length else value[:max_length]


def _normalize_tool_name(raw_tool: Any) -> str:
    if not isinstance(raw_tool, str):
        return "tool"

    tool = raw_tool.strip()
    if not tool:
        return "tool"

    if "<tool_call>" in tool:
        tool = tool.split("<tool_call>", 1)[0]
    if tool.endswith(")") and "(" in tool:
        inner = tool.rsplit("(", 1)[1][:-1].strip()
        if inner:
            tool = inner
    if tool.endswith("_args"):
        tool = tool[: -len("_args")]
    if tool.startswith("_"):
        tool = tool[1:]

    return _bounded_text(tool or "tool", TOOL_NAME_MAX_LENGTH)


def _tool_output_preview(event: dict[str, Any], tool_name: str) -> str:
    preview = event.get("output_preview")
    if event.get("type") == "tool_end" and isinstance(preview, str) and preview:
        return _bounded_text(preview, OUTPUT_PREVIEW_MAX_LENGTH)
    return _bounded_text(f"{event.get('type')}:{tool_name}", OUTPUT_PREVIEW_MAX_LENGTH)


def _usage_with_ttft(usage: dict[str, Any] | None, ttft_ms: int | None) -> dict[str, Any] | None:
    if ttft_ms is None:
        return usage

    next_usage = dict(usage or {})
    perf = next_usage.get("perf")
    next_perf = dict(perf) if isinstance(perf, dict) else {}
    next_perf.setdefault("ttft_ms", ttft_ms)
    next_usage["perf"] = next_perf
    return next_usage


def _append_timeline_event(timeline: list[dict[str, Any]], event: dict[str, Any]) -> None:
    if event.get("type") == "thinking_delta":
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        kind = event.get("kind") if isinstance(event.get("kind"), str) else "reasoning"
        if timeline and timeline[-1].get("type") == "thinking" and timeline[-1].get("kind") == kind:
            timeline[-1]["content"] = f"{timeline[-1].get('content', '')}{delta}"
            timeline[-1]["thinking"] = {kind: timeline[-1]["content"]}
            return
        timeline.append(
            {
                "type": "thinking",
                "kind": kind,
                "content": delta,
                "thinking": {kind: delta},
            }
        )
        return

    if event.get("type") in {"tool_start", "tool_end"}:
        timeline.append(event)


async def _persist_timeline_event(
    thread_repo: ThreadRepo,
    thread_id: int,
    event: dict[str, Any],
) -> None:
    if event.get("type") == "thinking":
        await thread_repo.add_message(
            thread_id,
            role="thinking",
            content=str(event.get("content") or ""),
            thinking_json=event.get("thinking"),
        )
        return

    tool_name = _normalize_tool_name(event.get("tool"))
    is_start = event.get("type") == "tool_start"
    output_preview = _tool_output_preview(event, tool_name)
    await thread_repo.add_message(
        thread_id,
        role="tool",
        content=f"[{'calling' if is_start else 'finished'} {tool_name}]",
        tool_name=tool_name,
        tool_call_id=event.get("call_id"),
        tool_input=event.get("args") if is_start else None,
        output_preview=output_preview,
    )


def _thinking_kind(provider: str | None) -> str:
    return "summary" if provider == "openai" else "reasoning"


def _thinking_text(thinking: Any, provider: str | None) -> str:
    if isinstance(thinking, str):
        return thinking
    if not isinstance(thinking, dict):
        return ""

    preferred = _thinking_kind(provider)
    value = thinking.get(preferred)
    if isinstance(value, str) and value:
        return value

    fallback = thinking.get("reasoning") or thinking.get("summary")
    if isinstance(fallback, str):
        return fallback
    return ""
