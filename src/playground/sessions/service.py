from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime

from playground.db.connection import Database
from playground.db.models import Message, ModelThread
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


@dataclass(frozen=True)
class PreparedModelSelection:
    provider: str
    model_name: str
    reasoning_effort: str | None
    model_id: int | None
    thread: ModelThread | None


@dataclass(frozen=True)
class RegenerationPreparation:
    source_runtime_session_id: str
    prior_user_turns: int
    reasoning_effort: str | None


class PlaygroundError(Exception):
    status_code = 400

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class PlaygroundNotFoundError(PlaygroundError):
    status_code = 404


class ModelNotFoundError(PlaygroundError):
    status_code = 400


class MessageNotFoundError(PlaygroundError):
    status_code = 404


class ThreadChangedError(PlaygroundError):
    status_code = 409


class SystemPromptLockedError(PlaygroundError):
    status_code = 409


class RuntimeForkError(PlaygroundError):
    status_code = 502


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
        skills: list[str] | None = None,
        system_prompt_name: str | None = None,
        system_prompt_content: str | None = None,
    ) -> PlaygroundOut:
        if system_prompt_name is None or system_prompt_content is None:
            system_prompt_name, system_prompt_content = (
                await self._get_default_system_prompt()
            )
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            playground = await session_repo.create(
                user_id=user_id,
                title=title,
                tools=tools,
                skills=skills,
                system_prompt_name=system_prompt_name,
                system_prompt_content=system_prompt_content,
            )
            return PlaygroundOut(
                id=encode(playground.id),
                title=playground.title,
                tools=playground.tools_json,
                skills=playground.skills_json,
                system_prompt_name=playground.system_prompt_name,
                system_prompt_content=playground.system_prompt_content,
                created_at=playground.created_at,
            )

    async def _get_default_system_prompt(self) -> tuple[str, str]:
        payload = await self.runtime.list_prompts()
        prompts = payload.get("prompts", [])
        default = next(
            (prompt for prompt in prompts if prompt.get("name") == "default"),
            prompts[0] if prompts else None,
        )
        if not default or not default.get("content"):
            raise PlaygroundError("No default system prompt is available")
        return "Default", str(default["content"])

    async def _ensure_system_prompt_snapshot(
        self,
        session_id: int,
        user_id: int,
    ) -> None:
        async with self.db.session() as session:
            repo = SessionRepo(session)
            playground = await repo.get_if_owner(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")
            if playground.system_prompt_content is not None:
                return

        name, content = await self._get_default_system_prompt()
        async with self.db.session() as session:
            repo = SessionRepo(session)
            playground = await repo.get_if_owner_for_update(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")
            if playground.system_prompt_content is None:
                await repo.update_system_prompt(session_id, user_id, name, content)

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
                    skills=s.skills_json,
                    system_prompt_name=s.system_prompt_name,
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
                sorted_messages = sorted(thread.messages, key=lambda m: (m.created_at, m.id))
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
                        selected_skill=message.selected_skill,
                        turn_id=message.turn_id,
                        transcript_sequence=message.transcript_sequence,
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
                skills=playground.skills_json,
                system_prompt_name=playground.system_prompt_name,
                system_prompt_content=playground.system_prompt_content,
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
        skills: list[str] | None = None,
        update_skills: bool = False,
        system_prompt_name: str | None = None,
        system_prompt_content: str | None = None,
        update_system_prompt: bool = False,
    ) -> PlaygroundOut:
        session_id = _decode_id(encoded_id, "Playground not found")
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            playground = await session_repo.get_if_owner_for_update(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")
            if title is not None:
                playground = await session_repo.update_title(session_id, user_id, title)
            if update_tools:
                playground = await session_repo.update_tools(session_id, user_id, tools)
            if update_skills:
                playground = await session_repo.update_skills(session_id, user_id, skills)
            if update_system_prompt:
                if system_prompt_name is None or system_prompt_content is None:
                    raise PlaygroundError(
                        "System prompt name and content must be provided together"
                    )
                if playground.comparison_started_at is not None:
                    raise SystemPromptLockedError(
                        "System prompt cannot be changed after a comparison has started"
                    )
                playground = await session_repo.update_system_prompt(
                    session_id,
                    user_id,
                    system_prompt_name,
                    system_prompt_content,
                )
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")
            return PlaygroundOut(
                id=encode(playground.id),
                title=playground.title,
                tools=playground.tools_json,
                skills=playground.skills_json,
                system_prompt_name=playground.system_prompt_name,
                system_prompt_content=playground.system_prompt_content,
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
        skills: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        session_id = _decode_id(encoded_id, "Playground not found")
        await self._ensure_system_prompt_snapshot(session_id, user_id)
        threads = await self._prepare_multi_chat(
            session_id, user_id, message, models, tools, skills
        )

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

            try:
                async for event in fanout_chat(self.runtime, threads, message, tools, skills):
                    capture = captures.get(event.get("thread_id"))
                    if capture is None or event.get("type") == "thread_start":
                        yield _sse(event)
                        continue

                    observed = capture.observe(event)
                    if event.get("type") == "done":
                        yield _sse(capture.thread_done_event())
                    elif observed is not None:
                        yield _sse(observed)
            except (GeneratorExit, asyncio.CancelledError):
                for capture in captures.values():
                    capture.cancel()
                raise
            finally:
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
        skills: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        session_id = _decode_id(encoded_id, "Playground not found")
        thread_id = _decode_id(thread_encoded_id, "Thread not found")
        thread = await self._prepare_single_chat(session_id, thread_id, user_id, message)
        return self._stream_thread_chat(thread, message, tools, skills)

    async def stream_regenerated_chat(
        self,
        encoded_id: str,
        thread_encoded_id: str,
        message_id: int,
        user_id: int,
        message: str,
        tools: list[str] | None = None,
        skills: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        session_id = _decode_id(encoded_id, "Playground not found")
        thread_id = _decode_id(thread_encoded_id, "Thread not found")
        preparation = await self._prepare_regeneration(
            session_id,
            thread_id,
            message_id,
            user_id,
        )

        try:
            forked_runtime_session_id = await self.runtime.fork_session(
                preparation.source_runtime_session_id,
                preparation.prior_user_turns,
            )
        except Exception as exc:
            raise RuntimeForkError("Could not prepare message regeneration") from exc

        thread = await self._replace_regeneration_tail(
            session_id,
            thread_id,
            message_id,
            user_id,
            preparation,
            forked_runtime_session_id,
            message,
        )
        return self._stream_thread_chat(
            thread,
            message,
            tools,
            skills,
            reasoning_effort=preparation.reasoning_effort,
        )

    def _stream_thread_chat(
        self,
        thread: ModelThread,
        message: str,
        tools: list[str] | None,
        skills: list[str] | None,
        *,
        reasoning_effort: str | None = None,
    ) -> AsyncGenerator[str, None]:

        async def _stream() -> AsyncGenerator[str, None]:
            thread_id_enc = encode(thread.id)
            capture = ChatThreadCapture(
                ChatThreadInfo(
                    id=thread.id,
                    encoded_id=thread_id_enc,
                    provider=thread.provider,
                    model_name=thread.model_name,
                    request_options=_request_options(
                        thread.provider,
                        thread.model_name,
                        reasoning_effort,
                    ),
                )
            )
            yield _sse(capture.start_event())
            failed = False
            try:
                async for event in self.runtime.chat_stream(
                    thread.runtime_session_id,
                    message,
                    provider=thread.provider,
                    model=thread.model_name,
                    reasoning_effort=reasoning_effort,
                    tools=tools,
                    skills=skills,
                ):
                    event["thread_id"] = thread_id_enc
                    if event.get("type") == "error":
                        failed = True
                        capture.observe(event)
                        yield _sse(event)
                        break
                    observed = capture.observe(event)
                    if observed is not None:
                        yield _sse(observed)
            except Exception as exc:
                failed = True
                yield _sse(capture.error_event(exc))
            except (GeneratorExit, asyncio.CancelledError):
                capture.cancel()
                raise
            finally:
                async with self.db.session() as session:
                    thread_repo = ThreadRepo(session)
                    await _persist_captured_messages(thread_repo, thread.id, capture)

            if not failed:
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
        skills: list[str] | None = None,
    ) -> list[tuple[ModelThread, str | None]]:
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            thread_repo = ThreadRepo(session)
            model_repo = ModelRepo(session)

            playground = await session_repo.get_if_owner_for_update(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")
            playground.comparison_started_at = (
                playground.comparison_started_at or datetime.now(UTC)
            )
            prompt_content = playground.system_prompt_content
            await session.flush()

            prepared: list[PreparedModelSelection] = []
            for provider, model_name, reasoning_effort in models:
                model = await model_repo.get_by_provider_model(provider, model_name)
                if model is None:
                    raise ModelNotFoundError(f"Model not found: {provider}/{model_name}")

                thread = await thread_repo.get_by_session_and_model(
                    session_id,
                    provider,
                    model_name,
                )
                prepared.append(
                    PreparedModelSelection(
                        provider=provider,
                        model_name=model_name,
                        reasoning_effort=reasoning_effort,
                        model_id=model.id,
                        thread=thread,
                    )
                )

        runtime_session_ids: dict[tuple[str, str], str] = {}
        for item in prepared:
            if item.thread is not None:
                continue
            key = (item.provider, item.model_name)
            if key not in runtime_session_ids:
                runtime_session_ids[key] = await self.runtime.create_session(
                    title=f"{item.provider}/{item.model_name}",
                    tools=tools,
                    skills=skills,
                    system_prompt=prompt_content,
                )

        async with self.db.session() as session:
            thread_repo = ThreadRepo(session)
            threads: list[tuple[ModelThread, str | None]] = []

            for item in prepared:
                thread = item.thread
                if thread is None:
                    thread = await thread_repo.get_by_session_and_model(
                        session_id,
                        item.provider,
                        item.model_name,
                    )
                if thread is None:
                    thread = await thread_repo.create(
                        playground_session_id=session_id,
                        provider=item.provider,
                        model_name=item.model_name,
                        runtime_session_id=runtime_session_ids[(item.provider, item.model_name)],
                        model_id=item.model_id,
                    )
                threads.append((thread, item.reasoning_effort))

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

    async def _prepare_regeneration(
        self,
        session_id: int,
        thread_id: int,
        message_id: int,
        user_id: int,
    ) -> RegenerationPreparation:
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            thread_repo = ThreadRepo(session)

            playground = await session_repo.get_if_owner(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")

            thread = await thread_repo.get(thread_id)
            if thread is None or thread.playground_session_id != session_id:
                raise PlaygroundNotFoundError("Thread not found")

            messages = sorted(thread.messages, key=lambda item: (item.created_at, item.id))
            target_index = next(
                (index for index, item in enumerate(messages) if item.id == message_id),
                None,
            )
            if target_index is None or messages[target_index].role != "user":
                raise MessageNotFoundError("User message not found")

            target = messages[target_index]
            prior_user_turns = sum(item.role == "user" for item in messages[:target_index])
            return RegenerationPreparation(
                source_runtime_session_id=thread.runtime_session_id,
                prior_user_turns=prior_user_turns,
                reasoning_effort=_reasoning_effort(target),
            )

    async def _replace_regeneration_tail(
        self,
        session_id: int,
        thread_id: int,
        message_id: int,
        user_id: int,
        preparation: RegenerationPreparation,
        forked_runtime_session_id: str,
        message: str,
    ) -> ModelThread:
        async with self.db.session() as session:
            session_repo = SessionRepo(session)
            thread_repo = ThreadRepo(session)

            playground = await session_repo.get_if_owner(session_id, user_id)
            if playground is None:
                raise PlaygroundNotFoundError("Playground not found")

            thread = await thread_repo.get_for_update(thread_id)
            if thread is None or thread.playground_session_id != session_id:
                raise PlaygroundNotFoundError("Thread not found")
            if thread.runtime_session_id != preparation.source_runtime_session_id:
                raise ThreadChangedError("Thread changed; reload before regenerating")

            target = next((item for item in thread.messages if item.id == message_id), None)
            if target is None or target.role != "user":
                raise MessageNotFoundError("User message not found")

            return await thread_repo.replace_tail_with_user_message(
                thread,
                message_id,
                forked_runtime_session_id,
                message,
                _request_options(
                    thread.provider,
                    thread.model_name,
                    preparation.reasoning_effort,
                ),
            )


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _reasoning_effort(message: Message) -> str | None:
    options = message.request_options_json
    if not isinstance(options, dict):
        return None
    effort = options.get("reasoning_effort")
    return effort if isinstance(effort, str) and effort else None


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
        selected_skill=message.selected_skill,
        turn_id=message.turn_id,
        transcript_sequence=message.transcript_sequence,
    )
